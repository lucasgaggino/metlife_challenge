"""Unit tests for pure functions in monitoring.py."""
import numpy as np
import pandas as pd
import pytest

import monitoring
from monitoring import (
    ALERT,
    OK,
    WARNING,
    check_schema,
    compute_performance,
    compute_prediction_drift,
    compute_psi,
    compute_psi_categorical,
    evaluate_online_retrain_trigger,
    snapshot_from_report,
    worst_status,
)


class TestComputePsi:
    def test_identical_distributions_near_zero(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        assert compute_psi(ref, ref.copy()) == pytest.approx(0.0, abs=1e-6)

    def test_shifted_distribution_positive(self):
        ref = np.linspace(0, 1, 100)
        cur = np.linspace(0.5, 1.5, 100)
        assert compute_psi(ref, cur) > 0.0

    def test_constant_reference_returns_zero(self):
        ref = np.array([5.0, 5.0, 5.0])
        cur = np.array([1.0, 9.0, 5.0])
        assert compute_psi(ref, cur) == 0.0


class TestComputePsiCategorical:
    def test_same_categories_near_zero(self):
        ref = ['a', 'b', 'a', 'b', 'a']
        assert compute_psi_categorical(ref, ref) == pytest.approx(0.0, abs=1e-6)

    def test_new_category_positive(self):
        ref = ['a', 'a', 'b', 'b']
        cur = ['a', 'c', 'c', 'c']
        assert compute_psi_categorical(ref, cur) > 0.0


class TestWorstStatus:
    def test_ok_and_warning(self):
        assert worst_status(OK, WARNING) == WARNING

    def test_warning_and_alert(self):
        assert worst_status(WARNING, ALERT) == ALERT

    def test_empty_returns_ok(self):
        assert worst_status() == OK


class TestPredictionDriftStatus:
    def test_identical_predictions_ok(self):
        preds = np.array([1000.0, 2000.0, 3000.0] * 50)
        result = compute_prediction_drift(preds, preds.copy())
        assert result['status'] == OK

    def test_heavily_shifted_predictions_alert(self):
        ref = np.linspace(1000, 5000, 200)
        cur = np.linspace(20000, 80000, 200)
        result = compute_prediction_drift(ref, cur)
        assert result['status'] == ALERT


class TestCheckSchema:
    def test_valid_rows_ok(self, sample_features_df, schema_test_config):
        result = check_schema(sample_features_df)
        assert result['status'] == OK
        assert result['violation_pct'] == 0.0

    def test_out_of_range_numeric_warning(self, schema_test_config):
        df = pd.DataFrame({
            'age': [25] * 99 + [10],
            'bmi': [22.0] * 100,
        })
        result = check_schema(df)
        assert result['status'] == WARNING
        assert result['violation_pct'] == pytest.approx(1.0)
        assert 'age' in result['details']

    def test_invalid_category_warning(self, schema_test_config):
        df = pd.DataFrame({'sex': ['male'] * 99 + ['invalid']})
        result = check_schema(df)
        assert result['status'] == WARNING
        assert 'sex' in result['details']


class TestComputePerformance:
    def test_perfect_predictions(self):
        actual = np.array([100.0, 200.0, 300.0])
        result = compute_performance(actual, actual.copy())
        assert result['rmse'] == pytest.approx(0.0)
        assert result['status'] == OK

    def test_rmse_ratio_alert(self):
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([300.0, 500.0, 700.0])
        baseline_rmse = 50.0
        result = compute_performance(actual, predicted, baseline_rmse=baseline_rmse)
        assert result['rmse_ratio'] > monitoring.PERF_ALERT_RATIO
        assert result['status'] == ALERT


class TestSnapshotFromReport:
    def test_extracts_expected_keys(self):
        report = {
            'n_rows': 10,
            'status': WARNING,
            'drift': {'max_psi': 0.15, 'status': WARNING, 'psi_by_feature': {'age': 0.15}},
            'prediction_drift': {'psi': 0.12, 'status': OK},
            'schema': {'violation_pct': 1.0, 'status': OK},
        }
        snap = snapshot_from_report(report)
        assert snap['n_rows'] == 10
        assert snap['max_psi'] == pytest.approx(0.15)
        assert snap['prediction_psi'] == pytest.approx(0.12)
        assert snap['schema_violation_pct'] == pytest.approx(1.0)
        assert snap['psi_by_feature'] == {'age': 0.15}


class TestEvaluateOnlineRetrainTrigger:
    def test_feature_psi_alert_triggers(self):
        snap = {'psi_by_feature': {'age': 0.30}, 'prediction_psi': 0.05}
        result = evaluate_online_retrain_trigger(snap, n_requests_done=1000)
        assert result is not None
        assert result['type'] == 'feature_psi_alert'
        assert result['feature'] == 'age'

    def test_low_sample_count_no_prediction_trigger(self):
        snap = {'psi_by_feature': {}, 'prediction_psi': 0.20}
        result = evaluate_online_retrain_trigger(
            snap, n_requests_done=100, min_samples_prediction=700,
        )
        assert result is None

    def test_no_trigger_when_psi_low(self):
        snap = {'psi_by_feature': {'age': 0.05}, 'prediction_psi': 0.02}
        result = evaluate_online_retrain_trigger(snap, n_requests_done=1000)
        assert result is None
