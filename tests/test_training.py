"""Unit tests for selected functions in training.py."""
import numpy as np
import pytest
from sklearn.compose import ColumnTransformer

from training import (
    _build_run_name,
    create_preprocessor,
    define_hyperparameter_grid,
    prepare_features_target,
    split_data,
)
from utils import transform_target


class TestBuildRunName:
    def test_default_prefix(self):
        name = _build_run_name(None)
        assert name.startswith('train_')
        assert len(name) == len('train_') + 15

    def test_feature_psi_alert(self):
        trigger = {'type': 'feature_psi_alert', 'feature': 'age'}
        name = _build_run_name(trigger)
        assert name.startswith('retrain_drift_feature_age_')

    def test_prediction_trigger(self):
        trigger = {'type': 'prediction_psi_warning'}
        name = _build_run_name(trigger)
        assert name.startswith('retrain_drift_prediction_')


class TestPrepareFeaturesTarget:
    def test_drops_id_and_created_at(self, sample_training_df):
        X, y_transformed, y_original = prepare_features_target(sample_training_df)
        assert 'id' not in X.columns
        assert 'created_at' not in X.columns
        assert 'charges' not in X.columns

    def test_adds_engineered_columns(self, sample_training_df):
        X, y_transformed, y_original = prepare_features_target(sample_training_df)
        for col in ('bmi_smoker', 'age_smoker', 'bmi_squared', 'age_squared', 'bmi_obese', 'age_senior'):
            assert col in X.columns

    def test_target_log_transform(self, sample_training_df):
        _, y_transformed, y_original = prepare_features_target(sample_training_df)
        expected = transform_target(sample_training_df['charges'], inverse=False)
        np.testing.assert_allclose(y_transformed, expected)


class TestSplitData:
    def test_split_sizes(self):
        import pandas as pd

        rows = []
        for i in range(100):
            rows.append({
                'age': 20 + (i % 50),
                'sex': 'male' if i % 2 else 'female',
                'bmi': 20.0 + (i % 20),
                'children': i % 4,
                'smoker': 'yes' if i % 5 == 0 else 'no',
                'region': 'southwest',
                'charges': 5000.0 + i * 100,
            })
        df = pd.DataFrame(rows)
        X, y_t, y_o = prepare_features_target(df)

        X_train, X_val, y_train, y_val, y_train_orig, y_val_orig = split_data(
            X, y_t, y_o, test_size=0.2, random_state=43,
        )
        assert len(X_train) == 80
        assert len(X_val) == 20
        assert len(y_train) == len(X_train)
        assert len(y_val) == len(X_val)
        assert len(y_train_orig) == len(X_train)
        assert len(y_val_orig) == len(X_val)


class TestDefineHyperparameterGrid:
    def test_returns_model_prefixed_keys(self):
        grid = define_hyperparameter_grid()
        assert len(grid) > 0
        assert all(k.startswith('model__') for k in grid)


class TestCreatePreprocessor:
    def test_returns_column_transformer(self):
        preprocessor = create_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)
        names = [name for name, _, _ in preprocessor.transformers]
        assert 'num' in names
        assert 'cat' in names
