"""Monitoreo de scoring por batch: performance, drift (PSI) y schema/calidad.

Umbrales y features se leen de config.yaml via config_loader.
"""
import logging
import numpy as np
import pandas as pd

from config_loader import get_monitoring_cfg

logger = logging.getLogger(__name__)

OK = "OK"
WARNING = "WARNING"
ALERT = "ALERT"
_SEVERITY = {OK: 0, WARNING: 1, ALERT: 2}

# Defaults (sobrescritos por apply_monitoring_config)
PSI_WARNING = 0.10
PSI_ALERT = 0.25
PERF_WARNING_RATIO = 1.25
PERF_ALERT_RATIO = 2.0
SCHEMA_WARNING_PCT = 0.0
SCHEMA_ALERT_PCT = 5.0
PSI_N_BINS = 10
PSI_EPSILON = 1e-6
PSI_PER_FEATURE: dict = {}
NUMERIC_RANGES: dict = {}
CATEGORICAL_DOMAINS: dict = {}
DRIFT_NUMERIC: list = []
DRIFT_CATEGORICAL: list = []
ENABLED_AXES: list = ['performance', 'drift', 'schema', 'prediction_drift']


def apply_monitoring_config():
    """Recarga umbrales desde config.yaml."""
    global PSI_WARNING, PSI_ALERT, PERF_WARNING_RATIO, PERF_ALERT_RATIO
    global SCHEMA_WARNING_PCT, SCHEMA_ALERT_PCT, PSI_N_BINS, PSI_EPSILON
    global PSI_PER_FEATURE

    m = get_monitoring_cfg()
    psi = m.get('psi', {})
    PSI_WARNING = float(psi.get('warning', 0.10))
    PSI_ALERT = float(psi.get('alert', 0.25))
    PSI_N_BINS = int(psi.get('n_bins', 10))
    PSI_EPSILON = float(psi.get('epsilon', 1e-6))
    PSI_PER_FEATURE = psi.get('per_feature') or {}

    perf = m.get('performance', {})
    PERF_WARNING_RATIO = float(perf.get('warning_ratio', 1.25))
    PERF_ALERT_RATIO = float(perf.get('alert_ratio', 2.0))

    schema = m.get('schema', {})
    SCHEMA_WARNING_PCT = float(schema.get('warning_pct', 0.0))
    SCHEMA_ALERT_PCT = float(schema.get('alert_pct', 5.0))
    NUMERIC_RANGES.clear()
    for k, v in (schema.get('numeric_ranges') or {}).items():
        NUMERIC_RANGES[k] = (float(v[0]), float(v[1]))
    CATEGORICAL_DOMAINS.clear()
    for k, v in (schema.get('categorical_domains') or {}).items():
        # YAML puede parsear yes/no como bool; normalizamos a strings del dataset
        domain = set()
        for x in v:
            if isinstance(x, bool):
                domain.add('yes' if x else 'no')
            else:
                domain.add(str(x))
        CATEGORICAL_DOMAINS[k] = domain

    drift = m.get('drift', {})
    DRIFT_NUMERIC[:] = list(drift.get('numeric_features', ['age', 'bmi', 'children']))
    DRIFT_CATEGORICAL[:] = list(drift.get('categorical_features', ['sex', 'smoker', 'region']))
    ENABLED_AXES[:] = list(m.get('enabled_axes', ['performance', 'drift', 'schema', 'prediction_drift']))


apply_monitoring_config()


def _psi_thresholds_for_feature(feat: str) -> tuple[float, float]:
    if feat in PSI_PER_FEATURE:
        pf = PSI_PER_FEATURE[feat]
        return float(pf.get('warning', PSI_WARNING)), float(pf.get('alert', PSI_ALERT))
    return PSI_WARNING, PSI_ALERT


def _status_from_psi(psi: float, warn: float, alert: float) -> str:
    if psi >= alert:
        return ALERT
    if psi >= warn:
        return WARNING
    return OK


def worst_status(*statuses):
    valid = [s for s in statuses if s is not None]
    if not valid:
        return OK
    return max(valid, key=lambda s: _SEVERITY[s])


def compute_psi(reference, current, n_bins=None, epsilon=None):
    n_bins = n_bins if n_bins is not None else PSI_N_BINS
    epsilon = epsilon if epsilon is not None else PSI_EPSILON

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    lo, hi = np.min(reference), np.max(reference)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0

    edges = np.linspace(lo, hi, n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_prop = ref_counts / max(ref_counts.sum(), 1)
    cur_prop = cur_counts / max(cur_counts.sum(), 1)

    ref_prop = np.clip(ref_prop, epsilon, None)
    cur_prop = np.clip(cur_prop, epsilon, None)

    psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
    return float(psi)


def compute_psi_categorical(reference, current, epsilon=None):
    epsilon = epsilon if epsilon is not None else PSI_EPSILON
    ref_series = pd.Series(reference).astype(str)
    cur_series = pd.Series(current).astype(str)
    categories = sorted(set(ref_series.unique()) | set(cur_series.unique()))

    ref_prop = ref_series.value_counts(normalize=True).reindex(categories, fill_value=0).values
    cur_prop = cur_series.value_counts(normalize=True).reindex(categories, fill_value=0).values

    ref_prop = np.clip(ref_prop, epsilon, None)
    cur_prop = np.clip(cur_prop, epsilon, None)

    psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
    return float(psi)


def compute_drift(reference_df, current_df):
    psi_by_feature = {}

    for feat in DRIFT_NUMERIC:
        if feat in reference_df.columns and feat in current_df.columns:
            psi_by_feature[feat] = compute_psi(
                reference_df[feat].dropna(), current_df[feat].dropna()
            )

    for feat in DRIFT_CATEGORICAL:
        if feat in reference_df.columns and feat in current_df.columns:
            psi_by_feature[feat] = compute_psi_categorical(
                reference_df[feat].dropna(), current_df[feat].dropna()
            )

    statuses = []
    drifted = []
    for feat, psi in psi_by_feature.items():
        warn, alert = _psi_thresholds_for_feature(feat)
        st = _status_from_psi(psi, warn, alert)
        statuses.append(st)
        if st != OK:
            drifted.append(feat)

    max_psi = max(psi_by_feature.values()) if psi_by_feature else 0.0
    status = worst_status(*statuses) if statuses else OK

    return {
        'psi_by_feature': psi_by_feature,
        'max_psi': max_psi,
        'drifted_features': drifted,
        'status': status,
    }


def check_schema(df):
    n = len(df)
    if n == 0:
        return {'violation_pct': 0.0, 'status': OK, 'details': {}}

    violated_mask = pd.Series(False, index=df.index)
    details = {}

    for feat, (lo, hi) in NUMERIC_RANGES.items():
        if feat in df.columns:
            col = pd.to_numeric(df[feat], errors='coerce')
            bad = (col < lo) | (col > hi) | col.isna()
            cnt = int(bad.sum())
            if cnt > 0:
                details[feat] = {'type': 'range', 'expected': [lo, hi], 'violations': cnt}
            violated_mask |= bad.fillna(True)

    for feat, domain in CATEGORICAL_DOMAINS.items():
        if feat in df.columns:
            bad = ~df[feat].astype(str).isin(domain)
            cnt = int(bad.sum())
            if cnt > 0:
                details[feat] = {'type': 'domain', 'expected': sorted(domain), 'violations': cnt}
            violated_mask |= bad

    violation_pct = 100.0 * violated_mask.sum() / n

    if violation_pct > SCHEMA_ALERT_PCT:
        status = ALERT
    elif violation_pct > SCHEMA_WARNING_PCT:
        status = WARNING
    else:
        status = OK

    return {'violation_pct': float(violation_pct), 'status': status, 'details': details}


def compute_performance(actual, predicted, baseline_rmse=None):
    if actual is None:
        return None

    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    rmse = float(root_mean_squared_error(actual, predicted))
    mae = float(mean_absolute_error(actual, predicted))
    r2 = float(r2_score(actual, predicted))
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)

    ratio = None
    status = OK
    if baseline_rmse:
        ratio = rmse / baseline_rmse
        if ratio > PERF_ALERT_RATIO:
            status = ALERT
        elif ratio > PERF_WARNING_RATIO:
            status = WARNING

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'rmse_ratio': ratio,
        'baseline_rmse': baseline_rmse,
        'status': status,
    }


def compute_prediction_drift(reference_pred, current_pred):
    if reference_pred is None or current_pred is None:
        return None

    reference_pred = np.asarray(reference_pred, dtype=float)
    current_pred = np.asarray(current_pred, dtype=float)
    if len(reference_pred) == 0 or len(current_pred) == 0:
        return None

    psi = compute_psi(reference_pred, current_pred)
    warn, alert = _psi_thresholds_for_feature('predicted_charges')
    status = _status_from_psi(psi, warn, alert)

    return {'psi': psi, 'status': status}


def monitor_batch(batch_name, features_df, predicted, actual, reference_df,
                  baseline_rmse=None, baseline_pred=None):
    apply_monitoring_config()

    schema = check_schema(features_df) if 'schema' in ENABLED_AXES else {
        'violation_pct': 0.0, 'status': OK, 'details': {},
    }
    drift = compute_drift(reference_df, features_df) if 'drift' in ENABLED_AXES else {
        'psi_by_feature': {}, 'max_psi': 0.0, 'drifted_features': [], 'status': OK,
    }
    performance = None
    if 'performance' in ENABLED_AXES:
        performance = compute_performance(actual, predicted, baseline_rmse)

    prediction_drift = None
    if 'prediction_drift' in ENABLED_AXES:
        prediction_drift = compute_prediction_drift(baseline_pred, predicted)

    perf_status = performance['status'] if performance else None
    pred_drift_status = prediction_drift['status'] if prediction_drift else None
    statuses = [s for s in (perf_status, drift['status'], schema['status'], pred_drift_status) if s is not None]
    overall = worst_status(*statuses) if statuses else OK

    return {
        'batch': batch_name,
        'n_rows': int(len(features_df)),
        'has_target': actual is not None,
        'status': overall,
        'performance': performance,
        'drift': drift,
        'prediction_drift': prediction_drift,
        'schema': schema,
    }


def snapshot_from_report(report: dict) -> dict:
    """Plano serializable para persistencia de snapshots online (Grafana)."""
    pred_drift = report.get('prediction_drift') or {}
    drift = report.get('drift') or {}
    schema = report.get('schema') or {}
    return {
        'n_rows': int(report.get('n_rows', 0)),
        'status': report.get('status'),
        'max_psi': float(drift.get('max_psi', 0.0)),
        'prediction_psi': float(pred_drift['psi']) if pred_drift.get('psi') is not None else None,
        'schema_violation_pct': float(schema.get('violation_pct', 0.0)),
        'drift_status': drift.get('status'),
        'prediction_drift_status': pred_drift.get('status'),
        'schema_status': schema.get('status'),
        'psi_by_feature': dict(drift.get('psi_by_feature') or {}),
    }


def evaluate_online_retrain_trigger(
    snap: dict,
    n_requests_done: int,
    *,
    min_samples_prediction: int = 700,
) -> dict | None:
    """Evalua si debe dispararse reentrenamiento automatico en modo online.

    - prediction_psi > umbral warning y n_requests_done > min_samples_prediction
    - cualquier PSI por feature >= umbral alert
    """
    apply_monitoring_config()

    psi_by_feature = snap.get('psi_by_feature') or {}
    for feat, psi in psi_by_feature.items():
        _, alert = _psi_thresholds_for_feature(feat)
        if float(psi) >= alert:
            return {
                'type': 'feature_psi_alert',
                'feature': feat,
                'feature_psi': float(psi),
                'alert_threshold': alert,
                'prediction_psi': snap.get('prediction_psi'),
                'reason': (
                    f"PSI({feat})={float(psi):.4f} >= alert={alert:.4f} "
                    f"(request {n_requests_done})"
                ),
            }

    pred_psi = snap.get('prediction_psi')
    if pred_psi is not None and n_requests_done > min_samples_prediction:
        warn, _ = _psi_thresholds_for_feature('predicted_charges')
        if float(pred_psi) >= warn:
            return {
                'type': 'prediction_psi_warning',
                'feature': None,
                'feature_psi': None,
                'prediction_psi': float(pred_psi),
                'warning_threshold': warn,
                'reason': (
                    f"PSI(predicted_charges)={float(pred_psi):.4f} >= warning={warn:.4f} "
                    f"y muestras={n_requests_done} > {min_samples_prediction}"
                ),
            }

    return None


def flatten_psi_rows(
    report: dict,
    environment: str,
    session_id: str,
    window_index: int,
    measured_at,
) -> list[dict]:
    """Filas largas (feature, psi) para online_monitoring_psi."""
    snap = snapshot_from_report(report)
    rows = []
    for feature, psi in snap['psi_by_feature'].items():
        rows.append({
            'environment': environment,
            'session_id': session_id,
            'window_index': window_index,
            'feature': feature,
            'psi': float(psi),
            'measured_at': measured_at,
        })
    if snap['prediction_psi'] is not None:
        rows.append({
            'environment': environment,
            'session_id': session_id,
            'window_index': window_index,
            'feature': 'predicted_charges',
            'psi': snap['prediction_psi'],
            'measured_at': measured_at,
        })
    return rows
