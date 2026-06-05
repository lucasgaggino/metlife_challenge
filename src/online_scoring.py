"""Simulacion de inferencia online (requests sin target) sobre samples de dataset.csv.

- Tasa configurable (ej. 10 req/s, 1200 requests).
- Champion del entorno activo (sandbox/prod via ML_ENV).
- Persistencia en online_predictions + monitoreo agregado por sesion.
- Un run MLflow por sesion (stage=online).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from sqlalchemy import text

os.environ.setdefault('GIT_PYTHON_REFRESH', 'quiet')

import monitoring
import grafana_utils
import training
from config_loader import (
    get_environment,
    get_online_cfg,
    log_config_summary,
)
from scoring import (
    load_champion_model,
    load_reference_features,
    resolve_path,
    save_monitoring_run,
    score_batch,
    setup_mlflow,
)
from utils import get_db_engine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def bmi_anomaly_multiplier(
    request_seq: int,
    *,
    start_at: int = 500,
    duration: int = 1000,
    max_multiplier: float = 4.0,
) -> float:
    """Multiplicador lineal de BMI inyectado en el sampler.

    request_seq < start_at: 1.0
    [start_at, start_at + duration - 1]: rampa 1.0 -> max_multiplier
    request_seq >= start_at + duration: max_multiplier (anomalia sostenida)
    """
    if request_seq < start_at:
        return 1.0
    end = start_at + duration - 1
    if request_seq >= end:
        return max_multiplier
    if duration <= 1:
        return max_multiplier
    progress = (request_seq - start_at) / (duration - 1)
    return 1.0 + progress * (max_multiplier - 1.0)


def resolve_bmi_anomaly_settings(cfg: dict, cli_enabled: bool) -> dict | None:
    """None si la inyeccion esta desactivada; si no, parametros de la rampa."""
    block = cfg.get('bmi_anomaly') or {}
    enabled = cli_enabled or bool(block.get('enabled', False))
    if not enabled:
        return None
    return {
        'start_at': int(block.get('start_at_sample', 500)),
        'duration': int(block.get('duration_samples', 1000)),
        'max_multiplier': float(block.get('max_multiplier', 4.0)),
    }


def ensure_online_tables(engine):
    """Crea tablas online si no existen (idempotente)."""
    grafana_utils.create_online_monitoring_tables(engine)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS online_predictions (
                id SERIAL PRIMARY KEY,
                environment VARCHAR(10) NOT NULL DEFAULT 'sandbox',
                session_id VARCHAR(64) NOT NULL,
                request_seq INTEGER NOT NULL,
                age INTEGER,
                sex VARCHAR(10),
                bmi FLOAT,
                children INTEGER,
                smoker VARCHAR(5),
                region VARCHAR(20),
                predicted_charges FLOAT,
                model_uri VARCHAR(256),
                latency_ms FLOAT,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_online_predictions_env_session
            ON online_predictions (environment, session_id, request_seq)
        """))
        conn.commit()


def flush_predictions(engine, rows: list[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_sql(
        'online_predictions',
        engine,
        if_exists='append',
        index=False,
        method='multi',
    )
    logger.info(f"[online] {len(rows)} predicciones persistidas.")


def _monitoring_window_data(
    all_feats: list[pd.DataFrame],
    all_preds: list[float],
    max_samples: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Ultimas max_samples requests para monitoreo (ventana deslizante)."""
    if max_samples <= 0 or not all_feats:
        return pd.DataFrame(), np.asarray([], dtype=float)
    n = min(len(all_feats), max_samples)
    feats = pd.concat(all_feats[-n:], ignore_index=True)
    preds = np.asarray(all_preds[-n:], dtype=float)
    return feats, preds


def apply_online_monitoring_axes():
    """Restringe ejes de monitoreo para modo sin target."""
    axes = get_online_cfg().get('monitoring_axes')
    if axes:
        monitoring.ENABLED_AXES = list(axes)
        logger.info(f"Monitoreo online: axes={monitoring.ENABLED_AXES}")


def _log_session_metrics(
    session_id: str,
    model_uri: str,
    report: dict,
    duration_s: float,
    n_requests: int,
    mean_latency_ms: float,
    sample_df: pd.DataFrame | None = None,
):
    """Metricas finales dentro del run MLflow activo."""
    mlflow.set_tag('status', report['status'])
    mlflow.log_param('session_id', session_id)
    mlflow.log_param('model_uri', str(model_uri))
    mlflow.log_param('n_requests', n_requests)
    mlflow.log_param('has_target', False)

    mlflow.log_metric('duration_s', duration_s)
    mlflow.log_metric('throughput_rps', n_requests / duration_s if duration_s > 0 else 0)
    mlflow.log_metric('mean_latency_ms', mean_latency_ms)
    mlflow.log_metric('max_psi', report['drift']['max_psi'])
    mlflow.log_metric('schema_violation_pct', report['schema']['violation_pct'])

    pred_drift = report.get('prediction_drift')
    if pred_drift:
        mlflow.log_metric('prediction_psi', pred_drift['psi'])

    for feat, psi in report['drift']['psi_by_feature'].items():
        mlflow.log_metric(f'psi_{feat}', psi)

    if sample_df is not None and len(sample_df) > 0:
        tmp = f'/tmp/online_sample_{session_id}.csv'
        try:
            sample_df.to_csv(tmp, index=False)
            mlflow.log_artifact(tmp, artifact_path='online')
        except Exception as e:
            logger.warning(f"No se pudo loguear muestra online: {e}")

    tmp_json = f'/tmp/online_report_{session_id}.json'
    try:
        with open(tmp_json, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        mlflow.log_artifact(tmp_json, artifact_path='online')
    except Exception as e:
        logger.warning(f"No se pudo loguear reporte online: {e}")


def main(*, inject_bmi_anomaly: bool = False) -> bool:
    try:
        cfg = get_online_cfg()
        bmi_anomaly = resolve_bmi_anomaly_settings(cfg, inject_bmi_anomaly)
        log_config_summary()
        window_size = int(cfg.get('monitoring_window_size', 100))
        monitoring_max_samples = int(cfg.get('monitoring_max_samples', 1500))
        auto_retrain_cfg = cfg.get('auto_retrain') or {}
        auto_retrain_enabled = bool(auto_retrain_cfg.get('enabled', True))
        min_samples_pred_retrain = int(
            auto_retrain_cfg.get('min_samples_prediction', 700),
        )
        logger.info(
            "Modo online: n_samples=%s rate=%s/s flush_every=%s window=%s max_mon=%s "
            "bmi_anomaly=%s auto_retrain=%s",
            cfg.get('n_samples', 1200),
            cfg.get('rate_per_second', 10),
            cfg.get('flush_every', 50),
            window_size,
            monitoring_max_samples,
            bmi_anomaly is not None,
            auto_retrain_enabled,
        )
        if bmi_anomaly:
            logger.info(
                "Inyeccion BMI: seq>=%s rampa %s muestras -> x%s",
                bmi_anomaly['start_at'],
                bmi_anomaly['duration'],
                bmi_anomaly['max_multiplier'],
            )

        monitoring.apply_monitoring_config()
        apply_online_monitoring_axes()
        setup_mlflow()

        engine = get_db_engine()
        ensure_online_tables(engine)

        model, model_uri = load_champion_model()
        reference_df = load_reference_features(engine)

        csv_path = resolve_path(cfg.get('source_csv', 'data/dataset.csv'))
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV no encontrado: {csv_path}")

        feature_cols = list(cfg.get('feature_columns', [
            'age', 'sex', 'bmi', 'children', 'smoker', 'region',
        ]))
        source_df = pd.read_csv(csv_path)
        missing = set(feature_cols) - set(source_df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en {csv_path}: {missing}")

        prefix = cfg.get('session_prefix', 'online')
        session_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        n_samples = int(cfg.get('n_samples', 1200))
        rate = float(cfg.get('rate_per_second', 10))
        flush_every = int(cfg.get('flush_every', 50))
        interval = 1.0 / rate if rate > 0 else 0.0
        env = get_environment()

        baseline_pred = score_batch(model, reference_df)
        grafana_utils.save_baseline_predictions(engine, baseline_pred, model_uri)

        buffer: list[dict] = []
        all_feats: list[pd.DataFrame] = []
        all_preds: list[float] = []
        latencies: list[float] = []
        t0 = time.perf_counter()

        logger.info(f"Sesion online '{session_id}' iniciada ({n_samples} requests)...")
        sample_rows: list[dict] = []
        window_count = 0
        last_checkpoint_seq = -1
        retrain_done = False

        def maybe_checkpoint(seq: int, force: bool = False):
            nonlocal window_count, last_checkpoint_seq, model, model_uri
            nonlocal baseline_pred, retrain_done
            n_done = seq + 1
            is_window = n_done % window_size == 0
            is_last = seq == n_samples - 1
            if not force and not is_window and not is_last:
                return
            if last_checkpoint_seq == seq and not force:
                return
            if not all_feats:
                return

            window_feats, window_preds = _monitoring_window_data(
                all_feats, all_preds, monitoring_max_samples,
            )
            measured_at = buffer[-1]['prediction_time'] if buffer else datetime.now()
            report = monitoring.monitor_batch(
                batch_name=session_id,
                features_df=window_feats,
                predicted=window_preds,
                actual=None,
                reference_df=reference_df,
                baseline_rmse=None,
                baseline_pred=baseline_pred,
            )
            grafana_utils.save_online_monitoring_snapshot(
                engine,
                session_id,
                window_count,
                seq,
                report,
                measured_at=measured_at,
            )
            snap = monitoring.snapshot_from_report(report)
            if snap.get('prediction_psi') is not None:
                mlflow.log_metric(
                    'prediction_psi',
                    snap['prediction_psi'],
                    step=window_count,
                )
            mlflow.log_metric('max_psi', snap['max_psi'], step=window_count)
            for feat, psi in snap['psi_by_feature'].items():
                mlflow.log_metric(f'psi_{feat}', psi, step=window_count)

            if auto_retrain_enabled and not retrain_done:
                trigger = monitoring.evaluate_online_retrain_trigger(
                    snap,
                    n_done,
                    min_samples_prediction=min_samples_pred_retrain,
                )
                if trigger:
                    trigger['end_request_seq'] = seq
                    trigger['window_index'] = window_count
                    logger.warning(
                        "Alerta drift online -> reentrenamiento (1x por sesion): %s",
                        trigger['reason'],
                    )
                    mlflow.set_tag('retrain_triggered', 'true')
                    mlflow.log_param('retrain_trigger_reason', trigger['reason'])
                    mlflow.log_param('retrain_trigger_type', trigger['type'])
                    retrain_done = True

                    retrain_run_name = None
                    retrain_run_id = None
                    try:
                        train_result = training.run_training_pipeline(
                            engine=engine,
                            trigger=trigger,
                            nested=True,
                            online_session_id=session_id,
                        )
                        retrain_run_name = train_result.get('run_name')
                        retrain_run_id = train_result.get('run_id')
                        mlflow.log_param('retrain_mlflow_run_name', retrain_run_name)

                        model, model_uri = load_champion_model()
                        baseline_pred = score_batch(model, reference_df)
                        grafana_utils.save_baseline_predictions(
                            engine, baseline_pred, model_uri,
                        )
                        logger.info(
                            "Reentrenamiento completado: run=%s champion=%s",
                            retrain_run_name,
                            train_result.get('is_champion'),
                        )
                    except Exception as retrain_err:
                        logger.error(
                            "Reentrenamiento automatico fallo: %s",
                            retrain_err,
                            exc_info=True,
                        )
                        retrain_run_name = 'FAILED'

                    grafana_utils.save_online_retrain_alert(
                        engine,
                        session_id,
                        window_count,
                        seq,
                        trigger,
                        measured_at=measured_at,
                        retrain_run_name=retrain_run_name,
                        retrain_mlflow_run_id=retrain_run_id,
                    )

            window_count += 1
            last_checkpoint_seq = seq
            logger.info(
                f"  Checkpoint window={window_count - 1} @ request {n_done} "
                f"status={report['status']} pred_psi={snap.get('prediction_psi')}"
            )

        with mlflow.start_run(run_name=session_id):
            mlflow.set_tag('stage', 'online')
            mlflow.set_tag('mode', 'simulation')
            mlflow.set_tag('environment', env)
            if bmi_anomaly:
                mlflow.set_tag('bmi_anomaly', 'true')
                mlflow.log_param('bmi_anomaly_start_at', bmi_anomaly['start_at'])
                mlflow.log_param('bmi_anomaly_duration', bmi_anomaly['duration'])
                mlflow.log_param('bmi_anomaly_max_multiplier', bmi_anomaly['max_multiplier'])

            for seq in range(n_samples):
                row = source_df.sample(1, replace=True).iloc[0]
                feats = pd.DataFrame([row[feature_cols].to_dict()])
                bmi_val = float(row['bmi'])
                if bmi_anomaly:
                    mult = bmi_anomaly_multiplier(
                        seq,
                        start_at=bmi_anomaly['start_at'],
                        duration=bmi_anomaly['duration'],
                        max_multiplier=bmi_anomaly['max_multiplier'],
                    )
                    bmi_val *= mult
                    feats.loc[0, 'bmi'] = bmi_val

                t_req = time.perf_counter()
                pred = float(score_batch(model, feats)[0])
                latency_ms = (time.perf_counter() - t_req) * 1000.0

                row_dict = {
                    'environment': env,
                    'session_id': session_id,
                    'request_seq': seq,
                    'age': int(row['age']),
                    'sex': str(row['sex']),
                    'bmi': bmi_val,
                    'children': int(row['children']),
                    'smoker': str(row['smoker']),
                    'region': str(row['region']),
                    'predicted_charges': pred,
                    'model_uri': str(model_uri),
                    'latency_ms': latency_ms,
                    'prediction_time': datetime.now(),
                }
                buffer.append(row_dict)
                if len(sample_rows) < 100:
                    sample_rows.append(row_dict)
                all_feats.append(feats)
                all_preds.append(pred)
                latencies.append(latency_ms)

                if len(buffer) >= flush_every:
                    flush_predictions(engine, buffer)
                    buffer.clear()
                    mlflow.log_metric(
                        'cumulative_requests',
                        seq + 1,
                        step=(seq + 1) // flush_every,
                    )

                if interval > 0 and seq < n_samples - 1:
                    time.sleep(interval)

                maybe_checkpoint(seq)

                if (seq + 1) % max(1, n_samples // 10) == 0:
                    logger.info(f"  Progreso: {seq + 1}/{n_samples} requests")

            flush_predictions(engine, buffer)
            duration_s = time.perf_counter() - t0
            mean_latency = float(np.mean(latencies)) if latencies else 0.0

            session_feats, session_preds = _monitoring_window_data(
                all_feats, all_preds, monitoring_max_samples,
            )

            report = monitoring.monitor_batch(
                batch_name=session_id,
                features_df=session_feats,
                predicted=session_preds,
                actual=None,
                reference_df=reference_df,
                baseline_rmse=None,
                baseline_pred=baseline_pred,
            )
            save_monitoring_run(engine, report)
            _log_session_metrics(
                session_id,
                model_uri,
                report,
                duration_s,
                n_samples,
                mean_latency,
                sample_df=pd.DataFrame(sample_rows) if sample_rows else None,
            )

        logger.info(
            f"Sesion '{session_id}' completada en {duration_s:.1f}s | "
            f"status={report['status']} | throughput={n_samples / duration_s:.2f} req/s"
        )
        return True

    except Exception as e:
        logger.error(f"Error en modo online: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulacion de inferencia online (requests sin target).',
    )
    parser.add_argument(
        '--bmi-anomaly',
        action='store_true',
        help=(
            'Inyecta drift en BMI: desde request_seq 500, rampa lineal 1x->4x '
            'en 1000 muestras (override de online.bmi_anomaly.enabled).'
        ),
    )
    args = parser.parse_args()
    ok = main(inject_bmi_anomaly=args.bmi_anomaly)
    sys.exit(0 if ok else 1)
