"""Persistencia auxiliar para los dashboards de Grafana.

Centraliza TODA la logica de escritura en Postgres que alimenta a Grafana, de
modo que `db_setup.py`, `training.py` y `scoring.py` solo importan y llaman a
estas funciones (cambios minimos en esos archivos).

Reglas de diseno:
- Cada funcion publica envuelve su cuerpo en try/except y solo loguea un
  warning ante un fallo: la persistencia para Grafana NUNCA debe abortar el
  pipeline principal (training/scoring).
- Las tablas se crean con CREATE TABLE IF NOT EXISTS y NO se dropean en el
  fresh start de db_setup, para acumular historial multi-run.
"""
import json
import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from config_loader import get_environment
import monitoring

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
def create_grafana_tables(engine):
    """Crea (si no existen) las tablas que alimentan los dashboards de Grafana.

    - training_runs: una fila por run de entrenamiento (metricas, dataset, tiempos).
    - training_feature_importance: importancia de features por run.
    - baseline_predictions: distribucion de predicciones del champion sobre training.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id SERIAL PRIMARY KEY,
                    environment VARCHAR(10) NOT NULL DEFAULT 'sandbox',
                    run_name VARCHAR(64) NOT NULL,
                    mlflow_run_id VARCHAR(64),
                    model_name VARCHAR(128),
                    model_version INTEGER,
                    is_champion BOOLEAN DEFAULT FALSE,
                    dataset_name VARCHAR(128),
                    dataset_rows INTEGER,
                    test_size FLOAT,
                    cv_folds INTEGER,
                    n_iter INTEGER,
                    duration_seconds FLOAT,
                    train_rmse FLOAT, train_mae FLOAT, train_r2 FLOAT,
                    train_adj_r2 FLOAT, train_mape FLOAT,
                    val_rmse FLOAT, val_mae FLOAT, val_r2 FLOAT,
                    val_adj_r2 FLOAT, val_mape FLOAT,
                    overfitting_score FLOAT,
                    cv_best_rmse FLOAT,
                    best_params JSONB,
                    started_at TIMESTAMP,
                    finished_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS training_feature_importance (
                    id SERIAL PRIMARY KEY,
                    environment VARCHAR(10) NOT NULL DEFAULT 'sandbox',
                    run_name VARCHAR(64) NOT NULL,
                    feature VARCHAR(64) NOT NULL,
                    importance FLOAT
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS baseline_predictions (
                    id SERIAL PRIMARY KEY,
                    environment VARCHAR(10) NOT NULL DEFAULT 'sandbox',
                    model_uri VARCHAR(256),
                    row_id INTEGER,
                    predicted_charges FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
        from db_setup import migrate_environment_columns
        migrate_environment_columns(engine)
        logger.info("Tablas de Grafana verificadas/creadas (training_runs, "
                    "training_feature_importance, baseline_predictions).")
    except Exception as e:
        logger.warning(f"No se pudieron crear las tablas de Grafana: {e}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _extract_feature_importances(best_model):
    """Devuelve lista de (feature, importance) a partir del pipeline entrenado.

    El pipeline es Pipeline([('preprocessor', ColumnTransformer), ('model', XGB)]).
    Se limpia el prefijo de ColumnTransformer ('num__'/'cat__') para legibilidad.
    """
    preprocessor = best_model.named_steps['preprocessor']
    model = best_model.named_steps['model']
    names = list(preprocessor.get_feature_names_out())
    importances = np.asarray(model.feature_importances_, dtype=float)

    pairs = []
    for name, imp in zip(names, importances):
        clean = name.split('__', 1)[-1]
        pairs.append((clean, float(imp)))
    return pairs


def save_training_run(engine, run_name, mlflow_run_id, model_version,
                      is_champion, df, metrics, random_search, best_model,
                      duration_seconds, started_at=None, finished_at=None,
                      dataset_name='training_dataset'):
    """Persiste un run de entrenamiento + importancias de features.

    Args:
        engine: SQLAlchemy engine.
        run_name: nombre del run (igual al de MLflow, ej. train_YYYYMMDD_HHMMSS).
        mlflow_run_id: run_id de MLflow.
        model_version: ModelVersion registrado (o None).
        is_champion: bool, si este run quedo como @champion.
        df: DataFrame de training (para dataset_rows).
        metrics: dict con 'train', 'validation' y 'overfitting_score'.
        random_search: objeto RandomizedSearchCV ajustado.
        best_model: pipeline entrenado (para feature importances).
        duration_seconds: duracion del run en segundos.
    """
    try:
        train = metrics.get('train', {})
        val = metrics.get('validation', {})

        best_params = {
            k.replace('model__', ''): v
            for k, v in random_search.best_params_.items()
        }

        model_version_num = None
        model_name = None
        if model_version is not None:
            try:
                model_version_num = int(model_version.version)
                model_name = model_version.name
            except Exception:
                pass

        env = get_environment()
        row = {
            'environment': env,
            'run_name': run_name,
            'mlflow_run_id': mlflow_run_id,
            'model_name': model_name,
            'model_version': model_version_num,
            'is_champion': bool(is_champion),
            'dataset_name': dataset_name,
            'dataset_rows': int(len(df)),
            'test_size': 0.2,
            'cv_folds': int(getattr(random_search, 'cv', None) or 0) or None,
            'n_iter': int(getattr(random_search, 'n_iter', None) or 0) or None,
            'duration_seconds': float(duration_seconds) if duration_seconds is not None else None,
            'train_rmse': train.get('rmse'),
            'train_mae': train.get('mae'),
            'train_r2': train.get('r2'),
            'train_adj_r2': train.get('adj_r2'),
            'train_mape': train.get('mape'),
            'val_rmse': val.get('rmse'),
            'val_mae': val.get('mae'),
            'val_r2': val.get('r2'),
            'val_adj_r2': val.get('adj_r2'),
            'val_mape': val.get('mape'),
            'overfitting_score': metrics.get('overfitting_score'),
            'cv_best_rmse': float(-random_search.best_score_) if hasattr(random_search, 'best_score_') else None,
            'best_params': json.dumps(best_params, default=str),
            'started_at': started_at,
            'finished_at': finished_at,
        }

        insert_sql = text("""
            INSERT INTO training_runs (
                environment, run_name, mlflow_run_id, model_name, model_version, is_champion,
                dataset_name, dataset_rows, test_size, cv_folds, n_iter,
                duration_seconds,
                train_rmse, train_mae, train_r2, train_adj_r2, train_mape,
                val_rmse, val_mae, val_r2, val_adj_r2, val_mape,
                overfitting_score, cv_best_rmse, best_params, started_at, finished_at
            ) VALUES (
                :environment, :run_name, :mlflow_run_id, :model_name, :model_version, :is_champion,
                :dataset_name, :dataset_rows, :test_size, :cv_folds, :n_iter,
                :duration_seconds,
                :train_rmse, :train_mae, :train_r2, :train_adj_r2, :train_mape,
                :val_rmse, :val_mae, :val_r2, :val_adj_r2, :val_mape,
                :overfitting_score, :cv_best_rmse, CAST(:best_params AS JSONB),
                :started_at, :finished_at
            )
            ON CONFLICT (environment, run_name) DO UPDATE SET
                is_champion = EXCLUDED.is_champion,
                model_version = EXCLUDED.model_version
        """)

        with engine.connect() as conn:
            conn.execute(insert_sql, row)
            conn.commit()
        logger.info(f"[grafana] Run '{run_name}' guardado en training_runs.")

        _save_feature_importances(engine, run_name, best_model)
    except Exception as e:
        logger.warning(f"[grafana] No se pudo guardar training_run '{run_name}': {e}")


def _save_feature_importances(engine, run_name, best_model):
    """Persiste las importancias de features del run (idempotente por run_name)."""
    try:
        pairs = _extract_feature_importances(best_model)
        if not pairs:
            return
        env = get_environment()
        fi_df = pd.DataFrame(pairs, columns=['feature', 'importance'])
        fi_df.insert(0, 'environment', env)
        fi_df.insert(1, 'run_name', run_name)
        with engine.connect() as conn:
            conn.execute(
                text(
                    "DELETE FROM training_feature_importance "
                    "WHERE environment = :env AND run_name = :rn"
                ),
                {'env': env, 'rn': run_name},
            )
            conn.commit()
        fi_df.to_sql('training_feature_importance', engine, if_exists='append', index=False)
        logger.info(f"[grafana] {len(fi_df)} importancias guardadas para '{run_name}'.")
    except Exception as e:
        logger.warning(f"[grafana] No se pudieron guardar importancias de '{run_name}': {e}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def save_baseline_predictions(engine, baseline_pred, model_uri):
    """Refresca baseline_predictions con las predicciones del champion sobre training.

    Se usa como baseline en el dashboard de produccion para comparar la
    distribucion de predicciones de cada batch contra la 'normal'.
    """
    try:
        env = get_environment()
        baseline_pred = np.asarray(baseline_pred, dtype=float).ravel()
        df = pd.DataFrame({
            'environment': env,
            'model_uri': str(model_uri),
            'row_id': np.arange(len(baseline_pred)),
            'predicted_charges': baseline_pred,
        })
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS baseline_predictions (
                    id SERIAL PRIMARY KEY,
                    environment VARCHAR(10) NOT NULL DEFAULT 'sandbox',
                    model_uri VARCHAR(256),
                    row_id INTEGER,
                    predicted_charges FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(
                text("DELETE FROM baseline_predictions WHERE environment = :env"),
                {'env': env},
            )
            conn.commit()
        df.to_sql('baseline_predictions', engine, if_exists='append', index=False, method='multi')
        logger.info(f"[grafana] {len(df)} baseline_predictions refrescadas (model={model_uri}).")
    except Exception as e:
        logger.warning(f"[grafana] No se pudieron guardar baseline_predictions: {e}")


# ---------------------------------------------------------------------------
# Online monitoring (time series)
# ---------------------------------------------------------------------------
def create_online_monitoring_tables(engine):
    """Crea tablas de snapshots y PSI por ventana para Grafana online."""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS online_monitoring_snapshots (
                    id SERIAL PRIMARY KEY,
                    environment VARCHAR(10) NOT NULL DEFAULT 'sandbox',
                    session_id VARCHAR(64) NOT NULL,
                    window_index INTEGER NOT NULL,
                    end_request_seq INTEGER NOT NULL,
                    n_rows INTEGER NOT NULL,
                    measured_at TIMESTAMP NOT NULL,
                    max_psi FLOAT,
                    prediction_psi FLOAT,
                    schema_violation_pct FLOAT,
                    status VARCHAR(10),
                    drift_status VARCHAR(10),
                    prediction_drift_status VARCHAR(10),
                    schema_status VARCHAR(10)
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_online_mon_snap_env_session
                ON online_monitoring_snapshots (environment, session_id, measured_at)
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS online_monitoring_psi (
                    id SERIAL PRIMARY KEY,
                    environment VARCHAR(10) NOT NULL DEFAULT 'sandbox',
                    session_id VARCHAR(64) NOT NULL,
                    window_index INTEGER NOT NULL,
                    feature VARCHAR(64) NOT NULL,
                    psi FLOAT NOT NULL,
                    measured_at TIMESTAMP NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_online_mon_psi_env_session
                ON online_monitoring_psi (environment, session_id, feature, measured_at)
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS online_retrain_alerts (
                    id SERIAL PRIMARY KEY,
                    environment VARCHAR(10) NOT NULL DEFAULT 'sandbox',
                    session_id VARCHAR(64) NOT NULL,
                    window_index INTEGER NOT NULL,
                    end_request_seq INTEGER NOT NULL,
                    measured_at TIMESTAMP NOT NULL,
                    alert_type VARCHAR(48) NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    prediction_psi FLOAT,
                    feature_name VARCHAR(64),
                    feature_psi FLOAT,
                    retrain_run_name VARCHAR(128),
                    retrain_mlflow_run_id VARCHAR(64)
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_online_retrain_alerts_session
                ON online_retrain_alerts (environment, session_id, measured_at)
            """))
            conn.commit()
        logger.info(
            "[grafana] Tablas online_monitoring_* y online_retrain_alerts verificadas."
        )
    except Exception as e:
        logger.warning(f"[grafana] No se pudieron crear tablas de monitoreo online: {e}")


def save_online_monitoring_snapshot(
    engine,
    session_id: str,
    window_index: int,
    end_request_seq: int,
    report: dict,
    measured_at=None,
):
    """Persiste snapshot agregado + PSI por feature (ventana online)."""
    try:
        env = get_environment()
        if measured_at is None:
            measured_at = pd.Timestamp.utcnow()
        snap = monitoring.snapshot_from_report(report)
        row = {
            'environment': env,
            'session_id': session_id,
            'window_index': int(window_index),
            'end_request_seq': int(end_request_seq),
            'n_rows': snap['n_rows'],
            'measured_at': measured_at,
            'max_psi': snap['max_psi'],
            'prediction_psi': snap['prediction_psi'],
            'schema_violation_pct': snap['schema_violation_pct'],
            'status': snap['status'],
            'drift_status': snap['drift_status'],
            'prediction_drift_status': snap['prediction_drift_status'],
            'schema_status': snap['schema_status'],
        }
        create_online_monitoring_tables(engine)
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO online_monitoring_snapshots (
                        environment, session_id, window_index, end_request_seq,
                        n_rows, measured_at, max_psi, prediction_psi,
                        schema_violation_pct, status, drift_status,
                        prediction_drift_status, schema_status
                    ) VALUES (
                        :environment, :session_id, :window_index, :end_request_seq,
                        :n_rows, :measured_at, :max_psi, :prediction_psi,
                        :schema_violation_pct, :status, :drift_status,
                        :prediction_drift_status, :schema_status
                    )
                """),
                row,
            )
            conn.commit()

        psi_rows = monitoring.flatten_psi_rows(
            report, env, session_id, window_index, measured_at,
        )
        if psi_rows:
            pd.DataFrame(psi_rows).to_sql(
                'online_monitoring_psi',
                engine,
                if_exists='append',
                index=False,
                method='multi',
            )
        logger.info(
            f"[grafana] Snapshot online window={window_index} session={session_id} "
            f"(n_rows={snap['n_rows']}, pred_psi={snap['prediction_psi']})"
        )
    except Exception as e:
        logger.warning(
            f"[grafana] No se pudo guardar snapshot online "
            f"session={session_id} window={window_index}: {e}"
        )


def save_online_retrain_alert(
    engine,
    session_id: str,
    window_index: int,
    end_request_seq: int,
    trigger: dict,
    measured_at=None,
    retrain_run_name: str | None = None,
    retrain_mlflow_run_id: str | None = None,
):
    """Persiste alerta de reentrenamiento para marcadores en Grafana."""
    try:
        env = get_environment()
        if measured_at is None:
            measured_at = pd.Timestamp.utcnow()
        create_online_monitoring_tables(engine)
        row = {
            'environment': env,
            'session_id': session_id,
            'window_index': int(window_index),
            'end_request_seq': int(end_request_seq),
            'measured_at': measured_at,
            'alert_type': trigger['type'],
            'trigger_reason': trigger['reason'],
            'prediction_psi': trigger.get('prediction_psi'),
            'feature_name': trigger.get('feature'),
            'feature_psi': trigger.get('feature_psi'),
            'retrain_run_name': retrain_run_name,
            'retrain_mlflow_run_id': retrain_mlflow_run_id,
        }
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO online_retrain_alerts (
                        environment, session_id, window_index, end_request_seq,
                        measured_at, alert_type, trigger_reason, prediction_psi,
                        feature_name, feature_psi, retrain_run_name, retrain_mlflow_run_id
                    ) VALUES (
                        :environment, :session_id, :window_index, :end_request_seq,
                        :measured_at, :alert_type, :trigger_reason, :prediction_psi,
                        :feature_name, :feature_psi, :retrain_run_name, :retrain_mlflow_run_id
                    )
                """),
                row,
            )
            conn.commit()
        logger.info(
            f"[grafana] Alerta retrain online session={session_id} "
            f"type={trigger['type']} req={end_request_seq}"
        )
    except Exception as e:
        logger.warning(
            f"[grafana] No se pudo guardar alerta retrain session={session_id}: {e}"
        )
