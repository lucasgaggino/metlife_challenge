"""Pipeline de scoring sobre batches de produccion (data/prod/).

- Consume el mejor modelo registrado en MLflow (alias @champion); fallback a models/best_model.pkl.
- Procesa cada batch de data/prod/, genera predicciones y las persiste en Postgres.
- Calcula monitoreo por batch (performance + drift PSI + schema) con estado OK/WARNING/ALERT.
- Registra resultados en MLflow (un run por batch) y en reportes JSON/TXT en results/.
"""
import pandas as pd
import numpy as np
from sqlalchemy import text
import joblib
import logging
import os
import sys
import json
from datetime import datetime

# Evita el warning de MLflow al no encontrar git (no usamos git SHA en los runs)
os.environ.setdefault('GIT_PYTHON_REFRESH', 'quiet')
import mlflow
import mlflow.sklearn

from utils import get_db_engine, feature_engineering, transform_target
import monitoring
import grafana_utils
from config_loader import (
    get_environment,
    get_model_cfg,
    get_mlflow_cfg,
    get_scoring_cfg,
    get_prod_batches,
    log_config_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def _feature_columns():
    return list(get_scoring_cfg().get('feature_columns', ['age', 'sex', 'bmi', 'children', 'smoker', 'region']))


def _champion_alias():
    return get_model_cfg().get('champion_alias', 'champion')


def resolve_path(rel_path):
    """Resuelve una ruta relativa al cwd o a la ubicacion del script."""
    if os.path.exists(rel_path):
        return rel_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt = os.path.normpath(os.path.join(script_dir, '..', rel_path))
    return alt


def setup_mlflow():
    """Configura MLflow desde config.yaml + env overrides."""
    mlf = get_mlflow_cfg()
    mlflow.set_tracking_uri(mlf['tracking_uri'])
    mlflow.set_experiment(mlf['experiment_name'])
    logger.info(f"MLflow configurado: tracking={mlf['tracking_uri']}, experimento={mlf['experiment_name']}")


def load_champion_model():
    """Carga el modelo @champion del Model Registry; fallback a models/best_model.pkl."""
    model_name = get_mlflow_cfg()['model_name']
    uri = f"models:/{model_name}@{_champion_alias()}"
    try:
        logger.info(f"Cargando modelo champion desde MLflow: {uri}")
        model = mlflow.sklearn.load_model(uri)
        logger.info(f"Modelo champion cargado ({type(model)}).")
        return model, uri
    except Exception as e:
        logger.warning(f"No se pudo cargar el champion desde MLflow ({e}). Usando fallback local.")
        local_path = resolve_path('models/best_model.pkl')
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Modelo no encontrado en MLflow ni en {local_path}")
        model = joblib.load(local_path)
        logger.info(f"Modelo cargado desde fallback local: {local_path}")
        return model, local_path


def load_baseline_rmse():
    """Lee el RMSE de validacion del modelo champion desde la metadata local."""
    metadata_path = resolve_path('models/best_model_metadata.json')
    try:
        with open(metadata_path) as f:
            meta = json.load(f)
        rmse = meta.get('validation_metrics', {}).get('rmse')
        logger.info(f"Baseline RMSE (validacion) = {rmse}")
        return rmse
    except Exception as e:
        logger.warning(f"No se pudo leer baseline RMSE de metadata: {e}")
        return None


def load_reference_features(engine):
    """Carga las features del training_dataset como referencia para drift."""
    logger.info("Cargando features de referencia (training_dataset) para drift.")
    df = pd.read_sql(text("SELECT age, sex, bmi, children, smoker, region FROM training_dataset"), engine)
    logger.info(f"Referencia cargada: {df.shape[0]} filas.")
    return df


def load_prod_batch(batch):
    """Carga features (+ target si existe) de un batch de produccion.

    El target se parsea con decimal=',' (formato declarado de los archivos).
    """
    feats_path = resolve_path(batch['feats'])
    logger.info(f"[{batch['name']}] Cargando features desde {feats_path}")
    feats = pd.read_csv(feats_path)
    feats = feats[_feature_columns()].copy()

    actual = None
    if batch['target']:
        target_path = resolve_path(batch['target'])
        logger.info(f"[{batch['name']}] Cargando target desde {target_path} (coma decimal)")
        # IMPORTANTE: el target es una sola columna con coma decimal (ej '14700,80931').
        # Usamos sep=';' (inexistente) para no partir el campo por la coma, y luego
        # convertimos la coma decimal a punto. Asi NO se altera el valor: prod1 queda en
        # escala valida y prod2 queda x100 (anomalia que el monitoreo debe detectar).
        target = pd.read_csv(target_path, sep=';', header=0)
        actual = pd.to_numeric(
            target.iloc[:, 0].astype(str).str.replace(',', '.', regex=False),
            errors='coerce',
        ).values
        if len(actual) != len(feats):
            logger.warning(
                f"[{batch['name']}] Desalineacion feats={len(feats)} vs target={len(actual)}; "
                f"se trunca al minimo."
            )
            n = min(len(feats), len(actual))
            feats = feats.iloc[:n].reset_index(drop=True)
            actual = actual[:n]
    else:
        logger.info(f"[{batch['name']}] Sin archivo target (batch sin etiquetas).")

    return feats, actual


def score_batch(model, feats):
    """Aplica feature engineering, predice y revierte la transformacion log del target."""
    fe = feature_engineering(feats.copy(), is_training=False)
    pred_log = model.predict(fe)
    predictions = transform_target(pred_log, inverse=True)
    return np.asarray(predictions, dtype=float)


def build_predictions_df(batch_name, feats, predictions, actual):
    """Arma el DataFrame de predicciones para persistir en prod_predictions."""
    n = len(feats)
    if actual is not None:
        abs_err = np.abs(actual - predictions)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_err = np.abs((actual - predictions) / actual) * 100
        actual_col = actual
    else:
        abs_err = [None] * n
        pct_err = [None] * n
        actual_col = [None] * n

    return pd.DataFrame({
        'environment': get_environment(),
        'batch': batch_name,
        'row_id': np.arange(n),
        'age': feats['age'].values,
        'sex': feats['sex'].values,
        'bmi': feats['bmi'].values,
        'children': feats['children'].values,
        'smoker': feats['smoker'].values,
        'region': feats['region'].values,
        'actual_charges': actual_col,
        'predicted_charges': predictions,
        'absolute_error': abs_err,
        'percentage_error': pct_err,
        'prediction_time': datetime.now(),
    })


def save_prod_predictions(engine, predictions_df):
    """Persiste predicciones del batch en la tabla prod_predictions."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prod_predictions (
                id SERIAL PRIMARY KEY,
                environment VARCHAR(10) NOT NULL DEFAULT 'sandbox',
                batch VARCHAR(50) NOT NULL,
                row_id INTEGER NOT NULL,
                age INTEGER, sex VARCHAR(10), bmi FLOAT, children INTEGER,
                smoker VARCHAR(5), region VARCHAR(20),
                actual_charges FLOAT, predicted_charges FLOAT,
                absolute_error FLOAT, percentage_error FLOAT,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()
    predictions_df.to_sql('prod_predictions', engine, if_exists='append', index=False, method='multi')
    logger.info(f"{len(predictions_df)} predicciones guardadas en 'prod_predictions'.")


def save_monitoring_run(engine, report):
    """Persiste el resumen de monitoreo del batch en monitoring_runs."""
    perf = report.get('performance') or {}
    pred_drift = report.get('prediction_drift') or {}
    row = {
        'environment': get_environment(),
        'batch': report['batch'],
        'status': report['status'],
        'n_rows': report['n_rows'],
        'rmse': perf.get('rmse'),
        'mae': perf.get('mae'),
        'r2': perf.get('r2'),
        'mape': perf.get('mape'),
        'rmse_ratio': perf.get('rmse_ratio'),
        'max_psi': report['drift']['max_psi'],
        'prediction_psi': pred_drift.get('psi'),
        'schema_violation_pct': report['schema']['violation_pct'],
        'performance_status': perf.get('status'),
        'drift_status': report['drift']['status'],
        'prediction_drift_status': pred_drift.get('status'),
        'schema_status': report['schema']['status'],
        'has_target': report['has_target'],
        'run_time': datetime.now(),
    }
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS monitoring_runs (
                id SERIAL PRIMARY KEY,
                environment VARCHAR(10) NOT NULL DEFAULT 'sandbox',
                batch VARCHAR(50) NOT NULL, status VARCHAR(10) NOT NULL,
                n_rows INTEGER, rmse FLOAT, mae FLOAT, r2 FLOAT, mape FLOAT,
                rmse_ratio FLOAT, max_psi FLOAT, prediction_psi FLOAT,
                schema_violation_pct FLOAT,
                performance_status VARCHAR(10), drift_status VARCHAR(10),
                prediction_drift_status VARCHAR(10),
                schema_status VARCHAR(10), has_target BOOLEAN,
                run_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()
    pd.DataFrame([row]).to_sql('monitoring_runs', engine, if_exists='append', index=False)
    logger.info(f"[{report['batch']}] Resumen de monitoreo guardado en 'monitoring_runs'.")


def log_batch_to_mlflow(report):
    """Loguea metricas y estado del batch en un run de MLflow."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with mlflow.start_run(run_name=f"score_{report['batch']}_{ts}"):
        mlflow.set_tag('stage', 'scoring')
        mlflow.set_tag('batch', report['batch'])
        mlflow.set_tag('status', report['status'])
        mlflow.log_param('n_rows', report['n_rows'])
        mlflow.log_param('has_target', report['has_target'])

        perf = report.get('performance')
        if perf:
            mlflow.log_metric('rmse', perf['rmse'])
            mlflow.log_metric('mae', perf['mae'])
            mlflow.log_metric('r2', perf['r2'])
            mlflow.log_metric('mape', perf['mape'])
            if perf['rmse_ratio'] is not None:
                mlflow.log_metric('rmse_ratio', perf['rmse_ratio'])

        for feat, psi in report['drift']['psi_by_feature'].items():
            mlflow.log_metric(f'psi_{feat}', psi)
        mlflow.log_metric('max_psi', report['drift']['max_psi'])

        pred_drift = report.get('prediction_drift')
        if pred_drift:
            mlflow.log_metric('prediction_psi', pred_drift['psi'])
            mlflow.set_tag('prediction_drift_status', pred_drift['status'])

        mlflow.log_metric('schema_violation_pct', report['schema']['violation_pct'])

        # Reporte del batch como artefacto JSON
        tmp = f"/tmp/monitoring_{report['batch']}_{ts}.json"
        try:
            with open(tmp, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            mlflow.log_artifact(tmp, artifact_path='monitoring')
        except Exception as e:
            logger.warning(f"No se pudo loguear artefacto de monitoreo: {e}")


def write_consolidated_report(reports, output_dir='results'):
    """Escribe el reporte consolidado de monitoreo en JSON y TXT."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_path = os.path.join(output_dir, f"monitoring_report_{ts}.json")
    with open(json_path, 'w') as f:
        json.dump({'generated_at': ts, 'batches': reports}, f, indent=2, default=str)

    txt_path = os.path.join(output_dir, f"monitoring_report_{ts}.txt")
    with open(txt_path, 'w') as f:
        f.write("METLIFE INSURANCE - REPORTE DE MONITOREO DE PRODUCCION\n")
        f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        for r in reports:
            f.write(f"BATCH: {r['batch']}  ->  ESTADO: {r['status']}\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Filas: {r['n_rows']} | Target disponible: {r['has_target']}\n")

            perf = r.get('performance')
            if perf:
                ratio = f"{perf['rmse_ratio']:.2f}x" if perf['rmse_ratio'] is not None else "n/a"
                f.write(f"  Performance [{perf['status']}]: "
                        f"RMSE=${perf['rmse']:,.2f} (baseline x{ratio}), "
                        f"MAE=${perf['mae']:,.2f}, R2={perf['r2']:.4f}, MAPE={perf['mape']:.1f}%\n")
            else:
                f.write("  Performance: sin target (no evaluable)\n")

            drift = r['drift']
            f.write(f"  Drift [{drift['status']}]: max_PSI={drift['max_psi']:.4f}\n")
            for feat, psi in drift['psi_by_feature'].items():
                flag = " <== DRIFT" if psi >= monitoring.PSI_WARNING else ""
                f.write(f"      PSI {feat:10s}: {psi:.4f}{flag}\n")

            pred_drift = r.get('prediction_drift')
            if pred_drift:
                flag = " <== DRIFT" if pred_drift['psi'] >= monitoring.PSI_WARNING else ""
                f.write(f"  Prediction drift [{pred_drift['status']}]: "
                        f"PSI(predicted_charges)={pred_drift['psi']:.4f}{flag}\n")
            else:
                f.write("  Prediction drift: n/a\n")

            schema = r['schema']
            f.write(f"  Schema [{schema['status']}]: {schema['violation_pct']:.2f}% filas con violaciones\n")
            for feat, det in schema['details'].items():
                f.write(f"      {feat}: {det['type']} esperado={det['expected']} viol={det['violations']}\n")
            f.write("\n")

    logger.info(f"Reporte consolidado: {json_path} | {txt_path}")
    return json_path, txt_path


def main():
    """Pipeline principal de scoring sobre produccion + monitoreo."""
    try:
        logger.info("Iniciando pipeline de scoring sobre produccion.")
        log_config_summary()
        monitoring.apply_monitoring_config()
        setup_mlflow()

        engine = get_db_engine()
        model, model_uri = load_champion_model()
        baseline_rmse = load_baseline_rmse()
        reference_df = load_reference_features(engine)

        frac = float(get_scoring_cfg().get('reference_sample_frac', 1.0))
        if 0 < frac < 1.0 and len(reference_df) > 0:
            reference_df = reference_df.sample(frac=frac, random_state=42).reset_index(drop=True)
            logger.info(f"Referencia muestreada: frac={frac} -> {len(reference_df)} filas.")

        # Baseline de salida para prediction drift: lo que el modelo predice
        # "normalmente" sobre el set de referencia (training).
        baseline_pred = score_batch(model, reference_df)
        logger.info(f"Baseline de predicciones (referencia): "
                    f"min=${baseline_pred.min():,.2f}, max=${baseline_pred.max():,.2f}, "
                    f"mean=${baseline_pred.mean():,.2f}")

        # Persistir baseline para Grafana (no aborta el pipeline si falla)
        grafana_utils.save_baseline_predictions(engine, baseline_pred, model_uri)

        prod_batches = get_prod_batches()
        logger.info(f"Batches a procesar: {[b['name'] for b in prod_batches]}")

        reports = []
        for batch in prod_batches:
            logger.info("\n" + "=" * 70)
            logger.info(f"Procesando batch: {batch['name']}")
            logger.info("=" * 70)

            feats, actual = load_prod_batch(batch)
            predictions = score_batch(model, feats)

            logger.info(f"[{batch['name']}] Predicciones: "
                        f"min=${predictions.min():,.2f}, max=${predictions.max():,.2f}, "
                        f"mean=${predictions.mean():,.2f}")

            predictions_df = build_predictions_df(batch['name'], feats, predictions, actual)
            save_prod_predictions(engine, predictions_df)

            report = monitoring.monitor_batch(
                batch_name=batch['name'],
                features_df=feats,
                predicted=predictions,
                actual=actual,
                reference_df=reference_df,
                baseline_rmse=baseline_rmse,
                baseline_pred=baseline_pred,
            )
            save_monitoring_run(engine, report)
            log_batch_to_mlflow(report)
            reports.append(report)

            pred_drift = report.get('prediction_drift')
            logger.info(f"[{batch['name']}] ESTADO: {report['status']} "
                        f"(perf={report['performance']['status'] if report['performance'] else 'n/a'}, "
                        f"drift={report['drift']['status']}, "
                        f"pred_drift={pred_drift['status'] if pred_drift else 'n/a'}, "
                        f"schema={report['schema']['status']})")

        json_path, txt_path = write_consolidated_report(reports)

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE DE SCORING + MONITOREO COMPLETADO")
        logger.info("=" * 70)
        logger.info(f"Modelo usado: {model_uri}")
        for r in reports:
            logger.info(f"  {r['batch']:8s} -> {r['status']}")
        logger.info(f"Reporte: {json_path}")

        return True

    except Exception as e:
        logger.error(f"Error en pipeline de scoring: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
