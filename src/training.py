import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error,mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import logging
import os
import sys
import json
import time
from datetime import datetime
from utils import get_db_engine, feature_engineering, transform_target
import grafana_utils
from config_loader import (
    get_config, get_training_cfg, get_model_cfg, get_paths, get_mlflow_cfg,
    get_promotion_cfg, numerical_features_for_training, categorical_features_for_training,
    resolve_promotion_metric_value, is_promotion_better, log_config_summary,
)
import shutil
# Evita el warning de MLflow al no encontrar git (no usamos git SHA en los runs)
os.environ.setdefault('GIT_PYTHON_REFRESH', 'quiet')
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def _champion_alias():
    return get_model_cfg().get('champion_alias', 'champion')
# Setup de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def load_training_data(engine):
    """Cargo los datos de entrenamiento desde la db"""
    
    logger.info("Cargando datos de entrenamiento desde la base de datos...")
    
    query = "SELECT * FROM training_dataset";
    df = pd.read_sql(query, engine)
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
    logger.info(f"Columnas: {df.columns.tolist()}")
    
    return df

def prepare_features_target(df):
    """Separar features y target con transformaciones y agregado de variables para mejorar predicciones"""

    # Eliminar columnas no necesarias
    columns_to_drop = ['id', 'created_at']
    df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)

    # Separar X e y
    X = df.drop('charges', axis=1)
    y = df['charges']

    logger.info(f"Features (X): {X.columns.tolist()}")
    logger.info(f"Target original(y): charges")
    logger.info(f"  - Min: ${y.min():,.2f}")
    logger.info(f"  - Max: ${y.max():,.2f}")
    logger.info(f"  - Mean: ${y.mean():,.2f}")
    logger.info(f"  - Median: ${y.median():,.2f}")
    logger.info(f"  - Skewness: {y.skew():.3f}")

    # Aplicar feature engineering
    X = feature_engineering(X, is_training=True)

    # Aplicar transform de target
    y_original=y.copy()
    y_transformed = transform_target(y, inverse=False)

    return X, y_transformed, y_original

def split_data(X, y_transformed, y_original, test_size=0.2, random_state=43):
    """Split train/validation"""

    X_train, X_val, y_train_log, y_val_log, y_train_orig, y_val_orig = train_test_split(
        X, y_transformed, y_original, test_size=test_size, random_state=random_state, shuffle=True
    )

    logger.info(f"Train set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    logger.info(f"Validation set: {X_val.shape[0]} samples ({test_size*100:.0f}%)")
    
    # Stats del target TRANSFORMADO (log)
    logger.info(f"\nTarget TRANSFORMADO (log):")
    logger.info(f"  Train - mean: {y_train_log.mean():.3f}, std: {y_train_log.std():.3f}")
    logger.info(f"  Val   - mean: {y_val_log.mean():.3f}, std: {y_val_log.std():.3f}")
    
    # Stats del target ORIGINAL ($) - para referencia
    logger.info(f"\nTarget ORIGINAL ($):")
    logger.info(f"  Train - mean: ${y_train_orig.mean():,.2f}, std: ${y_train_orig.std():,.2f}")
    logger.info(f"  Val   - mean: ${y_val_orig.mean():,.2f}, std: ${y_val_orig.std():,.2f}")
    
    return X_train, X_val, y_train_log, y_val_log, y_train_orig, y_val_orig

def create_preprocessor():
    """Pipeline de preprocesamiento de features"""
    categorical_features = categorical_features_for_training()
    numerical_features = numerical_features_for_training()
    
    preprocessor= ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )
    
    logger.info("Preprocesador configurado")
    logging.info(f"  - Features numéricas: {numerical_features}")
    logging.info(f"  - Features categóricas: {categorical_features}")
    
    return preprocessor

def define_hyperparameter_grid():
    """Grid de hiperparametros desde config.yaml"""
    param_distributions = dict(get_training_cfg().get('hyperparam_search', {}).get('grid', {}))
    if not param_distributions:
        param_distributions = {
            'model__n_estimators': [100, 200, 300, 500],
            'model__max_depth': [3, 5, 7, 9],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__reg_alpha': [0, 0.1, 1],
            'model__reg_lambda': [1, 10, 100],
        }

    total_combinations = np.prod([len(v) for v in param_distributions.values()])
    logger.info(f"Hyperparameter grid definido:")
    logger.info(f"  - Total combinaciones posibles: {total_combinations:,}")
    logger.info(f"  - Parámetros: {list(param_distributions.keys())}")

    return param_distributions


def create_estimator():
    """Factory de estimador segun model.type en config (xgboost | sklearn_hist_gbm)."""
    tcfg = get_training_cfg()
    mtype = get_model_cfg().get('type', 'xgboost').lower()
    rs = int(tcfg.get('random_state_model', 42))

    if mtype == 'sklearn_hist_gbm':
        from sklearn.ensemble import HistGradientBoostingRegressor
        gbm = tcfg.get('sklearn_hist_gbm', {})
        kwargs = {
            'max_iter': int(gbm.get('max_iter', 300)),
            'learning_rate': float(gbm.get('learning_rate', 0.1)),
            'random_state': rs,
        }
        if gbm.get('max_depth') is not None:
            kwargs['max_depth'] = int(gbm['max_depth'])
        logger.info("Estimador: HistGradientBoostingRegressor")
        return HistGradientBoostingRegressor(**kwargs)

    xgb_cfg = tcfg.get('xgboost', {})
    logger.info("Estimador: XGBRegressor")
    return xgb.XGBRegressor(
        objective=xgb_cfg.get('objective', 'reg:squarederror'),
        random_state=rs,
        n_jobs=int(xgb_cfg.get('n_jobs', -1)),
        verbosity=int(xgb_cfg.get('verbosity', 0)),
    )


def train_model(X_train, y_train):
    """Entrenamiento del modelo con RandomizedSearchCV"""

    logger.info("Iniciando entrenamiento del modelo con RandomizedSearchCV...")

    preprocessor = create_preprocessor()
    model = create_estimator()

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    hp = get_training_cfg().get('hyperparam_search', {})
    param_grid = define_hyperparameter_grid()
    n_iter = int(hp.get('n_iter', 50))
    cv_folds = int(hp.get('cv_folds', 5))
    scoring = hp.get('scoring', 'neg_root_mean_squared_error')
    rs = int(get_training_cfg().get('random_state_model', 42))

    logger.info(f"Configuracion de busqueda:")
    logger.info(f"  - Metodo: RandomizedSearchCV")
    logger.info(f"  - Iteraciones: {n_iter}")
    logger.info(f"  - Cross-validation folds: {cv_folds}")
    logger.info(f"  - Scoring metric: {scoring}")

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1,
        random_state=rs,
        verbose=2,
        return_train_score=True,
        refit=True
    )
    
    logger.info("Ejecutando RandomizedSearchCV...")
    logger.info(f"  - Esto puede tardar varios minutos dependiendo del tamaño del grid y la cantidad de iteraciones.")
    random_search.fit(X_train, y_train)
    
    logger.info("RandomizedSearchCV completado.")
    logger.info(f"Mejores hiperparámetros encontrados:")
    for param, value in random_search.best_params_.items():
        logger.info(f"  - {param}: {value}")
    logger.info(f"Mejor RMSE en validación: {-random_search.best_score_:.2f}")
    
    return random_search.best_estimator_,random_search

def evaluate_model(model, X_train, y_train, X_val, y_val, y_train_original, y_val_original):
    """Evaluar modelo con el set de validacion con y cin transformacion del target
    
    Args:
        model: Modelo entrenado
        X_train, y_train: Datos de entrenamiento (y_train en escala LOG)
        X_val, y_val: Datos de validación (y_val en escala LOG)
        y_train_original, y_val_original: Target en escala ORIGINAL ($$)
    """
    logger.info("EVALUACIÓN DEL MODELO")
    
    #predicciones con escala log
    y_train_pred_log = model.predict(X_train)
    y_val_pred_log = model.predict(X_val)
    
    #predicciones con escala original
    y_train_pred = transform_target(y_train_pred_log, inverse=True)
    y_val_pred = transform_target(y_val_pred_log, inverse=True)
    
    #Calculo metricas en las dos escalas
    def calculate_metrics(y_true_log, y_pred_log, y_true_original, y_pred_original, dataset_name):
        
        #Metricas log para verificar ajuste
        rmse_log = root_mean_squared_error(y_true_log, y_pred_log)
        mae_log = mean_absolute_error(y_true_log, y_pred_log)
        r2_log = r2_score(y_true_log, y_pred_log)
        mape_log= np.mean(np.abs((y_true_original - y_pred_original) / y_true_original)) * 100
        
        #Metricas originales, las que me importan
        rmse = root_mean_squared_error(y_true_original, y_pred_original)
        mae = mean_absolute_error(y_true_original, y_pred_original)
        r2 = r2_score(y_true_original, y_pred_original)
        mape = np.mean(np.abs((y_true_original - y_pred_original) / y_true_original)) * 100
        
        
        # Adjusted R² (penaliza complejidad del modelo)
        n = len(y_true_original)
        p = X_train.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        metrics = {
            'rmse_log': rmse_log,
            'mae_log': mae_log,
            'r2_log': r2_log,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'adj_r2': adj_r2,
            'mape': mape
        }
        
        logger.info(f" {dataset_name.upper()} SET:")
        logger.info(f" Metricas en escala LOG (para diagnóstico de ajuste):")
        logger.info(f"  RMSE (log):        ${rmse_log:,.2f}")
        logger.info(f"  MAE (log):         ${mae_log:,.2f}")
        logger.info(f"  R² (log):          {r2_log:.4f}")
        logger.info(f"Metricas en escala original($):")
        logger.info(f"  RMSE:        ${rmse:,.2f}")
        logger.info(f"  MAE:         ${mae:,.2f}")
        logger.info(f"  R²:          {r2:.4f}")
        logger.info(f"  Adjusted R²: {adj_r2:.4f}")
        logger.info(f"  MAPE:        {mape:.2f}%")

        return metrics
    
    train_metrics = calculate_metrics(y_train, y_train_pred_log, y_train_original, y_train_pred, "train")
    val_metrics = calculate_metrics(y_val, y_val_pred_log, y_val_original, y_val_pred, "validation")
    
    #Analisis de overfitting
    r2_diff = train_metrics['r2'] - val_metrics['r2']
    logger.info(f" OVERFITTING ANALYSIS:")
    logger.info(f"  R² difference (train - val): {r2_diff:.4f}")

    of_cfg = get_training_cfg().get('overfitting_log', {})
    severe = float(of_cfg.get('severe_r2_diff', 0.15))
    moderate = float(of_cfg.get('moderate_r2_diff', 0.10))
    minor = float(of_cfg.get('minor_r2_diff', 0.05))
    if r2_diff > severe:
        logger.warning("SEVERE overfitting detected!")
    elif r2_diff > moderate:
        logger.warning("Moderate overfitting detected")
    elif r2_diff > minor:
        logger.info("Minor overfitting (acceptable)")
    else:
        logger.info("No significant overfitting")

    return {
        'train': train_metrics,
        'validation': val_metrics,
        'overfitting_score': r2_diff
    }, y_val_pred
    
    
def save_model(model, metrics, best_params, output_dir=None):
    if output_dir is None:
        output_dir = get_paths().get('models_dir', 'models')
    """Guarda modelo y metadata de los mismos"""
    
    logger.info("Guardando modelo y metadata...")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    #Guardo modelo
    model_filename= f"model_{timestamp}.pkl"
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(model, model_path,compress=3)
    logger.info(f"Modelo guardado en: {model_path}")
    
    #Guardo metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': get_model_cfg().get('type', 'xgboost'),
        'best_params': best_params,
        'train_metrics': metrics['train'],
        'validation_metrics': metrics['validation'],
        'overfitting_score': metrics['overfitting_score']
    }
    metadata_filename = f"model_metadata_{timestamp}.json"
    metadata_path = os.path.join(output_dir, metadata_filename)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata guardada en: {metadata_path}")
    
    #Creo un symlink "latest" para siempre tener referencia al modelo más reciente
    latest_model_path = os.path.join(output_dir, "best_model.pkl")
    latest_metadata_path = os.path.join(output_dir, "best_model_metadata.json")
    
    shutil.copy2(model_path, latest_model_path)
    shutil.copy2(metadata_path, latest_metadata_path)
    logger.info(f"Symlink actualizado: {latest_model_path} -> {model_path}")
    
    return model_path

def generate_report(metrics, best_params, search_results, output_dir=None):
    if output_dir is None:
        output_dir = get_paths().get('results_dir', 'results')
    """Generar reporte de evaluacion del modelo"""
    
    logger.info("Generando reporte de evaluación...")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"training_report_{timestamp}.txt")
    
    with open(report_path, 'w') as f:
        f.write("METLIFE INSURANCE COST PREDICTION - TRAINING REPORT\n")
        
        f.write(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("Modelo seleccionado: XGBoostRegressor\n")
        f.write(f"\nJustificacion\n")
        f.write(f"XGBoost fue seleccionado por su excelente performance en problemas de\n")
        f.write(f"regresión tabular, su capacidad para manejar relaciones no lineales y\n")
        f.write(f"su robustez frente a outliers. Además, su eficiencia computacional\n")
        f.write(f"permite realizar una búsqueda de hiperparámetros más exhaustiva.\n")
        
        f.write("\nMejores hiperparámetros encontrados:\n")
        f.write("-"*70 + "\n")
        for param, value in best_params.items():
            param_clean = param.replace('model__', '')
            f.write(f"{param_clean:25s}: {value}\n")
        f.write("-"*70 + "\n")
        
        f.write("\nMétricas de evaluación:\n")
        f.write("-"*70 + "\n")
        f.write("TRAIN SET:\n")
        f.write(f"  RMSE:        ${metrics['train']['rmse']:>12,.2f}\n")
        f.write(f"  MAE:         ${metrics['train']['mae']:>12,.2f}\n")
        f.write(f"  R²:          {metrics['train']['r2']:>13.4f}\n")
        f.write(f"  Adjusted R²: {metrics['train']['adj_r2']:>13.4f}\n")
        f.write(f"  MAPE:        {metrics['train']['mape']:>12.2f}%\n")
        f.write("\nVALIDATION SET:\n")
        f.write(f"  RMSE:        ${metrics['validation']['rmse']:>12,.2f}\n")
        f.write(f"  MAE:         ${metrics['validation']['mae']:>12,.2f}\n")
        f.write(f"  R²:          {metrics['validation']['r2']:>13.4f}\n")
        f.write(f"  Adjusted R²: {metrics['validation']['adj_r2']:>13.4f}\n")
        f.write(f"  MAPE:        {metrics['validation']['mape']:>12.2f}%\n")
        
        f.write("Interpretación de resultados:\n")
        f.write("-"*70 + "\n")
        r2_pct = metrics['validation']['r2'] * 100
        f.write(f"El modelo explica aproximadamente {r2_pct:.2f}% de la varianza en los costos\n")
        f.write(f"de seguros en el set de validación.\n\n")
        f.write(f"Error promedio absoluto (MAE) de ${metrics['validation']['mae']:,.2f} por prediccion\n")
        f.write(f"Error porcentual medio (MAPE) de {metrics['validation']['mape']:.2f}% \n\n")
        
        overfitting = metrics['overfitting_score']
        report_thr = float(get_training_cfg().get('overfitting_log', {}).get('report_threshold', 0.1))
        if overfitting > report_thr:
            f.write(f"Se detecta un posible overfitting (R² train - R² val = {overfitting:.4f}).\n")
            f.write(f"Considerar técnicas de regularización o más datos para mejorar generalización.\n")
        else:
            f.write(f"No se detecta un overfitting significativo (R² train - R² val = {overfitting:.4f}).\n")
            f.write(f"El modelo parece generalizar bien al set de validación.\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Busqueda de hiperparametros\n")
        f.write("-"*70 + "\n")
        f.write(f"Metodo: RandomizedSearchCV\n")
        f.write(f"Iteraciones: {search_results.n_iter}\n")
        f.write(f"Cross-validation folds: {search_results.cv}\n")
        f.write(f"Scoring metric: {search_results.scoring}\n")
        f.write(f"Mejor score (CV RMSE): {-search_results.best_score_:.2f}\n")
        
        f.write("Top 5 combinaciones de hiperparámetros:\n")
        f.write("-"*70 + "\n")
        results_df = pd.DataFrame(search_results.cv_results_)
        results_df= results_df.sort_values('rank_test_score')
        
        for idx, row in results_df.head(5).iterrows():
            f.write(f"\nRank {int(row['rank_test_score'])}:\n")
            f.write(f"  RMSE={-row['mean_test_score']:,.2f}\n")
            f.write(f"  Params: {row['params']}\n")

    logger.info(f"Reporte generado en: {report_path}")
    return report_path

def setup_mlflow():
    """Configura MLflow desde config.yaml + env overrides."""
    mlf = get_mlflow_cfg()
    mlflow.set_tracking_uri(mlf['tracking_uri'])
    mlflow.set_experiment(mlf['experiment_name'])
    logger.info(f"MLflow configurado:")
    logger.info(f"  - Tracking URI: {mlf['tracking_uri']}")
    logger.info(f"  - Experimento: {mlf['experiment_name']}")
    logger.info(f"  - Model name: {mlf['model_name']}")
    return mlf['tracking_uri'], mlf['experiment_name']

def log_training_to_mlflow(best_model, metrics, random_search, report_path, model_dir=None):
    if model_dir is None:
        model_dir = get_paths().get('models_dir', 'models')
    """Loguea params, metricas y artefactos del entrenamiento en MLflow y registra el modelo.

    Devuelve el ModelVersion registrado (o None si falla el registro).
    """
    model_name = get_mlflow_cfg()['model_name']
    tcfg = get_training_cfg()

    params = {k.replace('model__', ''): v for k, v in random_search.best_params_.items()}
    params.update({
        'n_iter': random_search.n_iter,
        'cv_folds': random_search.cv,
        'random_state_split': tcfg.get('random_state_split', 43),
        'random_state_model': tcfg.get('random_state_model', 42),
        'test_size': tcfg.get('test_size', 0.2),
        'target_transform': tcfg.get('target_transform', 'log1p'),
        'model_type': get_model_cfg().get('type', 'xgboost'),
        'numerical_features': ",".join(numerical_features_for_training()),
        'categorical_features': ",".join(categorical_features_for_training()),
        'search_scoring': random_search.scoring,
        'promotion_metric': get_promotion_cfg().get('metric'),
    })
    mlflow.log_params(params)

    # Metricas train/val en escala original ($) y diagnostico log
    for split_name in ('train', 'validation'):
        for metric_name, value in metrics[split_name].items():
            mlflow.log_metric(f"{split_name}_{metric_name}", float(value))
    mlflow.log_metric('overfitting_score', float(metrics['overfitting_score']))
    mlflow.log_metric('cv_best_rmse', float(-random_search.best_score_))

    # Artefactos: reporte y metadata local
    if report_path and os.path.exists(report_path):
        mlflow.log_artifact(report_path, artifact_path='reports')
    metadata_local = os.path.join(model_dir, 'best_model_metadata.json')
    if os.path.exists(metadata_local):
        mlflow.log_artifact(metadata_local, artifact_path='metadata')

    # Modelo + registro en Model Registry
    input_example = None
    try:
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path='model',
            registered_model_name=model_name,
            input_example=input_example,
        )
        logger.info(f"Modelo logueado y registrado en MLflow como '{model_name}'.")
    except Exception as e:
        logger.warning(f"No se pudo registrar el modelo en MLflow Registry: {e}")
        return None

    # Recuperar la ultima version registrada para este run
    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id
    versions = client.search_model_versions(f"name='{model_name}'")
    this_version = None
    for v in versions:
        if v.run_id == run_id:
            this_version = v
            break
    return this_version

def promote_if_better(model_version, metrics):
    """Promueve a champion si mejora la metrica configurada (promotion.metric / direction)."""
    if model_version is None:
        logger.warning("Sin version registrada; se omite promocion.")
        return False

    prom = get_promotion_cfg()
    alias = _champion_alias()
    metric_key = prom.get('metric', 'validation_rmse')

    if prom.get('block_if_overfitting') and metrics.get('overfitting_score', 0) > float(prom.get('max_r2_diff', 0.15)):
        logger.warning(
            f"Promocion bloqueada: overfitting_score={metrics['overfitting_score']:.4f} > "
            f"max_r2_diff={prom.get('max_r2_diff')}"
        )
        return False

    new_val = resolve_promotion_metric_value(metrics)
    if new_val is None:
        logger.warning(f"No se pudo resolver metrica de promocion '{metric_key}'.")
        return False

    model_name = model_version.name
    client = MlflowClient()
    current_val = None
    try:
        champion = client.get_model_version_by_alias(model_name, alias)
        run = client.get_run(champion.run_id)
        current_val = run.data.metrics.get(metric_key)
        logger.info(f"Champion actual: v{champion.version} ({metric_key}={current_val})")
    except Exception:
        logger.info("No existe champion previo; se promovera este modelo.")

    if is_promotion_better(new_val, current_val):
        client.set_registered_model_alias(model_name, alias, model_version.version)
        logger.info(f"Modelo v{model_version.version} promovido a '@{alias}' ({metric_key}={new_val}).")
        return True

    logger.info(
        f"Modelo v{model_version.version} ({metric_key}={new_val}) NO supera al champion "
        f"({metric_key}={current_val}); se mantiene el champion actual."
    )
    return False

def _build_run_name(trigger: dict | None) -> str:
    if not trigger:
        return f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if trigger.get('type') == 'feature_psi_alert':
        feat = str(trigger.get('feature', 'unknown'))
        feat_slug = ''.join(c if c.isalnum() else '_' for c in feat).strip('_') or 'unknown'
        return f"retrain_drift_feature_{feat_slug}_{ts}"
    return f"retrain_drift_prediction_{ts}"


def _apply_trigger_tags(trigger: dict | None, session_id: str | None = None) -> None:
    if not trigger:
        return
    mlflow.set_tag('trigger_source', 'online_auto_retrain')
    mlflow.set_tag('trigger_type', trigger['type'])
    mlflow.set_tag(
        'trigger_description',
        trigger.get('reason', 'Reentrenamiento disparado por monitoreo online'),
    )
    if session_id:
        mlflow.set_tag('trigger_online_session', session_id)
    if trigger.get('feature'):
        mlflow.log_param('trigger_feature', trigger['feature'])
    if trigger.get('feature_psi') is not None:
        mlflow.log_metric('trigger_feature_psi', float(trigger['feature_psi']))
    if trigger.get('prediction_psi') is not None:
        mlflow.log_metric('trigger_prediction_psi', float(trigger['prediction_psi']))
    if trigger.get('end_request_seq') is not None:
        mlflow.log_param('trigger_at_request_seq', int(trigger['end_request_seq']))


def run_training_pipeline(
    engine=None,
    *,
    run_name: str | None = None,
    trigger: dict | None = None,
    nested: bool = False,
    online_session_id: str | None = None,
) -> dict:
    """Ejecuta entrenamiento completo. Retorna resumen con run_id y paths."""
    if engine is None:
        engine = get_db_engine()

    df = load_training_data(engine)
    run_name = run_name or _build_run_name(trigger)

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.set_tag('stage', 'training')
        if trigger:
            mlflow.set_tag('triggered_by', 'online_drift')
            _apply_trigger_tags(trigger, online_session_id)
        started_at = datetime.now()
        t0 = time.time()

        logger.info("Preparando features y target...")
        X, y_transformed, y_original = prepare_features_target(df)

        logger.info("Dividiendo datos en train y validation...")
        tcfg = get_training_cfg()
        X_train, X_val, y_train, y_val, y_train_orig, y_val_orig = split_data(
            X, y_transformed, y_original,
            test_size=float(tcfg.get('test_size', 0.2)),
            random_state=int(tcfg.get('random_state_split', 43)),
        )

        logger.info("Entrenando modelo con RandomizedSearchCV...")
        best_model, random_search = train_model(X_train, y_train)

        logger.info("Evaluando modelo en el set de validación...")
        metrics, _y_val_pred = evaluate_model(
            best_model, X_train, y_train, X_val, y_val, y_train_orig, y_val_orig,
        )

        logger.info("Guardando modelo y metadata...")
        model_path = save_model(best_model, metrics, random_search.best_params_)

        report_path = generate_report(metrics, random_search.best_params_, random_search)

        logger.info("Logueando entrenamiento en MLflow y registrando modelo...")
        model_version = log_training_to_mlflow(best_model, metrics, random_search, report_path)

        is_champion = promote_if_better(model_version, metrics)

        grafana_utils.save_training_run(
            engine=engine,
            run_name=run_name,
            mlflow_run_id=run.info.run_id,
            model_version=model_version,
            is_champion=is_champion,
            df=df,
            metrics=metrics,
            random_search=random_search,
            best_model=best_model,
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=datetime.now(),
        )

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("=" * 70)
        logger.info(f"MLflow run_id: {run.info.run_id}")
        logger.info(f"Modelo guardado en: {model_path}")
        logger.info(f"Validation R²: {metrics['validation']['r2']:.4f}")
        logger.info(f"Validation RMSE: ${metrics['validation']['rmse']:,.2f}")

        return {
            'success': True,
            'run_id': run.info.run_id,
            'run_name': run_name,
            'is_champion': is_champion,
            'model_path': model_path,
            'report_path': report_path,
            'metrics': metrics,
        }


def main():
    """Función principal para ejecutar el proceso de entrenamiento."""
    try:
        logger.info("Iniciando proceso de entrenamiento...")
        log_config_summary()
        setup_mlflow()
        run_training_pipeline()
        return True
    except Exception as e:
        logger.error(f"\nError en el proceso de entrenamiento: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)