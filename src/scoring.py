import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import joblib
import logging
import os
import sys
from datetime import datetime
from utils import get_db_engine, feature_engineering, transform_target

"""Pipeline de scroring para el proyecto."""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def create_scoring_dataset(engine, n_samples=10):
    """Crea dataset de scoring para muestras aleatorias del dataset"""
    
    logger.info(f"Creando dataset de scoring con {n_samples} muestras aleatorias.")
    
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scoring_dataset"))
        conn.commit()
        
        # Creo dataset de scoring con muestras aleatorias
        query = text(f"""
                     CREATE TABLE scoring_dataset AS
                     SELECT * FROM training_dataset
                     ORDER BY RANDOM()
                     LIMIT :n_samples
                     """)
        conn.execute(query, {"n_samples": n_samples})
        conn.commit()
        logger.info("Dataset de scoring creado exitosamente.")
        
        # Verifico que se hayan insertado las muestras
        result = conn.execute(text("SELECT COUNT(*) FROM scoring_dataset"))
        count = result.scalar()
        logger.info(f"Cantidad de muestras en scoring_dataset: {count}")
        return True

def load_scoring_data(engine):
    """Cargo el dataset de scoring desde la base de datos."""
    
    logger.info("Cargando dataset de scoring desde la base de datos.")
    
    query = text("SELECT * FROM scoring_dataset")
    df = pd.read_sql_query(query, engine)
    
    logger.info(f"Dataset de scoring cargado con {df.shape[0]} muestras.")
    
    return df

def load_training_model(model_path='models/best_model.pkl'):
    """Cargo el modelo previamente entrenado"""
    
    logger.info(f"Cargando modelo entrenado desde {model_path}.")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    model=joblib.load(model_path)
    logger.info("Modelo cargado exitosamente.")
    logger.info(f"Tipo de modelo cargado: {type(model)}")
    
    return model

def generate_predictions(model,df):
    """Genera predicciones sobre datos de scoring con Transformaciones iguales a las de training"""
    
    logger.info("Generando predicciones con el modelo cargado.")
    
    if 'id' in df.columns:
        ids=df['id'].values
    else:
        ids=np.arange(len(df))
        
    actual_charges=df['charges'].values
    
    #para hacer las predicciones elimino las columnas que no necesito
    columns_to_drop=['id','charges', 'created_at']
    df_features=df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
    
    logger.info(f"Features originales: {df_features.columns.tolist()}")
    
    logger.info("Aplicando feature engineering a los datos de scoring.")
    df_features=feature_engineering(df_features, is_training=False)
    logger.info(f"Features después de feature engineering: {df_features.columns.tolist()}")
    
    #predictions log-transformadas, vuelvo a escala original después
    predictions_log=model.predict(df_features)
    #predictions con escala original
    predictions=transform_target(predictions_log, inverse=True)
    
    #Info de las predicciones
    logger.info(f"✓ Predicciones generadas para {len(predictions)} registros")
    logger.info(f"  Predicciones (log): min={predictions_log.min():.3f}, max={predictions_log.max():.3f}")
    logger.info(f"  Predicciones ($):   min=${predictions.min():,.2f}, max=${predictions.max():,.2f}")
    
    # #predecir charges con el modelo
    # predictions=model.predict(df_features)
    logger.info("Predicciones generadas exitosamente.")
    
    #creo df con los resultados
    results_df=pd.DataFrame({
        'scoring_id': ids,
        'age': df['age'].values,
        'sex': df['sex'].values,
        'bmi': df['bmi'].values,
        'children': df['children'].values,
        'smoker': df['smoker'].values,
        'region': df['region'].values,
        'actual_charges': actual_charges,
        'predicted_charges': predictions,
        'absolute_error': np.abs(actual_charges - predictions),
        'percentage_error': np.abs((actual_charges - predictions) / actual_charges) * 100,
        'prediction_time': datetime.now()
    })
    
    return results_df

def save_predictions_to_db(engine, results_df):
    """Guarda las predicciones en la db"""
    
    logger.info("Guardando predicciones en la base de datos.")
    
    with engine.connect() as conn:
        # Creo tabla de resultados si no existe (funciona como check esto)
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                scoring_id INTEGER,
                age INTEGER NOT NULL,
                sex VARCHAR(10) NOT NULL,
                bmi FLOAT NOT NULL,
                children INTEGER NOT NULL,
                smoker VARCHAR(5) NOT NULL,
                region VARCHAR(20) NOT NULL,
                actual_charges FLOAT,
                predicted_charges FLOAT,
                absolute_error FLOAT,
                percentage_error FLOAT,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(create_table_query)
        conn.commit()
        logger.info("Tabla 'predictions' verificada/creada.")
        
    # Inserto los resultados en la tabla
    results_df_to_insert = results_df.copy()
    results_df_to_insert.to_sql(
        'predictions',
        engine,
        if_exists='append',
        index=False,
        method='multi'
    )
    logger.info(f"{len(results_df)} predicciones guardadas en la base de datos.")
    return True

def evaluate_predictions(results_df):
    """Evalua performance de las predicciones"""
    logger.info("Evaluando performance de las predicciones.")
    
    actual = results_df['actual_charges']
    predicted = results_df['predicted_charges']
    
    # Calculo métricas de evaluación
    rmse=root_mean_squared_error(actual, predicted)
    mae=mean_absolute_error(actual, predicted)
    r2=r2_score(actual, predicted)
    mape=np.mean(np.abs((actual - predicted) / actual)) * 100
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'n_samples': len(results_df)
    }
    
    logger.info(f"\nMuestras evaluadas: {len(results_df)}")
    logger.info(f"RMSE: {rmse:,.2f}")
    logger.info(f"MAE:  {mae:,.2f}")
    logger.info(f"R2:   {r2:.4f}")
    logger.info(f"MAPE: {mape:.2f}%")
    
    # Mostrar desgloce de predicciones
    logger.info("\nDesgloce de predicciones:")
    logger.info(f"{'#':<4} {'Actual':>12} {'Predicted':>12} {'Error $':>12} {'Error %':>10}")
    logger.info("-"*80)

    for idx, row in results_df.iterrows():
        logger.info(
            f"{idx+1:<4} "
            f"{row['actual_charges']:>11,.2f} "
            f"{row['predicted_charges']:>11,.2f} "
            f"{row['absolute_error']:>11,.2f} "
            f"{row['percentage_error']:>9.1f}%"
        )
    
    logger.info("-"*80)
    
    #Analisis de errores
    logger.info("\nAnálisis de errores:")
    logger.info(f"  Error mínimo: ${results_df['absolute_error'].min():,.2f}")
    logger.info(f"  Error máximo: ${results_df['absolute_error'].max():,.2f}")
    logger.info(f"  Error mediano: ${results_df['absolute_error'].median():,.2f}")
    logger.info(f"  Error promedio: ${results_df['absolute_error'].mean():,.2f}")
    
    
    #Casos con mayor error
    worst_cases=results_df.nlargest(3, 'absolute_error')
    logger.info("\nCasos con mayor error:")
    for idx, row in worst_cases.iterrows():
        logger.info(
            f"  ID {row['scoring_id']}: "
            f"Error=${row['absolute_error']:,.2f} "
            f"({row['percentage_error']:.1f}%) "
            f"- Smoker: {row['smoker']}, BMI: {row['bmi']}"
        )
    return metrics

    
def main():
    """Pipeline principal de scoring."""
    
    try:
        logger.info("Iniciando pipeline de scoring.")
        
        #config
        n_samples= int(os.getenv('SCORING_SAMPLE_SIZE', '10'))
        
        # Paso 1: Conectando a la db
        logger.info("Conectando a la base de datos.")
        engine = get_db_engine()
        
        # Paso 2: Creando dataset de scoring
        create_scoring_dataset(engine, n_samples=n_samples)
        
        # Paso 3: Cargando dataset de scoring
        scoring_df = load_scoring_data(engine)
        
        # Paso 4: Cargando modelo entrenado
        model = load_training_model()
        
        # Paso 5: Generando predicciones
        results_df = generate_predictions(model, scoring_df)
        print(results_df)
        
        save_predictions_to_db(engine, results_df)
        
        # Paso 6: Evaluando performance
        logger.info("Evaluando performance de las predicciones.")
        metrics = evaluate_predictions(results_df)
        
        logger.info("Pipeline de scoring completado exitosamente.")
        logger.info(f"\nPuntos clave:")
        logger.info(f"  - {metrics['n_samples']} predicciones generadas")
        logger.info(f"  - R²: {metrics['r2']:.4f}")
        logger.info(f"  - RMSE: ${metrics['rmse']:,.2f}")
        logger.info(f"  - MAE: ${metrics['mae']:,.2f}")
        logger.info(f"\nPredicciones guardadas en tabla 'predictions'")

        
        return True

    except Exception as e:
        logger.error(f"Error en pipeline de scoring: str({e})", exc_info=True)
        return False
    
if __name__ == "__main__":
    succcess = main()
    sys.exit(0 if succcess else 1)