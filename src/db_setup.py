import pandas as pd
from sqlalchemy import create_engine, text
import logging
import os
import sys

# Setup de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_to_db():
    """Conectar a PostgreSQL y retornar el engine."""

    db_config={
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'user': os.getenv('DB_USER', 'metlife_user'),
        'password': os.getenv('DB_PASSWORD', 'metlife_pass'),
        'database': os.getenv('DB_NAME', 'metlife_db')
    }
    
    connection_string = (
        f"postgresql://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    
    try:
        engine = create_engine(connection_string)
        logger.info("Conexión a la base de datos exitosa.")
        return engine
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos: {e}")
        raise

def drop_tables(engine):
    """Elimina tablas existentes (para fresh start)."""
    logger.info("Eliminando tablas existentes (si existen)...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS predictions CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS scoring_dataset CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS training_dataset CASCADE;"))
        conn.commit()
    logger.info("Tablas eliminadas.")

def create_tables(engine):
    """Crea tanlas necesarias para el proyecto."""
    
    logger.info("Creando tablas...")
    
    with engine.connect() as conn:
        # Tabla para daos de entrenamiento
        create_trainnig_table = text("""
            CREATE TABLE training_dataset (
                id SERIAL PRIMARY KEY,
                age INTEGER NOT NULL,
                sex VARCHAR(10) NOT NULL,
                bmi FLOAT NOT NULL,
                children INTEGER NOT NULL,
                smoker VARCHAR(5) NOT NULL,
                region VARCHAR(20) NOT NULL,
                charges FLOAT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(create_trainnig_table)
        conn.commit()
        logger.info("Tabla 'training_dataset' creada.")
        

def load_dataset(engine, csv_path='data/dataset.csv'):
    """Carga el dataset desde un CSV a la tabla"""
    
    logger.info(f"Cargando dataset desde {csv_path}...")
    
    #Verifico que el archivo existe
    if not os.path.exists(csv_path):
        # Intentar con ruta absoluta basada en la ubicación del script si la relativa falla
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, '..', 'data', 'dataset.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Archivo CSV no encontrado en {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas.")
    
    # Validar columnas esperadas
    expected_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
    missing_columns = set(expected_columns) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Faltan columnas en el dataset: {missing_columns}")
    
    # Validar datos
    logger.info("Validando datos del dataset...")
    
    # Veo valores nulos
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Valores nulos encontrados:\n{null_counts}")
    else:
        logger.info("No se encontraron valores nulos.")
    
    # Veo si hay duplicados
    duplicates = df[df.duplicated()]
    print (duplicates)
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        logger.warning(f"Se encontraron {duplicate_count} filas duplicadas.")
        df = df.drop_duplicates()
        logger.info(f"Filas duplicadas eliminadas. Nuevo tamaño del dataset: {df.shape[0]} filas.")
    else:
        logger.info("No se encontraron filas duplicadas.")
        
    # Ahora inserto el df en la DB
    logger.info("Insertando datos en la base de datos...")
    df[expected_columns].to_sql(
        'training_dataset',
        con=engine,
        if_exists='append',
        index=False
    )
    
    logger.info(f"{len(df)} datos insertados exitosamente en la DB.")
    
    # Verifico que se hayan insertado correctamente
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM training_dataset;"))
        count_in_db = result.scalar()
        logger.info(f"Total de filas en 'training_dataset' después de la inserción: {count_in_db}")
        
        # Mostrar la muestra que se inserto 
        result = conn.execute(text("SELECT * FROM training_dataset LIMIT 5;"))
        logger.info("Muestra de datos insertados:")
        for row in result.mappings():
            logger.info(f"{dict(row)}")

def create_additional_tables(engine):
    """Crea tablas addicionales para scoring y predicciones"""
    
    with engine.connect() as conn:
        
        create_predictions_table = text("""
            CREATE TABLE predictions (
                id SERIAL PRIMARY KEY,
                scoring_id INTEGER NOT NULL,
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
        
        conn.execute(create_predictions_table)
        conn.commit()
        logger.info("Tabla 'predictions' creada.")

def main():
    """Pipeline principal para setup de la base de datos."""
    
    try:
        logger.info("Iniciando setup de la base de datos...")
        engine = connect_to_db()
        drop_tables(engine)
        create_tables(engine)
        load_dataset(engine)
        create_additional_tables(engine)
        logger.info("Setup de la base de datos completado exitosamente.")
        return True
    
    except Exception as e:
        logger.error(f"Error durante el setup de la base de datos: {e}", exc_info=True)
        return False
    
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)