import os
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

from config_loader import get_training_cfg

"""Script de utilidades para el proyecto."""


def get_db_engine():
    db_config = {
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

    return create_engine(connection_string)


def feature_engineering(X, is_training=True):
    """Feature engineering aplicable a training y a scoring (toggles en config.yaml)."""
    t = get_training_cfg()
    fe_cfg = t.get('feature_engineering', {})
    enabled = fe_cfg.get('enabled', {})
    bmi_thr = float(fe_cfg.get('bmi_obese_threshold', 30))
    age_thr = float(fe_cfg.get('age_senior_threshold', 50))

    X = X.copy()
    applied = []

    if enabled.get('bmi_smoker', True):
        X['bmi_smoker'] = X['bmi'] * (X['smoker'] == 'yes').astype(int)
        applied.append('bmi_smoker')
    if enabled.get('age_smoker', True):
        X['age_smoker'] = X['age'] * (X['smoker'] == 'yes').astype(int)
        applied.append('age_smoker')
    if enabled.get('bmi_squared', True):
        X['bmi_squared'] = X['bmi'] ** 2
        applied.append('bmi_squared')
    if enabled.get('age_squared', True):
        X['age_squared'] = X['age'] ** 2
        applied.append('age_squared')
    if enabled.get('bmi_obese', True):
        X['bmi_obese'] = (X['bmi'] > bmi_thr).astype(int)
        applied.append('bmi_obese')
    if enabled.get('age_senior', True):
        X['age_senior'] = (X['age'] > age_thr).astype(int)
        applied.append('age_senior')

    if is_training:
        import logging
        log = logging.getLogger(__name__)
        log.info("Feature engineering aplicado:")
        log.info(f"  - Features activas: {applied}")
        log.info(f"  - Shape resultante: {X.shape}")

    return X


def transform_target(y, inverse=False):
    """Transforma target segun training.target_transform en config (log1p | none)."""
    transform = get_training_cfg().get('target_transform', 'log1p')
    y = np.asarray(y, dtype=float)

    if transform == 'none':
        return y

    if inverse:
        return np.expm1(y)
    return np.log1p(y)
