"""Constantes y helpers de entorno MLOps (sandbox vs prod)."""
from __future__ import annotations

import os

from config_loader import get_config, get_mlflow_cfg, get_model_cfg

SANDBOX = 'sandbox'
PROD = 'prod'

SANDBOX_EXPERIMENT = 'metlife_insurance'
SANDBOX_REGISTRY = 'metlife_insurance_xgb'
PROD_EXPERIMENT = 'metlife_insurance_prod'
PROD_REGISTRY = 'metlife_insurance_xgb_prod'
CHAMPION_ALIAS = 'champion'


def _ml_env_raw() -> str:
    return os.getenv('ML_ENV', SANDBOX).strip().lower()


def get_environment() -> str:
    """Entorno activo: 'sandbox' o 'prod'."""
    cfg = get_config()
    env = (cfg.get('environment') or _ml_env_raw() or SANDBOX).strip().lower()
    if env not in (SANDBOX, PROD):
        return SANDBOX
    return env


def sandbox_mlflow_cfg() -> dict:
    """Nombres MLflow fijos de sandbox (promocion cross-env)."""
    return {
        'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
        'experiment_name': SANDBOX_EXPERIMENT,
        'model_name': SANDBOX_REGISTRY,
        'champion_alias': CHAMPION_ALIAS,
    }


def prod_mlflow_cfg() -> dict:
    """Nombres MLflow fijos de prod."""
    return {
        'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
        'experiment_name': PROD_EXPERIMENT,
        'model_name': PROD_REGISTRY,
        'champion_alias': CHAMPION_ALIAS,
    }


def active_mlflow_cfg() -> dict:
    """Config MLflow del entorno activo."""
    mlf = get_mlflow_cfg()
    mlf['champion_alias'] = get_model_cfg().get('champion_alias', CHAMPION_ALIAS)
    return mlf
