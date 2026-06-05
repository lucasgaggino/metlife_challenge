"""Carga centralizada de config.yaml para el pipeline MLOps.

Data Science edita config.yaml; ops mantiene secretos en .env.
Variables de entorno pueden overridear claves puntuales (compat Docker).
"""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIG: dict[str, Any] | None = None

_ENV_OVERRIDES = {
    'HYPERPARAM_ITERATIONS': ('training', 'hyperparam_search', 'n_iter'),
    'CV_FOLDS': ('training', 'hyperparam_search', 'cv_folds'),
    'MLFLOW_TRACKING_URI': ('mlflow', 'tracking_uri'),
    'MLFLOW_EXPERIMENT_NAME': ('mlflow', 'experiment_name'),
    'MLFLOW_MODEL_NAME': ('model', 'registry_name'),
    'LOG_LEVEL': ('logging', 'level'),
    'ONLINE_N_SAMPLES': ('online', 'n_samples'),
    'ONLINE_RATE_PER_SECOND': ('online', 'rate_per_second'),
    'ONLINE_MONITORING_WINDOW_SIZE': ('online', 'monitoring_window_size'),
    'ONLINE_MONITORING_MAX_SAMPLES': ('online', 'monitoring_max_samples'),
}

_VALID_ML_ENV = frozenset({'sandbox', 'prod'})


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _ml_env_from_os() -> str:
    raw = os.getenv('ML_ENV', 'sandbox').strip().lower()
    return raw if raw in _VALID_ML_ENV else 'sandbox'


def _load_prod_overlay() -> dict[str, Any]:
    prod_path = _project_root() / 'config.prod.yaml'
    if not prod_path.exists():
        example = _project_root() / 'config.prod.yaml.example'
        if example.exists():
            prod_path = example
        else:
            return {}
    with open(prod_path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _default_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    example = config_path.parent / 'config.yaml.example'
    if example.exists():
        with open(example, encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    raise FileNotFoundError(
        f"No se encontro config.yaml ni config.yaml.example en {config_path.parent}"
    )


def _deep_set(d: dict, keys: tuple[str, ...], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _apply_env_overrides(cfg: dict) -> None:
    env = (cfg.get('environment') or _ml_env_from_os()).strip().lower()
    skip_mlflow_names = env == 'prod'
    for env_key, path in _ENV_OVERRIDES.items():
        if skip_mlflow_names and env_key in ('MLFLOW_EXPERIMENT_NAME', 'MLFLOW_MODEL_NAME'):
            continue
        val = os.getenv(env_key)
        if val is not None and val != '':
            if path[-1] in (
                'n_iter', 'cv_folds', 'n_samples', 'flush_every',
                'monitoring_window_size', 'monitoring_max_samples',
            ):
                _deep_set(cfg, path, int(val))
            elif path[-1] == 'rate_per_second':
                _deep_set(cfg, path, float(val))
            else:
                _deep_set(cfg, path, val)


def load_config(force_reload: bool = False) -> dict[str, Any]:
    global _CONFIG
    if _CONFIG is not None and not force_reload:
        return _CONFIG

    path = os.getenv('CONFIG_PATH')
    ml_env = _ml_env_from_os()

    if path and Path(path).exists():
        with open(path, encoding='utf-8') as f:
            loaded = yaml.safe_load(f) or {}
        path_name = Path(path).name
        if path_name.startswith('config.prod') or loaded.get('environment') == 'prod':
            cfg = _deep_merge(_default_config(), loaded)
        else:
            cfg = copy.deepcopy(loaded)
    elif ml_env == 'prod':
        cfg = _deep_merge(_default_config(), _load_prod_overlay())
    else:
        cfg = copy.deepcopy(_default_config())

    if 'environment' not in cfg or not cfg.get('environment'):
        cfg['environment'] = ml_env
    _apply_env_overrides(cfg)
    _CONFIG = cfg
    return cfg


def get_environment() -> str:
    """Entorno activo del pipeline: sandbox o prod."""
    env = (get_config().get('environment') or _ml_env_from_os()).strip().lower()
    return env if env in _VALID_ML_ENV else 'sandbox'


def get_config() -> dict[str, Any]:
    return load_config()


def log_config_summary() -> None:
    cfg = get_config()
    prom = cfg.get('promotion', {})
    mon = cfg.get('monitoring', {})
    train = cfg.get('training', {})
    hp = train.get('hyperparam_search', {})
    mlf = cfg.get('mlflow', {})
    logger.info(
        "Config: environment=%s | model=%s | registry=%s | experiment=%s | "
        "champion_metric=%s (%s) | HP n_iter=%s cv=%s | PSI warn/alert=%s/%s | axes=%s",
        get_environment(),
        cfg.get('model', {}).get('type'),
        cfg.get('model', {}).get('registry_name'),
        mlf.get('experiment_name'),
        prom.get('metric'),
        prom.get('direction'),
        hp.get('n_iter'),
        hp.get('cv_folds'),
        mon.get('psi', {}).get('warning'),
        mon.get('psi', {}).get('alert'),
        mon.get('enabled_axes'),
    )


def get_paths() -> dict:
    return get_config().get('paths', {})


def get_model_cfg() -> dict:
    return get_config().get('model', {})


def get_promotion_cfg() -> dict:
    return get_config().get('promotion', {})


def get_training_cfg() -> dict:
    return get_config().get('training', {})


def get_monitoring_cfg() -> dict:
    return get_config().get('monitoring', {})


def get_scoring_cfg() -> dict:
    return get_config().get('scoring', {})


def get_online_cfg() -> dict:
    return get_config().get('online', {})


def get_mlflow_cfg() -> dict:
    m = get_config().get('mlflow', {})
    cfg_exp = m.get('experiment_name', 'metlife_insurance')
    cfg_model = get_model_cfg().get('registry_name', 'metlife_insurance_xgb')
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME') or cfg_exp
    model_name = os.getenv('MLFLOW_MODEL_NAME') or cfg_model
    # Compose inyecta nombres sandbox; en ML_ENV=prod priorizar config _prod
    if get_environment() == 'prod':
        if experiment_name == 'metlife_insurance':
            experiment_name = cfg_exp
        if model_name == 'metlife_insurance_xgb':
            model_name = cfg_model
    return {
        'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', m.get('tracking_uri', 'http://localhost:5000')),
        'experiment_name': experiment_name,
        'model_name': model_name,
    }


def numerical_features_for_training() -> list[str]:
    t = get_training_cfg()
    raw = [c for c in t.get('features', {}).get('raw', []) if c not in ('sex', 'smoker', 'region')]
    eng = t.get('features', {}).get('numerical_engineered', [])
    enabled = t.get('feature_engineering', {}).get('enabled', {})
    out = list(raw)
    for feat in eng:
        if enabled.get(feat, True):
            out.append(feat)
    return out


def categorical_features_for_training() -> list[str]:
    return list(get_training_cfg().get('features', {}).get('categorical', ['sex', 'smoker', 'region']))


def resolve_promotion_metric_value(metrics: dict) -> float | None:
    prom = get_promotion_cfg()
    metric_key = prom.get('metric', 'validation_rmse')
    split = prom.get('split', 'validation')

    if '_' in metric_key and metric_key.startswith(split + '_'):
        inner = metric_key[len(split) + 1:]
    elif metric_key in ('rmse', 'mae', 'r2', 'mape', 'adj_r2'):
        inner = metric_key
    else:
        parts = metric_key.split('_', 1)
        inner = parts[1] if len(parts) == 2 else metric_key

    split_metrics = metrics.get(split, metrics.get('validation', {}))
    return split_metrics.get(inner)


def is_promotion_better(new_val: float, current_val: float | None) -> bool:
    if current_val is None:
        return True
    direction = get_promotion_cfg().get('direction', 'minimize')
    if direction == 'maximize':
        return new_val > current_val
    return new_val < current_val


def get_prod_batches() -> list[dict]:
    sc = get_scoring_cfg()
    batches = sc.get('prod_batches', [])
    filt = sc.get('prod_batches_filter')
    if filt:
        names = set(filt)
        batches = [b for b in batches if b.get('name') in names]
    result = []
    for b in batches:
        bb = dict(b)
        if bb.get('target') in (None, 'null', ''):
            bb['target'] = None
        result.append(bb)
    return result
