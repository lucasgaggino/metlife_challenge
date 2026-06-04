"""Promueve el modelo @champion de Sandbox al Model Registry de Prod.

Accion explicita (no corre en entrypoint). Opcionalmente lanza scoring en prod.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

os.environ.setdefault('GIT_PYTHON_REFRESH', 'quiet')

import mlflow
from mlflow import MlflowClient

from environment import CHAMPION_ALIAS, prod_mlflow_cfg, sandbox_mlflow_cfg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def promote_sandbox_champion_to_prod() -> int:
    sb = sandbox_mlflow_cfg()
    pr = prod_mlflow_cfg()
    mlflow.set_tracking_uri(sb['tracking_uri'])
    client = MlflowClient()

    try:
        sandbox_champ = client.get_model_version_by_alias(sb['model_name'], CHAMPION_ALIAS)
    except Exception as e:
        logger.error(
            f"No hay champion en sandbox ({sb['model_name']}@{CHAMPION_ALIAS}): {e}"
        )
        return 1

    source = f"models:/{sb['model_name']}/{sandbox_champ.version}"
    logger.info(
        f"Promoviendo sandbox v{sandbox_champ.version} (run_id={sandbox_champ.run_id}) "
        f"-> registry {pr['model_name']}"
    )

    try:
        run = client.get_run(sandbox_champ.run_id)
        val_rmse = run.data.metrics.get('validation_rmse')
        if val_rmse is not None:
            logger.info(f"Metrica origen validation_rmse={val_rmse:.4f}")
    except Exception:
        val_rmse = None

    try:
        client.get_registered_model(pr['model_name'])
    except Exception:
        logger.info(f"Creando registered model '{pr['model_name']}'...")
        client.create_registered_model(pr['model_name'])

    prod_version = client.create_model_version(
        name=pr['model_name'],
        source=source,
        description=(
            f"Promoted from sandbox {sb['model_name']} v{sandbox_champ.version} "
            f"(run {sandbox_champ.run_id})"
        ),
    )
    client.set_registered_model_alias(
        pr['model_name'],
        CHAMPION_ALIAS,
        prod_version.version,
    )
    logger.info(
        f"Prod champion actualizado: {pr['model_name']}@{CHAMPION_ALIAS} "
        f"-> v{prod_version.version}"
    )
    return 0


def run_prod_scoring() -> int:
    os.environ['ML_ENV'] = 'prod'
    root = os.path.dirname(os.path.abspath(__file__))
    prod_cfg = os.path.join(root, '..', 'config.prod.yaml')
    if os.path.exists(prod_cfg):
        os.environ['CONFIG_PATH'] = os.path.abspath(prod_cfg)

    from config_loader import load_config
    load_config(force_reload=True)

    import scoring
    logger.info("Ejecutando scoring en entorno prod...")
    ok = scoring.main()
    return 0 if ok else 1


def main() -> bool:
    parser = argparse.ArgumentParser(
        description='Promueve el champion de Sandbox al registry de Prod.'
    )
    parser.add_argument(
        '--run-scoring',
        action='store_true',
        help='Tras promover, ejecutar scoring.py en entorno prod',
    )
    args = parser.parse_args()

    code = promote_sandbox_champion_to_prod()
    if code != 0:
        return False

    if args.run_scoring:
        code = run_prod_scoring()
        if code != 0:
            return False

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
