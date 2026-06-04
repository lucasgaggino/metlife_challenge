#!/bin/bash
set -e

# Config (con defaults alineados a docker-compose)
DB_HOST="${MLFLOW_DB_HOST:-postgres}"
DB_PORT="${MLFLOW_DB_PORT:-5432}"
DB_USER="${MLFLOW_DB_USER:-metlife_user}"
DB_PASSWORD="${MLFLOW_DB_PASSWORD:-metlife_pass}"
DB_NAME="${MLFLOW_DB_NAME:-mlflow_db}"
ARTIFACTS_DIR="${MLFLOW_ARTIFACTS_DIR:-/mlartifacts}"

echo "[mlflow-entrypoint] Esperando a PostgreSQL en ${DB_HOST}:${DB_PORT}..."
until python -c "import psycopg2, os; psycopg2.connect(host='${DB_HOST}', port='${DB_PORT}', user='${DB_USER}', password='${DB_PASSWORD}', dbname='postgres').close()" 2>/dev/null; do
    echo "[mlflow-entrypoint] PostgreSQL no disponible aun, reintentando..."
    sleep 2
done
echo "[mlflow-entrypoint] PostgreSQL disponible."

# Crear base de datos de MLflow si no existe (idempotente, robusto ante volumen existente)
echo "[mlflow-entrypoint] Asegurando base de datos '${DB_NAME}'..."
python - <<PYEOF
import psycopg2
from psycopg2 import sql
conn = psycopg2.connect(host="${DB_HOST}", port="${DB_PORT}", user="${DB_USER}", password="${DB_PASSWORD}", dbname="postgres")
conn.autocommit = True
cur = conn.cursor()
cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", ("${DB_NAME}",))
if cur.fetchone() is None:
    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier("${DB_NAME}")))
    print("[mlflow-entrypoint] Base de datos '${DB_NAME}' creada.")
else:
    print("[mlflow-entrypoint] Base de datos '${DB_NAME}' ya existe.")
cur.close()
conn.close()
PYEOF

mkdir -p "${ARTIFACTS_DIR}"

BACKEND_URI="postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
echo "[mlflow-entrypoint] Iniciando MLflow server (backend=${DB_NAME}, artifacts=${ARTIFACTS_DIR})..."
exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "${BACKEND_URI}" \
    --artifacts-destination "${ARTIFACTS_DIR}" \
    --serve-artifacts
