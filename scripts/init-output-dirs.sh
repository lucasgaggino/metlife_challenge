#!/usr/bin/env bash
# Crea carpetas de salida con permisos amplios para Docker (evita Permission denied en ./logs).
set -e
cd "$(dirname "$0")/.."
mkdir -p models results logs models/prod results/prod logs/prod
chmod -R 777 models results logs 2>/dev/null || true
echo "OK: models/, results/, logs/ listos para docker compose."
