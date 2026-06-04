#!/usr/bin/env bash
# Crea carpetas de salida del pipeline (idempotente). Se commitean via .gitkeep;
# este script sirve si alguien clono sin esas rutas o las borro localmente.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for d in logs logs/prod models models/prod results results/prod mlartifacts; do
  mkdir -p "$ROOT/$d"
  touch "$ROOT/$d/.gitkeep"
done
echo "Directorios de salida listos en $ROOT"
