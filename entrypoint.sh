#!/bin/bash

set -e #salgo en error
set -u #salgo si uso una variable no definida
set -o pipefail #si un comando falla, el pipeline falla

#Config

# Colores para output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color


# --- Permisos en volumenes montados (./logs, ./models, ./results) ---
# El contenedor corre como appuser; si el host creo carpetas como root, tee falla.
dir_writable() {
    local d="$1"
    mkdir -p "$d" 2>/dev/null || return 1
    touch "$d/.write_test" 2>/dev/null && rm -f "$d/.write_test" 2>/dev/null
}

init_log_dir() {
    LOG_DIR="${PIPELINE_LOG_DIR:-/app/logs}"
    if ! dir_writable "$LOG_DIR"; then
        LOG_DIR="/tmp/metlife-pipeline-logs"
        mkdir -p "$LOG_DIR"
        LOG_DIR_FALLBACK=1
    fi
    LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
    touch "$LOG_FILE" 2>/dev/null || LOG_FILE=""
}

check_output_dirs() {
    local failed=0
    for d in /app/models /app/results; do
        if ! dir_writable "$d"; then
            echo -e "${RED}[ERROR]${NC} Sin permiso de escritura en ${d}." >&2
            echo "  El volumen montado desde el host no es escribible por el usuario del contenedor (appuser)." >&2
            echo "  En la raiz del repo (host), ejecuta:" >&2
            echo "    mkdir -p models results logs models/prod results/prod logs/prod" >&2
            echo "    chmod -R 777 models results logs   # Linux/Mac" >&2
            echo "  O borra las carpetas logs/models/results y vuelve a crearlas con tu usuario." >&2
            failed=1
        fi
    done
    return $failed
}

init_log_dir
check_output_dirs || exit 1

_append_log() {
    if [ -n "${LOG_FILE:-}" ]; then
        tee -a "$LOG_FILE" 2>/dev/null || cat
    else
        cat
    fi
}

# Función de logging
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | _append_log
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $@" | _append_log
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $@" | _append_log
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $@" | _append_log
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $@" | _append_log
}

if [ "${LOG_DIR_FALLBACK:-0}" = "1" ]; then
    log_warn "/app/logs no es escribible (permisos del volumen host). Logs en: ${LOG_DIR}"
fi

# Banner
echo -e "${BLUE}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        MetLife Insurance Prediction - ML Pipeline           ║
║                     Version 1.0.0                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# ========== FUNCIONES ==========

wait_for_postgres() {
    log_info "Esperando a que PostgreSQL esté listo..."

    local max_retries=30
    local retry_count=0
    local wait_seconds=2

    until PGPASSWORD=$DB_PASSWORD psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c '\q' 2>/dev/null; do
        retry_count=$((retry_count + 1))

        if [ $retry_count -ge $max_retries ]; then
            log_error "PostgreSQL no disponible después de $max_retries intentos"
            exit 1
        fi

        log_warn "Intento $retry_count/$max_retries - PostgreSQL aún no está listo"
        sleep $wait_seconds
    done

    log_success "PostgreSQL está listo!"
}

wait_for_mlflow() {
    local tracking_uri="${MLFLOW_TRACKING_URI:-http://mlflow:5000}"
    log_info "Esperando a que MLflow esté listo en $tracking_uri ..."

    local max_retries=30
    local retry_count=0
    local wait_seconds=2

    until curl -sf "${tracking_uri}/health" >/dev/null 2>&1; do
        retry_count=$((retry_count + 1))

        if [ $retry_count -ge $max_retries ]; then
            log_error "MLflow no disponible después de $max_retries intentos"
            exit 1
        fi

        log_warn "Intento $retry_count/$max_retries - MLflow aún no está listo"
        sleep $wait_seconds
    done

    log_success "MLflow está listo!"
}

_resolve_output_paths() {
    local out
    out=$(python -c "
import os, sys
sys.path.insert(0, '/app/src')
from config_loader import load_config, get_paths
load_config(force_reload=True)
p = get_paths()
def to_abs(path, default):
    path = path or default
    return path if os.path.isabs(path) else os.path.join('/app', path)
print(to_abs(p.get('models_dir'), 'models'))
print(to_abs(p.get('results_dir'), 'results'))
" 2>/dev/null) || true
    if [ -n "$out" ]; then
        OUTPUT_MODELS_DIR=$(echo "$out" | sed -n '1p')
        OUTPUT_RESULTS_DIR=$(echo "$out" | sed -n '2p')
    else
        if [ "${ML_ENV:-sandbox}" = "prod" ]; then
            OUTPUT_MODELS_DIR="/app/models/prod"
            OUTPUT_RESULTS_DIR="/app/results/prod"
        else
            OUTPUT_MODELS_DIR="/app/models"
            OUTPUT_RESULTS_DIR="/app/results"
        fi
    fi
}

run_python_script() {
    local script_name=$1
    local script_path="src/$script_name"

    echo ""
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${YELLOW}Ejecutando: $script_name${NC}"
    echo -e "${BLUE}==========================================${NC}"

    log_info "Iniciando $script_name"

    # Ejecutar con logging
    if python "$script_path" 2>&1 | _append_log; then
        log_success "$script_name completado exitosamente"
        return 0
    else
        local exit_code=$?
        log_error "$script_name falló con código $exit_code"
        return 1
    fi
}

# ========== MAIN EXECUTION ==========

log_info "Iniciando ML Pipeline"
log_info "Configuración:"
log_info "  ML Environment: ${ML_ENV:-sandbox}"
log_info "  Config path: ${CONFIG_PATH:-/app/config.yaml}"
log_info "  DB Host: $DB_HOST"
log_info "  DB Name: $DB_NAME"
log_info "  DB User: $DB_USER"
log_info "  Log Level: ${LOG_LEVEL:-INFO}"
log_info "  Hyperparam Iterations: ${HYPERPARAM_ITERATIONS:-50}"
log_info "  CV Folds: ${CV_FOLDS:-5}"
log_info "  Scoring Sample Size: ${SCORING_SAMPLE_SIZE:-10}"

# 1. Wait for PostgreSQL
wait_for_postgres

# 1b. Wait for MLflow
wait_for_mlflow

# 2. Database Setup
if ! run_python_script "db_setup.py"; then
    log_error "Pipeline abortado: fallo en db_setup"
    exit 1
fi

# 3. Training Pipeline
if ! run_python_script "training.py"; then
    log_error "Pipeline abortado: fallo en training"
    exit 1
fi

# 4. Scoring Pipeline
if ! run_python_script "scoring.py"; then
    log_error "Pipeline abortado: fallo en scoring"
    exit 1
fi

# ========== SUCCESS ==========

echo ""
echo -e "${GREEN}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          ✓ PIPELINE COMPLETADO EXITOSAMENTE                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

_resolve_output_paths

log_success "Pipeline completado exitosamente"
log_info "Outputs generados (rutas en el contenedor):"
log_info "  - Modelos: ${OUTPUT_MODELS_DIR}/"
log_info "  - Reportes: ${OUTPUT_RESULTS_DIR}/"
if [ -n "${LOG_FILE:-}" ]; then
    log_info "  - Log pipeline: ${LOG_FILE}"
else
    log_info "  - Log pipeline: solo consola (no se pudo escribir archivo)"
fi
if [ "${LOG_DIR_FALLBACK:-0}" = "1" ]; then
    log_warn "  El log NO esta en ./logs del host; uso interno: ${LOG_DIR}"
else
    log_info "  En tu PC (repo): ./models ./results ./logs (mapeo Docker)"
fi

echo ""
echo "Resumen de archivos generados:"
echo "────────────────────────────────────────────"
echo "MODELOS (${OUTPUT_MODELS_DIR}):"
ls -lh "${OUTPUT_MODELS_DIR}"/*.pkl 2>/dev/null || echo "  (sin archivos .pkl en esta carpeta)"
echo ""
echo "RESULTADOS (${OUTPUT_RESULTS_DIR}):"
ls -lh "${OUTPUT_RESULTS_DIR}"/*.txt 2>/dev/null || echo "  (sin archivos .txt en esta carpeta)"
echo ""
echo "LOGS (${LOG_DIR}):"
if [ -n "${LOG_FILE:-}" ]; then
    ls -lh "${LOG_FILE}" 2>/dev/null || echo "  (archivo de log no listable)"
else
    echo "  (sin archivo en disco)"
fi
echo "────────────────────────────────────────────"
echo ""

exit 0