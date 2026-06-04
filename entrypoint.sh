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


# Logging (subcarpeta prod si ML_ENV=prod; alinear con config.prod.yaml paths.logs_dir)
if [ "${ML_ENV:-sandbox}" = "prod" ]; then
    LOG_DIR="/app/logs/prod"
else
    LOG_DIR="/app/logs"
fi
mkdir -p /app/logs /app/logs/prod /app/models /app/models/prod /app/results /app/results/prod
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
if ! touch "$LOG_FILE" 2>/dev/null; then
    echo "[WARN] No se puede escribir en $LOG_DIR (permisos del volumen). Logs solo en consola." >&2
    LOG_FILE="/dev/null"
fi

# Función de logging
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { 
    echo -e "${BLUE}[INFO]${NC} $@" | tee -a "$LOG_FILE"
}

log_warn() { 
    echo -e "${YELLOW}[WARN]${NC} $@" | tee -a "$LOG_FILE"
}

log_error() { 
    echo -e "${RED}[ERROR]${NC} $@" | tee -a "$LOG_FILE"
}

log_success() { 
    echo -e "${GREEN}[SUCCESS]${NC} $@" | tee -a "$LOG_FILE"
}

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

run_python_script() {
    local script_name=$1
    local script_path="src/$script_name"

    echo ""
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${YELLOW}Ejecutando: $script_name${NC}"
    echo -e "${BLUE}==========================================${NC}"

    log_info "Iniciando $script_name"

    # Ejecutar con logging
    if python "$script_path" 2>&1 | tee -a "$LOG_FILE"; then
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

log_success "Pipeline completado exitosamente"
log_info "Outputs generados:"
log_info "  - Modelos: /app/models/"
log_info "  - Reportes: /app/results/"
log_info "  - Logs: $LOG_FILE"

echo ""
echo "Resumen de archivos generados:"
echo "────────────────────────────────────────────"
echo "MODELOS:"
ls -lh /app/models/*.pkl 2>/dev/null || echo "  (sin archivos .pkl)"
echo ""
echo "RESULTADOS:"
ls -lh /app/results/*.txt 2>/dev/null || echo "  (sin archivos .txt)"
echo ""
echo "LOGS:"
ls -lh /app/logs/*.log 2>/dev/null || echo "  (sin archivos .log)"
echo "────────────────────────────────────────────"
echo ""

exit 0