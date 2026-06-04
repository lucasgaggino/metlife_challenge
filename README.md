# ML-Ops-Challenge

## Objetivo
Desarrollar un modelo de Machine Learning que prediga costos de seguros médicos basándose en:

- **Atributos personales**: edad, género, BMI, número de hijos, hábito de fumar
- **Factores geográficos**: región de cobertura


**Dataset:**

El dataset (`dataset.csv`) contiene 1,338 registros con las siguientes variables:

| Variable | Descripción | Tipo |
|----------|-------------|------|
| `age` | Edad del asegurado | Numérica (18-64 años) |
| `sex` | Género (male/female) | Categórica |
| `bmi` | Índice de Masa Corporal | Numérica (15.96-53.13) |
| `children` | Número de dependientes cubiertos | Numérica (0-5) |
| `smoker` | Si el asegurado fuma (yes/no) | Categórica |
| `region` | Área geográfica (northeast/northwest/southeast/southwest) | Categórica |
| `charges` | **TARGET** - Costos del seguro médico | Numérica ($1,121 - $63,770) |

---


## Enfoque de Modelado

### **Algoritmo Seleccionado: XGBoost (Extreme Gradient Boosting)**

**Justificación:**

He seleccionado **XGBoost** como algoritmo principal por las siguientes razones:

- Excelente performance en datos tabulares
- Captura de relaciones no lineales. El dataset presenta interacciones complejas, especialmente entre `smoker` y `bmi` (fumadores con alto BMI tienen costos exponencialmente mayores). XGBoost captura naturalmente estas no linealidades mediante árboles de decisión.

- Manejo robusto de features heterogéneas: Con variables numéricas (age, bmi) y categóricas (sex, smoker, region) de diferentes escalas, XGBoost no requiere normalización extensiva y maneja bien la mezcla de tipos de datos.

- Regularización incorporada. Los parámetros `reg_alpha` (L1) y `reg_lambda` (L2) ayudan a prevenir overfitting, crucial con un dataset relativamente pequeño (~1,300 registros).

- XGBoost proporciona métricas de importancia de variables, permitiendo identificar qué factores impactan más en los costos (esencial para explicar decisiones de pricing a stakeholders de MetLife).

- Eficiencia computacional: El entrenamiento con hyperparameter tuning (50 iteraciones de RandomizedSearchCV) se completa en minutos, no horas, facilitando iteración rápida.


**Alternativas consideradas:**
- **Linear Regression**: Descartada por asumir linealidad (inadecuado para interacciones multiplicativas smoker×BMI)
- **Random Forest**: Buen candidato, pero XGBoost generalmente supera en accuracy y velocidad
- **Neural Networks**: Overkill para este tamaño de dataset; requiere más datos para evitar overfitting

**Transformación de la Target:**

Aplicamos `log1p(charges)` como variable objetivo porque:
- La distribución original de `charges` tiene **skewness = 1.52** (fuertemente asimétrica)
- La transformación logarítmica reduce skewness a **0.09**, aproximándose a normalidad
- Esto estabiliza la varianza y mejora el ajuste del modelo en todo el rango de precios
- Después de predecir, invertimos con `exp(pred) - 1` para volver a escala de dólares


**Justificación de feature engineering aplicado**

Se agregaron variables derivadas como bmi², age², bmi×smoker y age×smoker para capturar no linealidades e interacciones reales que no están representadas en las variables originales.
Estas features permiten modelar efectos curvos y efectos condicionados (por ejemplo, que el impacto del BMI y la edad cambia en fumadores), mejorando la capacidad predictiva del modelo y su generalización sin aumentar excesivamente la complejidad.

---


**Flujo de Datos:**

1. **Ingesta**: `db_setup.py` lee `dataset.csv` y lo carga en PostgreSQL (`training_dataset`)
2. **Entrenamiento**: `training.py` aplica feature engineering, optimiza hiperparámetros, entrena XGBoost con target logarítmico
3. **Scoring**: `scoring.py` crea tabla temporal con 10 muestras aleatorias, predice, invierte log, guarda resultados
4. **Persistencia**: Modelo en `models/best_model.pkl`, métricas en `results/training_report.txt`, predicciones en tabla `predictions`

---

## 📁 Estructura del Proyecto

```
metlife-insurance-prediction/
│
├── data/
│   └── dataset.csv                    # Dataset original (1,338 registros)
│
├── notebooks/
│   └── exploratory_analysis_improved.ipynb  # EDA completo con Seaborn
│
├── src/
│   ├── db_setup.py                    # Crea DB y carga training_dataset
│   ├── training.py                    # Pipeline de entrenamiento
│   ├── scoring.py                     # Pipeline de scoring
│   └── utils.py                       # Funciones reutilizables
│
├── models/
│   ├── best_model.pkl                 # Modelo entrenado (symlink)
│   ├── model_YYYYMMDD_HHMMSS.pkl     # Modelos versionados
│   └── model_metadata_*.json          # Metadata de modelos
│
├── results/
│   └── training_report_*.txt          # Reporte de evaluación
│
├── docker/
│   └── Dockerfile                     # Multi-stage build optimizado
│
├── docker-compose.yml                 # Orquestación PostgreSQL + ML
├── entrypoint.sh                      # Script de ejecución secuencial
├── requirements.txt                   # Dependencias Python
├── .env.example                       # Template de variables de entorno
├── .gitignore                         # Archivos excluidos de Git
└── README.md                          # Este archivo
```


## 🚀 Instalación y Configuración

Guía paso a paso (build → parámetros → reentrenar): **[GUIDE.md](GUIDE.md)**.

### **Requisitos Previos**

- **Python**: 3.10 (probado con 3.10.19)
- **Docker**: 20.10+ (para ejecución containerizada)
- **Git**: Para clonar el repositorio

### Opción 1: Con Docker (Recomendado)

**✅ Ventajas:**
- Setup automático de PostgreSQL
- Entorno reproducible
- No contamina sistema local
- Ejecuta todo el pipeline secuencialmente

#### **Paso 1: Unzip Folder**
```bash
cd metlife-insurance-prediction
```

#### **Paso 2: Configurar variables de entorno**
```bash
# Copiar template
cp .env.example .env

# Editar .env (opcional - valores por defecto funcionan)
nano .env
```

**Contenido de `.env`:**
```bash
# PostgreSQL Configuration
DB_HOST=postgres
DB_PORT=5432
DB_NAME=metlife_db
DB_USER=metlife_user
DB_PASSWORD=metlife_pass

# ML Pipeline Configuration
HYPERPARAM_ITERATIONS=50
CV_FOLDS=5
SCORING_SAMPLE_SIZE=10
```


#### **Paso 3: Construir y ejecutar**
```bash
# Construir imagen Docker
docker compose build

# Ejecutar pipeline completo (db_setup → training → scoring)
docker compose up

# O en background
docker compose up -d

# Ver logs en tiempo real
docker compose logs -f ml_pipeline
```

**Output esperado:**
```
✓ PostgreSQL ready
✓ Database 'metlife_db' created
✓ Table 'training_dataset' created and populated (1337 rows)
✓ Training completed - R² = 0.835, RMSE = $4,835
✓ Model saved: models/best_model.pkl
✓ Scoring completed - 10 predictions generated
✓ All pipelines executed successfully
```


#### **Paso 4: Verificar resultados**
```bash
# Listar modelos entrenados
docker compose exec ml_pipeline ls -lh models/

# Ver reporte de entrenamiento
docker compose exec ml_pipeline cat results/training_report_*.txt

# Consultar predicciones en DB
docker compose exec postgres psql -U metlife_user -d metlife_db \
  -c "SELECT * FROM predictions ORDER BY prediction_time DESC LIMIT 5;"
```

#### **Paso 5: Detener y limpiar**
```bash
# Detener contenedores
docker compose down

# Eliminar también volúmenes (resetea DB)
docker compose down -v
```


---

### Opción 2: Sin Docker (Local)

**⚠️ Requiere:**
- PostgreSQL instalado localmente
- Python 3.10 (usado 3.10.19)
- Entorno virtual Python

#### **Paso 1: Instalar PostgreSQL**

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**macOS (Homebrew):**
```bash
brew install postgresql@15
brew services start postgresql@15
```

**Windows:**
Descargar desde [postgresql.org](https://www.postgresql.org/download/windows/)

#### **Paso 2: Crear base de datos**
```bash
# Conectarse a PostgreSQL
sudo -u postgres psql

# Dentro de psql:
CREATE DATABASE metlife_db;
CREATE USER metlife_user WITH PASSWORD 'metlife_pass';
GRANT ALL PRIVILEGES ON DATABASE metlife_db TO metlife_user;
\q
```

#### **Paso 3: Clonar repositorio**
```bash
git clone https://github.com/TU_USUARIO/metlife-insurance-prediction.git
cd metlife-insurance-prediction
```

#### **Paso 4: Crear entorno virtual**
```bash
# Crear virtualenv
python3.10 -m venv venv

# Activar
source venv/bin/activate  # Linux/Mac
# O en Windows: venv\Scripts\activate
```

#### **Paso 5: Instalar dependencias**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Contenido de `requirements.txt`:**
```
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
xgboost==2.0.3
psycopg2-binary==2.9.9
SQLAlchemy==2.0.23
joblib==1.3.2
python-dotenv==1.0.0
matplotlib==3.8.2
seaborn==0.13.0
scipy==1.11.4
statsmodels==0.14.1
```


#### **Paso 6: Configurar variables de entorno**
```bash
# Copiar template
cp .env.example .env

# Editar .env para apuntar a localhost
nano .env
```

**Modificar `.env`:**
```bash
DB_HOST=localhost  # ← Cambiar de 'postgres' a 'localhost'
DB_PORT=5432
DB_NAME=metlife_db
DB_USER=metlife_user
DB_PASSWORD=metlife_pass

HYPERPARAM_ITERATIONS=50
CV_FOLDS=5
SCORING_SAMPLE_SIZE=10
```

---


## ▶️ Ejecución del Pipeline

### **Pipeline Completo (Secuencial)**

```bash
# 1. Setup de base de datos
python src/db_setup.py

# 2. Entrenamiento
python src/training.py

# 3. Scoring
python src/scoring.py

```

---

## 🔬 MLOps: MLflow, scoring de producción y monitoreo

El pipeline incorpora tracking de experimentos, registro de modelo y monitoreo por batch.

### 1) Entrenamiento con MLflow

`src/training.py` registra cada corrida en MLflow:

- **Params**: hiperparámetros, seed, features, `n_iter`, `cv_folds`.
- **Métricas**: RMSE, MAE, R², Adjusted R², MAPE (train y validation) + overfitting.
- **Artefactos**: modelo (`mlflow.sklearn`), reporte de training y metadata JSON.
- **Registro + promoción**: el modelo se registra como `metlife_insurance_xgb` y se promueve al alias `@champion` si mejora (menor RMSE de validación) al champion vigente.

Config vía variables de entorno: `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `MLFLOW_MODEL_NAME`.

UI de MLflow disponible en **http://localhost:5000** (servicio `mlflow` en docker-compose, backend Postgres `mlflow_db`, artefactos en `./mlartifacts`).

### 2) Scoring sobre producción

`src/scoring.py` consume el modelo `@champion` (fallback a `models/best_model.pkl`) y procesa los batches de `data/prod/`:

| Batch | Target | Anomalía intencional | Resultado esperado |
|-------|--------|----------------------|--------------------|
| `prod1` | sí | ninguna (datos sanos) | **OK** |
| `prod2` | sí | target ×100 (coma decimal mal ubicada) | **ALERT** (performance) |
| `prod3` | no | `bmi` ×1000 (sin decimal) | **ALERT** (drift de covariables) |

Los targets se parsean con `decimal=','`. Las predicciones se guardan en la tabla `prod_predictions`.

### 3) Monitoreo por batch

`src/monitoring.py` calcula tres ejes y consolida el peor estado:

- **Performance** (si hay target): RMSE/MAE/R²/MAPE; ratio vs baseline de validación. Umbrales: <1.25 OK, 1.25–2.0 WARNING, >2.0 ALERT.
- **Drift**: PSI por feature vs `training_dataset`. Umbrales: <0.1 OK, 0.1–0.25 WARNING, ≥0.25 ALERT.
- **Schema**: rangos numéricos y dominios categóricos. Umbrales: 0% OK, ≤5% WARNING, >5% ALERT.

Salidas: tabla `monitoring_runs`, reportes `results/monitoring_report_*.json` y `.txt`, y un run de MLflow por batch.

### 4) Revisar resultados

```bash
# MLflow UI
# Abrir http://localhost:5000

# Predicciones de producción
docker compose exec postgres psql -U metlife_user -d metlife_db \
  -c "SELECT batch, COUNT(*), AVG(predicted_charges) FROM prod_predictions GROUP BY batch;"

# Estado de monitoreo por batch
docker compose exec postgres psql -U metlife_user -d metlife_db \
  -c "SELECT batch, status, rmse, rmse_ratio, max_psi, schema_violation_pct FROM monitoring_runs ORDER BY run_time DESC;"

# Reporte legible
cat results/monitoring_report_*.txt
```

### 5) Configuración Data Science (`config.yaml`)

Parámetros de modelo, entrenamiento, promoción de champion y monitoreo están centralizados en [`config.yaml`](config.yaml) (plantilla: [`config.yaml.example`](config.yaml.example)).

**Editar ahí (sin tocar código):**
- `model.type`: `xgboost` o `sklearn_hist_gbm`
- `promotion.metric` / `direction`: criterio de champion (ej. `validation_rmse`, `minimize`)
- `promotion.block_if_overfitting` / `max_r2_diff`: bloquear promoción si hay overfitting (v1.1)
- `training.hyperparam_search`: `n_iter`, `cv_folds`, `scoring`, grid de hiperparámetros
- `training.feature_engineering.enabled`: activar/desactivar features compuestas (v1.1)
- `monitoring.psi` / `performance` / `schema`: umbrales OK/WARNING/ALERT
- `monitoring.enabled_axes`: ej. `[drift, schema, prediction_drift]` sin `performance` (v1.1)
- `monitoring.psi.per_feature`: umbrales PSI por feature (v1.1)
- `scoring.prod_batches_filter`: procesar solo algunos batches (v1.1)
- `scoring.reference_sample_frac`: fracción del training para drift (v1.1)

Secretos e infra (`DB_*`, `MLFLOW_TRACKING_URI`) siguen en `.env`. Variables `HYPERPARAM_ITERATIONS`, `CV_FOLDS`, `MLFLOW_*` pueden **override** valores del YAML.

Tras editar `config.yaml`, reiniciar el pipeline: `docker compose up --build`.

### 6) Dashboards en Grafana

Al hacer `docker compose up` se levanta también un servicio **Grafana** (provisionado como código) conectado a Postgres.

- URL: **http://localhost:3000** — usuario `admin`, contraseña `admin` (configurables vía `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`).
- Datasource Postgres y dashboards se cargan automáticamente desde `grafana/provisioning/` (carpeta `MetLife MLOps`).

Tres dashboards:

1. **Training data review** — análisis exploratorio genérico de `training_dataset` (distribuciones de `charges`/`age`/`bmi`, conteos categóricos, correlaciones, charges por categoría y scatter por `smoker`).
2. **Training review** — selector de `run` (mismo nombre que MLflow, ej. `train_20260604_100848`): duración, dataset usado, métricas train/val, hiperparámetros, feature importances e historial de runs. Datos en la tabla `training_runs` (poblada por `training.py`).
3. **Prod Predictions** — selector de `batch`: estado de monitoreo y comparaciones del batch vs baseline (features y predicciones) y predicciones vs target. Usa `prod_predictions`, `monitoring_runs` y `baseline_predictions`.

Variable **Environment** (`sandbox` / `prod`) en los dashboards para filtrar filas por entorno MLOps.

### 7) Sandbox y Producción (MLOps)

- **Sandbox** (default): experiment `metlife_insurance`, registry `metlife_insurance_xgb` — sin cambio de nombres.
- **Prod**: experiment `metlife_insurance_prod`, registry `metlife_insurance_xgb_prod` — ver [`config.prod.yaml`](config.prod.yaml).
- Mismo pipeline (`training` + `scoring`); champion auto-promovido **dentro** de cada registry.
- Promoción explícita Sandbox → Prod: `src/promote_sandbox_to_prod.py` (opcional `--run-scoring`).
- Servicio Docker: `docker compose --profile prod run --rm ml_pipeline_prod`.

Guía operativa: [`GUIDE.md`](GUIDE.md) sección 3.7.

Las tablas extra (`training_runs`, `training_feature_importance`, `baseline_predictions`) las gestiona `src/grafana_utils.py` y **no alteran el flujo del pipeline** (escritura tolerante a fallos). Para validar las queries SQL de los dashboards:

```bash
docker compose up -d postgres
python sql_test/run_tests.py
```

---

## 📈 Resultados del Modelo

### **Métricas de Performance**

| Métrica | Train | Validation | Interpretación |
|---------|-------|------------|----------------|
| **R²** | 0.8806 | **0.8353** | Explica 83.5% de la varianza ✅ |
| **Adjusted R²** | 0.8793 | **0.8276** | Ajustado por número de features |
| **RMSE** | $4,199 | **$4,835** | Error cuadrático medio ✅ |
| **MAE** | $1,814 | **$2,102** | Error absoluto promedio ✅ |
| **MAPE** | 14.51% | **17.70%** | Error porcentual ✅ |

**Análisis de Overfitting:**
- Diferencia R² (train - val): **4.53%**
- Status: **No significant overfitting** ✅
- Interpretación: El modelo generaliza muy bien

### **Contexto de las Métricas**

**R² = 0.8353:**
- El modelo explica **83.53%** de la variabilidad en los costos
- Para un problema de seguros con variables no observadas (historial médico, medicaciones), esto es **excelente**
- Benchmark típico para seguros: 0.75-0.85

**MAE = $2,102:**
- Error promedio absoluto de **$2,102**
- Sobre una media de ~$13,300, esto es **15.8%** del valor promedio
- **Muy bueno** para decisiones de pricing

**MAPE = 17.70%:**
- Error porcentual promedio de **17.7%**
- Estándar de la industria: 15-25% es excelente
- Nuestro modelo está en el **rango superior**


### **Hiperparámetros Finales**

```python
{
    'n_estimators': 500,        # Suficientes árboles para convergencia
    'max_depth': 3,             # Profundidad conservadora - evita overfitting
    'learning_rate': 0.01,      # Aprendizaje gradual
    'reg_alpha': 0.1,           # L1 regularization (feature selection)
    'reg_lambda': 1             # L2 regularization (estabilidad)
}
```

**Análisis:** Configuración conservadora que prioriza **generalización** sobre ajuste perfecto en train.


### **Feature Importance (Top 10)**

**Según correlación con target (del EDA):**

| Rank | Feature | Correlación | Insight |
|------|---------|-------------|---------|
| 1 | `bmi_smoker` | **0.845** | ⭐ **MEJOR PREDICTOR** (interacción crítica) |
| 2 | `age_smoker` | **0.789** | Interacción importante |
| 3 | `smoker` | **0.787** | Factor dominante base |
| 4 | `age_squared` | 0.300 | Captura no linealidad |
| 5 | `age` | 0.298 | Efecto consistente |
| 6 | `age_senior` | 0.239 | Umbral relevante |
| 7 | `bmi_obese` | 0.201 | Umbral clínico |
| 8 | `bmi` | 0.198 | Efecto base |
| 9 | `bmi_squared` | 0.193 | No linealidad sutil |
| 10 | `children` | 0.067 | Impacto menor |

**Conclusión:** Las 3 features más importantes (`bmi_smoker`, `age_smoker`, `smoker`) acumulan **~85%** de la capacidad predictiva.