# Guía de uso del repositorio MetLife MLOps

Guía práctica: desde el **build** con Docker hasta **modificar parámetros** y **reentrenar** el modelo, sin tocar código Python.

---

## 1. Qué hace este repo

Pipeline automatizado que:

1. Carga `data/dataset.csv` en PostgreSQL.
2. Entrena un modelo (XGBoost por defecto), lo registra en **MLflow** y promueve un **champion** si mejora la métrica configurada.
3. Hace **scoring** sobre `data/prod/` (prod1, prod2, prod3), guarda predicciones y ejecuta **monitoreo** (drift, performance, schema).
4. Opcionalmente visualiza todo en **Grafana** (http://localhost:3000).

Orquestación por defecto: `entrypoint.sh` → `db_setup.py` → `training.py` → `scoring.py`.

**Entornos MLOps:** el flujo anterior es **Sandbox** (pruebas). Existe un entorno **Prod** paralelo (mismo pipeline, otros nombres en MLflow y filas `environment=prod` en Postgres). La promoción explícita Sandbox → Prod es un script aparte (ver sección 9).

---

## 2. Requisitos

| Herramienta | Versión mínima |
|-------------|----------------|
| Docker | 20.10+ |
| Docker Compose | v2 |
| Git | cualquiera reciente |

Puertos libres en el host: **5432** (Postgres), **5000** (MLflow), **3000** (Grafana).

---

## 3. Primera vez: build y ejecución completa

### 3.1 Clonar y entrar al proyecto

```bash
cd metlife-challenge-mlops
```

### 3.2 Variables de entorno (infra y secretos)

```bash
cp .env.template .env
```

Editá `.env` solo si necesitás cambiar credenciales de DB o la URI de MLflow. Los valores por defecto funcionan con Docker Compose.

| Variable | Uso |
|----------|-----|
| `DB_*` | Conexión PostgreSQL del pipeline |
| `MLFLOW_TRACKING_URI` | En Docker: `http://mlflow:5000` |
| `HYPERPARAM_ITERATIONS` | Override opcional de `config.yaml` → `training.hyperparam_search.n_iter` |
| `CV_FOLDS` | Override opcional de `config.yaml` → `training.hyperparam_search.cv_folds` |

### 3.3 Configuración de ML

Los parámetros de modelo, entrenamiento, champion y monitoreo están en **`config.yaml`** en la raíz del repo.

- Archivo activo: [`config.yaml`](config.yaml)
- Plantilla de referencia: [`config.yaml.example`](config.yaml.example)

No hace falta copiar nada si ya existe `config.yaml`; solo editarlo antes de reentrenar.

### 3.4 Build de imágenes

```bash
docker compose build
```

Construye:

- `ml_pipeline` — pipeline Python (training + scoring)
- `mlflow` — servidor de tracking
- `grafana` — dashboards (imagen oficial)

### 3.5 Levantar todo y correr el pipeline

```bash
docker compose up --build
```

O en segundo plano:

```bash
docker compose up -d --build
docker compose logs -f ml_pipeline
```

**Qué ocurre:**

1. Postgres y MLflow esperan healthcheck.
2. `ml_pipeline` ejecuta el pipeline completo (puede tardar varios minutos por la búsqueda de hiperparámetros).
3. Al terminar, el contenedor `ml_pipeline` sale con código 0 si todo fue bien.

**Servicios que quedan corriendo** (con `up -d`): `postgres`, `mlflow`, `grafana`.

### 3.6 URLs útiles

| Servicio | URL | Credenciales |
|----------|-----|--------------|
| MLflow UI | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | `admin` / `admin` |
| PostgreSQL | `localhost:5432` | `metlife_user` / `metlife_pass`, DB `metlife_db` |

### 3.7 Entornos Sandbox y Prod (MLOps)

| | Sandbox (default) | Prod |
|--|-------------------|------|
| Variable | `ML_ENV=sandbox` | `ML_ENV=prod` |
| MLflow experiment | `metlife_insurance` | `metlife_insurance_prod` |
| Model Registry | `metlife_insurance_xgb` | `metlife_insurance_xgb_prod` |
| Config | `config.yaml` | `config.prod.yaml` (merge sobre base) |
| Artefactos locales | `models/`, `results/`, `logs/` | `models/prod/`, etc. |
| Postgres | `environment='sandbox'` | `environment='prod'` |

**Sandbox** (`docker compose up`): entrenar, probar HP, monitoreo; el champion se actualiza solo si mejora la métrica en el registry sandbox.

**Prod** (mismo pipeline, otro registry):

```bash
docker compose --profile prod run --rm ml_pipeline_prod
```

O:

```bash
docker compose run --rm -e ML_ENV=prod -e CONFIG_PATH=/app/config.prod.yaml \
  -v ./config.prod.yaml:/app/config.prod.yaml:ro ml_pipeline
```

**Promover el champion de Sandbox a Prod** (acción manual, no corre en el entrypoint):

```bash
docker compose run --rm --entrypoint python ml_pipeline src/promote_sandbox_to_prod.py
```

Con scoring en prod tras la promoción:

```bash
docker compose run --rm --entrypoint python ml_pipeline src/promote_sandbox_to_prod.py --run-scoring
```

En Grafana, usá la variable **Environment** (`sandbox` / `prod`) en los dashboards.

**Nota:** los batches `prod1`/`prod2`/`prod3` en `data/prod/` son datos de scoring del challenge; no son el entorno MLOps Prod.

---

## 4. Dónde se guarda cada cosa

| Salida | Ubicación |
|--------|-----------|
| Modelo local | `./models/best_model.pkl` |
| Metadata del modelo | `./models/best_model_metadata.json` |
| Reporte de entrenamiento | `./results/training_report_*.txt` |
| Reporte de monitoreo | `./results/monitoring_report_*.json` y `.txt` |
| Logs del pipeline | `./logs/pipeline_*.log` |
| MLflow runs / registry | UI + `./mlartifacts/` |
| Datos en DB | tablas `training_dataset`, `prod_predictions`, `monitoring_runs`, `training_runs` (columna `environment`) |
| Prod (MLOps) | `config.prod.yaml`, registry `metlife_insurance_xgb_prod`, script `src/promote_sandbox_to_prod.py` |
| Grafana dashboards | carpeta `MetLife MLOps` en la UI |

---

## 5. Cómo modificar parámetros (sin tocar código)

### 5.1 Regla general

| Archivo | Quién lo edita | Para qué |
|---------|----------------|----------|
| **`config.yaml`** | Data Science | Modelo, HP, champion, features, umbrales de monitoreo, batches |
| **`.env`** | Ops / Dev | Passwords, hosts, overrides puntuales (`HYPERPARAM_ITERATIONS`, etc.) |

Después de cambiar `config.yaml`, hay que **volver a ejecutar** el pipeline (o al menos training/scoring según el cambio).

### 5.2 Parámetros más usados en `config.yaml`

#### Modelo y champion

```yaml
model:
  type: xgboost              # o sklearn_hist_gbm
  registry_name: metlife_insurance_xgb
  champion_alias: champion

promotion:
  metric: validation_rmse    # validation_rmse | validation_mae | validation_r2
  direction: minimize        # minimize | maximize (usar maximize con r2)
  block_if_overfitting: false
  max_r2_diff: 0.15
```

El champion se promueve solo si la métrica del run nuevo es **mejor** que la del champion actual en MLflow.

#### Entrenamiento

```yaml
training:
  target_transform: log1p    # log1p | none
  test_size: 0.2
  hyperparam_search:
    n_iter: 50               # más iteraciones = más lento, más exploración
    cv_folds: 5
    scoring: neg_root_mean_squared_error
    grid:
      model__n_estimators: [100, 200, 300, 500]
      # ... resto del grid
```

#### Feature engineering (activar/desactivar)

```yaml
training:
  feature_engineering:
    enabled:
      bmi_smoker: true
      age_smoker: false      # ejemplo: desactivar una feature
```

#### Monitoreo (umbrales de alerta)

```yaml
monitoring:
  enabled_axes: [performance, drift, schema, prediction_drift]
  psi:
    warning: 0.10
    alert: 0.25
    per_feature:
      bmi: {warning: 0.08, alert: 0.20}
  performance:
    warning_ratio: 1.25
    alert_ratio: 2.0
  schema:
    alert_pct: 5.0
```

#### Scoring (batches de producción)

```yaml
scoring:
  prod_batches_filter: null           # null = todos; o [prod1, prod3]
  reference_sample_frac: 1.0          # < 1.0 acelera drift usando muestra
```

### 5.3 Importante: valores categóricos en YAML

Usá **comillas** en dominios categóricos; si no, YAML interpreta `yes`/`no` como booleanos:

```yaml
# Correcto
smoker: ["yes", "no"]

# Evitar
smoker: [yes, no]
```

### 5.4 Ver la config efectiva en los logs

Al iniciar cada etapa verás una línea similar a:

```
Config: model=xgboost | champion_metric=validation_rmse (minimize) | HP n_iter=50 cv=5 | PSI warn/alert=0.1/0.25 | axes=[...]
```

---

## 6. Reentrenar el modelo

“Reentrenar” implica volver a correr **training** (y normalmente **db_setup** antes, porque recrea tablas de producción).

### 6.1 Flujo recomendado (cambió `config.yaml`)

```bash
# 1. Editar config.yaml (métricas, grid, modelo, etc.)

# 2. Rebuild solo si cambiaste requirements.txt o código en src/
docker compose build ml_pipeline

# 3. Pipeline completo (db_setup + training + scoring)
docker compose up ml_pipeline
```

Con `docker compose up ml_pipeline` se levantan dependencias (postgres, mlflow) y se ejecuta solo el servicio del pipeline una vez.

### 6.2 Solo reentrenar (sin rescoring)

Útil para iterar rápido en hiperparámetros:

```bash
docker compose up -d postgres mlflow

docker compose run --rm --entrypoint python ml_pipeline src/db_setup.py
docker compose run --rm --entrypoint python ml_pipeline src/training.py
```

`db_setup` **borra y recrea** `training_dataset`, `prod_predictions` y `monitoring_runs`. No borra `training_runs` (historial para Grafana).

### 6.3 Solo scoring (modelo ya entrenado)

Si ya tenés un champion en MLflow y solo querés reprocesar batches de prod:

```bash
docker compose up -d postgres mlflow

docker compose run --rm --entrypoint python ml_pipeline src/scoring.py
```

### 6.4 Cambiar dataset de entrenamiento

1. Reemplazá o actualizá `data/dataset.csv` (mismas columnas: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`).
2. Opcional: en `config.yaml` → `paths.training_csv`.
3. Corré `db_setup` + `training` (o pipeline completo).

---

## 7. Verificar que el reentrenamiento funcionó

### 7.1 Logs

```bash
docker compose logs ml_pipeline
# o
type logs\pipeline_*.log    # Windows
cat logs/pipeline_*.log     # Linux/Mac
```

Buscá: `TRAINING PIPELINE COMPLETADO`, `promovido a '@champion'` o mensaje de que no superó al champion.

### 7.2 MLflow

Abrir http://localhost:5000 → experimento `metlife_insurance` → último run `train_YYYYMMDD_HHMMSS` → métricas y artefactos.

Model Registry → modelo `metlife_insurance_xgb` → alias **champion**.

### 7.3 Archivos locales

```bash
dir models
dir results\training_report_*.txt
```

### 7.4 PostgreSQL (opcional)

```bash
docker compose exec postgres psql -U metlife_user -d metlife_db -c "SELECT run_name, val_rmse, is_champion FROM training_runs ORDER BY started_at DESC LIMIT 5;"
```

### 7.5 Monitoreo post-scoring

```bash
docker compose exec postgres psql -U metlife_user -d metlife_db -c "SELECT batch, status, rmse, max_psi FROM monitoring_runs ORDER BY run_time DESC LIMIT 5;"
```

Resultados esperados del challenge (con config por defecto):

| Batch | Estado típico | Motivo |
|-------|---------------|--------|
| prod1 | OK | Datos y target coherentes |
| prod2 | ALERT | Target corrupto (performance) |
| prod3 | ALERT | Drift en BMI (sin target) |

---

## 8. Grafana (después del pipeline)

1. `docker compose up -d grafana` (o ya está arriba con `compose up`).
2. http://localhost:3000 → login `admin` / `admin`.
3. Carpeta **MetLife MLOps**:
   - **Training data review** — EDA de `training_dataset`
   - **Training review** — selector de run (`train_...`)
   - **Prod Predictions** — selector de batch + comparación vs baseline

Si editás dashboards en la UI y querés persistirlos en git, exportá el JSON y reemplazá los archivos en `grafana/dashboards/`.

---

## 9. Escenarios frecuentes

### Más iteraciones de hiperparámetros (más lento)

En `config.yaml`:

```yaml
training:
  hyperparam_search:
    n_iter: 100
```

O en `.env`: `HYPERPARAM_ITERATIONS=100` (override).

### Probar otro algoritmo

```yaml
model:
  type: sklearn_hist_gbm
```

Rebuild y `docker compose up ml_pipeline`.

### Entrenar con menos features

Desactivá features en `training.feature_engineering.enabled` y reentrená.

### Monitorear solo drift (sin performance)

```yaml
monitoring:
  enabled_axes: [drift, schema, prediction_drift]
```

Útil cuando no hay target (prod3).

### Reset total de base de datos

```bash
docker compose down -v
docker compose up --build
```

Borra volúmenes de Postgres y Grafana; pierde historial de `training_runs` en DB.

---

## 10. Solución de problemas

| Problema | Qué revisar |
|----------|-------------|
| `ml_pipeline` sale al instante | `docker compose logs ml_pipeline` — fallo en db_setup/training |
| MLflow no responde | `docker compose ps` — servicio `mlflow` healthy en :5000 |
| Todos los batches schema ALERT | `smoker`/`sex` en YAML deben ir entre comillas `"yes"` |
| `docker compose run ... scoring.py` corre todo el pipeline | Usar `--entrypoint python` como en la sección 6.3 |
| Cambios en `config.yaml` no aplican | Montaje `./config.yaml:/app/config.yaml:ro` en compose; reiniciar run |
| Champion no se actualiza | El nuevo run debe **mejorar** `promotion.metric` vs el champion actual |

---

## 11. Resumen rápido (cheat sheet)

```bash
# Primera vez / reentrenar todo
docker compose up --build

# Editar parámetros ML
notepad config.yaml   # o tu editor

# Solo training
docker compose run --rm --entrypoint python ml_pipeline src/training.py

# Solo scoring
docker compose run --rm --entrypoint python ml_pipeline src/scoring.py

# MLflow
start http://localhost:5000

# Grafana
start http://localhost:3000

# Parar
docker compose down
```

---

## 12. Referencias en el repo

| Documento | Contenido |
|-----------|-----------|
| [README.md](README.md) | Overview y detalle MLOps |
| [challenge_ml.md](challenge_ml.md) | Enunciado del challenge |
| [CHALLENGE_ENTREGA.md](CHALLENGE_ENTREGA.md) | Entrega y decisiones de diseño |
| [config.yaml.example](config.yaml.example) | Todos los parámetros comentables |

Para profundizar en el código: `src/training.py`, `src/scoring.py`, `src/monitoring.py`, `src/config_loader.py`.
