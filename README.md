# MetLife MLOps Challenge — Guía de uso

Guía práctica para reproducir el challenge: **entrenamiento con MLflow**, **scoring** sobre `data/prod/`, **monitoreo** por batch y **modo online** opcional. Cubre build con Docker, parámetros en `config.yaml` y reentrenamiento sin tocar código Python.

---

## 1. Qué hace este repo

Pipeline automatizado que:

1. Carga `data/dataset.csv` en PostgreSQL.
2. Entrena un modelo (XGBoost por defecto), lo registra en **MLflow** y promueve un **champion** si mejora la métrica configurada.
3. Hace **scoring** sobre `data/prod/` (prod1, prod2, prod3), guarda predicciones y ejecuta **monitoreo** (drift, performance, schema) vía `src/monitoring.py` **dentro de** `scoring.py`.
4. Opcionalmente visualiza todo en **Grafana** (http://localhost:3000).

**Orquestación por defecto** (`docker compose up`): `entrypoint.sh` → `db_setup.py` → `training.py` → `scoring.py`.

**No incluidos en el `up` automático** (se ejecutan a mano):

- `src/online_scoring.py` — simulación de inferencia online (sección 3.8).
- `src/promote_sandbox_to_prod.py` — copia del champion sandbox al registry prod (sección 3.7).

**Entornos MLOps:** el flujo anterior es **Sandbox** (pruebas). Existe un entorno **Prod** paralelo (mismo pipeline, otros nombres en MLflow y filas `environment=prod` en Postgres). La promoción explícita Sandbox → Prod es el script anterior (ver sección 3.7).

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

Las carpetas de salida (`logs/`, `logs/prod/`, `models/`, `models/prod/`, `results/`, `results/prod/`, `mlartifacts/`) vienen en el repo con `.gitkeep` para que Docker pueda montarlas sin error de permisos. Si faltan:

```bash
bash scripts/init-output-dirs.sh
```

En **Windows**, ese script requiere Git Bash o WSL. Si no lo tenés, `docker compose up` suele crear las carpetas al montar volúmenes; el script solo asegura `.gitkeep` y permisos.

Si ya corriste Docker y `logs/` quedó creada como root (error `Permission denied` en `tee`), corregí permisos en el host:

```bash
sudo chown -R "$(id -u):$(id -g)" logs models results mlartifacts
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
2. `ml_pipeline` ejecuta el pipeline completo 
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

### 3.8 Modo online (simulación productiva v1)

Simula requests **sin target** muestreando features de `data/dataset.csv`, a tasa configurable (default **10 req/s**, **1600** requests en `config.yaml` → ~**160 s**). Usa el champion del entorno activo y guarda en `online_predictions` (sin `actual_charges`). En MLflow: **1 run por sesión** (`stage=online`).

**Requisito:** haber corrido `scoring.py` antes en el mismo entorno (para `baseline_predictions`).

**Rebuild:** si modificaste código en `src/`, corré `docker compose build ml_pipeline` antes del run. Cambios solo en `config.yaml` no requieren rebuild.

```bash
# Sandbox (defaults en config.yaml)
docker compose run --rm --entrypoint python ml_pipeline src/online_scoring.py

# Profile online
docker compose --profile online run --rm ml_pipeline_online

# Prod
docker compose run --rm -e ML_ENV=prod -e CONFIG_PATH=/app/config.prod.yaml \
  --entrypoint python ml_pipeline src/online_scoring.py
```

Parámetros en `config.yaml` → sección `online` (ver también section 5.2). Overrides por env: `ONLINE_N_SAMPLES`, `ONLINE_RATE_PER_SECOND`, `ONLINE_MONITORING_WINDOW_SIZE`, `ONLINE_MONITORING_MAX_SAMPLES`.

**Drift sintético en BMI:** flag `--bmi-anomaly` (o `online.bmi_anomaly.enabled: true`). Desde `request_seq` 500, el BMI muestreado se multiplica en rampa lineal hasta `max_multiplier` (configurable en `online.bmi_anomaly.max_multiplier`; p. ej. **1.7** en `config.yaml`, **4.0** en `config.yaml.example`) en las siguientes **1000** muestras; después se mantiene en ese factor. Útil para validar PSI y auto-retrain en Grafana.

```bash
docker compose run --rm --entrypoint python ml_pipeline \
  src/online_scoring.py --bmi-anomaly
```

**Reentrenamiento automático** (`online.auto_retrain`, máximo **1** por sesión online):

- `prediction_psi` ≥ umbral **warning** de `monitoring.psi` y más de `min_samples_prediction` requests (default 700).
- Cualquier PSI por feature ≥ umbral **alert**.
- Persiste fila en `online_retrain_alerts` (marcadores en el time series de Prediction PSI en Grafana).
- Lanza un run MLflow hijo con nombre `retrain_drift_feature_{feature}_YYYYMMDD_HHMMSS` o `retrain_drift_prediction_YYYYMMDD_HHMMSS`, y tags `triggered_by=online_drift`, `trigger_description`, etc.

**Monitoreo temporal:** cada `monitoring_window_size` requests (default 100) se ejecuta `monitor_batch` sobre las **últimas** `monitoring_max_samples` predicciones (default 1500), no todo el historial de la sesión. Se guarda en `online_monitoring_snapshots` + `online_monitoring_psi`. Al cierre, `monitoring_runs` usa la misma ventana (máx. 1500).

**Grafana:** dashboard **Online Predictions** — fila superior con filas analizadas, gauges de PSI, series temporales de prediction PSI y evolución de todos los PSI; debajo, comparación vs baseline de entrenamiento.

---

## 4. Dónde se guarda cada cosa

En Docker, las rutas locales de la tabla (salvo Postgres) son **bind mounts**: carpetas/archivos del host montados dentro del contenedor. En `ml_pipeline` / `ml_pipeline_online` / `ml_pipeline_prod` (`docker-compose.yaml`):

| Host (tu repo) | Contenedor |
|----------------|------------|
| `./models` | `/app/models` |
| `./results` | `/app/results` |
| `./logs` | `/app/logs` |
| `./config.yaml` | `/app/config.yaml` (solo lectura) |
| `./config.prod.yaml` | `/app/config.prod.yaml` (solo lectura) |

Lo que el pipeline escribe en `/app/models`, `/app/results` o `/app/logs` queda **en el host** al salir el contenedor. MLflow usa `./mlartifacts` → `/mlartifacts` en el servicio `mlflow`.

| Salida | Ubicación en el host |
|--------|-----------|
| Modelo local (sandbox) | `./models/best_model.pkl` |
| Modelo local (prod MLOps) | `./models/prod/best_model.pkl` |
| Metadata del modelo | `./models/best_model_metadata.json` (o `models/prod/`) |
| Reporte de entrenamiento | `./results/training_report_*.txt` (o `results/prod/`) |
| Reporte de monitoreo | `./results/monitoring_report_*.json` y `.txt` |
| Logs del pipeline | `./logs/pipeline_*.log` (o `logs/prod/`) |
| MLflow runs / registry | UI + `./mlartifacts/` (bind mount del servicio `mlflow`) |
| Datos en DB (batch) | `training_dataset`, `prod_predictions`, `monitoring_runs`, `training_runs`, `baseline_predictions` (columna `environment`) |
| Datos en DB (online) | `online_predictions`, `online_monitoring_snapshots`, `online_monitoring_psi`, `online_retrain_alerts` |
| Prod (MLOps) | `config.prod.yaml`, registry `metlife_insurance_xgb_prod`, script `src/promote_sandbox_to_prod.py` |
| Online | sesiones `online_YYYYMMDD_HHMMSS`, dashboard Grafana **Online Predictions** |
| Grafana dashboards | carpeta `MetLife MLOps` en la UI; JSON en `grafana/dashboards/` |

---

## 5. Cómo modificar parámetros (sin tocar código)

### 5.1 Regla general

| Archivo | Quién lo edita | Para qué |
|---------|----------------|----------|
| **`config.yaml`** | Data Science /MLOPs| Modelo, HP, champion, features, umbrales de monitoreo, batches |
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

#### Modo online (simulación y auto-retrain)

```yaml
online:
  n_samples: 1600
  rate_per_second: 10
  flush_every: 50
  monitoring_window_size: 100
  monitoring_max_samples: 1500
  monitoring_axes: [drift, schema, prediction_drift]
  bmi_anomaly:
    enabled: false
    start_at_sample: 500
    duration_samples: 1000
    max_multiplier: 1.7              # rampa 1x -> max_multiplier
  auto_retrain:
    enabled: true
    min_samples_prediction: 700        # prediction PSI > warning solo tras N requests
```

Ver plantilla comentada en [`config.yaml.example`](config.yaml.example).

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

`db_setup` **borra y recrea** `training_dataset`, `prod_predictions`, `monitoring_runs` y también las tablas **online** (`online_predictions`, `online_monitoring_snapshots`, `online_monitoring_psi`, `online_retrain_alerts`). No borra `training_runs` (historial para Grafana).

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
   - **Online Predictions** — variables `environment` y `session`; fila superior con gauges PSI y series temporales (Prediction PSI con **marcadores de retrain** desde `online_retrain_alerts`); debajo, comparación vs `baseline_predictions`

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

### Probar drift online + reentrenamiento automático

1. Pipeline completo o al menos scoring (para `baseline_predictions`):

   ```bash
   docker compose up ml_pipeline
   # o solo: docker compose run --rm --entrypoint python ml_pipeline src/scoring.py
   ```

2. Asegurate de `online.auto_retrain.enabled: true` en `config.yaml`.

3. Simulá drift en BMI y corré online:

   ```bash
   docker compose build ml_pipeline   # si hubo cambios en src/
   docker compose run --rm --entrypoint python ml_pipeline \
     src/online_scoring.py --bmi-anomaly
   ```

4. Verificá en Grafana (**Online Predictions** → elegir `session`) el punto de alerta en Prediction PSI; en MLflow, run online + hijo `retrain_drift_feature_*` o `retrain_drift_prediction_*`.

### Reset total de base de datos

```bash
docker compose down -v
docker compose up --build
```

Borra volúmenes de Postgres y Grafana; pierde historial de `training_runs` en DB.

---

## 10. Tests y CI

### Ejecutar tests localmente

```bash
pip install -r requirements.txt
pytest tests/
```

Los tests unitarios cubren funciones puras de `src/monitoring.py` y `src/training.py`. No requieren PostgreSQL ni MLflow.

### Catálogo de tests (28)

#### `tests/test_monitoring.py` — monitoreo (`src/monitoring.py`)

| Test | Descripción breve |
|------|-------------------|
| `test_identical_distributions_near_zero` | PSI numérico ≈ 0 cuando referencia y actual son iguales. |
| `test_shifted_distribution_positive` | PSI numérico > 0 cuando la distribución actual está desplazada. |
| `test_constant_reference_returns_zero` | PSI numérico = 0 si la referencia es constante (sin bins útiles). |
| `test_same_categories_near_zero` | PSI categórico ≈ 0 con las mismas categorías en ambos lados. |
| `test_new_category_positive` | PSI categórico > 0 cuando aparece una categoría nueva en el batch actual. |
| `test_ok_and_warning` | `worst_status` devuelve `WARNING` si algún eje está en warning. |
| `test_warning_and_alert` | `worst_status` devuelve `ALERT` si algún eje está en alerta. |
| `test_empty_returns_ok` | `worst_status` sin estados de entrada devuelve `OK`. |
| `test_identical_predictions_ok` | Drift de predicciones en `OK` si referencia y actual coinciden. |
| `test_heavily_shifted_predictions_alert` | Drift de predicciones en `ALERT` con distribución muy desplazada. |
| `test_valid_rows_ok` | `check_schema` en `OK` con filas dentro de rangos y dominios válidos. |
| `test_out_of_range_numeric_warning` | `check_schema` en `WARNING` por valor numérico fuera de rango (1 % de filas). |
| `test_invalid_category_warning` | `check_schema` en `WARNING` por categoría inválida (1 % de filas). |
| `test_perfect_predictions` | `compute_performance` con RMSE 0 y estado `OK` si predicción = real. |
| `test_rmse_ratio_alert` | `compute_performance` en `ALERT` cuando el ratio RMSE supera el umbral vs baseline. |
| `test_extracts_expected_keys` | `snapshot_from_report` extrae PSI, schema y metadatos del reporte de batch. |
| `test_feature_psi_alert_triggers` | `evaluate_online_retrain_trigger` dispara retrain por PSI de feature ≥ alerta. |
| `test_low_sample_count_no_prediction_trigger` | No dispara retrain por PSI de predicción si hay pocas muestras. |
| `test_no_trigger_when_psi_low` | No dispara retrain cuando PSI de features y predicciones están bajo umbral. |

#### `tests/test_training.py` — entrenamiento (`src/training.py`)

| Test | Descripción breve |
|------|-------------------|
| `test_default_prefix` | `_build_run_name(None)` genera nombre con prefijo `train_` + timestamp. |
| `test_feature_psi_alert` | `_build_run_name` con drift de feature → `retrain_drift_feature_{feat}_…`. |
| `test_prediction_trigger` | `_build_run_name` con drift de predicción → `retrain_drift_prediction_…`. |
| `test_drops_id_and_created_at` | `prepare_features_target` elimina `id`, `created_at` y `charges` de X. |
| `test_adds_engineered_columns` | `prepare_features_target` agrega las 6 features derivadas del config. |
| `test_target_log_transform` | `prepare_features_target` aplica `log1p` al target según `config.yaml`. |
| `test_split_sizes` | `split_data` respeta `test_size=0.2` y alinea longitudes train/val. |
| `test_returns_model_prefixed_keys` | `define_hyperparameter_grid` devuelve claves `model__…` desde config. |
| `test_returns_column_transformer` | `create_preprocessor` devuelve `ColumnTransformer` num + cat. |

### GitHub Actions

El workflow [`.github/workflows/python-app.yml`](.github/workflows/python-app.yml) corre en cada push y pull request hacia `main`:

1. **flake8** — errores de sintaxis y lint básico.
2. **pytest tests/** — suite unitaria.

### Bloquear merge a `main` sin CI verde

La configuración del workflow no impide merges por sí sola. En GitHub, después de que el workflow haya corrido al menos una vez en un PR:

1. **Settings → Branches** (o **Rules → Rulesets**) → regla para la rama `main`.
2. Activar **Require a pull request before merging** (si usás PRs).
3. Activar **Require status checks to pass before merging**.
4. Marcar el check **`build`** (puede aparecer como **Python application / build**).
5. Recomendado: **Require branches to be up to date before merging**.

Hasta configurar el paso 3–4, un PR puede mergearse aunque falle pytest.

---

## 11. Referencias en el repo

| Documento | Contenido |
|-----------|-----------|
| **README.md** (este archivo) | Cómo usar el challenge de punta a punta |
| [challenge_ml.md](challenge_ml.md) | Enunciado y criterios de aceptación |
| [CHALLENGE_ENTREGA.md](CHALLENGE_ENTREGA.md) | Arquitectura, decisiones de diseño y resultados de entrega |
| [config.yaml.example](config.yaml.example) | Todos los parámetros comentables |

Para profundizar en el código: `src/training.py`, `src/scoring.py`, `src/monitoring.py`, `src/config_loader.py`.
