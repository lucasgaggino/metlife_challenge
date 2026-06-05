# Entrega — Challenge MLOps (MetLife Insurance)

Documento de resumen para evaluación: qué pedía el challenge, problemas en los datos de producción, cómo se resolvieron, solución implementada, métricas y resultados.

Requisitos originales: [challenge_ml.md](challenge_ml.md).

---

## 1. Resumen del challenge

El repositorio predice el costo de seguro médico (`charges`, en USD) a partir de variables demográficas y de salud (`age`, `sex`, `bmi`, `children`, `smoker`, `region`). El challenge pide evolucionar el pipeline hacia un enfoque **MLOps observable**:

- **Entrenamiento** con MLflow: parámetros, métricas, artefactos y registro del mejor modelo.
- **Scoring** sobre los CSV de `data/prod/`, consumiendo ese modelo (no un `.pkl` manual aislado).
- **Monitoreo simulado** por batch: performance, drift y calidad de schema, con estado consolidado `OK` / `WARNING` / `ALERT`.

El monitoreo **no corrige** anomalías en los datos; las **detecta y clasifica**.

**Extensiones implementadas** (más allá del enunciado mínimo en `challenge_ml.md`):

- Entornos **Sandbox** y **Prod** (registries MLflow separados, columna `environment` en Postgres).
- **Grafana** con dashboards de entrenamiento, scoring batch y modo online.
- **Modo online** (`online_scoring.py`): simulación de requests sin target, monitoreo temporal (PSI) y **reentrenamiento automático** ante drift (máx. 1× por sesión).
- Promoción manual Sandbox → Prod (`promote_sandbox_to_prod.py`).
- Tests unitarios (`tests/`) y CI con GitHub Actions.

Guía operativa: [README.md](README.md).

---

## 2. Arquitectura de la solución

### 2.1 Pipeline batch (challenge core)

```mermaid
flowchart LR
  subgraph infra [Infra]
    PG[(PostgreSQL)]
    MLF[MLflow :5000]
    GF[Grafana :3000]
  end
  DB[db_setup.py]
  TR[training.py]
  SC[scoring.py + monitoring.py]
  PG --> DB
  MLF --> TR
  DB --> TR
  TR -->|register @champion| MLF
  TR --> SC
  MLF --> SC
  DB --> SC
  SC --> T1[(prod_predictions)]
  SC --> T2[(monitoring_runs)]
  SC --> T3[(baseline_predictions)]
  SC --> R[results/ bind mount]
  TR --> M[models/ bind mount]
  SC --> MLF
  TR --> GF
  SC --> GF
  PG --> GF
```

Orquestación default: `entrypoint.sh` → `db_setup` → `training` → `scoring`. Salidas locales (`models/`, `results/`, `logs/`, `config.yaml`) son **bind mounts** del host al contenedor `ml_pipeline` (ver README section 4).

### 2.2 Modo online y auto-retrain (extensión)

```mermaid
flowchart TB
  SC[scoring.py previo]
  ON[online_scoring.py]
  MON[monitoring.py checkpoints]
  TR2[training.py retrain_drift_*]
  SC -->|baseline_predictions| ON
  ON -->|cada N requests| MON
  MON -->|PSI alert o pred warning| TR2
  TR2 -->|nuevo @champion| ON
  MON --> AL[(online_retrain_alerts)]
  AL --> GF[Grafana Online Predictions]
```

No forma parte de `docker compose up`; se ejecuta a mano o con profile `online`.

---

## 3. Problemas con los datos de producción

Los batches en `data/prod/` simulan escenarios reales de degradación. Ejemplos de la **fila 1**:

| Batch | Problema | Ejemplo en archivo | Tras parseo / lectura |
|-------|----------|-------------------|------------------------|
| **prod1** | Target con coma decimal europea (una columna) | `14700,80931` | **14700.81** USD — batch sano |
| **prod2** | Misma estructura, coma “corrida” (~100×) | `1470080,931` | **1470080.93** USD — target corrupto |
| **prod3** | BMI sin punto decimal (×1000), sin target | `bmi=27929` (vs ~27.9 en training) | Covariate drift; sin métricas de performance |
| *(transversal)* | `age` en prod = training + 1 | age 20 vs 19 en fila 1 | Shift leve; no se corrige |

**Features prod1/prod2** son equivalentes entre sí (mismas columnas que training); solo cambia el archivo de target.

---

## 4. Cómo se resolvieron los problemas de datos

### 4.1 Ingesta y parseo del target (`src/scoring.py`)

**Problema técnico:** usar `pd.read_csv(..., decimal=',')` **partía** el valor porque la coma actuaba como separador de columnas (ej. `14700,80931` → dos columnas; se tomaba solo `80931`).

**Solución:** leer el CSV con `sep=';'` (una columna) y reemplazar coma decimal por punto en la serie (`str.replace(',', '.')`). Así prod1 conserva ~14 700 USD y prod2 queda con el valor corrupto ~1.47M para que el eje **performance** dispare ALERT.

### 4.2 Drift — PSI (`src/monitoring.py`)

**Problema:** bins por **cuantiles** en variables enteras (`age` con corrimiento +1 en todo el batch) generaban PSI ~0.57 y **falsos ALERT** en prod1/prod2.

**Solución:** PSI con bins de **ancho fijo** sobre el rango de `training_dataset`:

- `age` +1 → PSI ~**0.006** (OK).
- `bmi` en prod3 (escala ×1000) → PSI ~**16.5** (ALERT).

### 4.3 Schema / calidad

Reglas fijas: `age` 18–100, `bmi` 10–60, `children` 0–10, dominios categóricos `sex` / `smoker` / `region`. En prod3, **99.9%** de filas violan el rango de `bmi`.

---

## 5. Cómo se resolvió el challenge (checklist)

| Requisito (`challenge_ml.md`) | Implementación |
|------------------------------|----------------|
| MLflow en entrenamiento | Experimento `metlife_insurance`, params/métricas/artefactos por run |
| Mejor modelo registrado | Model Registry `metlife_insurance_xgb`, alias **`@champion`** |
| Criterio de mejor modelo | Menor **`validation_rmse`** en USD (`promote_if_better` en `training.py`) |
| Scoring desde artefacto MLflow | `models:/metlife_insurance_xgb@champion` (fallback `models/best_model.pkl`) |
| Procesar `data/prod/` | prod1, prod2 (con target), prod3 (sin target) |
| Salida reproducible | Tablas `prod_predictions`, `monitoring_runs`; reportes `results/monitoring_report_*` |
| Monitoreo OK/WARNING/ALERT | Tres ejes en `monitoring.py`; estado = peor eje |
| Docker / env | `docker-compose.yaml`, `.env.template`, `MLFLOW_*` |
| Documentación reproducible | [README.md](README.md) (entrenamiento, scoring, monitoreo, online) |

### 5.1 Bonus del enunciado (`challenge_ml.md`)

| Bonus | Estado |
|-------|--------|
| Model Registry + alias champion | `@champion` en sandbox y prod |
| Script de promoción | `promote_sandbox_to_prod.py` (sandbox → registry prod) |
| Dashboards de monitoreo | Grafana: Training data review, Training review, Prod Predictions, **Online Predictions** |
| Tests unitarios | `tests/test_monitoring.py`, `tests/test_training.py`; CI en `.github/workflows/python-app.yml` |

**Modelo:** XGBoost en `Pipeline` (OneHot + regresión), target `log1p(charges)`, mismas features derivadas en train y score (`utils.feature_engineering`). Config centralizada en `config.yaml` + `config_loader.py`.

**Champion — hiperparámetros (última corrida):**

| Parámetro | Valor |
|-----------|-------|
| learning_rate | 0.05 |
| max_depth | 3 |
| n_estimators | 300 |
| reg_alpha | 0.1 |
| reg_lambda | 100 |

Ver también: `models/best_model_metadata.json` y run de MLflow asociado al `@champion`.

---

## 6. Métricas e indicadores

### 6.1 Entrenamiento

Métricas en **dólares** (predicciones con `expm1` tras entrenar en log):

| Métrica | Train | Validation |
|---------|-------|------------|
| RMSE | $4,202 | **$4,897** (baseline monitoreo) |
| MAE | $1,869 | $2,251 |
| R² | 0.880 | 0.831 |
| Adjusted R² | 0.879 | 0.823 |
| MAPE | 14.2% | 17.8% |

**Overfitting:** `R²_train − R²_val` ≈ **0.049** (aceptable).

La búsqueda de hiperparámetros (`RandomizedSearchCV`) optimiza MSE en escala **log**; la promoción a champion usa solo **RMSE de validación en $**.

### 6.2 Monitoreo por batch (tres ejes)

El **estado del batch** es el **peor** entre performance, drift y schema.

| Eje | Indicadores | Umbrales |
|-----|-------------|----------|
| **Performance** (si hay target) | RMSE, MAE, R², MAPE, **rmse_ratio** = RMSE_batch / RMSE_baseline | ratio &lt; 1.25 OK; 1.25–2.0 WARNING; &gt; 2.0 ALERT |
| **Drift** | **PSI** por `age`, `bmi`, `children`, `sex`, `smoker`, `region` vs `training_dataset` | PSI &lt; 0.1 OK; 0.1–0.25 WARNING; ≥ 0.25 ALERT |
| **Schema** | % filas fuera de rango o categoría inválida | 0% OK; &gt; 0% WARNING; &gt; 5% ALERT |

**PSI (Population Stability Index):** mide cuánto cambió la distribución de una variable entre referencia (training) y el batch. Se comparan proporciones por bins; valores altos indican que los datos de entrada ya no se parecen a los de entrenamiento (no mide calidad de predicción por sí solo).

---

## 7. Resultados finales (corrida validada)

Fuente: `results/monitoring_report_20260603_100049.json` (1.338 filas por batch).

| Batch | Estado | Performance | Drift (max PSI) | Schema | Interpretación |
|-------|--------|-------------|-----------------|--------|----------------|
| **prod1** | **OK** | RMSE $4,843; ratio **0.99×**; R² 0.85 | 0.006 | 0% | Pipeline sano; modelo alineado al baseline |
| **prod2** | **ALERT** | RMSE $1.79M; ratio **366×**; R² −1.12 | 0.006 (OK) | 0% (OK) | Target corrupto; features y modelo coherentes |
| **prod3** | **ALERT** | *(sin target)* | **16.54** (bmi) | **99.9%** viol. bmi | Drift de covariables; sin evaluar performance |

**Por batch en una frase:**

- **prod1:** referencia de producción correcta.
- **prod2:** demuestra detección de **target drift** sin confundirlo con drift de features.
- **prod3:** demuestra **covariate drift** y fallo de schema cuando no hay etiquetas.

---

## 8. Extensiones MLOps (post-challenge)

### 8.1 Entornos Sandbox y Prod

| | Sandbox | Prod (MLOps) |
|--|---------|----------------|
| MLflow experiment | `metlife_insurance` | `metlife_insurance_prod` |
| Model Registry | `metlife_insurance_xgb` | `metlife_insurance_xgb_prod` |
| Config | `config.yaml` | `config.prod.yaml` (merge) |
| Postgres | `environment='sandbox'` | `environment='prod'` |

Los batches `prod1`/`prod2`/`prod3` en `data/prod/` son **datos de scoring del challenge**, no el entorno MLOps Prod.

### 8.2 Modo online (`src/online_scoring.py`)

- Simula inferencia sin target muestreando `data/dataset.csv` (tasa y volumen en `config.yaml` → `online`).
- Persiste `online_predictions`; checkpoints cada `monitoring_window_size` sobre ventana deslizante (`monitoring_max_samples`).
- Tablas temporales: `online_monitoring_snapshots`, `online_monitoring_psi`.
- Flag `--bmi-anomaly`: rampa de multiplicador en BMI para simular covariate drift (parámetros en `online.bmi_anomaly`).

### 8.3 Reentrenamiento automático por drift online

Disparadores (máximo **un** reentrenamiento por sesión online), evaluados en cada checkpoint:

| Condición | Umbral |
|-----------|--------|
| PSI de predicción | ≥ `monitoring.psi.warning` y más de `online.auto_retrain.min_samples_prediction` requests (default 700) |
| PSI por feature | ≥ `monitoring.psi.alert` (cualquier covariable) |

Acciones: run MLflow hijo con nombre `retrain_drift_feature_{feature}_YYYYMMDD_HHMMSS` o `retrain_drift_prediction_YYYYMMDD_HHMMSS`, tags `triggered_by=online_drift`, fila en `online_retrain_alerts` (marcadores en Grafana), recarga del `@champion` y actualización de `baseline_predictions`.

### 8.4 Grafana

Dashboards provisionados desde `grafana/dashboards/` (carpeta **MetLife MLOps**):

- **Training data review** — EDA de `training_dataset`
- **Training review** — métricas e importancia por run
- **Prod Predictions** — batch vs baseline (scoring)
- **Online Predictions** — PSI temporal, alertas de retrain, comparación vs baseline

Datos vía PostgreSQL (`training_runs`, `prod_predictions`, `online_monitoring_*`, etc.).

### 8.5 Calidad y CI

- `pytest tests/` — PSI, schema, performance, triggers de auto-retrain (`evaluate_online_retrain_trigger`).
- GitHub Actions: flake8 + pytest en push/PR a `main`.

---

## 9. Decisiones documentadas

- **Target prod:** una columna con coma decimal; no se “arregla” prod2 en ingestión (se detecta por performance).
- **Champion:** solo compara `validation_rmse` en USD; no MAE ni R² (`promotion` en `config.yaml`).
- **PSI:** bins de ancho fijo (no cuantiles) para evitar falsos positivos en `age`.
- **Baseline de monitoreo batch:** RMSE de validación del champion; predicciones de referencia en `baseline_predictions` (scoring).
- **Online sin target:** ejes `performance` desactivados vía `online.monitoring_axes`; drift + schema + prediction_drift.
- **Auto-retrain online:** un solo intento por sesión; el run de entrenamiento queda anidado bajo el run `stage=online` en MLflow.
- **Sandbox vs Prod:** entrenar y experimentar en sandbox; prod registry se alimenta con promoción explícita, no con el `up` default.

---

## 10. Reproducibilidad

| Objetivo | Dónde |
|----------|-------|
| Ejecutar el challenge de punta a punta | [README.md](README.md) section 3 |
| Parámetros ML sin tocar código | `config.yaml` / `config.yaml.example` |
| Modo online + drift sintético | README section 3.8 y section 9 |
| Enunciado y criterios de aceptación | [challenge_ml.md](challenge_ml.md) |

---

