# Challenge ML Ops

## Contexto
Este repositorio fue creada por el equipo de ciencia de datos (entrenamiento + scoring) para predecir costos médicos. El objetivo de este challenge es evolucionar esa solución hacia un enfoque más productivo y observable, incorporando prácticas de ML Ops.

## Objetivo
Actualizar el proyecto para que:
1. El pipeline de entrenamiento registre experimentos con MLflow y persista los mejores artefactos.
2. El pipeline de scoring consuma esos artefactos para predecir sobre archivos de producción en `data/prod/`, registre resultados y simule monitoreo.

---

## Alcance del desafío

### 1) Actualizar entrenamiento (`src/training.py`)
Implementar un flujo de entrenamiento con MLflow que permita trazabilidad completa.

#### Requerimientos mínimos
- Integrar MLflow Tracking en `src/training.py`.
- Registrar por cada ejecución:
  - parámetros (hiperparámetros, semilla, features usadas, etc.),
  - métricas de evaluación (al menos 2 métricas relevantes de regresión),
  - artefactos (modelo serializado, reporte de métricas y cualquier archivo útil para auditoría).
- Definir y registrar un criterio de "mejor modelo" (por ejemplo, menor RMSE o mayor R2).
- Guardar/registrar de forma explícita los artefactos del mejor run para que luego puedan ser consumidos por scoring.
- Incluir una configuración simple de experimento (nombre de experimento y tracking URI) mediante variables de entorno o constantes centralizadas.

#### Entregables esperados
- `src/training.py` actualizado.
- Evidencia de runs en MLflow (capturas, logs o instrucciones reproducibles).
- Estructura clara de artefactos del mejor modelo.

---

### 2) Actualizar scoring (`src/scoring.py`)
Implementar un flujo de inferencia que use el mejor artefacto registrado y procese datos de producción.

#### Requerimientos mínimos
- Consumir el modelo/artefactos generados por entrenamiento (vía MLflow o ruta de artefactos definida).
- Leer los CSV de `data/prod/`:
  - `data/prod/dataset_prod1_feats.csv.csv`
  - `data/prod/dataset_prod1_target.csv.csv`
  - `data/prod/dataset_prod2_feats.csv.csv`
  - `data/prod/dataset_prod2_target.csv.csv`
  - `data/prod/dataset_prod3_feats.csv.csv` (si no tiene target, tratarlo como batch sin etiquetas)
- Generar predicciones para cada batch disponible.
- Registrar resultados de scoring en un output reproducible (tabla en DB, CSV, o artefacto versionado).
- Cuando exista target, calcular y registrar métricas de desempeño por batch.

#### Monitoreo simulado (mínimo)
Agregar una simulación de monitoreo con al menos:
- Métricas de performance por batch (por ejemplo RMSE/MAE/R2 cuando hay target).
- Comparativa simple contra training o contra un umbral definido.
- Una señal básica de drift o cambio de distribución (por ejemplo diferencia en medias, desviaciones, PSI simplificado o chequeo de rangos) para features clave.
- Un reporte final consolidado (texto, JSON o CSV) con estado por batch: `OK`, `WARNING` o `ALERT`.

#### Entregables esperados
- `src/scoring.py` actualizado.
- Salidas de predicción y reporte de monitoreo.
- Evidencia de que se procesaron los lotes de `data/prod/`.

---

## Requisitos técnicos
- Mantener compatibilidad con la ejecución actual del proyecto (incluyendo Docker si ya está configurado).
- Documentar en `README.md` cómo ejecutar:
  1. entrenamiento con tracking,
  2. scoring sobre producción,
  3. revisión de resultados/monitoreo.
- Agregar dependencias necesarias en `requirements.txt` (por ejemplo `mlflow`) si corresponde.
- Evitar hardcodeos sensibles (usar variables de entorno para configuración principal).

---

## Criterios de aceptación
La solución se considera completa si:
1. Se pueden ejecutar entrenamiento y scoring de punta a punta sin errores.
2. Quedan registrados runs con parámetros, métricas y artefactos en MLflow.
3. Scoring usa efectivamente el mejor artefacto entrenado y no un modelo aislado/manual.
4. Se generan predicciones sobre los lotes de `data/prod/`.
5. Existe un reporte de monitoreo por batch con métricas y estado.
6. La documentación permite reproducir el flujo completo.

---

## Criterios de evaluación
- Calidad técnica de la implementación (robustez, claridad, manejo de errores).
- Trazabilidad de experimentos y artefactos.
- Calidad del diseño de scoring batch y monitoreo.
- Reproducibilidad (setup y ejecución simples).
- Buenas prácticas de ingeniería (estructura, logging, documentación).

---

## Bonus (opcional)
- Registro de modelo en MLflow Model Registry con etapa (`Staging`/`Production`).
- Script de promoción de modelo basado en métricas.
- Dashboards simples o visualizaciones de monitoreo.
- Tests unitarios básicos para utilidades de tracking y monitoreo.

---

## Nota
Si detectás ambigüedades, asumí una decisión razonable y documentala explícitamente en la solución.
