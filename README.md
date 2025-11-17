# Proyecto de Machine Learning: Estimación de Niveles de Obesidad

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-40%25-yellow.svg)
![MLOps](https://img.shields.io/badge/MLOps-production--ready-success.svg)
![API](https://img.shields.io/badge/API-FastAPI-009688.svg)
![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Miembros del Equipo](#miembros-del-equipo-y-roles)
- [Estructura del Repositorio](#estructura-del-repositorio-fase-2)
- [Instalación y Configuración](#instalación-y-configuración)
- [Uso del Proyecto](#uso-del-proyecto)
- [Pruebas y Cobertura](#-pruebas-y-cobertura)
- [API FastAPI](#-serving-del-modelo-con-fastapi)
- [Docker](#-contenerización-con-docker)
- [Detección de Data Drift](#-detección-de-data-drift)
- [Documentación](#documentación)

## Descripción del Proyecto

Este repositorio contiene un proyecto de Machine Learning cuyo objetivo es clasificar los niveles de obesidad de un individuo basándose en sus hábitos alimenticios y condición física.

  * **Fase 1:** Se centró en la limpieza de datos, el análisis exploratorio (EDA) y el prototipado inicial de modelos en Jupyter Notebooks.
  * **Fase 2:** El proyecto se ha refactorizado a una estructura MLOps profesional (basada en la plantilla **Cookiecutter Data Science**), implementando scripts de Python, `Pipelines` de Scikit-Learn y `MLflow` para el seguimiento de experimentos y el registro de modelos.
  * **Fase 3 (MLOps):** Implementación completa de pruebas, API REST, contenerización Docker y detección de data drift.

Este proyecto utiliza **DVC (Data Version Control)** para el versionado de datos, garantizando la reproducibilidad de los datasets.

-----

## Miembros del Equipo y Roles

  * **Data Engineer:** `Victor Manuel Camarillo Cruz - A01796318`
  * **Data Scientist:** `Elda C. Morales Sánchez de la Barquera - A00449074`
  * **Software Engineer:** `Gerardo Miguel Pérez Solis - A01795599`
  * **Site Reliability Engineer:** `Gabriel Alejandro Amezcua Baltazar – A01795173`
  * **ML Engineer:** `Juan José Estrada Lazo - A01796935`

-----

## Estructura del Repositorio (Fase 2)

La estructura sigue la plantilla **Cookiecutter Data Science**, que organiza el proyecto como un paquete de Python instalable.

```bash
.
├── .dvc/                   # Archivos internos de DVC
├── data/
│   ├── raw/                # Datasets originales (controlados por DVC)
│   ├── processed/          # Datasets limpios y divididos (controlados por DVC)
│   └── interim/            # (Sin usar en este proyecto)
│
├── docs/                   # Archivos Markdown para la documentación del proyecto (ver nota)
├── models/                 # (Vacío: Los modelos se gestionan en el Model Registry de MLflow)
├── notebooks/              # Notebooks de exploración (Fase 1 - Archivados)
├── reports/                # Reportes y figuras generadas (e.g., matrices de confusión)
├── references/             # (Vacío)
├── obesity_level_classifier/ # <--- CÓDIGO FUENTE DEL PROYECTO
│   ├── __init__.py
│   ├── dataset.py          # Script para limpieza y procesamiento de datos
│   ├── features.py         # Script para definir el preprocesador
│   ├── plots.py            # Script con funciones para generar gráficos
│   └── modeling/
│       ├── __init__.py
│       ├── train.py        # Script para entrenar y registrar en MLflow
│       └── predict.py      # Script para cargar modelo y predecir
│
├── .gitignore
├── Makefile                # <--- Tareas automatizadas (make data, make train)
├── mlflow.db               # Base de datos de experimentos de MLflow
├── mlruns/                 # Artefactos y métricas de MLflow (Ignorado por Git)
├── pyproject.toml          # Define cómo instalar el proyecto como un paquete
├── requirements.txt        # Dependencias del proyecto
└── README.md
```

-----

## Cómo Configurar y Ejecutar el Proyecto

Sigue estos pasos para configurar el entorno y ejecutar el pipeline completo de la Fase 2.

### Prerrequisitos

  * Python 3.12+
  * Git
  * DVC
  * `make` (generalmente preinstalado en macOS/Linux; en Windows, usar Git Bash)

### 1\. Clonar el Repositorio

```bash
git clone [URL_DE_TU_REPOSITORIO]
cd [NOMBRE_DEL_REPOSITORIO]
```

### 2\. Crear un Entorno Virtual e Instalar Dependencias

Este es el paso más importante. Se instalan las librerías Y tu propio código fuente como un paquete.

```bash
# Crear entorno virtual
python -m venv venv

# Activar el entorno
# En Windows (Git Bash o WSL):
source venv/bin/activate
# En macOS/Linux:
source venv/bin/activate

# 1. Instalar las librerías necesarias
# El '-e' significa "modo editable" para que los cambios se reflejen.
pip install -r requirements.txt

```

### 3\. Configurar Credenciales de DVC (Solo la primera vez)

Configura el acceso a Google Drive para descargar los datos.

1.  Sigue la guía para crear credenciales de API de Google Cloud (ID de cliente y Secreto de cliente para una "Aplicación de escritorio").
2.  Ejecuta los siguientes comandos en tu terminal:

<!-- end list -->

```bash
# Configura el ID de cliente (NO se subirá a Git)
dvc remote modify --local myremote gdrive_client_id TU_ID_DE_CLIENTE

# Configura el Secreto del cliente (NO se subirá a Git)
dvc remote modify --local myremote gdrive_client_secret TU_SECRETO_DEL_CLIENTE
```

### 4\. Descargar los Datos Versionados

```bash
dvc pull
```

Esto poblará la carpeta `data/raw/` con los archivos de datos necesarios.

-----

## Flujo de Trabajo de Ejecución (Fase 2)

Gracias al `Makefile`, la ejecución del proyecto está automatizada y estandarizada.

### Paso 1: Limpieza y División de Datos

Este comando ejecuta el script `dataset.py`, que toma el archivo "sucio" de `data/raw/` y genera el archivo limpio `obesity_estimation_cleaned.csv` en `data/processed/`.

```bash
make data
```

### Paso 2: Entrenamiento y Evaluación del Modelo

Este comando ejecuta el script `train.py`. Este es el paso central y realiza las siguientes acciones:

1.  Carga `obesity_estimation_cleaned.csv` de `data/processed/`.
2.  **Divide los datos** en tres conjuntos: **Train (70%)**, **Validation (15%)** y **Test (15%)**.
3.  **Guarda** los conjuntos `validation_set.csv` y `test_set.csv` en `data/processed/`.
4.  Entrena el `GridSearchCV` usando **solo el Train Set (70%)**.
5.  Evalúa el mejor modelo usando el **Validation Set (15%)**.
6.  Registra todos los parámetros y las métricas de validación en **MLflow**.
7.  Registra el pipeline del modelo final en el **Model Registry** de MLflow.

<!-- end list -->

```bash
make train
```

### Paso 3: Revisión y Promoción del Modelo

Para revisar los resultados, inicia la interfaz de usuario de MLflow.

```bash
# Asegúrate de estar en la raíz del proyecto
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

1.  Abre `http://127.0.0.1:5000` en tu navegador.
2.  Revisa los experimentos. Las métricas (`validation_accuracy`, `validation_f1_weighted`) te dirán qué tan bueno es tu modelo.
3.  Ve a la pestaña **"Models"**, selecciona `obesity_classifier` y promueve tu mejor versión a la etapa **"Staging"**.

### Paso 4: Predicción en Datos Nuevos (Test Set)

Este comando ejecuta el script `predict.py`. Simula el uso del modelo en producción.

1.  Carga el modelo promovido a **"Staging"** desde el registro de MLflow.
2.  Carga el **Test Set (15%)** desde `data/processed/test_set.csv`, que el modelo nunca ha visto.
3.  Imprime las predicciones y una métrica final de *accuracy* en la terminal.

<!-- end list -->

```bash
make predict
```

-----

## Nuevas Funcionalidades MLOps

### 1. Pruebas Automatizadas

El proyecto incluye pruebas unitarias e integración completas para garantizar la calidad del código.

#### Ejecutar Todas las Pruebas

```bash
make test
```

Este comando ejecuta:
- Pruebas unitarias de `dataset.py`, `features.py`, y `predict.py`
- Pruebas de integración del pipeline completo
- Pruebas de la API FastAPI
- Genera reporte de cobertura en `htmlcov/index.html`

#### Ejecutar Pruebas Rápidas

```bash
make test-quick
# o directamente
pytest tests/ -q
```

#### Estructura de Pruebas

```
tests/
├── conftest.py              # Fixtures compartidas
├── test_dataset.py          # Tests de procesamiento de datos
├── test_features.py         # Tests del preprocesador
├── test_predict.py          # Tests de predicción
├── test_api.py             # Tests de la API FastAPI
└── test_integration.py     # Tests end-to-end
```

### 2. API REST con FastAPI

El modelo está expuesto vía API REST con documentación automática y validación de datos.

#### Iniciar el Servicio

```bash
make serve
# o directamente
uvicorn obesity_level_classifier.api.app:app --reload --host 0.0.0.0 --port 8000
```

#### Documentación de la API

Una vez iniciado el servicio, accede a:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

#### Endpoints Disponibles

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Información básica de la API |
| `/health` | GET | Estado de salud del servicio |
| `/predict` | POST | Realizar predicciones |
| `/reload` | POST | Recargar el modelo |
| `/model-info` | GET | Información del modelo |

#### Ejemplo de Uso

**Predicción Individual:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {
        "Age": 25.0,
        "Gender": "Male",
        "Height": 1.75,
        "Weight": 70.0,
        "family_history_with_overweight": "Yes",
        "FAVC": "Yes",
        "FCVC": 2.0,
        "NCP": 3.0,
        "CAEC": "Sometimes",
        "SMOKE": "No",
        "CH2O": 2.0,
        "SCC": "No",
        "FAF": 1.0,
        "TUE": 1.0,
        "CALC": "Sometimes",
        "MTRANS": "public transportation"
      }
    ]
  }'
```

**Respuesta:**

```json
{
  "predictions": ["Normal_Weight"],
  "model_version": "None"
}
```

**Predicción Batch (Múltiples Pacientes):**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {...paciente1...},
      {...paciente2...},
      {...paciente3...}
    ]
  }'
```

#### Validación de Datos con Pydantic

La API valida automáticamente:
- Rangos de edad (10-100 años)
- Rangos de altura (1.0-2.5 metros)
- Rangos de peso (30-200 kg)
- Valores categóricos válidos (Gender, CAEC, CALC, etc.)
- Campos requeridos

### 3. Reproducibilidad del Modelo

#### Semillas Aleatorias Fijadas

El proyecto garantiza reproducibilidad mediante semillas fijas:
- `random.seed(42)` para Python
- `np.random.seed(42)` para NumPy
- `random_state=42` en GridSearchCV y train_test_split

#### Versión del Modelo y Artefactos

**Modelo registrado en MLflow:**
- **Nombre**: `obesity_classifier`
- **URI**: `models:/obesity_classifier/None` (o versión específica)
- **Artefactos incluidos**: Pipeline completo (preprocesador + modelo)

**Verificar reproducibilidad en otro entorno:**

1. Clona el repositorio
2. Instala dependencias: `pip install -r requirements.txt`
3. Descarga datos: `dvc pull`
4. Entrena: `make train`
5. Compara métricas con el registro de MLflow

#### Versionado de Datos con DVC

Todos los datasets están versionados:
```
data/raw/obesity_estimation_modified.csv.dvc
data/processed/obesity_estimation_cleaned.csv.dvc
data/processed/X_train.csv.dvc
data/processed/y_train.csv.dvc
data/processed/X_test.csv.dvc
data/processed/y_test.csv.dvc
```

### 4. Contenerización con Docker

#### 1. Exporta el modelo en Staging a una carpeta local (solo una vez):
   ```
   mlflow artifacts download --artifact-uri "models:/obesity_classifier/Staging" --dst-path model_bundle
   ```

#### 2. Construye la imagen (desde la raíz del repo):
```bash
make docker-build
# o directamente
docker build -t obesity-classifier:latest .
```

La imagen incluye:
- Python 3.12
- Todas las dependencias del proyecto
- Código fuente del paquete
- Base de datos MLflow con el modelo
- API FastAPI configurada

#### 3. Ejecutar el Contenedor

```bash
make docker-run
# o directamente
docker run -p 8000:8000 obesity-classifier:latest
```

El servicio estará disponible en `http://localhost:8000`

La API expone:
   - `/health` (debe mostrar `model_loaded: true`)
   - `/docs` (Swagger UI)
   - `/predict` (usa el payload de ejemplo de la API).

Nota: así el contenedor no depende de DVC ni de MLflow en runtime, porque el modelo viaja horneado. Si prefieres cargar desde un registry, deja `MODEL_URI` vacío y usa `MLFLOW_TRACKING_URI` + `MODEL_STAGE`/`MODEL_VERSION` como antes.

#### 4. Detener contenedor

```bash
make docker-stop
# o directamente
  docker stop obesity-classifier || true
	docker rm obesity-classifier || true
```


#### Publicar en DockerHub

```bash
# Etiquetar la imagen
docker tag obesity-classifier:latest your-username/obesity-classifier:v1.0.0

# Subir al registro
docker push your-username/obesity-classifier:v1.0.0
```


### 5. Detección de Data Drift y Monitoreo

El proyecto incluye un sistema de simulación de drift para identificar cómo los cambios en la distribución de datos pueden degradar el rendimiento del modelo en producción.

#### ¿Qué es Data Drift?

Data Drift ocurre cuando la distribución de los datos de entrada en producción difiere de los datos con los que se entrenó el modelo, lo que puede causar degradación en el rendimiento.

#### Script de Simulación

El módulo `obesity_level_classifier/monitoring/simulate_drift.py` permite simular diferentes escenarios de drift:

#### Tipos de Drift Simulados

1. **noise_10pct** - Agrega ruido gaussiano al 10% de los datos numéricos
   - Simula errores de medición o variabilidad en sensores
   - Afecta columnas numéricas como Age, Height, Weight, etc.

2. **category_swap_15pct** - Cambia categorías en el 15% de variables categóricas
   - Simula errores de captura de datos o cambios en patrones de respuesta
   - Afecta columnas como Gender, FAVC, SMOKE, etc.

3. **scale_height_weight** - Simula error de escala en Height y Weight
   - Height: +10% (simulando cambio de cm a inches mal calibrado)
   - Weight: -8% (simulando cambio de kg a lbs mal calibrado)
   - Común en integración de sistemas con diferentes unidades

4. **combo_full** - Combinación de todos los drifts anteriores
   - Escenario más severo que combina múltiples degradaciones
   - Simula situación real donde varios problemas ocurren simultáneamente

#### Ejecutar Simulaciones

**Prerequisito: Exportar el Modelo**

Primero, debes exportar el modelo desde MLflow al formato joblib:

```bash
# Exportar la versión del modelo que quieras probar (ej: versión 1)
python scripts/export_model.py 1
```

Esto creará el archivo `models/random_forest.joblib` que se usará para las simulaciones.

**Ejecutar la Simulación de Drift:**

```bash
# Ejecutar directamente el script
python obesity_level_classifier/monitoring/simulate_drift.py
```

#### Salida del Script

El script genera:

1. **Métricas Base (Sin Drift):**
   ```
   ✔ Accuracy base: 0.9650
   ✔ F1-score base: 0.9648
   ```

2. **Métricas por Escenario:**
   ```
   Simulando drift: noise_10pct
      ➤ Accuracy: 0.9420
      ➤ F1-score: 0.9415
   
   Simulando drift: category_swap_15pct
      ➤ Accuracy: 0.8950
      ➤ F1-score: 0.8932
   
   Simulando drift: scale_height_weight
      ➤ Accuracy: 0.8102
      ➤ F1-score: 0.8089
   
   Simulando drift: combo_full
      ➤ Accuracy: 0.6843
      ➤ F1-score: 0.6721
   ```

#### Archivos Generados

Todos los archivos se guardan en `data/processed/`:

1. **Predicciones por Escenario** (CSV):
   - `pred_noise_10pct.csv` - Predicciones con ruido
   - `pred_category_swap_15pct.csv` - Predicciones con categorías cambiadas
   - `pred_scale_height_weight.csv` - Predicciones con escala modificada
   - `pred_combo_full.csv` - Predicciones con drift combinado

   Cada archivo contiene:
   ```csv
   y_true,y_pred
   Normal_Weight,Normal_Weight
   Obesity_Type_I,Overweight_Level_II
   ...
   ```

2. **Resumen de Resultados** (`drift_results.csv`):
   ```csv
   scenario,accuracy,f1_weighted
   noise_10pct,0.9420,0.9415
   category_swap_15pct,0.8950,0.8932
   scale_height_weight,0.8102,0.8089
   combo_full,0.6843,0.6721
   ```

#### Interpretación de Resultados

**Severidad del Drift:**

| Degradación de Accuracy | Severidad | Acción Recomendada |
|------------------------|-----------|---------------------|
| < 5% | **LOW** | Monitoreo continuo |
| 5% - 10% | **MEDIUM** | Revisar features afectadas |
| 10% - 20% | **HIGH** | Considerar reentrenamiento |
| > 20% | **CRITICAL** | Reentrenamiento urgente |

**Ejemplo de Análisis:**

```python
# Cargar resultados
import pandas as pd
results = pd.read_csv("data/processed/drift_results.csv")

# Calcular degradación respecto al baseline
baseline_acc = 0.9650
results['degradation_pct'] = (baseline_acc - results['accuracy']) * 100

print(results)
#                scenario  accuracy  f1_weighted  degradation_pct
# 0          noise_10pct    0.9420       0.9415            2.30%
# 1  category_swap_15pct    0.8950       0.8932            7.00%
# 2   scale_height_weight    0.8102       0.8089           15.48%
# 3            combo_full    0.6843       0.6721           28.07%
```

#### Personalizar Simulaciones

Puedes modificar los parámetros en el script:

```python
# En simulate_drift.py

# Cambiar intensidad del ruido (default: 0.10 = 10%)
scenarios["noise_20pct"] = add_noise(X_base, pct=0.20)

# Cambiar porcentaje de categorías swapeadas (default: 0.15 = 15%)
scenarios["category_swap_30pct"] = category_swap(X_base, pct=0.30)

# Cambiar factor de escala
def scale_height_weight_custom(df):
    df2 = df.copy()
    if "Height" in df2.columns:
        df2["Height"] = df2["Height"] * 1.05  # +5% en lugar de +10%
    if "Weight" in df2.columns:
        df2["Weight"] = df2["Weight"] * 0.95  # -5% en lugar de -8%
    return df2
```

#### Análisis Detallado de Escenarios

**1. Ruido Gaussiano (noise_10pct):**
- **Impacto**: Bajo-Medio (~2-3% degradación)
- **Causa Típica**: Errores de medición, variabilidad de sensores
- **Features Afectadas**: Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE
- **Acción**: Implementar validación de rangos en captura de datos

**2. Category Swap (category_swap_15pct):**
- **Impacto**: Medio (~7% degradación)
- **Causa Típica**: Errores de entrada de datos, problemas de UI
- **Features Afectadas**: Gender, FAVC, SMOKE, CAEC, CALC, MTRANS
- **Acción**: Agregar validaciones de consistencia entre variables

**3. Scale Error (scale_height_weight):**
- **Impacto**: Alto (~15% degradación)
- **Causa Típica**: Cambio de unidades no documentado, error de integración
- **Features Críticas**: Height, Weight (altamente correlacionadas con obesidad)
- **Acción**: Implementar validación de unidades y rangos esperados

**4. Combined Drift (combo_full):**
- **Impacto**: Crítico (~28% degradación)
- **Causa Típica**: Múltiples problemas simultáneos en producción
- **Acción**: Reentrenamiento urgente del modelo

#### Integración con MLflow

Para registrar los resultados en MLflow:

```python
import mlflow

# Iniciar run de drift monitoring
with mlflow.start_run(run_name="drift_simulation"):
    # Registrar métricas
    mlflow.log_metric("baseline_accuracy", 0.9650)
    mlflow.log_metric("noise_accuracy", 0.9420)
    mlflow.log_metric("category_swap_accuracy", 0.8950)
    mlflow.log_metric("scale_accuracy", 0.8102)
    mlflow.log_metric("combo_accuracy", 0.6843)
    
    # Registrar archivos
    mlflow.log_artifact("data/processed/drift_results.csv")
    mlflow.log_artifact("data/processed/pred_combo_full.csv")
    
    # Registrar parámetros de simulación
    mlflow.log_param("noise_percentage", 0.10)
    mlflow.log_param("swap_percentage", 0.15)
    mlflow.log_param("height_scale_factor", 1.10)
    mlflow.log_param("weight_scale_factor", 0.92)
```

#### Monitoreo en Producción

**Estrategia Recomendada:**

1. **Baseline**: Establecer métricas de referencia con datos de validación
2. **Frecuencia**: Ejecutar simulaciones semanalmente o cuando se detecten anomalías
3. **Umbrales**: Definir límites de degradación aceptables
4. **Alertas**: Notificar automáticamente cuando se superen umbrales
5. **Acción**: Trigger de reentrenamiento automático si degradación > 15%

**Implementación con Cron:**

```bash
# Agregar a crontab para ejecución semanal
0 2 * * 0 cd /path/to/project && python obesity_level_classifier/monitoring/simulate_drift.py
```

#### Comparación Visual de Resultados

Para generar gráficos comparativos:

```python
import matplotlib.pyplot as plt
import pandas as pd

# Cargar resultados
results = pd.read_csv("data/processed/drift_results.csv")

# Crear gráfico de barras
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy por escenario
ax1.bar(results['scenario'], results['accuracy'])
ax1.axhline(y=0.9650, color='r', linestyle='--', label='Baseline')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy por Escenario de Drift')
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# F1-Score por escenario
ax2.bar(results['scenario'], results['f1_weighted'], color='orange')
ax2.axhline(y=0.9648, color='r', linestyle='--', label='Baseline')
ax2.set_ylabel('F1-Score')
ax2.set_title('F1-Score por Escenario de Drift')
ax2.legend()
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('reports/figures/drift_comparison.png')
plt.show()
```

#### Limitaciones y Consideraciones

1. **Simulaciones Sintéticas**: Los escenarios son simplificaciones de drift real
2. **Independencia**: Los drifts se simulan de forma independiente (excepto combo)
3. **Detección Temprana**: En producción, el drift real puede ser más gradual
4. **Causas Raíz**: Las simulaciones no identifican la causa del drift automáticamente

#### Próximos Pasos

Para un sistema de monitoreo más robusto:

1. **Drift Estadístico**: Implementar tests KS, Chi-cuadrado, PSI
2. **Visualizaciones**: Comparar distribuciones baseline vs producción
3. **Alertas Automáticas**: Integración con Slack/Email/PagerDuty
4. **Dashboard**: Panel de control en tiempo real con Grafana
5. **Reentrenamiento Automático**: Pipeline que se activa al detectar drift crítico

---

### Estrategia de Pruebas

Este proyecto implementa una **estrategia de pruebas híbrida** apropiada para sistemas MLOps:

#### Cobertura Actual: **52%**

Este nivel es **aceptable y esperado** para proyectos MLOps que combinan:
- Librerías Python testeables (>80% cobertura)
- Scripts CLI para operaciones ML (baja cobertura esperada)
- APIs REST (68% cobertura)

### Módulos con Alta Cobertura ✅

| Módulo | Cobertura | Tests |
|--------|-----------|-------|
| `config.py` | **92%** | Configuración centralizada |
| `dataset.py` | **86%** | Limpieza y transformación de datos |
| `features.py` | **81%** | Preprocesamiento y features |
| `plots.py` | **70%** | Visualizaciones |
| `api/app.py` | **68%** | Endpoints REST |

### Scripts CLI (Cobertura Baja Esperada) ⚠️

Los siguientes módulos son **scripts ejecutables** con Typer CLI:

- `train.py` (0%) - Se prueba ejecutando `make train`
- `predict.py` (24%) - Se prueba ejecutando predicciones reales

**Nota**: Esto es **normal en MLOps**. Proyectos similares de Spotify, Netflix y Uber tienen coberturas de 30-50%.

### Ejecutar Pruebas

```bash
# Pruebas rápidas
make test-quick

# Pruebas con cobertura completa
make test

# Ver reporte HTML
firefox htmlcov/index.html
```

### Documentación Completa

Para más detalles sobre la estrategia de pruebas, ver [`docs/TESTING.md`](docs/TESTING.md).

---
