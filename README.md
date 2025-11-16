# Proyecto de Machine Learning: Estimación de Niveles de Obesidad

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Descripción del Proyecto

Este repositorio contiene un proyecto de Machine Learning cuyo objetivo es clasificar los niveles de obesidad de un individuo basándose en sus hábitos alimenticios y condición física.

  * **Fase 1:** Se centró en la limpieza de datos, el análisis exploratorio (EDA) y el prototipado inicial de modelos en Jupyter Notebooks.
  * **Fase 2:** El proyecto se ha refactorizado a una estructura MLOps profesional (basada en la plantilla **Cookiecutter Data Science**), implementando scripts de Python, `Pipelines` de Scikit-Learn y `MLflow` para el seguimiento de experimentos y el registro de modelos.

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

## Servir la API en Docker (modelo incluido)

Para empaquetar la API y el modelo en una imagen auto-contenida:

1. Exporta el modelo en Staging a una carpeta local (solo una vez):
   ```
   mlflow artifacts download --artifact-uri "models:/obesity_classifier/Staging" --dst-path model_bundle
   ```
2. El Dockerfile copia `model_bundle/` y usa `MODEL_URI` para cargarlo. Si ajustas dependencias, usa una versión de pandas con wheel para Python 3.12 (por ejemplo, `pandas==2.1.4`).
3. Construye la imagen (desde la raíz del repo):
   ```
   docker build -t obesity-api .
   ```
4. Ejecuta el contenedor:
   ```
   docker run -p 8000:8000 obesity-api
   ```
   La API expone:
   - `/health` (debe mostrar `model_loaded: true`)
   - `/docs` (Swagger UI)
   - `/predict` (usa el payload de ejemplo de la API).

Nota: así el contenedor no depende de DVC ni de MLflow en runtime, porque el modelo viaja horneado. Si prefieres cargar desde un registry, deja `MODEL_URI` vacío y usa `MLFLOW_TRACKING_URI` + `MODEL_STAGE`/`MODEL_VERSION` como antes.
