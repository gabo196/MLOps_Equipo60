# Proyecto de Machine Learning: Estimación de Niveles de Obesidad

## Descripción del Proyecto

Este repositorio contiene un proyecto de Machine Learning cuyo objetivo es clasificar los niveles de obesidad de un individuo basándose en sus hábitos alimenticios y condición física.

  * **Fase 1:** Se centró en la limpieza de datos, el análisis exploratorio (EDA) y el prototipado inicial de modelos en Jupyter Notebooks.
  * **Fase 2:** El proyecto se ha refactorizado a una estructura MLOps profesional, implementando scripts de Python, `Pipelines` de Scikit-Learn y `MLflow` para el seguimiento de experimentos y el registro de modelos.

Este proyecto utiliza **DVC (Data Version Control)** para el versionado de datos, garantizando la reproducibilidad de los datasets.

-----

## Miembros del Equipo y Roles

  - **Data Engineer:** ` Victor Manuel Camarillo Cruz - A01796318  `
  - **Data Scientist:** ` Elda C. Morales Sánchez de la Barquera - A00449074  `
  - **Software Engineer:** `Gerardo Miguel Pérez Solis - A01795599`
  - **Site Reliability Engineer:** ` Gabriel Alejandro Amezcua Baltazar - A01795173  `
  - **ML Engineer:** ` Juan José Estrada Lazo - A01796935  `

-----

## Estructura del Repositorio (Fase 2)

La estructura sigue una plantilla profesional (tipo Cookiecutter) que separa el código fuente, los datos y los notebooks.

```bash
.
├── .dvc/                   # Archivos internos de DVC
├── data/
│   ├── raw/                # Datasets originales (controlados por DVC)
│   │   ├── obesity_estimation_modified.csv.dvc
│   │   └── obesity_estimation_original.csv.dvc
│   └── processed/          # Datasets limpios (controlados por DVC)
│       └── obesity_estimation_cleaned.csv.dvc
├── models/                 # (Vacío: Los modelos ahora se registran en MLflow)
├── notebooks/              # Notebooks de exploración (Fase 1 - Archivados)
│   ├── 1_Limpieza_y_EDA.ipynb
│   └── 2_Modelo_Predictivo.ipynb
├── reports/                # Reportes y figuras generadas
├── src/                    # Código fuente refactorizado
│   ├── __init__.py
│   ├── data/
│   │   └── make_dataset.py     # Script para limpieza y procesamiento de datos
│   ├── features/
│   │   └── build_features.py   # Script para definir el preprocesador
│   └── models/
│       ├── train_model.py      # Script para entrenar y registrar en MLflow
│       └── predict_model.py    # Script para cargar modelo y predecir
├── .gitignore
├── mlflow.db               # Base de datos de experimentos de MLflow
├── mlruns/                 # Artefactos y métricas de MLflow (Ignorado por Git)
├── requirements.txt        # Dependencias del proyecto
└── README.md
```

-----

## Cómo Configurar y Ejecutar el Proyecto

Sigue estos pasos para configurar el entorno y ejecutar el pipeline completo de la Fase 2.

### Prerrequisitos

  - Python 3.8+
  - Git
  - DVC

### 1\. Clonar el Repositorio

```bash
git clone [URL_DE_TU_REPOSITORIO]
cd [NOMBRE_DEL_REPOSITORIO]
```

### 2\. Crear un Entorno Virtual e Instalar Dependencias

Asegúrate de que tu archivo `requirements.txt` contenga todas las librerías necesarias.

```bash
# Crear entorno virtual
python -m venv venv

# Activar el entorno
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# Instalar las librerías necesarias desde requirements.txt
pip install -r requirements.txt

# Si no tienes un requirements.txt, instala las librerías clave:
# pip install pandas scikit-learn dvc[gdrive] mlflow typer
```

### 3\. Configurar Credenciales de DVC (Solo la primera vez)

Este proyecto utiliza Google Drive como almacenamiento remoto. Cada miembro del equipo debe configurar su acceso de forma local y segura.

**Instrucciones:**

1.  Sigue la guía para crear credenciales de API de Google Cloud (ID de cliente y Secreto de cliente para una "Aplicación de escritorio").
2.  Ejecuta los siguientes comandos en tu terminal, reemplazando con tus credenciales personales:

<!-- end list -->

```bash
# Configura el ID de cliente (NO se subirá a Git)
dvc remote modify --local myremote gdrive_client_id TU_ID_DE_CLIENTE

# Configura el Secreto del cliente (NO se subirá a Git)
dvc remote modify --local myremote gdrive_client_secret TU_SECRETO_DEL_CLIENTE
```

### 4\. Descargar los Datos Versionados

Una vez configuradas las credenciales, descarga la última versión de los datos desde el almacenamiento remoto.

```bash
dvc pull
```

Esto poblará las carpetas `data/raw/` y `data/processed/` con los archivos de datos correspondientes.

-----

## Flujo de Trabajo de Ejecución (Fase 2)

Con el proyecto refactorizado, la ejecución se basa en scripts de Python.

### Paso 1: Limpieza de Datos

Ejecuta el script `make_dataset.py` para tomar los datos "sucios" de `data/raw/`, limpiarlos y guardar la versión procesada en `data/processed/`.

```bash
python src/data/make_dataset.py data/raw/obesity_estimation_modified.csv data/processed/obesity_estimation_cleaned.csv
```

### Paso 2: Entrenamiento y Registro del Modelo

Ejecuta el script `train_model.py`. Este script hará lo siguiente:

1.  Cargará los datos limpios de `data/processed/`.
2.  Ejecutará el `GridSearchCV` para encontrar el mejor modelo.
3.  Registrará todos los parámetros, métricas y el modelo final en **MLflow**.

<!-- end list -->

```bash
python src/models/train_model.py
```

### Paso 3: Revisión de Experimentos y Promoción del Modelo

Para revisar los resultados y gestionar los modelos, inicia la interfaz de usuario de MLflow.

```bash
# Asegúrate de estar en la raíz del proyecto
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

1.  Abre `http://127.0.0.1:5000` en tu navegador.
2.  Explora los experimentos. Verás el "run" de `RandomForest_GridSearch`.
3.  Ve a la pestaña **"Models"**.
4.  Haz clic en el modelo `obesity_classifier`.
5.  Haz clic en la última versión.
6.  Usa el botón azul **"Stage"** para promover el modelo a la etapa **"Staging"** o **"Production"**.

### Paso 4: Predicción con el Modelo Registrado

Una vez que un modelo ha sido promovido (ej. a "Staging"), puedes usar el script `predict_model.py` para cargar el modelo desde el registro de MLflow y hacer predicciones.

```bash
# Ejecutar predicciones usando el modelo en "Staging"
python src/models/predict_model.py --data-path data/processed/obesity_estimation_cleaned.csv --model-stage "Staging"

# Si quieres probar con la última versión sin promoverla:
# python src/models/predict_model.py --data-path data/processed/obesity_estimation_cleaned.csv --model-stage "None"
```

-----

## Flujo de Trabajo (Fase 1 - Archivado)

Para fines de consulta, los notebooks originales de la Fase 1 se encuentran en la carpeta `notebooks/`.

1.  `1_Limpieza_y_EDA.ipynb`: Contiene el análisis exploratorio y la lógica de limpieza original.
2.  `2_Modelo_Predictivo.ipynb`: Contiene el prototipo original del modelo y su evaluación.