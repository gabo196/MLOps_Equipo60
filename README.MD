# Proyecto de Machine Learning: Estimación de Niveles de Obesidad

## Descripción del Proyecto

Este repositorio contiene el código y los datos para la primera fase de un proyecto de Machine Learning cuyo objetivo es clasificar los niveles de obesidad de un individuo basándose en sus hábitos alimenticios y condición física. El proyecto abarca desde la limpieza y el análisis exploratorio de datos (EDA) hasta el preprocesamiento de los mismos para prepararlos para la fase de modelado.

Este proyecto utiliza **DVC (Data Version Control)** para el versionado de datos, garantizando la reproducibilidad de los resultados.

## Fase 1 – Procesamiento y Modelado Inicial

---

## Miembros del Equipo y Roles

-   **Data Engineer:** `Victor Manuel Camarillo Cruz A01796318 `
-   **Data Scientist:** `Elda C. Morales Sánchez de la Barquera A00449074 `
-   **Software Engineer:** `Gerardo Miguel Pérez Solis A01795599`
-   **Site Reliability Engineer:** `Gabriel Alejandro Amezcua Baltazar – A01795173 `
-   **ML Engineer:** `Juan José Estrada Lazo A01796935 `

---

## Estructura del Repositorio

```bash
├── .dvc/                   # Archivos internos de DVC
├── data/
│   ├── raw/                # Datasets originales (controlados por DVC)
│   │   ├── obesity_estimation_modified.csv.dvc
│   │   └── obesity_estimation_original.csv.dvc
│   └── processed/          # Datasets limpios (controlados por DVC)
│       └── obesity_clean.csv.dvc
├── notebooks/
│   └── 1_Limpieza_y_EDA.ipynb  # Jupyter Notebook con el análisis y limpieza
│   └── 2_Modelo_Predictivo.ipynb  # Jupyter Notebook con el modelo predictivo
├── .gitignore
└── README.md
```


## Cómo Configurar y Ejecutar el Proyecto

### Prerrequisitos

-   Python 3.8+
-   Git
-   DVC

### 1. Clonar el Repositorio

```bash
git clone [URL_DE_TU_REPOSITORIO]
cd [NOMBRE_DEL_REPOSITORIO]
```

### 2. Crear un Entorno Virtual e Instalar Dependencias

Es altamente recomendable utilizar un entorno virtual para aislar las dependencias del proyecto.

```bash
# Crear entorno virtual
python -m venv venv

# Activar el entorno
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# Instalar las librerías necesarias
pip install -r requirements.txt
```

### 3. Configurar Credenciales de DVC (Solo la primera vez)

Este proyecto utiliza Google Drive como almacenamiento remoto. Cada miembro del equipo debe configurar su acceso de forma local y segura.

#### Instrucciones:

* Sigue la guía para crear credenciales de API de Google Cloud (ID de cliente y Secreto de cliente para una "Aplicación de escritorio").

* Ejecuta los siguientes comandos en tu terminal, reemplazando con tus credenciales personales:


```bash
# Configura el ID de cliente (NO se subirá a Git)
dvc remote modify --local myremote gdrive_client_id TU_ID_DE_CLIENTE

# Configura el Secreto del cliente (NO se subirá a Git)
dvc remote modify --local myremote gdrive_client_secret TU_SECRETO_DEL_CLIENTE
```

### 4. Descargar los Datos Versionados

Una vez configuradas las credenciales, descarga la última versión de los datos desde el almacenamiento remoto.

```bash
dvc pull
```

Esto poblará las carpetas data/raw/ y data/processed/ con los archivos de datos correspondientes.

### 5. Ejecutar el Notebook

Ahora puedes ejecutar el notebook para explorar el proceso de limpieza y análisis. 1_Limpieza_y_EDA.ipynb

Ahora se ejecuta notebook para entrenar, optimizar y evaluar un modelo de Machine Learning capaz de predecir el nivel de obesidad. 2_Modelo_Predictivo.ipynb