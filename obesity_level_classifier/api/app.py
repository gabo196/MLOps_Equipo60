"""
Servicio FastAPI para servir el modelo de clasificación de obesidad.
Expone el modelo vía API REST con validación Pydantic.

Para ejecutar:
    uvicorn obesity_level_classifier.api.app:app --reload --host 0.0.0.0 --port 8000

Documentación disponible en:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from loguru import logger
import os

# Configuración de MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "obesity_classifier")
MODEL_STAGE = os.getenv("MODEL_STAGE", "None")
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
MODEL_URI = os.getenv("MODEL_URI")

# Inicializar FastAPI
app = FastAPI(
    title="Obesity Level Classifier API",
    description="API para predecir niveles de obesidad basado en hábitos alimenticios y condición física",
    version="1.0.0"
)

# Variable global para el modelo
model = None

# --- Modelos Pydantic para Validación ---

class PatientData(BaseModel):
    """Esquema de entrada para un paciente individual."""
    Age: float = Field(..., ge=10, le=100, description="Edad del paciente (10-100 años)")
    Gender: str = Field(..., description="Género: Male o Female")
    Height: float = Field(..., ge=1.0, le=2.5, description="Altura en metros (1.0-2.5)")
    Weight: float = Field(..., ge=30, le=200, description="Peso en kg (30-200)")
    family_history_with_overweight: str = Field(..., description="Historial familiar: Yes o No")
    FAVC: str = Field(..., description="Consumo frecuente de alimentos altos en calorías: Yes o No")
    FCVC: float = Field(..., ge=1, le=3, description="Frecuencia de consumo de vegetales (1-3)")
    NCP: float = Field(..., ge=1, le=4, description="Número de comidas principales (1-4)")
    CAEC: str = Field(..., description="Consumo de alimentos entre comidas: No, Sometimes, Frequently, Always")
    SMOKE: str = Field(..., description="¿Fuma?: Yes o No")
    CH2O: float = Field(..., ge=0, le=5, description="Consumo de agua diario en litros (0-5)")
    SCC: str = Field(..., description="Monitoreo de consumo calórico: Yes o No")
    FAF: float = Field(..., ge=0, le=5, description="Frecuencia de actividad física (0-5)")
    TUE: float = Field(..., ge=0, le=5, description="Tiempo usando dispositivos tecnológicos (0-5 horas)")
    CALC: str = Field(..., description="Consumo de alcohol: No, Sometimes, Frequently, Always")
    MTRANS: str = Field(..., description="Medio de transporte usado: Automobile, Motorbike, Bike, Public Transportation, Walking")
    
    @field_validator('Gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female', 'male', 'female']:
            raise ValueError('Gender debe ser Male o Female')
        return v.title()
    
    @field_validator('family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC')
    def validate_yes_no(cls, v):
        if v.lower() not in ['yes', 'no']:
            raise ValueError('El valor debe ser Yes o No')
        return v.title()
    
    @field_validator('CAEC', 'CALC')
    def validate_frequency(cls, v):
        valid_values = ['no', 'sometimes', 'frequently', 'always']
        if v.lower() not in valid_values:
            raise ValueError(f'El valor debe ser uno de: {", ".join(valid_values)}')
        return v.title()
    
    @field_validator('MTRANS')
    def validate_transport(cls, v):
        valid_values = ['automobile', 'motorbike', 'bike', 'public transportation', 'walking']
        if v.lower() not in valid_values:
            raise ValueError(f'El valor debe ser uno de: {", ".join(valid_values)}')
        return v.title()

class PredictionRequest(BaseModel):
    """Esquema para una solicitud de predicción (puede ser batch)."""
    patients: List[PatientData] = Field(..., description="Lista de pacientes para predecir")

class PredictionResponse(BaseModel):
    """Esquema de respuesta de predicción."""
    predictions: List[str] = Field(..., description="Lista de predicciones de nivel de obesidad")
    model_version: Optional[str] = Field(None, description="Versión del modelo usada")
    
class HealthResponse(BaseModel):
    """Esquema de respuesta para el endpoint de salud."""
    status: str
    model_loaded: bool
    model_info: Optional[dict] = None

# --- Funciones de utilidad ---

def load_model():
    """Carga el modelo desde MLflow."""
    global model
    try:
        # Si se provee una ruta local del modelo (bundle exportado), úsala.
        if MODEL_URI:
            model_uri = MODEL_URI
        else:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            if MODEL_VERSION:
                model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
            else:
                model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            
        logger.info(f"Cargando modelo desde: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Modelo cargado exitosamente")
        return True
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        return False

# --- Event Handlers ---

@app.on_event("startup")
async def startup_event():
    """Carga el modelo al iniciar la aplicación."""
    logger.info("Iniciando aplicación...")
    success = load_model()
    if not success:
        logger.warning("No se pudo cargar el modelo al inicio. El endpoint /predict no funcionará.")

# --- Endpoints ---

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz con información básica de la API."""
    return {
        "message": "Obesity Level Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Verifica el estado de salud de la API y el modelo."""
    model_loaded = model is not None
    
    model_info = None
    if model_loaded:
        model_info = {
            "model_name": MODEL_NAME,
            "model_stage": MODEL_STAGE,
            "model_version": MODEL_VERSION
        }
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_info=model_info
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Realiza predicciones de nivel de obesidad para uno o más pacientes.
    
    - **patients**: Lista de datos de pacientes
    
    Returns:
        Predicciones de nivel de obesidad para cada paciente
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. Intente recargar con /reload"
        )
    
    try:
        # Convertir los datos de entrada a DataFrame
        patients_data = [patient.dict() for patient in request.patients]
        df = pd.DataFrame(patients_data)
        
        logger.info(f"Realizando predicción para {len(df)} paciente(s)")
        
        # Realizar predicción
        predictions = model.predict(df)
        
        # Convertir a lista de strings
        predictions_list = predictions.tolist() if isinstance(predictions, np.ndarray) else list(predictions)
        
        logger.info(f"Predicción exitosa: {predictions_list}")
        
        return PredictionResponse(
            predictions=predictions_list,
            model_version=MODEL_VERSION or MODEL_STAGE
        )
        
    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la predicción: {str(e)}"
        )

@app.post("/reload", tags=["Management"])
async def reload_model():
    """
    Recarga el modelo desde MLflow.
    Útil si se ha actualizado el modelo en el Model Registry.
    """
    logger.info("Recargando modelo...")
    success = load_model()
    
    if success:
        return {"status": "success", "message": "Modelo recargado exitosamente"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al recargar el modelo"
        )

@app.get("/model-info", tags=["Management"])
async def get_model_info():
    """Retorna información sobre el modelo actualmente cargado."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    return {
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_version": MODEL_VERSION,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI
    }

# --- Manejo de Errores ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Manejador personalizado de excepciones HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Manejador general de excepciones."""
    logger.error(f"Error no manejado: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Error interno del servidor", "details": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
