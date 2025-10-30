"""
Script para cargar un modelo registrado desde MLflow y hacer predicciones.
Esto simula cómo el modelo se usaría en producción.

Ejemplo de uso:
    python src/models/predict_model.py --model-name "obesity_classifier" --data-path data/processed/obesity_estimation_cleaned.csv --model-stage "None"
"""

import mlflow
import pandas as pd
import typer
import logging
from pathlib import Path

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_path: Path) -> pd.DataFrame:
    """Carga datos para predicción."""
    try:
        data = pd.read_csv(data_path)        
        # Se asume que el input es el formato "procesado"
        # y solo quitamos el target si existe.
        if "NObesity" in data.columns:
            data = data.drop("NObesity", axis=1)
        
        logging.info(f"Datos de entrada cargados desde {data_path}. Dimensiones: {data.shape}")
        return data
        
    except FileNotFoundError:
        logging.error(f"Error: No se encontró el archivo en {data_path}")
        return None

def main(
    data_path: Path = typer.Option(..., help="Ruta al archivo CSV con datos para predecir."),
    model_name: str = typer.Option("obesity_classifier", help="Nombre del modelo en el Model Registry."),
    model_stage: str = typer.Option("Staging", help="Etapa del modelo a usar (ej. 'Staging', 'Production').")
):
    """
    Carga el modelo especificado y predice sobre los datos de entrada.
    """
    logging.info(f"Cargando modelo '{model_name}' en etapa '{model_stage}' desde MLflow...")
    
    # 1. Configurar MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # 2. Cargar el modelo desde el Model Registry
    # mlflow.pyfunc carga el modelo como una función genérica de Python,
    # que ya incluye el preprocesador.
    try:
        model_uri = f"models:/{model_name}/{model_stage}"
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logging.error(f"Error al cargar el modelo desde MLflow: {e}")
        return

    logging.info("Modelo cargado exitosamente.")
    
    # 3. Cargar datos de entrada
    X_new = load_data(data_path)
    
    if X_new is not None:
        # 4. Realizar predicciones
        logging.info("Realizando predicciones...")
        try:
            predictions = model.predict(X_new)
            
            # 5. Mostrar resultados
            logging.info("Predicciones completadas.")
            print("\n--- Primeras 10 Predicciones ---")
            result_df = X_new.copy()
            result_df['PREDICCION_NIVEL_OBESIDAD'] = predictions
            print(result_df.head(10))
            
            # Opcional: Guardar predicciones
            output_path = "reports/predictions.csv"
            result_df.to_csv(output_path, index=False)
            logging.info(f"Predicciones guardadas en {output_path}")

        except Exception as e:
            logging.error(f"Error durante la predicción: {e}")

if __name__ == "__main__":
    typer.run(main)