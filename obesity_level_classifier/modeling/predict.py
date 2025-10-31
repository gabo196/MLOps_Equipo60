"""
Script para cargar un modelo registrado desde MLflow y hacer predicciones.
Esto simula cómo el modelo se usaría en producción.

Ejemplo de uso:
    python obesity_level_classifier/modeling/predict.py --model-name "obesity_classifier" --data-path data/processed/obesity_estimation_test.csv --model-stage "None"
"""

import mlflow
import pandas as pd
import typer
import logging
from pathlib import Path
from obesity_level_classifier.config import MODEL_NAME, PROCESSED_DATA_DIR, TEST_SET_FILE_NAME

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = typer.Typer()

@app.command()
def main(
    data_path: Path = typer.Option(PROCESSED_DATA_DIR / TEST_SET_FILE_NAME, help="Ruta al archivo CSV con datos para predecir."),
    model_name: str = typer.Option(MODEL_NAME, help="Nombre del modelo en el Model Registry."),
    model_stage: str = typer.Option("None", help="Etapa del modelo a usar (ej. 'Staging', 'Production').")
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
    try:
        X_test = pd.read_csv(data_path)
        y_test = None # Asumimos que podemos tener el target para comparar
        if "NObesity" in X_test.columns:
            y_test = X_test["NObesity"]
            X_test = X_test.drop("NObesity", axis=1)
        logging.info(f"Datos de Test cargados. Dimensiones: {X_test.shape}")
    except FileNotFoundError:
        logging.error(f"Error: No se encontró el archivo en {data_path}")
        return

    logging.info("Realizando predicciones en el Test Set...")
    
    try:
        predictions = model.predict(X_test)
        
        logging.info("Predicciones completadas.")
        print("\n--- Primeras 10 Predicciones (Test Set) ---")
        result_df = X_test.head(10).copy()
        result_df['PREDICCION_NIVEL_OBESIDAD'] = predictions[:10]
        print(result_df)
        
        # Si el test set tenía etiquetas, podemos imprimir un reporte final
        if y_test is not None:
            from sklearn.metrics import accuracy_score
            test_acc = accuracy_score(y_test, predictions)
            print("\n--- Evaluación Final en Test Set ---")
            print(f"Accuracy en Test Set (Datos nuevos): {test_acc:.4f}")

    except Exception as e:
        logging.error(f"Error durante la predicción: {e}")

if __name__ == "__main__":
    app()