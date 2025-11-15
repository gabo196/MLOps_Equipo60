import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "obesity_classifier"
VERSION = 3  # la versión que quieras usar (la 3 que se ve en tu UI)

def export_model():
    print(f"Obteniendo modelo '{MODEL_NAME}' versión {VERSION} desde el Model Registry...")

    # 👉 Usar el MISMO tracking URI que en train.py
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Traer info de la versión del modelo
    mv = client.get_model_version(MODEL_NAME, str(VERSION))
    print(f"  - Status de la versión: {mv.status}")
    print(f"  - Source (carpeta del modelo): {mv.source}")

    # 2. Cargar modelo desde el source
    model = mlflow.sklearn.load_model(mv.source)

    # 3. Guardarlo como random_forest.joblib en models/
    os.makedirs("models", exist_ok=True)
    output_path = os.path.join("models", "random_forest.joblib")
    joblib.dump(model, output_path)

    print(f"\n✅ Modelo exportado correctamente a: {output_path}")

if __name__ == "__main__":
    export_model()
