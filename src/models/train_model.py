"""
Script principal para entrenar el modelo de Machine Learning.
Este script carga los datos procesados, construye el pipeline completo
(preprocesador + modelo), y realiza el seguimiento de experimentos con MLflow.

Ejemplo de uso (desde la raíz del proyecto):
    python src/models/train_model.py
"""

import pandas as pd
import logging
import typer
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

# Importar el constructor del preprocesador desde nuestro módulo de features
from src.features.build_features import build_preprocessor

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes ---
PROCESSED_DATA_PATH = Path("data/processed/obesity_estimation_cleaned.csv")
MODEL_OUTPUT_DIR = Path("models/")
TARGET_COLUMN = "NObesity"
MLFLOW_EXPERIMENT_NAME = "Obesity_Level_Estimation"

def load_data(data_path: Path) -> (pd.DataFrame, pd.Series):
    """Carga los datos procesados y los divide en X (features) e y (target)."""
    logging.info(f"Cargando datos procesados desde {data_path}")
    df = pd.read_csv(data_path)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    logging.info(f"Datos cargados. X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def build_full_pipeline() -> Pipeline:
    """Construye el pipeline completo de Scikit-Learn."""
    preprocessor = build_preprocessor()
    rf_model = RandomForestClassifier(random_state=42)
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', rf_model)
    ])
    logging.info("Pipeline completo (preprocesador + modelo) construido.")
    return full_pipeline

def plot_confusion_matrix(y_true, y_pred, labels, filename="reports/figures/confusion_matrix.png"):
    """Genera y guarda un gráfico de la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de Confusión")
    plt.ylabel("Etiqueta Verdadera")
    plt.xlabel("Etiqueta Predicha")
    plt.savefig(filename)
    logging.info(f"Matriz de confusión guardada en {filename}")
    return filename

def main():
    """Función principal para ejecutar el pipeline de entrenamiento."""
    
    # 0. Configurar MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # 1. Cargar datos
    X, y = load_data(PROCESSED_DATA_PATH)
    
    # 2. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Construir Pipeline
    pipeline = build_full_pipeline()

    # 4. Definir Grid de Hiperparámetros
    # Nota: Los parámetros del modelo deben prefijarse con 'model__'
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20],
        'model__min_samples_leaf': [2, 4]
    }

    # 5. Configurar GridSearchCV
    # Usamos 'f1_weighted' como métrica principal, es más robusta para multiclase
    grid_search = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        cv=5, 
        scoring='f1_weighted', 
        n_jobs=-1, 
        verbose=1
    )

    logging.info("Iniciando GridSearchCV para encontrar los mejores hiperparámetros...")
    
    # Iniciar un "run" padre de MLflow para el GridSearchCV
    with mlflow.start_run(run_name="RandomForest_GridSearch") as parent_run:
        grid_search.fit(X_train, y_train)
        
        logging.info("GridSearchCV completado.")
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # 6. Evaluar el mejor modelo
        y_pred = best_pipeline.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)
        class_labels = sorted(y.unique())

        # Loggear parámetros y métricas en MLflow
        logging.info("Loggeando resultados en MLflow...")
        mlflow.log_params({k.replace('model__', ''): v for k, v in best_params.items()})
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1)
        
        # Loggear métricas por clase
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                mlflow.log_metric(f"f1_{class_name}", metrics['f1-score'])
        
        # 7. Loggear artefactos (matriz de confusión)
        cm_path = plot_confusion_matrix(y_test, y_pred, labels=class_labels)
        mlflow.log_artifact(cm_path)

        # 8. Loggear y registrar el modelo
        # Esto es para que MLflow infiera la firma del modelo.
        input_example = X_train.head(5)

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            input_example=input_example,
            registered_model_name="obesity_classifier" # Registra el modelo en el Model Registry
        )
        
        logging.info(f"Modelo registrado en MLflow como 'obesity_classifier' con run_id: {parent_run.info.run_id}")
        
    print("\n--- Resultados del Mejor Modelo ---")
    print(f"Mejores Hiperparámetros: {best_params}")
    print(f"Accuracy en Test: {accuracy:.4f}")
    print(f"F1-Score (Weighted) en Test: {f1:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    typer.run(main)