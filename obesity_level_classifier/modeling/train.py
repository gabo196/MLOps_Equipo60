"""
Script principal para entrenar el modelo de Machine Learning.
Este script carga los datos procesados, construye el pipeline completo
(preprocesador + modelo), y realiza el seguimiento de experimentos con MLflow.

Ejemplo de uso (desde la raíz del proyecto):
    python obesity_level_classifier/modeling/train.py
"""

import pandas as pd
import logging
import typer
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
from obesity_level_classifier.features import build_preprocessor
from obesity_level_classifier.plots import plot_confusion_matrix

# Importar constantes
from obesity_level_classifier.config import (
    PROCESSED_DATA_DIR,
    PROCESSED_FILE_NAME,
    TARGET_COLUMN,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_NAME,
    TEST_SET_FILE_NAME,
    FIGURES_DIR
)
# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = typer.Typer()

def load_data(data_path: Path) -> pd.DataFrame:
    """Carga los datos procesados."""
    logging.info(f"Cargando datos procesados desde {data_path}")
    return pd.read_csv(data_path)

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

@app.command()
def main():
    """Función principal para ejecutar el pipeline de entrenamiento."""
    
    # 0. Configurar MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # 1. Cargar datos completos
    full_df = load_data(PROCESSED_DATA_DIR / PROCESSED_FILE_NAME)
    
    # 2. Primera división: Apartar el Test Set (15%)
    logging.info("Dividiendo datos (Paso 1): 85% Train-Val / 15% Test")
    train_val_df, test_df = train_test_split(
        full_df, 
        test_size=0.15, # 15% para Test
        random_state=42, 
        stratify=full_df[TARGET_COLUMN]
    )

    # 3. Segunda división: Separar Train (70%) y Validation (15%)
    # test_size = 0.15 / 0.85 = 0.1765 (para obtener 15% del total)
    logging.info("Dividiendo datos (Paso 2): 70% Train / 15% Validation")
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.1765, 
        random_state=42,
        stratify=train_val_df[TARGET_COLUMN]
    )
    
    # 4. GUARDAR LOS SETS DE VALIDACIÓN Y PRUEBA
    logging.info(f"Guardando Test Set en {TEST_SET_FILE_NAME}")
    test_df.to_csv(PROCESSED_DATA_DIR / TEST_SET_FILE_NAME, index=False)
    
    # 5. Definir X_train, y_train, X_val, y_val
    X_train = train_df.drop(TARGET_COLUMN, axis=1)
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df.drop(TARGET_COLUMN, axis=1)
    y_val = val_df[TARGET_COLUMN]
    
    class_labels = sorted(y_train.unique())
    logging.info(f"Datos listos: Train (n={len(X_train)}), Val (n={len(X_val)}), Test (n={len(test_df)})")

    # 6. Construir Pipeline y Grid de Hiperparámetros
    pipeline = build_full_pipeline()
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20],
        'model__min_samples_leaf': [2, 4]
    }

    # 7. Configurar GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        cv=5, # 5-fold cross-validation DENTRO del train_set (70%)
        scoring='f1_weighted', 
        n_jobs=-1, 
        verbose=1
    )

    logging.info("Iniciando GridSearchCV (Entrenando en el 70% de los datos)...")
    
    with mlflow.start_run(run_name="RandomForest_GridSearch") as parent_run:
        grid_search.fit(X_train, y_train) # <-- SOLO SE ENTRENA CON X_train (70%)
        
        logging.info("GridSearchCV completado.")
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # 8. Evaluar el mejor modelo en el VALIDATION SET (15%)
        logging.info("Evaluando el mejor modelo en el Validation Set (15%)...")
        y_pred_val = best_pipeline.predict(X_val) # <-- PREDICCIÓN EN X_val
        
        # Estas son las métricas clave que reportamos
        val_accuracy = accuracy_score(y_val, y_pred_val)
        val_f1 = f1_score(y_val, y_pred_val, average='weighted')
        
        # 9. Loggear parámetros y métricas
        logging.info("Loggeando resultados en MLflow...")
        mlflow.log_params({k.replace('model__', ''): v for k, v in best_params.items()})
        mlflow.log_metric("validation_accuracy", val_accuracy) # <-- MÉTRICA CLAVE
        mlflow.log_metric("validation_f1_weighted", val_f1) # <-- MÉTRICA CLAVE
        
        # 10. Loggear artefactos (Matriz de Confusión del Validation Set)
        cm_path = FIGURES_DIR / "validation_confusion_matrix.png"
        plot_confusion_matrix(y_val, y_pred_val, class_labels, cm_path)
        mlflow.log_artifact(str(cm_path))

        # 11. Loggear y registrar el modelo
        input_example = X_train.head(5)
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            input_example=input_example,
            registered_model_name=MODEL_NAME
        )
        
        logging.info(f"Modelo registrado en MLflow como 'obesity_classifier'")
        
    print("\n--- Resultados de Evaluación (en Validation Set) ---")
    print(f"Mejores Hiperparámetros: {best_params}")
    print(f"Accuracy en Validación: {val_accuracy:.4f}")
    print(f"F1-Score (Weighted) en Validación: {val_f1:.4f}")
    print("\nReporte de Clasificación (en Validación):")
    print(classification_report(y_val, y_pred_val))

if __name__ == "__main__":
    app()