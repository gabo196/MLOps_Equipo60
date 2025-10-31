"""
Módulo para funciones de visualización reutilizables.
Cada función guarda el gráfico en un archivo y retorna la ruta.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

def plot_confusion_matrix(
    y_true: pd.Series, 
    y_pred: pd.Series, 
    labels: list, 
    output_path: Path
) -> Path:
    """
    Genera, guarda y retorna la ruta a una matriz de confusión.
    """
    logging.info(f"Generando matriz de confusión en {output_path}...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=labels, 
        yticklabels=labels
    )
    plt.title("Matriz de Confusión")
    plt.ylabel("Etiqueta Verdadera")
    plt.xlabel("Etiqueta Predicha")
    plt.savefig(output_path)
    plt.close() # Cierra la figura para liberar memoria
    logging.info(f"Matriz de confusión guardada.")
    return output_path

def plot_feature_importance(
    pipeline: Pipeline, 
    feature_names: list, 
    output_path: Path
) -> Path:
    """
    Extrae, grafica y guarda la importancia de las características de un 
    pipeline que contiene un modelo tipo árbol (ej. RandomForest).
    """
    logging.info(f"Generando gráfico de importancia de características en {output_path}...")
    
    # Asume que el modelo es el último paso del pipeline
    model = pipeline.named_steps['model']
    
    if not hasattr(model, 'feature_importances_'):
        logging.warning("El modelo no tiene 'feature_importances_'. Saltando gráfico.")
        return None

    importances_values = model.feature_importances_
    
    importances_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances_values
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=importances_df)
    plt.title('Importancia de las Características en el Modelo')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Gráfico de importancia de características guardado.")
    return output_path

if __name__ == "__main__":
    # Puedes añadir un pequeño test aquí si lo deseas
    logging.basicConfig(level=logging.INFO)
    print("Módulo de plots cargado. Contiene funciones reutilizables.")