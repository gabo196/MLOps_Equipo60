"""
Módulo para construir el pipeline de preprocesamiento de características (features).
Este script define las transformaciones que se aplicarán a los datos.
"""
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# --- Listas de Características ---
# Estas listas son la "fuente de verdad" para el preprocesamiento.

NUMERIC_FEATURES = [
    'Age', 
    'Height', 
    'Weight', 
    'FCVC', 
    'NCP', 
    'CH2O', 
    'FAF', 
    'TUE'
]

# Variables categóricas con un orden inherente
ORDINAL_FEATURES = ['CAEC', 'CALC']

# Variables categóricas sin orden
NOMINAL_FEATURES = [
    'Gender', 
    'family_history_with_overweight', 
    'FAVC', 
    'SMOKE', 
    'SCC', 
    'MTRANS'
]

# --- Definición de Categorías Ordinales ---
# El orden es crucial para el OrdinalEncoder
CAEC_CATEGORIES = ['No', 'Sometimes', 'Frequently', 'Always']
CALC_CATEGORIES = ['No', 'Sometimes', 'Frequently', 'Always']

def build_preprocessor() -> ColumnTransformer:
    """
    Construye y retorna un ColumnTransformer que aplica las transformaciones
    correctas a las columnas del dataset.

    Returns:
        ColumnTransformer: Objeto de Scikit-Learn listo para ser ajustado.
    """
    logging.info("Construyendo el pipeline de preprocesamiento...")

    # --- Creación de los Pipelines de Transformación ---
    
    # Pipeline para variables numéricas: solo escalar
    numeric_transformer = StandardScaler()

    # Pipeline para variables ordinales: codificar según el orden
    ordinal_transformer = OrdinalEncoder(
        categories=[CAEC_CATEGORIES, CALC_CATEGORIES],
        handle_unknown='use_encoded_value',  # Asigna un valor (ej. -1) a categorías desconocidas
        unknown_value=-1
    )

    # Pipeline para variables nominales: One-Hot Encoding
    nominal_transformer = OneHotEncoder(
        handle_unknown='ignore',  # Ignora categorías desconocidas en la predicción
        sparse_output=False
    )

    # --- Combinación en un ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('ord', ordinal_transformer, ORDINAL_FEATURES),
            ('nom', nominal_transformer, NOMINAL_FEATURES)
        ],
        remainder='drop'  # Elimina columnas no especificadas
    )
    
    logging.info("Pipeline de preprocesamiento construido exitosamente.")
    return preprocessor

if __name__ == "__main__":
    # Esto permite probar el módulo de forma independiente
    logging.basicConfig(level=logging.INFO)
    preprocessor = build_preprocessor()
    print("Pipeline de Preprocesamiento:")
    print(preprocessor)