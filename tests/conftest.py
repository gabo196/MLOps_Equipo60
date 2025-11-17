"""
Configuración de fixtures compartidas para pytest.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def sample_raw_data():
    """
    Genera un DataFrame de muestra simulando datos crudos con problemas.
    """
    data = {
        'Age': [25, 30, 'error', 45, 22],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Height': [1.75, 1.65, 1.80, '?', 1.70],
        'Weight': [70, 65, 85, 75, 68],
        'family_history_with_overweight': ['yes', 'yes', 'no', 'yes', 'no'],
        'FAVC': ['yes', 'no', 'yes', 'yes', 'no'],
        'FCVC': [2.0, 3.0, 1.5, 2.5, 3.0],
        'NCP': [3.0, 4.0, 2.0, 3.0, 3.0],
        'CAEC': ['Sometimes', 'No', 'Frequently', 'Sometimes', 'No'],
        'SMOKE': ['no', 'no', 'no', 'no', 'no'],
        'CH2O': [2.0, 2.5, 1.5, 2.0, 2.5],
        'SCC': ['no', 'yes', 'no', 'no', 'yes'],
        'FAF': [1.0, 2.0, 0.5, 1.5, 2.5],
        'TUE': [1.0, 0.5, 2.0, 1.0, 0.5],
        'CALC': ['Sometimes', 'No', 'Sometimes', 'Frequently', 'No'],
        'MTRANS': ['Public_Transportation', 'Walking', 'Automobile', 'Public_Transportation', 'Walking'],
        'NObesity': ['Normal_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Normal_Weight', 'Normal_Weight']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_clean_data():
    """
    Genera un DataFrame limpio para probar preprocesamiento y predicción.
    """
    data = {
        'Age': [25, 30, 35, 45, 22],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Height': [1.75, 1.65, 1.80, 1.70, 1.70],
        'Weight': [70.0, 65.0, 85.0, 75.0, 68.0],
        'family_history_with_overweight': ['Yes', 'Yes', 'No', 'Yes', 'No'],
        'FAVC': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'FCVC': [2.0, 3.0, 2.0, 2.0, 3.0],
        'NCP': [3.0, 4.0, 2.0, 3.0, 3.0],
        'CAEC': ['Sometimes', 'No', 'Frequently', 'Sometimes', 'No'],
        'SMOKE': ['No', 'No', 'No', 'No', 'No'],
        'CH2O': [2.0, 2.5, 1.5, 2.0, 2.5],
        'SCC': ['No', 'Yes', 'No', 'No', 'Yes'],
        'FAF': [1.0, 2.0, 0.5, 1.5, 2.5],
        'TUE': [1.0, 0.5, 2.0, 1.0, 0.5],
        'CALC': ['Sometimes', 'No', 'Sometimes', 'Frequently', 'No'],
        'MTRANS': ['Public Transportation', 'Walking', 'Automobile', 'Public Transportation', 'Walking'],
        'NObesity': ['Normal Weight', 'Normal Weight', 'Overweight Level I', 'Normal Weight', 'Normal Weight']
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_data_dir():
    """
    Crea un directorio temporal para tests que requieren I/O.
    """
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_features():
    """
    Retorna listas de características para tests de preprocesamiento.
    """
    return {
        'numeric': ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'],
        'ordinal': ['CAEC', 'CALC'],
        'nominal': ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC', 'MTRANS']
    }
