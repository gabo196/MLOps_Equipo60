"""
Pruebas unitarias para el módulo features.py
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from obesity_level_classifier.features import build_preprocessor


class TestFeatures:
    """Tests para el módulo de features."""
    
    def test_build_preprocessor_returns_column_transformer(self):
        """Test que verifica que build_preprocessor retorna un ColumnTransformer."""
        preprocessor = build_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)
    
    def test_preprocessor_has_correct_transformers(self):
        """Test que verifica que el preprocessor tiene los transformers correctos."""
        preprocessor = build_preprocessor()
        
        # Verificar que tiene 3 transformers (num, ord, nom)
        assert len(preprocessor.transformers) == 3
        
        # Verificar nombres de los transformers
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert 'num' in transformer_names
        assert 'ord' in transformer_names
        assert 'nom' in transformer_names
    
    def test_preprocessor_numerical_features(self):
        """Test que verifica las características numéricas."""
        preprocessor = build_preprocessor()
        
        # Buscar el transformer numérico
        num_transformer = None
        for name, transformer, features in preprocessor.transformers:
            if name == 'num':
                num_transformer = (transformer, features)
                break
        
        assert num_transformer is not None
        scaler, num_features = num_transformer
        
        # Verificar que usa StandardScaler
        assert isinstance(scaler, StandardScaler)
        
        # Verificar que incluye las características numéricas esperadas
        expected_num_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        assert set(num_features) == set(expected_num_features)
    
    def test_preprocessor_ordinal_features(self):
        """Test que verifica las características ordinales."""
        preprocessor = build_preprocessor()
        
        # Buscar el transformer ordinal
        ord_transformer = None
        for name, transformer, features in preprocessor.transformers:
            if name == 'ord':
                ord_transformer = (transformer, features)
                break
        
        assert ord_transformer is not None
        encoder, ord_features = ord_transformer
        
        # Verificar que usa OrdinalEncoder
        assert isinstance(encoder, OrdinalEncoder)
        
        # Verificar que incluye características ordinales esperadas
        expected_ord_features = ['CAEC', 'CALC']
        assert set(ord_features) == set(expected_ord_features)
    
    def test_preprocessor_nominal_features(self):
        """Test que verifica las características nominales."""
        preprocessor = build_preprocessor()
        
        # Buscar el transformer nominal
        nom_transformer = None
        for name, transformer, features in preprocessor.transformers:
            if name == 'nom':
                nom_transformer = (transformer, features)
                break
        
        assert nom_transformer is not None
        encoder, nom_features = nom_transformer
        
        # Verificar que usa OneHotEncoder
        assert isinstance(encoder, OneHotEncoder)
        
        # Verificar que incluye características nominales esperadas
        expected_nom_features = ['Gender', 'family_history_with_overweight', 'FAVC', 
                                'SMOKE', 'SCC', 'MTRANS']
        assert set(nom_features) == set(expected_nom_features)
    
    def test_preprocessor_fit_transform(self):
        """Test que verifica que el preprocessor puede hacer fit y transform."""
        # Crear datos de prueba
        data = pd.DataFrame({
            'Age': [25, 30],
            'Gender': ['Male', 'Female'],
            'Height': [1.75, 1.65],
            'Weight': [70, 65],
            'family_history_with_overweight': ['Yes', 'No'],
            'FAVC': ['Yes', 'No'],
            'FCVC': [2.0, 3.0],
            'NCP': [3.0, 4.0],
            'CAEC': ['Sometimes', 'Frequently'],
            'SMOKE': ['No', 'No'],
            'CH2O': [2.0, 2.5],
            'SCC': ['No', 'Yes'],
            'FAF': [1.0, 2.0],
            'TUE': [1.0, 0.5],
            'CALC': ['Sometimes', 'No'],
            'MTRANS': ['Public_Transportation', 'Walking']
        })
        
        preprocessor = build_preprocessor()
        
        # Fit y transform
        X_transformed = preprocessor.fit_transform(data)
        
        # Verificar que la salida es un array numpy
        assert isinstance(X_transformed, np.ndarray)
        
        # Verificar que tiene la forma correcta
        assert X_transformed.shape[0] == 2  # 2 muestras
        assert X_transformed.shape[1] > 0  # Tiene columnas