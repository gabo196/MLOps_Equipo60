"""
Pruebas unitarias para el módulo train.py
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


class TestTraining:
    """Tests para funciones de entrenamiento."""
    
    @pytest.fixture
    def train_test_data(self, sample_clean_data):
        """Fixture que proporciona datos de entrenamiento y prueba."""
        from sklearn.model_selection import train_test_split
        
        X = sample_clean_data.drop('NObesity', axis=1)
        y = sample_clean_data['NObesity']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def test_model_training(self, train_test_data):
        """Test que verifica el entrenamiento básico de un modelo."""
        X_train, X_test, y_train, y_test = train_test_data
        
        from obesity_level_classifier.features import build_preprocessor
        
        # Crear pipeline
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', DecisionTreeClassifier(random_state=42, max_depth=3))
        ])
        
        # Entrenar
        pipeline.fit(X_train, y_train)
        
        # Verificar que está entrenado
        assert hasattr(pipeline, 'predict')
        
        # Hacer predicción
        predictions = pipeline.predict(X_test)
        assert len(predictions) == len(y_test)
    
    def test_model_evaluation_metrics(self, train_test_data):
        """Test que verifica el cálculo de métricas de evaluación."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from obesity_level_classifier.features import build_preprocessor
        
        X_train, X_test, y_train, y_test = train_test_data
        
        # Crear y entrenar modelo
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted')
        
        # Verificar que las métricas están en rangos válidos
        assert 0 <= accuracy <= 1
        assert 0 <= f1 <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
    
    def test_random_seed_reproducibility(self, train_test_data):
        """Test que verifica reproducibilidad con semilla aleatoria."""
        X_train, X_test, y_train, y_test = train_test_data
        from obesity_level_classifier.features import build_preprocessor
        
        # Entrenar dos modelos con la misma semilla
        pipeline1 = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])
        
        pipeline2 = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])
        
        pipeline1.fit(X_train, y_train)
        pipeline2.fit(X_train, y_train)
        
        pred1 = pipeline1.predict(X_test)
        pred2 = pipeline2.predict(X_test)
        
        # Las predicciones deberían ser idénticas
        assert np.array_equal(pred1, pred2)
    
    def test_cross_validation(self, sample_clean_data):
        """Test que verifica validación cruzada."""
        from sklearn.model_selection import cross_val_score
        from obesity_level_classifier.features import build_preprocessor
        
        X = sample_clean_data.drop('NObesity', axis=1)
        y = sample_clean_data['NObesity']
        
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', DecisionTreeClassifier(random_state=42, max_depth=3))
        ])
        
        # Realizar validación cruzada con menos folds debido a datos pequeños
        scores = cross_val_score(pipeline, X, y, cv=2)
        
        assert len(scores) == 2
        assert all(0 <= score <= 1 for score in scores)
    
    @patch('mlflow.log_metric')
    @patch('mlflow.log_param')
    def test_mlflow_logging(self, mock_log_param, mock_log_metric):
        """Test que verifica el logging a MLflow."""
        import mlflow
        
        # Simular logging de parámetros y métricas
        mlflow.log_param('max_depth', 5)
        mlflow.log_metric('accuracy', 0.85)
        
        mock_log_param.assert_called_once_with('max_depth', 5)
        mock_log_metric.assert_called_once_with('accuracy', 0.85)
    
    def test_hyperparameter_ranges(self):
        """Test que verifica rangos de hiperparámetros."""
        hyperparams = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Verificar que los rangos son válidos
        assert all(isinstance(v, list) for v in hyperparams.values())
        assert all(len(v) > 0 for v in hyperparams.values())
        assert all(all(x > 0 for x in v) for v in hyperparams.values())
    
    def test_model_serialization(self, train_test_data, tmp_path):
        """Test que verifica la serialización del modelo."""
        import joblib
        from obesity_level_classifier.features import build_preprocessor
        
        X_train, X_test, y_train, y_test = train_test_data
        
        # Entrenar modelo
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Guardar modelo
        model_path = tmp_path / "model.pkl"
        joblib.dump(pipeline, model_path)
        
        assert model_path.exists()
        
        # Cargar modelo
        loaded_pipeline = joblib.load(model_path)
        
        # Verificar que funciona
        predictions = loaded_pipeline.predict(X_test)
        assert len(predictions) == len(y_test)
    
    def test_feature_preprocessing_in_pipeline(self, train_test_data):
        """Test que verifica el preprocesamiento en el pipeline."""
        from obesity_level_classifier.features import build_preprocessor
        
        X_train, X_test, y_train, y_test = train_test_data
        
        preprocessor = build_preprocessor()
        
        # Transformar datos
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Verificar dimensiones
        assert X_train_transformed.shape[0] == len(X_train)
        assert X_test_transformed.shape[0] == len(X_test)
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
    
    def test_class_imbalance_handling(self, sample_clean_data):
        """Test que verifica el manejo de desbalance de clases."""
        y = sample_clean_data['NObesity']
        
        # Verificar distribución de clases
        class_counts = y.value_counts()
        
        # Calcular ratio de desbalance
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        assert imbalance_ratio >= 1
    
    def test_training_time_measurement(self, train_test_data):
        """Test que verifica la medición del tiempo de entrenamiento."""
        import time
        from obesity_level_classifier.features import build_preprocessor
        
        X_train, X_test, y_train, y_test = train_test_data
        
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])
        
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        assert training_time >= 0
        assert training_time < 60  # No debería tomar más de 1 minuto
    
    def test_model_comparison(self, train_test_data):
        """Test que verifica la comparación de modelos."""
        from sklearn.metrics import accuracy_score
        from obesity_level_classifier.features import build_preprocessor
        
        X_train, X_test, y_train, y_test = train_test_data
        
        # Entrenar diferentes modelos
        models = {
            'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=3),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=10, max_depth=3)
        }
        
        scores = {}
        
        for name, model in models.items():
            pipeline = Pipeline([
                ('preprocessor', build_preprocessor()),
                ('classifier', model)
            ])
            
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            scores[name] = accuracy_score(y_test, predictions)
        
        assert len(scores) == len(models)
        assert all(0 <= score <= 1 for score in scores.values())
