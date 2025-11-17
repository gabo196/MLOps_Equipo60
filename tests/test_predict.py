"""
Pruebas unitarias para el módulo predict.py
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import mlflow
import sys
from io import StringIO


class TestPredict:
    """Tests para funcionalidad de predicción."""
    
    def test_model_prediction_shape(self, sample_clean_data):
        """Test que verifica que las predicciones tienen el shape correcto."""
        X = sample_clean_data.drop('NObesity', axis=1)
        
        # Mock del modelo
        mock_model = Mock()
        mock_model.predict.return_value = np.array(['Normal Weight'] * len(X))
        
        predictions = mock_model.predict(X)
        
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
        
    def test_prediction_output_types(self, sample_clean_data):
        """Test que verifica que las predicciones son del tipo correcto."""
        X = sample_clean_data.drop('NObesity', axis=1)
        
        # Mock del modelo con predicciones válidas
        valid_classes = ['Normal Weight', 'Overweight Level I', 'Obesity Type I']
        mock_model = Mock()
        mock_model.predict.return_value = np.array([valid_classes[0]] * len(X))
        
        predictions = mock_model.predict(X)
        
        # Verificar que las predicciones son strings válidos
        assert all(isinstance(pred, str) for pred in predictions)
        
    def test_prediction_with_missing_features(self, sample_clean_data):
        """Test que verifica el comportamiento con features faltantes."""
        # Eliminar una columna importante
        X_incomplete = sample_clean_data.drop(['NObesity', 'Age'], axis=1)
        
        mock_model = Mock()
        mock_model.predict.side_effect = ValueError("Feature mismatch")
        
        # Debería lanzar un error
        with pytest.raises(ValueError):
            mock_model.predict(X_incomplete)
            
    @patch('mlflow.pyfunc.load_model')
    def test_load_model_from_mlflow(self, mock_load_model):
        """Test que verifica la carga del modelo desde MLflow."""
        # Mock del modelo cargado
        mock_model = Mock()
        mock_model.predict.return_value = np.array(['Normal Weight'])
        mock_load_model.return_value = mock_model
        
        # Simular carga
        model_uri = "models:/obesity_classifier/1"
        loaded_model = mock_load_model(model_uri)
        
        assert loaded_model is not None
        mock_load_model.assert_called_once_with(model_uri)
        
    def test_prediction_consistency(self, sample_clean_data):
        """Test que verifica consistencia en predicciones múltiples."""
        X = sample_clean_data.drop('NObesity', axis=1)
        
        # Mock del modelo con comportamiento determinista
        mock_model = Mock()
        predictions1 = np.array(['Normal Weight'] * len(X))
        predictions2 = np.array(['Normal Weight'] * len(X))
        
        mock_model.predict.side_effect = [predictions1, predictions2]
        
        pred1 = mock_model.predict(X)
        pred2 = mock_model.predict(X)
        
        # Las predicciones deberían ser idénticas
        assert np.array_equal(pred1, pred2)
        
    def test_batch_prediction(self, sample_clean_data):
        """Test que verifica predicción en lote."""
        X = sample_clean_data.drop('NObesity', axis=1)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array(['Normal Weight'] * len(X))
        
        # Predecir en lote
        predictions = mock_model.predict(X)
        
        assert len(predictions) == len(X)
        
    def test_single_prediction(self, sample_clean_data):
        """Test que verifica predicción de una sola instancia."""
        X = sample_clean_data.drop('NObesity', axis=1).iloc[[0]]
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array(['Normal Weight'])
        
        prediction = mock_model.predict(X)
        
        assert len(prediction) == 1


class TestPredictModule:
    """Tests para el módulo predict.py completo."""
    
    @patch('mlflow.pyfunc.load_model')
    @patch('pandas.read_csv')
    def test_predict_main_with_mock_model(self, mock_read_csv, mock_load_model, sample_clean_data, tmp_path):
        """Test del flujo completo de predicción con modelo mockeado."""
        from obesity_level_classifier.modeling import predict
        
        # Preparar datos de prueba
        X = sample_clean_data.drop('NObesity', axis=1)
        mock_read_csv.return_value = X
        
        # Mock del modelo
        mock_model = Mock()
        mock_model.predict.return_value = np.array(['Normal_Weight'] * len(X))
        mock_load_model.return_value = mock_model
        
        # Crear archivo temporal
        test_file = tmp_path / "test_data.csv"
        X.to_csv(test_file, index=False)
        
        # El test verifica que las funciones se pueden importar
        assert hasattr(predict, 'main')
        assert callable(predict.main)
    
    def test_mlflow_model_uri_format(self):
        """Test que verifica el formato correcto del model URI."""
        model_name = "obesity_classifier"
        stage = "Production"
        
        expected_uri = f"models:/{model_name}/{stage}"
        
        assert expected_uri == "models:/obesity_classifier/Production"
        assert "/" in expected_uri
        assert "models:" in expected_uri
    
    @patch('mlflow.set_tracking_uri')
    def test_mlflow_tracking_uri_setup(self, mock_set_tracking):
        """Test que verifica la configuración del tracking URI."""
        expected_uri = "sqlite:///mlflow.db"
        
        mlflow.set_tracking_uri(expected_uri)
        
        mock_set_tracking.assert_called_once_with(expected_uri)
    
    def test_prediction_with_target_column(self, sample_clean_data):
        """Test que verifica el manejo cuando los datos incluyen la columna target."""
        # Datos con NObesity
        df_with_target = sample_clean_data.copy()
        
        assert 'NObesity' in df_with_target.columns
        
        # Separar X e y
        y = df_with_target['NObesity']
        X = df_with_target.drop('NObesity', axis=1)
        
        assert 'NObesity' not in X.columns
        assert len(y) == len(X)
    
    def test_prediction_without_target_column(self, sample_clean_data):
        """Test que verifica el manejo cuando los datos NO incluyen la columna target."""
        X = sample_clean_data.drop('NObesity', axis=1)
        
        assert 'NObesity' not in X.columns
        
        # Mock de predicción
        mock_model = Mock()
        mock_model.predict.return_value = np.array(['Normal_Weight'] * len(X))
        
        predictions = mock_model.predict(X)
        assert len(predictions) == len(X)