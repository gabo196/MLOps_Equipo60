"""
Pruebas unitarias para el módulo plots.py
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Importar las funciones reales del módulo
from obesity_level_classifier.plots import plot_confusion_matrix, plot_feature_importance
from obesity_level_classifier.features import build_preprocessor


class TestPlotsModule:
    """Tests para el módulo plots.py real."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Fixture que proporciona predicciones de ejemplo."""
        y_true = pd.Series(['Normal_Weight', 'Overweight_Level_I', 'Obesity_Type_I'] * 10)
        y_pred = pd.Series(['Normal_Weight', 'Overweight_Level_I', 'Obesity_Type_I'] * 10)
        labels = ['Normal_Weight', 'Overweight_Level_I', 'Obesity_Type_I']
        return y_true, y_pred, labels
    
    @pytest.fixture
    def sample_clean_data(self):
        """Fixture que proporciona datos limpios de muestra."""
        return pd.DataFrame({
            'Age': [25, 30, 35, 40, 45, 22, 28, 33, 38, 42],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 
                      'Female', 'Male', 'Female', 'Male', 'Female'],
            'Height': [1.75, 1.65, 1.80, 1.70, 1.72, 1.68, 1.78, 1.62, 1.85, 1.73],
            'Weight': [70, 65, 95, 85, 78, 55, 72, 90, 110, 88],
            'family_history_with_overweight': ['Yes', 'No', 'Yes', 'Yes', 'No', 
                                              'Yes', 'No', 'Yes', 'Yes', 'No'],
            'FAVC': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No'],
            'FCVC': [2.0, 3.0, 3.0, 2.5, 2.0, 2.5, 3.0, 2.0, 3.0, 2.5],
            'NCP': [3.0, 4.0, 4.0, 3.5, 3.0, 3.0, 4.0, 3.5, 4.0, 3.0],
            'CAEC': ['Sometimes', 'Frequently', 'Sometimes', 'Frequently', 'Sometimes',
                    'Sometimes', 'Frequently', 'Sometimes', 'Frequently', 'Sometimes'],
            'SMOKE': ['No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No'],
            'CH2O': [2.0, 2.5, 1.5, 2.0, 2.5, 2.5, 2.0, 1.5, 1.0, 2.0],
            'SCC': ['No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No'],
            'FAF': [1.0, 2.0, 0.5, 1.5, 2.5, 2.0, 1.0, 0.5, 0.0, 1.5],
            'TUE': [1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 2.0, 2.5, 3.0, 1.0],
            'CALC': ['Sometimes', 'No', 'Frequently', 'Sometimes', 'No',
                    'Sometimes', 'Frequently', 'Sometimes', 'Frequently', 'No'],
            'MTRANS': ['Public_Transportation', 'Walking', 'Automobile', 'Public_Transportation', 'Walking',
                      'Walking', 'Automobile', 'Public_Transportation', 'Automobile', 'Walking'],
            'NObesity': ['Normal_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_I', 'Normal_Weight',
                        'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Obesity_Type_I', 'Overweight_Level_I']
        })
    
    def test_plot_confusion_matrix_creation(self, sample_predictions, tmp_path):
        """Test que verifica la creación real de matriz de confusión."""
        y_true, y_pred, labels = sample_predictions
        output_path = tmp_path / "confusion_matrix.png"
        
        # Ejecutar función real
        result_path = plot_confusion_matrix(y_true, y_pred, labels, output_path)
        
        # Verificar que el archivo se creó
        assert result_path is not None
        assert result_path.exists()
        assert result_path.stat().st_size > 0
    
    def test_plot_feature_importance_with_random_forest(self, sample_clean_data, tmp_path):
        """Test que verifica el plot de importancia con RandomForest."""
        X = sample_clean_data.drop('NObesity', axis=1)
        y = sample_clean_data['NObesity']
        
        # Crear pipeline con RandomForest
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('model', RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3))
        ])
        
        # Entrenar pipeline
        pipeline.fit(X, y)
        
        # AHORA obtener nombres de features después del fit
        preprocessor = pipeline.named_steps['preprocessor']
        feature_names = []
        
        # Iterar sobre los transformers fitted
        for name, transformer, features in preprocessor.transformers_:
            if name == 'num':
                # Features numéricas mantienen sus nombres
                feature_names.extend(features)
            elif name == 'ord':
                # Features ordinales mantienen sus nombres
                feature_names.extend(features)
            elif name == 'nom':
                # OneHotEncoder genera múltiples columnas
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out(features).tolist())
                else:
                    # Fallback: solo agregar los nombres base
                    feature_names.extend(features)
        
        output_path = tmp_path / "feature_importance.png"
        
        # Ejecutar función real
        result_path = plot_feature_importance(pipeline, feature_names, output_path)
        
        # Verificar que el archivo se creó
        assert result_path is not None
        assert result_path.exists()
        assert result_path.stat().st_size > 0
    
    def test_plot_confusion_matrix_with_different_predictions(self, tmp_path):
        """Test con predicciones diferentes."""
        y_true = pd.Series(['Normal_Weight'] * 20 + ['Overweight_Level_I'] * 15 + ['Obesity_Type_I'] * 10)
        y_pred = pd.Series(['Normal_Weight'] * 18 + ['Overweight_Level_I'] * 17 + ['Obesity_Type_I'] * 10)
        labels = ['Normal_Weight', 'Overweight_Level_I', 'Obesity_Type_I']
        
        output_path = tmp_path / "confusion_matrix_diff.png"
        
        result_path = plot_confusion_matrix(y_true, y_pred, labels, output_path)
        
        assert result_path is not None
        assert result_path.exists()
    
    def test_plot_feature_importance_model_without_feature_importances(self, sample_clean_data, tmp_path):
        """Test con modelo que no tiene feature_importances."""
        from sklearn.linear_model import LogisticRegression
        
        X = sample_clean_data.drop('NObesity', axis=1)
        y = sample_clean_data['NObesity']
        
        # Crear pipeline con modelo sin feature_importances
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('model', LogisticRegression(random_state=42, max_iter=100))
        ])
        
        pipeline.fit(X, y)
        
        feature_names = ['feat1', 'feat2', 'feat3']
        output_path = tmp_path / "no_importance.png"
        
        # Debería retornar None
        result_path = plot_feature_importance(pipeline, feature_names, output_path)
        
        assert result_path is None
    
    def test_plot_confusion_matrix_saves_to_correct_path(self, sample_predictions, tmp_path):
        """Test que verifica que el archivo se guarda en la ruta correcta."""
        y_true, y_pred, labels = sample_predictions
        output_path = tmp_path / "subfolder" / "confusion.png"
        
        # Crear subdirectorio
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_path = plot_confusion_matrix(y_true, y_pred, labels, output_path)
        
        assert result_path == output_path
        assert result_path.exists()


class TestPlotsUtility:
    """Tests de utilidad para plots."""
    
    def test_matplotlib_backend(self):
        """Test que verifica que matplotlib usa el backend correcto."""
        backend = matplotlib.get_backend()
        assert backend == 'agg' or backend == 'Agg'
    
    def test_figure_save_capability(self, tmp_path):
        """Test que verifica la capacidad de guardar figuras."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        output_file = tmp_path / "test_plot.png"
        fig.savefig(output_file)
        
        assert output_file.exists()
        assert output_file.stat().st_size > 0
        plt.close(fig)
    
    def test_plot_with_empty_data(self):
        """Test que verifica el manejo de datos vacíos."""
        fig, ax = plt.subplots()
        
        # Intentar plotear datos vacíos
        x = []
        y = []
        ax.plot(x, y)
        
        # No debería fallar
        assert fig is not None
        plt.close(fig)
    
    def test_multiple_subplots(self):
        """Test que verifica la creación de múltiples subplots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        assert fig is not None
        assert axes.shape == (2, 2)
        assert len(axes.flatten()) == 4
        plt.close(fig)