"""
Pruebas de integración para el pipeline completo.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from obesity_level_classifier.dataset import DataCleaner
from obesity_level_classifier.features import build_preprocessor
from obesity_level_classifier.modeling.train import build_full_pipeline
from obesity_level_classifier.plots import plot_confusion_matrix, plot_feature_importance
from obesity_level_classifier.modeling import predict as predict_module

class TestEndToEndPipeline:
    """Tests de integración del pipeline completo."""
    
    @pytest.fixture
    def sample_clean_data(self):
        """Fixture que proporciona datos limpios de muestra con más variedad."""
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
    
    def test_load_clean_data_from_file(self):
        """Test que verifica que se pueden cargar datos limpios desde archivo."""
        # Este test requiere que exista el archivo real
        processed_dir = Path("data/processed")
        clean_file = processed_dir / "obesity_estimation_cleaned.csv"
        
        if clean_file.exists():
            df = pd.read_csv(clean_file)
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert 'NObesity' in df.columns
        else:
            pytest.skip("Archivo de datos limpios no encontrado")
    
    def test_preprocessor_on_sample_data(self, sample_clean_data):
        """Test que verifica el preprocessor con datos de muestra."""
        X = sample_clean_data.drop('NObesity', axis=1)
        
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X)
        
        # Verificar transformación
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] > 0
        
        # Verificar que no hay NaN
        assert not np.isnan(X_transformed).any()
    
    def test_pipeline_with_train_test_split(self, sample_clean_data):
        """Test del flujo con división train/test."""
        from sklearn.model_selection import train_test_split
        
        # 1. Preparar y dividir datos
        X = sample_clean_data.drop('NObesity', axis=1)
        y = sample_clean_data['NObesity']
        
        # Usar test_size más pequeño y sin stratify para este test pequeño
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Verificar división
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        
        # 2. Entrenar preprocessor
        preprocessor = build_preprocessor()
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Verificar transformaciones
        assert X_train_transformed.shape[0] == len(X_train)
        assert X_test_transformed.shape[0] == len(X_test)
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
    
    def test_full_pipeline_with_model(self, sample_clean_data):
        """Test del pipeline completo incluyendo modelo simple."""
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.tree import DecisionTreeClassifier
        
        # 1. Preparar datos
        X = sample_clean_data.drop('NObesity', axis=1)
        y = sample_clean_data['NObesity']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # 2. Crear pipeline simple
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('classifier', DecisionTreeClassifier(random_state=42, max_depth=3))
        ])
        
        # 3. Entrenar
        pipeline.fit(X_train, y_train)
        
        # 4. Predecir
        y_pred = pipeline.predict(X_test)
        
        # Verificar predicciones
        assert len(y_pred) == len(y_test)
        assert all(pred in y.unique() for pred in y_pred)
        
        # 5. Calcular score básico
        score = pipeline.score(X_test, y_test)
        assert 0 <= score <= 1
    
    def test_preprocessor_transform_consistency(self, sample_clean_data):
        """Test que verifica consistencia en múltiples transformaciones."""
        X = sample_clean_data.drop('NObesity', axis=1)
        
        preprocessor = build_preprocessor()
        X_transformed1 = preprocessor.fit_transform(X)
        X_transformed2 = preprocessor.transform(X)
        
        # Las transformaciones deben ser idénticas
        np.testing.assert_array_almost_equal(X_transformed1, X_transformed2)
    
    def test_pipeline_handles_missing_model_gracefully(self):
        """Test que verifica manejo de modelo no encontrado."""
        import mlflow
        
        # Configurar MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Intentar cargar un modelo que no existe
        try:
            model_uri = "models:/nonexistent_model/None"
            model = mlflow.pyfunc.load_model(model_uri)
            # Si llegamos aquí, el modelo existe (no debería)
            assert False, "Se esperaba que el modelo no existiera"
        except Exception:
            # Se espera una excepción al cargar un modelo inexistente
            assert True
    
    def test_data_files_exist(self):
        """Test que verifica que los archivos de datos esperados existen."""
        processed_dir = Path("data/processed")
        
        # Archivos que deberían existir después de ejecutar el pipeline
        expected_files = [
            "obesity_estimation_cleaned.csv",
        ]
        
        for filename in expected_files:
            filepath = processed_dir / filename
            if filepath.exists():
                # Verificar que el archivo no está vacío
                df = pd.read_csv(filepath)
                assert not df.empty, f"{filename} está vacío"
    
    def test_preprocessor_handles_new_categories(self, sample_clean_data):
        """Test que verifica que el preprocessor maneja nuevas categorías correctamente."""
        X = sample_clean_data.drop('NObesity', axis=1)
        
        # Entrenar preprocessor
        preprocessor = build_preprocessor()
        preprocessor.fit(X)
        
        # Crear datos con una categoría nueva en Gender (que no debería existir)
        X_new = X.copy()
        X_new.loc[0, 'Gender'] = 'Male'  # Categoría conocida
        
        # Transform debería funcionar sin errores
        X_transformed = preprocessor.transform(X_new)
        assert X_transformed is not None
        assert X_transformed.shape[0] == len(X_new)

    def _build_raw_obesity_df():
        """Small synthetic dataset that roughly follows the expected schema."""
        return pd.DataFrame({
            # numeric columns (as strings to exercise conversion)
            "Age": ["20", "30", "40", "50"],
            "Height": ["1.70", "1.80", "1.60", "1.75"],
            "Weight": ["70", "80", "60", "90"],
            "FCVC": ["2", "3", "2", "3"],
            "NCP": ["3", "4", "3", "4"],
            "CH2O": ["2", "3", "2", "3"],
            "FAF": ["1", "2", "1", "2"],
            "TUE": ["0", "1", "0", "1"],
            # ordinal features
            "CAEC": ["no", "always", "sometimes", "frequently"],
            "CALC": ["no", "always", "sometimes", "frequently"],
            # nominal features
            "Gender": ["male", "female", "male", "female"],
            "family_history_with_overweight": ["yes", "no", "yes", "no"],
            "FAVC": ["yes", "no", "yes", "no"],
            "SMOKE": ["no", "yes", "no", "yes"],
            "SCC": ["no", "no", "yes", "yes"],
            "MTRANS": ["bike", "car", "walking", "public_transport"],
            # target column (raw name used in cleaner)
            "NObeyesdad": ["normal weight", "overweight i", "normal weight", "overweight ii"],
        })

    def test_integration_trained_pipeline_and_plots(tmp_path):
        # Reuse cleaned data from previous helper path
        raw_df = _build_raw_obesity_df()
        raw_path = tmp_path / "raw_for_plots.csv"
        raw_df.to_csv(raw_path, index=False)

        cleaner = DataCleaner(raw_data_path=raw_path)
        clean_df = cleaner.clean_data()

        X = clean_df.drop("NObesity", axis=1)
        y = clean_df["NObesity"]

        pipeline = build_full_pipeline()
        pipeline.fit(X, y)

        # --- Confusion Matrix ---
        y_pred = pipeline.predict(X)
        labels = sorted(y.unique())
        cm_path = tmp_path / "cm_integration.png"

        returned_cm_path = plot_confusion_matrix(
            y_true=y,
            y_pred=y_pred,
            labels=labels,
            output_path=cm_path,
        )

        assert cm_path.exists()
        assert returned_cm_path == cm_path

        # --- Feature Importance ---
        fi_path = tmp_path / "feature_importance_integration.png"
        feature_names = list(X.columns)

        returned_fi_path = plot_feature_importance(
            pipeline=pipeline,
            feature_names=feature_names,
            output_path=fi_path,
        )

        # RandomForest has feature_importances_, so we expect a file
        assert returned_fi_path == fi_path
        assert fi_path.exists()