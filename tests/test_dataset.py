"""
Pruebas unitarias para el módulo dataset.py
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from obesity_level_classifier.dataset import DataCleaner

class TestDataCleaner:
    """Tests para la clase DataCleaner."""
    
    def test_load_data(self, sample_raw_data, temp_data_dir):
        """Test que verifica la carga correcta de datos."""
        # Guardar datos temporales
        test_file = temp_data_dir / "test_data.csv"
        sample_raw_data.to_csv(test_file, index=False)
        
        # Inicializar cleaner y cargar
        cleaner = DataCleaner(test_file)
        cleaner.load_data()
        
        assert cleaner.df is not None
        assert cleaner.df.shape[0] == len(sample_raw_data)

    def test_load_data_file_not_found(temp_data_dir):
<<<<<<< Updated upstream
        """Test para verificar comportamiento en caso de no encontrar el archivo"""
=======
        """Test para verificar el comportamiento en caso de que no se encuentre el archivo"""
>>>>>>> Stashed changes
        cleaner = DataCleaner(temp_data_dir / "missing.csv")

        with pytest.raises(SystemExit):
            cleaner.load_data()
        
    def test_normalize_text(self, sample_raw_data, temp_data_dir):
        """Test que verifica la normalización de texto."""
        test_file = temp_data_dir / "test_data.csv"
        sample_raw_data.to_csv(test_file, index=False)
        
        cleaner = DataCleaner(test_file)
        cleaner.load_data()
        cleaner._normalize_text()
        
        # Verificar que el texto está en minúsculas y sin guiones bajos
        assert cleaner.df['Gender'].iloc[0] == 'male'
        assert 'public transportation' in cleaner.df['MTRANS'].values
        
    def test_handle_invalid_markers(self, sample_raw_data, temp_data_dir):
        """Test que verifica el reemplazo de marcadores inválidos."""
        test_file = temp_data_dir / "test_data.csv"
        sample_raw_data.to_csv(test_file, index=False)
        
        cleaner = DataCleaner(test_file)
        cleaner.load_data()
        cleaner._normalize_text()
        cleaner._handle_invalid_markers()
        
        # Verificar que '?' se convirtió en NaN
        assert pd.isna(cleaner.df['Height'].iloc[3])
        assert 'error' not in cleaner.df.values
        
    def test_convert_data_types(self, sample_raw_data, temp_data_dir):
        """Test que verifica la conversión de tipos de datos."""
        test_file = temp_data_dir / "test_data.csv"
        sample_raw_data.to_csv(test_file, index=False)
        
        cleaner = DataCleaner(test_file)
        cleaner.load_data()
        cleaner._normalize_text()
        cleaner._handle_invalid_markers()
        cleaner._convert_data_types()
        
        # Verificar que las columnas numéricas son del tipo correcto
        assert cleaner.df['Age'].dtype in [np.float64, np.int64, float, int]
        assert cleaner.df['Weight'].dtype in [np.float64, np.int64, float, int]
        
    def test_correct_domain_values(self, sample_clean_data, temp_data_dir):
        """Test que verifica la corrección de valores fuera de rango."""
        # Modificar datos para tener valores fuera de rango
        sample_clean_data.loc[0, 'FCVC'] = 5.0  # Fuera de rango [1-3]
        sample_clean_data.loc[1, 'NCP'] = 0.5   # Fuera de rango [1-4]
        
        test_file = temp_data_dir / "test_data.csv"
        sample_clean_data.to_csv(test_file, index=False)
        
        cleaner = DataCleaner(test_file)
        cleaner.load_data()
        cleaner._correct_domain_values()
        
        # Verificar que los valores están en rango
        assert cleaner.df['FCVC'].iloc[0] == 3.0
        assert cleaner.df['NCP'].iloc[1] == 1.0
        
    def test_impute_missing_values(self, sample_raw_data, temp_data_dir):
        """Test que verifica la imputación de valores faltantes."""
        test_file = temp_data_dir / "test_data.csv"
        sample_raw_data.to_csv(test_file, index=False)
        
        cleaner = DataCleaner(test_file)
        cleaner.load_data()
        cleaner._normalize_text()
        cleaner._handle_invalid_markers()
        cleaner._convert_data_types()
        cleaner._impute_missing_values()
        
        # Verificar que no hay NaN en columnas numéricas
        assert cleaner.df['Age'].isna().sum() == 0
        assert cleaner.df['Height'].isna().sum() == 0
        
    def test_handle_duplicates(self, sample_clean_data, temp_data_dir):
        """Test que verifica la eliminación de duplicados."""
        # Agregar una fila duplicada
        duplicated = pd.concat([sample_clean_data, sample_clean_data.iloc[[0]]], ignore_index=True)
        
        test_file = temp_data_dir / "test_data.csv"
        duplicated.to_csv(test_file, index=False)
        
        cleaner = DataCleaner(test_file)
        cleaner.load_data()
        initial_count = len(cleaner.df)
        cleaner._handle_duplicates()
        
        # Verificar que se eliminó al menos un duplicado
        assert len(cleaner.df) < initial_count

<<<<<<< Updated upstream
    def test_treat_outliers(temp_data_dir):
        """Test para verificar el tratamiento de outliers"""
        data = {
            "Age": [20, 25, 30, 35, 55],   
            "Height": [0.5, 1.5, 1.6, 1.7, 1.8],  
            "Weight": [50, 60, 70, 80, 120],  
        }
        test_file = temp_data_dir / "test_data.csv"
        cleaner = DataCleaner(test_file)
=======
    def create_temp_csv(tmp_path, data):
        """Helper to generate temporary CSV files."""
        f = tmp_path / "input.csv"
        pd.DataFrame(data).to_csv(f, index=False)
        return f
    
    def test_treat_outliers(temp_data_dir):
        """Test para verificar el tratamiento de outliers"""
        data = {
            "Age": [20, 25, 30, 35, 55],   # 200 is extreme
            "Height": [0.5, 1.5, 1.6, 1.7, 1.8],  # 10 is extreme
            "Weight": [50, 60, 70, 80, 120],  # 1000 is extreme
        }
        temp = create_temp_csv(temp_data_dir, data)
        cleaner = DataCleaner(temp)
>>>>>>> Stashed changes
        cleaner.load_data()

        cleaner._treat_outliers()

        # outliers should be clipped inside IQR bounds
        assert cleaner.df["Age"].max() < 200
        assert cleaner.df["Height"].max() < 10
        assert cleaner.df["Weight"].max() < 1000
        
    def test_clean_data_pipeline(self, sample_raw_data, temp_data_dir):
        """Test de integración del pipeline completo de limpieza."""
        test_file = temp_data_dir / "test_data.csv"
        sample_raw_data.to_csv(test_file, index=False)
        
        cleaner = DataCleaner(test_file)
        clean_df = cleaner.clean_data()
        
        # Verificar que el DataFrame limpio tiene sentido
        assert clean_df is not None
        assert len(clean_df) > 0
        assert 'NObesity' in clean_df.columns
        # Verificar que no hay NaN en datos numéricos después de limpieza
        numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
        assert clean_df[numeric_cols].isna().sum().sum() == 0
        
    def test_save_data(self, sample_clean_data, temp_data_dir):
        """Test que verifica el guardado de datos."""
        test_input = temp_data_dir / "input.csv"
        test_output = temp_data_dir / "output.csv"
        
        sample_clean_data.to_csv(test_input, index=False)
        
        cleaner = DataCleaner(test_input)
        cleaner.load_data()
        cleaner.save_data(test_output)
        
        # Verificar que el archivo se guardó
        assert test_output.exists()
        loaded_df = pd.read_csv(test_output)
        assert len(loaded_df) == len(sample_clean_data)

