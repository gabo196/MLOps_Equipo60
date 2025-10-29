"""
Script para procesar los datos crudos y guardarlos en la carpeta de datos procesados.
Ejemplo de uso desde la terminal (estando en la raíz del proyecto):
    python src/data/make_dataset.py data/raw/obesity_estimation_modified.csv data/processed/obesity_estimation_cleaned.csv
"""

import pandas as pd
import numpy as np
import sys
import logging
import typer
from pathlib import Path

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCleaner:
    """
    Clase para encapsular toda la lógica de limpieza del dataset de obesidad.
    Aplica principios de POO para organizar las responsabilidades de limpieza.
    """
    def __init__(self, raw_data_path: Path):
        self.raw_data_path = raw_data_path
        self.df = None
        logging.info(f"Iniciando limpieza para el archivo: {raw_data_path}")

    def load_data(self):
        """Carga los datos desde la ruta especificada."""
        try:
            self.df = pd.read_csv(self.raw_data_path)
            logging.info(f"Datos cargados exitosamente. Dimensiones: {self.df.shape}")
        except FileNotFoundError:
            logging.error(f"Error: No se encontró el archivo en {self.raw_data_path}")
            sys.exit(1)

    def _normalize_text(self):
        """Paso 1: Normaliza todo el texto (espacios, minúsculas, guiones bajos)."""
        logging.info("Paso 1: Normalizando texto...")
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].str.strip().str.lower().str.replace('_', ' ', regex=False)

    def _delete_innessary_columns(self):
        """Elimina columnas innecesarias si existen."""
        logging.info("Eliminando columnas innecesarias si existen...")        
        col_to_drop = 'mixed_type_col'
        if col_to_drop in self.df.columns:
            self.df.drop(columns=[col_to_drop], inplace=True)

    def _handle_invalid_markers(self):
        """Paso 2: Reemplaza marcadores de error explícitos por NaN."""
        logging.info("Paso 2: Reemplazando marcadores inválidos por NaN...")
        invalid_markers = ['nan', '?', 'error', 'invalid']
        self.df.replace(invalid_markers, np.nan, inplace=True)

    def _convert_data_types(self):
        """Paso 3: Convierte columnas numéricas, forzando errores a NaN."""
        logging.info("Paso 3: Convirtiendo tipos de datos numéricos...")
        cols_to_numeric = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        for col in cols_to_numeric:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def _correct_domain_values(self):
        """Paso 4: Corrige valores que están fuera del rango lógico del dominio."""
        logging.info("Paso 4: Corrigiendo valores fuera de rango lógico...")
        # FCVC (Frecuencia de consumo de vegetales) debe estar entre 1 y 3.
        self.df['FCVC'] = self.df['FCVC'].apply(lambda x: 3.0 if x > 3.0 else (1.0 if x < 1.0 else x))
        # NCP (Número de comidas principales) debe estar entre 1 y 4.
        self.df['NCP'] = self.df['NCP'].apply(lambda x: 4.0 if x > 4.0 else (1.0 if x < 1.0 else x))
        # Redondeamos para que los valores sean consistentes
        self.df['FCVC'] = self.df['FCVC'].round()
        self.df['NCP'] = self.df['NCP'].round()

    def _impute_missing_values(self):
        """Paso 5: Imputa los valores nulos (NaN) usando mediana y moda."""
        logging.info("Paso 5: Imputando valores nulos...")
        # Imputar numéricos con mediana
        for col in self.df.select_dtypes(include=np.number).columns:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
        # Imputar categóricos con moda
        for col in self.df.select_dtypes(include='object').columns:
            if self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode_val)

    def _standardize_text_casing(self):
        """Paso 6: Estandariza la capitalización del texto para presentación."""
        logging.info("Paso 6: Estandarizando capitalización de texto...")
        for col in self.df.select_dtypes(include=['object']).columns:
            if col == 'NObesity':
                self.df[col] = self.df[col].str.title().str.replace(' i', ' I')
            else:
                self.df[col] = self.df[col].str.title()

    def _handle_duplicates(self):
        """Paso 7: Elimina filas duplicadas."""
        logging.info("Paso 7: Eliminando duplicados...")
        initial_rows = self.df.shape[0]
        self.df.drop_duplicates(inplace=True)
        final_rows = self.df.shape[0]
        logging.info(f"Se eliminaron {initial_rows - final_rows} filas duplicadas.")

    def _treat_outliers(self):
        """Paso 8: Trata outliers en columnas clave usando el método IQR."""
        logging.info("Paso 8: Tratando outliers con método IQR...")
        numerical_cols = ['Age', 'Weight', 'Height']
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
            self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])

    def save_data(self, output_path: Path):
        """Guarda el DataFrame limpio en la ruta de salida."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(output_path, index=False)
            logging.info(f"Datos limpios guardados exitosamente en: {output_path}")
        except IOError as e:
            logging.error(f"Error al guardar el archivo en {output_path}: {e}")
            sys.exit(1)

    def clean_data(self) -> pd.DataFrame:
        """
        Ejecuta el pipeline de limpieza completo en orden.
        Esta es la única función pública además de load y save.
        """
        self.load_data()
        self._normalize_text()
        self._delete_innessary_columns()
        self._handle_invalid_markers()
        self._convert_data_types()
        self._correct_domain_values()
        self._impute_missing_values()
        self._standardize_text_casing()
        self._handle_duplicates()
        self._treat_outliers()
        
        # Correcciones finales
        if 'ID' in self.df.columns:
            self.df = self.df.drop(columns=['ID'])
        if 'NObeyesdad' in self.df.columns:
            self.df = self.df.rename(columns={'NObeyesdad': 'NObesity'})
        
        logging.info("¡Proceso de limpieza completado!")
        logging.info(f"Dimensiones finales del dataset limpio: {self.df.shape}")
        return self.df

# Usamos Typer para una interfaz de línea de comandos moderna
def main(
    input_path: Path = typer.Argument(..., help="Ruta al archivo CSV de datos crudos."),
    output_path: Path = typer.Argument(..., help="Ruta donde se guardará el CSV limpio.")
):
    cleaner = DataCleaner(raw_data_path=input_path)
    cleaned_df = cleaner.clean_data()
    cleaner.save_data(output_path=output_path)

if __name__ == "__main__":
    typer.run(main)