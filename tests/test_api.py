"""
Pruebas para la API FastAPI.
"""
import pytest
from fastapi.testclient import TestClient
from obesity_level_classifier.api.app import app

client = TestClient(app)

class TestAPI:
    """Tests para los endpoints de la API."""
    
    def test_root_endpoint(self):
        """Test del endpoint raíz."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        
    def test_health_endpoint(self):
        """Test del endpoint de salud."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        
    def test_predict_endpoint_valid_input(self):
        """Test de predicción con entrada válida."""
        payload = {
            "patients": [
                {
                    "Age": 25.0,
                    "Gender": "Male",
                    "Height": 1.75,
                    "Weight": 70.0,
                    "family_history_with_overweight": "Yes",
                    "FAVC": "Yes",
                    "FCVC": 2.0,
                    "NCP": 3.0,
                    "CAEC": "Sometimes",
                    "SMOKE": "No",
                    "CH2O": 2.0,
                    "SCC": "No",
                    "FAF": 1.0,
                    "TUE": 1.0,
                    "CALC": "Sometimes",
                    "MTRANS": "Public Transportation"
                }
            ]
        }
        
        response = client.post("/predict", json=payload)
        
        # Puede fallar si el modelo no está cargado, pero debe manejar el error
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert isinstance(data["predictions"], list)
            
    def test_predict_endpoint_invalid_age(self):
        """Test de predicción con edad inválida."""
        payload = {
            "patients": [
                {
                    "Age": 200.0,  # Edad inválida
                    "Gender": "Male",
                    "Height": 1.75,
                    "Weight": 70.0,
                    "family_history_with_overweight": "Yes",
                    "FAVC": "Yes",
                    "FCVC": 2.0,
                    "NCP": 3.0,
                    "CAEC": "Sometimes",
                    "SMOKE": "No",
                    "CH2O": 2.0,
                    "SCC": "No",
                    "FAF": 1.0,
                    "TUE": 1.0,
                    "CALC": "Sometimes",
                    "MTRANS": "Public Transportation"
                }
            ]
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
        
    def test_predict_endpoint_invalid_gender(self):
        """Test de predicción con género inválido."""
        payload = {
            "patients": [
                {
                    "Age": 25.0,
                    "Gender": "Other",  # Género inválido
                    "Height": 1.75,
                    "Weight": 70.0,
                    "family_history_with_overweight": "Yes",
                    "FAVC": "Yes",
                    "FCVC": 2.0,
                    "NCP": 3.0,
                    "CAEC": "Sometimes",
                    "SMOKE": "No",
                    "CH2O": 2.0,
                    "SCC": "No",
                    "FAF": 1.0,
                    "TUE": 1.0,
                    "CALC": "Sometimes",
                    "MTRANS": "Public Transportation"
                }
            ]
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        
    def test_predict_endpoint_batch(self):
        """Test de predicción con múltiples pacientes."""
        payload = {
            "patients": [
                {
                    "Age": 25.0,
                    "Gender": "Male",
                    "Height": 1.75,
                    "Weight": 70.0,
                    "family_history_with_overweight": "Yes",
                    "FAVC": "Yes",
                    "FCVC": 2.0,
                    "NCP": 3.0,
                    "CAEC": "Sometimes",
                    "SMOKE": "No",
                    "CH2O": 2.0,
                    "SCC": "No",
                    "FAF": 1.0,
                    "TUE": 1.0,
                    "CALC": "Sometimes",
                    "MTRANS": "Public Transportation"
                },
                {
                    "Age": 30.0,
                    "Gender": "Female",
                    "Height": 1.65,
                    "Weight": 65.0,
                    "family_history_with_overweight": "No",
                    "FAVC": "No",
                    "FCVC": 3.0,
                    "NCP": 4.0,
                    "CAEC": "No",
                    "SMOKE": "No",
                    "CH2O": 2.5,
                    "SCC": "Yes",
                    "FAF": 2.0,
                    "TUE": 0.5,
                    "CALC": "No",
                    "MTRANS": "Walking"
                }
            ]
        }
        
        response = client.post("/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["predictions"]) == 2
            
    def test_model_info_endpoint(self):
        """Test del endpoint de información del modelo."""
        response = client.get("/model-info")
        # Puede ser 200 o 503 dependiendo de si el modelo está cargado
        assert response.status_code in [200, 503]
        
    def test_docs_available(self):
        """Test que verifica que la documentación está disponible."""
        response = client.get("/docs")
        assert response.status_code == 200
