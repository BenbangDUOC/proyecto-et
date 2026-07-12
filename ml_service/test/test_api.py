from fastapi.testclient import TestClient
from ml_service.app import app
# Inicializamos el cliente de pruebas
client = TestClient(app)

def test_endpoint_inicio():
    """
    Prueba que el endpoint raíz esté levantado y devuelva el mensaje correcto.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"mensaje": "Servicio ML funcionando"}

def test_endpoint_dashboard_data():
    """
    Prueba que el endpoint que alimenta el dashboard responda con éxito 
    y contenga las llaves principales (clientes, centroides, metricas).
    """
    response = client.get("/dashboard-data")
    
    # Verificamos que no haya errores de servidor (Status 200 = OK)
    assert response.status_code == 200
    
    # Extraemos el JSON
    data = response.json()
    
    # Verificamos que la estructura enviada a Streamlit sea la correcta
    assert "clientes" in data
    assert "centroides" in data
    assert "metricas" in data

