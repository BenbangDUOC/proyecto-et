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

# se le crea un cuerpo a la peticicion simulando un cliente válido
payload_base = {
    "cantidad_contenidos_vistos": 15.0,
    "porcentaje_finalizacion": 85.5,
    "tiempo_promedio_sesion_min": 45.0,
    "cantidad_generos_consumidos": 3.0,
    "porcentaje_uso_promociones": 10.0,
    "antiguedad_cliente_meses": 12.0,
    "edad": 28.0,
    "dispositivos_registrados": 2.0,
    "porcentaje_uso_app_movil": 75.0,
    "cantidad_perfiles_creados": 1.0,
    "interacciones_mensuales_soporte": 0.0,
    "distancia_promedio_red_km": 5.5
}

def test_predecir_comportamiento_lineal():
    """
    Prueba que el modelo de Regresión Lineal retorne una estimación válida
    al recibir un payload correcto.
    """
    response = client.post("/predecir_comportamiento", json=payload_base)
    
    assert response.status_code == 200
    data = response.json()
    assert "gasto_mensual_estimado" in data
    # Verificamos que el resultado sea numérico
    assert isinstance(data["gasto_mensual_estimado"], float)

def test_predecir_comportamiento_arbol():
    """
    Prueba que el modelo Decision Tree retorne la estimación y el nombre del modelo.
    """
    response = client.post("/predecir_comportamiento_arbol", json=payload_base)
    
    assert response.status_code == 200
    data = response.json()
    assert "gasto_mensual_estimado" in data
    assert "modelo_utilizado" in data
    assert data["modelo_utilizado"] == "Decision Tree Regressor"
    assert isinstance(data["gasto_mensual_estimado"], float)

def test_predecir_error_validacion():
    """
    Prueba el manejo robusto de errores de la API. 
    Si falta un dato obligatorio (ej. edad), debe retornar error 422 (Unprocessable Entity).
    """
    payload_incompleto = payload_base.copy()
    del payload_incompleto["edad"]  # Forzamos un error eliminando una columna
    
    response = client.post("/predecir_comportamiento", json=payload_incompleto)
    
    # FastAPI usa 422 cuando falla la validación de esquemas (Pydantic)
    assert response.status_code == 422