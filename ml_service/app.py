import pandas as pd
import json
import pickle

from fastapi import FastAPI
import pickle
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="Servicio Segmentación Clientes")

# Carga la data que fue guardada en el entrenamiento
data = pd.read_csv("data/clientes_segmentados.csv")

# Carga el modelo
modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
# Carga data escalada
scaler = pickle.load(open("models/scaler.pkl", "rb"))
# Carga de modelos predictivos
modelo_lr = pickle.load(open("models/modelo_regresion.pkl", "rb"))
# Carga las métricas
with open("models/metricas.json") as f:
    metricas = json.load(f)

@app.get("/")
def inicio():
    return {
        "mensaje":
        "Servicio ML funcionando"
    }

@app.get("/dashboard-data")
def dashboard_data():
    clientes = pd.read_csv(
        "data/clientes_segmentados.csv"
    )

    centroides = pd.read_csv("data/centroides.csv")

    return {
        "clientes": clientes.to_dict(orient="records"),
        "centroides": centroides.to_dict(orient="records"),
        "metricas": metricas
    }

@app.post("/predict")
def predict(datos:dict):
    data = pd.DataFrame([datos])
    X = scaler.transform(data)
    # Realiza la predicción (determinar el cluster)
    cluster = modelo.predict(X)

    return {"cluster": int(cluster[0])}

import pickle
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

# Configuración de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Definición del esquema según las columnas numéricas que utiliza tu modelo
class PrediccionRequest(BaseModel):
    cantidad_contenidos_vistos: float
    porcentaje_finalizacion: float
    tiempo_promedio_sesion_min: float
    cantidad_generos_consumidos: float
    porcentaje_uso_promociones: float
    antiguedad_cliente_meses: float
    edad: float
    dispositivos_registrados: float
    porcentaje_uso_app_movil: float
    cantidad_perfiles_creados: float
    interacciones_mensuales_soporte: float
    distancia_promedio_red_km: float

# Carga del pipeline completo
try:
    with open('models/modelo_regresion.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    logger.info("Pipeline de regresión cargado correctamente.")
except Exception as e:
    logger.error(f"Error al cargar el pipeline: {e}")
    pipeline = None

@app.post("/predecir_comportamiento")
async def predecir_comportamiento(data: PrediccionRequest):
    """
    Recibe los datos del cliente, aplica el preprocesamiento del pipeline 
    y retorna la predicción del gasto mensual.
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible.")
    
    try:
        # Convertir el objeto Pydantic a DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        #imputa, escala y predice 
        prediccion = pipeline.predict(input_df)
        
        return {"gasto_mensual_estimado": float(prediccion[0])}
    
    except Exception as e:
        logger.error(f"Error en la inferencia: {e}")
        raise HTTPException(status_code=400, detail="Error procesando los datos de entrada.")
    
# Carga del pipeline completo del Decision Tree Regressor
try:
    with open('models/modelo_arbol.pkl', 'rb') as f:
        pipeline_dt = pickle.load(f)
    logger.info("Pipeline de Decision Tree cargado correctamente.")
except Exception as e:
    logger.error(f"Error al cargar el pipeline: {e}")
    pipeline_dt = None


@app.post("/predecir_comportamiento_arbol")
async def predecir_comportamiento_arbol(data: PrediccionRequest):
    """
    Recibe los datos del cliente, aplica el preprocesamiento del pipeline 
    y retorna la predicción del gasto mensual usando el modelo Decision Tree.
    Justificación: Este modelo permite capturar relaciones no lineales en los datos.
    """
    if pipeline_dt is None:
        raise HTTPException(status_code=500, detail="El modelo de árbol no está disponible.")
    
    try:
        # Convertir el objeto Pydantic a DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # El pipeline imputa, escala y predice 
        prediccion = pipeline_dt.predict(input_df)
        
        return {
            "modelo_utilizado": "Decision Tree Regressor",
            "gasto_mensual_estimado": float(prediccion[0])
        }
    
    except Exception as e:
        logger.error(f"Error en la inferencia con el árbol: {e}")
        raise HTTPException(status_code=400, detail="Error procesando los datos de entrada.")