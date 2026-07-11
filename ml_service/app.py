import pandas as pd
import json
import pickle

from fastapi import FastAPI
import os
import subprocess

# --- AUTO-ENTRENAMIENTO ---
# Si la API detecta que los modelos no existen, ejecuta train.py internamente
if not os.path.exists("models/decision_tree_optimizado.pkl"):
    print("Modelos no encontrados. Entrenando desde cero dentro de Docker...")
    subprocess.run(["python", "train.py"])
# --------------------------

app = FastAPI(title="Servicio Segmentación Clientes")

# Carga la data que fue guardada en el entrenamiento
data = pd.read_csv("data/clientes_segmentados.csv")

# Carga el modelo
modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
# Carga data escalada
scaler = pickle.load(open("models/scaler.pkl", "rb"))
# Carga de modelos predictivos
modelo_dtc = pickle.load(open("models/decision_tree_optimizado.pkl", "rb"))
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

@app.post("/predict-abandono")
def predict_abandono(datos: dict):
    # Usamos exactamente tu mismo formato de entrada
    data = pd.DataFrame([datos])
    
    # El pipeline guardado en el .pkl ya incluye el preprocesador y escalador interno
    prediccion = modelo_dtc.predict(data)
    return {"riesgo_abandono": int(prediccion[0])}