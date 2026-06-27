import pandas as pd
import json
import pickle

from fastapi import FastAPI

app = FastAPI(title="Servicio Segmentación Clientes")

# Carga la data que fue guardada en el entrenamiento
data = pd.read_csv("data/clientes_segmentados.csv")

# Carga el modelo
modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
# Carga data escalada
scaler = pickle.load(open("models/scaler.pkl", "rb"))

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