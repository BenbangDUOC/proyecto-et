import pandas as pd
import logging
import os
import pickle
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
from kneed import KneeLocator
from sklearn.decomposition import PCA


# ============================================================================
# BLOQUE 1 (INTEGRANTE 1): CONFIGURACIÓN DE LOGS DE AUDITORÍA Y COMPONENTE ETL
# ============================================================================

# Crear la carpeta de logs si no existe
os.makedirs("data", exist_ok=True)

# Inicializar sistema de logs en archivo físico (Exigido en la rúbrica)
logging.basicConfig(
    filename='data/etl_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def ejecutar_pipeline_etl():
    logging.info("Arrancando el pipeline ETL unificado de la plataforma...")
    try:
        # extracción desde la fuente local CSV
        logging.info("Extrayendo datos de comportamiento desde data/usuarios_streaming.csv...")
        if not os.path.exists("data/usuarios_streaming.csv"):
            raise FileNotFoundError("No se encontró el archivo usuarios_streaming.csv en data/")
        clientes_streaming = pd.read_csv("data/usuarios_streaming.csv")
        
        # extracción desde la base de datos contenerizada de Postgres
        logging.info("Conectando al motor PostgreSQL 'crm_clientes'...")
        # NOTA: Usamos el host de red interna de docker "postgres"
        engine = create_engine("postgresql://admin:password@postgres:5432/crm_clientes")
        perfil_usuarios = pd.read_sql("SELECT * FROM perfil_usuarios", engine)
        logging.info("Extracción de perfiles desde Postgres completada con éxito.")

        # integración de fuentes mediante identificador único de negocio
        logging.info("Ejecutando operación Merge JOIN entre streaming y perfiles relacionales...")
        data_consolidada = clientes_streaming.merge(perfil_usuarios, on="id_cliente", how="inner")
        
        # Validación de esquemas contra GIGO
        logging.info("Iniciando auditoría y validación de consistencia de esquemas...")
        # Validación A: Evitar nulos imprevistos
        if data_consolidada.isnull().sum().sum() > 0:
            logging.warning("Se detectaron registros nulos imprevistos. Ejecutando dropna de emergencia.")
            data_consolidada.dropna(inplace=True)
        # Validación B: Consistencia de tipos de datos
        if not pd.api.types.is_numeric_dtype(data_consolidada['gasto_mensual']):
            raise TypeError("Error de esquema crítico: 'gasto_mensual' contiene datos no numéricos.")
            
        logging.info("Validación de esquemas aprobada. Conjunto analítico íntegro.")

        # guardar el dataset integrado listo en la ruta compartida
        ruta_salida = "data/data_usuarios.csv"
        data_consolidada.to_csv(ruta_salida, index=False)
        logging.info(f"Fase de carga completada con éxito. Archivo disponible en: {ruta_salida}")
        
        return data_consolidada

    except Exception as e:
        logging.critical(f"El Pipeline ETL se ha detenido por un fallo catastrófico: {str(e)}")
        raise

# ============================================================================
# ORQUESTACIÓN DEL FLUJO MAESTRO DE DATOS
# ============================================================================
if __name__ == "__main__":
    # --- CONFIRMACION PIPELINE ---
    print("[Ejecutando Pipeline ETL, Logs y Validación...")
    data = ejecutar_pipeline_etl()
    print("Matriz consolidada y validada con éxito. Dimensiones:", data.shape)
    
    # --- ESPACIO PARA LA PROGRAMACIÓN DEL INTEGRANTE 2 (MODELAMIENTO) ---
    # El k sea Integrante2 continuará programando aquí abajo. Usando la variable "data" 
    # que está limpia y lista para entrenar el escalador y el KMeans:
    
    os.makedirs("models", exist_ok=True)
    X = data.drop(columns=["cliente_id"])

    # Escalamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    silhouettes = []
    for k in range(2,11):
        modelo = KMeans(n_clusters=k, random_state=29, n_init=10)
        modelo.fit(X_scaled)

        inertias.append(modelo.inertia_)
        silhouettes.append(silhouette_score(X_scaled, modelo.labels_))

    kl = KneeLocator(
        range(2,11),
        inertias,
        curve='convex',
        direction='decreasing'
    )

    # Modelo
    k_optimo = kl.elbow
    kmeans = KMeans(n_clusters=k_optimo, random_state=29, n_init=10)
    # Predicciones
    clusters = kmeans.fit_predict(X_scaled)
    data["cluster"] = clusters

    print("Modelo de segmentación creado!!!")

    pca = PCA(n_components=2)

    componentes = pca.fit_transform(X_scaled)

    data["pc1"] = componentes[:, 0]
    data["pc2"] = componentes[:, 1]

    # Guarda data con los cluster y dos componentes principales
    data.to_csv("data/clientes_segmentados.csv", index=False)

    # Guarda las métricas 
    metricas = {
        "k_optimo": int(k_optimo),
        "silhouette_score": silhouette_score(X_scaled, data["cluster"]),
        "n_clientes": int(len(data)),
        "n_clusters": int(k_optimo),
        "varianza_pca": float(
            pca.explained_variance_ratio_.sum()
        )
    }

    with open("models/metricas.json", "w") as f:
        json.dump(metricas, f, indent=4)

    # Guarda los cenroides
    centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)

    centroides_df = pd.DataFrame(
        centroides_original,
        columns=X.columns
    )

    centroides_df.to_csv("data/centroides.csv", index=False)

    # Guardar modelo y data escalada
    pickle.dump(kmeans, open("models/modelo_kmeans.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    pickle.dump(pca, open("models/pca.pkl", "wb"))

    print("Modelo guardado")