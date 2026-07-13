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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold, RandomizedSearchCV
import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
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
        perfil_usuario = pd.read_sql("SELECT * FROM perfil_usuarios", engine)
        logging.info("Extracción de perfiles desde Postgres completada con éxito.")

        # integración de fuentes mediante identificador único de negocio
        logging.info("Ejecutando operación Merge JOIN entre streaming y perfiles relacionales...")
        data_consolidada = clientes_streaming.merge(perfil_usuario, on="id_cliente", how="inner")
        
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
    clientes = pd.read_csv("data/usuarios_streaming.csv")

    # Fuente desde la BD
    engine = create_engine("postgresql://admin:password@postgres:5432/crm_clientes")

    perfil = pd.read_sql(
        """
        SELECT *
        FROM perfil_usuarios
        """,
        engine
    )

    # Integración
    data = clientes.merge(perfil, on="id_cliente")

    X = data.drop(columns=["id_cliente"])

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


    #Prediccion ---------------------------------------------------------
    seed = 67
    
    cols_num = ['cantidad_contenidos_vistos', 
    'porcentaje_finalizacion', 'tiempo_promedio_sesion_min', 
    'cantidad_generos_consumidos', 'porcentaje_uso_promociones', 
    'antiguedad_cliente_meses', 'edad', 'dispositivos_registrados', 
    'porcentaje_uso_app_movil', 'cantidad_perfiles_creados', 
    'interacciones_mensuales_soporte', 'distancia_promedio_red_km']

    # Pipeline numérico
    pipe_num = Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="mean")),
        ("escalado", StandardScaler())
    ])

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", pipe_num, cols_num),
        ],
        remainder='drop'
    )
    
    # Queremos predecir el gasto mensual numérico
    y = data['gasto_mensual']
    X = data.drop(columns=['id_cliente', 'cluster', 'pc1', 'pc2', 'gasto_mensual'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    pipeline_modelo_lr = Pipeline([
        ('preprocesamiento', preprocesador), 
        ('modelo', LinearRegression())
    ])

    print("Entrenando Regresión Lineal")
    pipeline_modelo_lr.fit(X_train, y_train)

    y_pred = pipeline_modelo_lr.predict(X_test)
    
    # Métricas de regresión 
    r2_lr = r2_score(y_test, y_pred)
    mse_lr = mean_squared_error(y_test, y_pred)
    mae_lr = mean_absolute_error(y_test, y_pred)

    print(f"R2 Score (Varianza explicada): {r2_lr}")
    print(f"Error Absoluto Medio (MAE): {mae_lr}")

    # Guardamos el pipeline lineal
    with open('models/modelo_regresion.pkl', 'wb') as f:
        pickle.dump(pipeline_modelo_lr, f)
    #  Decision Tree Regressor
    pipeline_dt = Pipeline([
        ('preprocesamiento', preprocesador), 
        ('modelo', DecisionTreeRegressor(max_depth=5, random_state=seed)) 
    ])


    pipeline_dt.fit(X_train, y_train)
        
    y_pred = pipeline_dt.predict(X_test)
    r2_dt = r2_score(y_test, y_pred)
    mae_dt = mean_absolute_error(y_test, y_pred)
    mse_dt = mean_squared_error(y_test, y_pred)

        
    print(f"Resultados Decision Tree Regression: R2={r2_dt:.4f}, MAE={mae_dt:.4f}")

    with open('models/modelo_arbol.pkl', 'wb') as f:
        pickle.dump(pipeline_dt, f)

    print("Modelos guardados en /models/")
    metricas_dict = {}
    
    #Se guardan las metricas de el modelo de regresion lineal
    metricas_dict['lr_r2'] = r2_lr
    metricas_dict['lr_mse'] = mae_lr
    metricas_dict['lr_mae'] = mae_lr
    
    #Se guardan las metricas del modelo de arbol de decision regresion
    metricas_dict['dt_r2'] = r2_dt
    metricas_dict['dt_mae'] = mae_dt
    metricas_dict['dt_mse'] = mse_dt



    # Guardamos las métricas K-Means en el mismo diccionario para unificar todo
    metricas_dict.update({
        "k_optimo": int(k_optimo),
        "silhouette_score": float(silhouette_score(X_scaled, data["cluster"])),
        "n_clientes": int(len(data)),
        "n_clusters": int(k_optimo),
        "varianza_pca": float(pca.explained_variance_ratio_.sum()),
        "inercia": float(kmeans.inertia_),          
        "lista_inercias": [float(i) for i in inertias], 
        "lista_k": list(range(2, 11))
    })

    with open("models/metricas.json", "w") as f:
        json.dump(metricas_dict, f, indent=4)
    # Guarda los cenroides
    centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)

    cols_para_centroides = [c for c in data.columns if c not in ["id_cliente", "cluster", "pc1", "pc2"]]
    centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)
    
    centroides_df = pd.DataFrame(
        centroides_original,
        columns=cols_para_centroides 
    )
    centroides_pca = pca.transform(kmeans.cluster_centers_)
    centroides_df["pc1"] = centroides_pca[:, 0]
    centroides_df["pc2"] = centroides_pca[:, 1]

    centroides_df.to_csv("data/centroides.csv", index=False)

    # Guardar modelo y data escalada
    pickle.dump(kmeans, open("models/modelo_kmeans.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    pickle.dump(pca, open("models/pca.pkl", "wb"))

    print("Modelo guardado")