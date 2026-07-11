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
    # --- 1. PREPARACIÓN DE DATOS ---
    # Definimos el riesgo de abandono: Menos de 10 horas al mes Y 1 o menos sesiones por semana
    seed = 67
    cols_num = ['gasto_mensual', 'cantidad_contenidos_vistos', 
    'porcentaje_finalizacion', 'tiempo_promedio_sesion_min', 
    'cantidad_generos_consumidos', 'porcentaje_uso_promociones', 
    'antiguedad_cliente_meses', 'edad', 'dispositivos_registrados', 
    'porcentaje_uso_app_movil', 'cantidad_perfiles_creados', 
    'interacciones_mensuales_soporte', 'distancia_promedio_red_km']

    # Pipeline para variables numéricas (Imputa nulos con la media y luego escala)
    pipe_num = Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="mean")),
        ("escalado", StandardScaler())
    ])

    # Unimos los pipelines
    preprocesador = ColumnTransformer(
        transformers=[
            ("num", pipe_num, cols_num),
        ],
        remainder='drop' # Ignora cualquier columna que no esté en las listas anteriores (como los IDs)
    )
    # Separamos las características (X) y la etiqueta a predecir (y)
# --- NUEVA VARIABLE OBJETIVO ---
    y = data['cluster']
    # Eliminamos columnas que no aportan al modelo de clasificación
    X = data.drop(columns=['id_cliente', 'cluster', 'pc1', 'pc2'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # --- 2. DEFINICIÓN DEL PIPELINE ---
    # Debes asegurarte de definir el pipeline antes de pasarlo al RandomizedSearchCV
    # Ahora el pipeline completo hace la limpieza, transformación y luego predice
    pipeline_modelo_dtc = Pipeline([
        ('preprocesamiento', preprocesador), 
        ('modelo', DecisionTreeClassifier(random_state=seed))
    ])

# Este pipeline_modelo_dtc es el que le pasas al RandomizedSearchCV

    # --- 3. TU CÓDIGO DE TUNING (Búsqueda de Hiperparámetros) ---
    param_random_dtc = {
        "modelo__criterion": ["gini", "entropy"],
        "modelo__splitter": ["best", "random"],
        "modelo__max_depth": stats.randint(3, 10),
        "modelo__class_weight": ["balanced", None],
        "modelo__min_samples_split": stats.randint(3, 20)
    }

    random_dtc = RandomizedSearchCV(
    pipeline_modelo_dtc,
    param_random_dtc,
    n_iter=20,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
    scoring="accuracy",  # Cambiado a 'accuracy' para clasificación multiclase
    n_jobs=-1
)

    print("Entrenando Decision Tree con RandomizedSearchCV...")
    random_dtc.fit(X_train, y_train)

    # --- 4. EVALUACIÓN DEL MEJOR MODELO ---
    # Extraemos el mejor modelo encontrado
    mejor_dtc = random_dtc.best_estimator_

    # Evaluamos con los datos de prueba
    y_pred = mejor_dtc.predict(X_test)
# Cambiado a 'weighted' para que funcione con múltiples clusters
    score_f1_test = f1_score(y_test, y_pred, average='weighted')
    print(f"Mejores hiperparámetros: {random_dtc.best_params_}")
    print(f"F1-Score en Test: {score_f1_test}")

    # --- 5. PERSISTENCIA (GUARDADO PARA LA API) ---
    with open('models/decision_tree_optimizado.pkl', 'wb') as f:
        pickle.dump(mejor_dtc, f)

    # Creamos un diccionario base en lugar de intentar leer uno que no existe
    metricas_dict = {}
    
    # Guardamos el F1-Score y los mejores parámetros
    metricas_dict['dtc_accuracy_score'] = float(random_dtc.best_score_)
    metricas_dict['dtc_f1_weighted'] = float(score_f1_test)

    # --- 6. MÉTRICAS KMEANS ---
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

    # Guardamos todo en un solo archivo final
    with open("models/metricas.json", "w") as f:
        json.dump(metricas_dict, f, indent=4)
    #FIN PREDICCION=======================================================
    # Guarda los cenroides
    centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)

    # Usamos las columnas originales del dataset "data" (antes de dropear nada)
    # Debes excluir las columnas que no eran numéricas/escaladas
    cols_para_centroides = [c for c in data.columns if c not in ["id_cliente", "cluster", "pc1", "pc2", "riesgo_abandono"]]
    
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