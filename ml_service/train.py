import pandas as pd
import logging
import os
import pickle
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
        # 1. Extracción desde la fuente local CSV
        logging.info("Extrayendo datos de comportamiento desde data/usuarios_streaming.csv...")
        if not os.path.exists("data/usuarios_streaming.csv"):
            raise FileNotFoundError("No se encontró el archivo usuarios_streaming.csv en data/")
        clientes_streaming = pd.read_csv("data/usuarios_streaming.csv")
        
        # 2. Extracción desde la base de datos contenerizada de Postgres
        logging.info("Conectando al motor PostgreSQL 'crm_clientes'...")
        # NOTA: Usamos el host de red interna de docker "postgres"
        engine = create_engine("postgresql://admin:admin@postgres:5432/crm_clientes")
        perfil_usuarios = pd.read_sql("SELECT * FROM perfil_usuarios", engine)
        logging.info("Extracción de perfiles desde Postgres completada con éxito.")

        # 3. Transformación: Integración de fuentes mediante identificador único de negocio
        logging.info("Ejecutando operación Merge JOIN entre streaming y perfiles relacionales...")
        data_consolidada = clientes_streaming.merge(perfil_usuarios, on="id_cliente", how="inner")
        
        # 4. Validación de Esquemas (Control IL 1.5 contra el GIGO)
        logging.info("Iniciando auditoría y validación de consistencia de esquemas...")
        # Validación A: Evitar nulos imprevistos
        if data_consolidada.isnull().sum().sum() > 0:
            logging.warning("Se detectaron registros nulos imprevistos. Ejecutando dropna de emergencia.")
            data_consolidada.dropna(inplace=True)
        # Validación B: Consistencia de tipos de datos
        if not pd.api.types.is_numeric_dtype(data_consolidada['gasto_mensual']):
            raise TypeError("Error de esquema crítico: 'gasto_mensual' contiene datos no numéricos.")
            
        logging.info("Validación de esquemas aprobada. Conjunto analítico íntegro.")

        # 5. Carga: Guardar el dataset integrado listo en la ruta compartida
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
    # --- EJECUCIÓN PARTE INTEGRANTE 1 ---
    # Este print y esta función se ejecutarán perfectamente de inmediato para testear tu ETL
    print("[INTEGRANTE 1] Ejecutando Pipeline ETL, Logs y Validación...")
    data = ejecutar_pipeline_etl()
    print("[INTEGRANTE 1] Matriz analítica consolidada y validada con éxito. Dimensiones:", data.shape)
    
    # --- ESPACIO PARA LA PROGRAMACIÓN DEL INTEGRANTE 2 (MODELAMIENTO) ---
    # Tu compañero (Integrante 2) continuará programando aquí abajo. Él usará la variable "data" 
    # que tú le dejaste limpia y lista para entrenar el escalador y el KMeans:
    
    # os.makedirs("models", exist_ok=True)
    # X = data.drop(columns=["id_cliente"])  # Restricción: Sin variables categóricas o identificadores
    # scaler = StandardScaler()
    # ... kmeans.fit() ...