# Sistema End-to-End de analítica y predicción para usuarios de streaming

## Contexto de Negocio
Una plataforma internacional de entretenimiento vía streaming digital busca mitigar la pérdida de clientes y maximizar el consumo dentro de la aplicación mediante estrategias comerciales diferenciadas (campañas de retención, recomendaciones de contenido dirigidas y beneficios de fidelización). Históricamente, la organización operaba bajo reglas heurísticas generales. 
---

## Arquitectura de la Solución (Multi-Container Architecture)

La solución analítica está diseñada bajo una arquitectura desacoplada y modular compuesta por tres servicios principales aislados en una red virtual dedicada, garantizando alta cohesión, portabilidad y reproducibilidad formal.

### Diagrama de Arquitectura Técnica

El sistema se compone de tres microservicios independientes e imbricados mediante una red virtual privada interna de Docker, aislando las responsabilidades analíticas, de almacenamiento y de interfaz de usuario.

```text
                  +---------------------------------------------+
                  |             MÁQUINA HOST LOCAL              |
                  |     (Directorio compartido: ./data)         |
                  |     - Archivo: usuarios_streaming.csv       |
                  +----------------------+----------------------+
                                         |
                                         | (Mapeo por Bind Mount)
                                         v
+-----------------------------------------------------------------------+
|               ENTORNO CONTENERIZADO (streaming_network)               |
|                                                                       |
|  +-------------------+      +-------------------+     +------------+  |
|  | Servicio postgres |      |Servicio ml-service|     | Servicio   |  |
|  |  (crm_database)   |      | (ml_container)    |     | dashboard  |  |
|  |   PostgreSQL 16   |      |API + train.py ETL |     | Streamlit  |  |
|  +--------+----------+      +--------+----------+     +-----+------+  |
|           |                          ^                      |         |
|           |                          |                      |         |
|           | (Extracción SQL de       | (Peticiones HTTP     |         |
|           |  perfil_usuarios)        |  POST /predict)      |         |
|           +--------------------------+                      |         |
|                                      |                      |         |
|                                      +----------------------+         |
|                                                                       |
|                   +-----------------------+                           |
|                   |   Volumen Nombrado    |                           |
|                   |  modelos           |                               | 
|                   |  (Intercambio .pkl)   |                           |
|                   +-----------+-----------+                           |
|                               ^                                       |
|                               | (Persistencia cruzada de artefactos)  |
|                               +------------------------------------+  |
+-----------------------------------------------------------------------+
```
## Estructura del Proyecto

El repositorio mantiene la siguiente jerarquía estricta de componentes para garantizar la modularidad y el aislamiento de responsabilidades:
```text
/ (Raíz del proyecto)
│
├── docker-compose.yml              # Orquestador maestro multiservicio
├── .gitignore                      # Exclusión de entornos virtuales y temporales
├── .env                            # Variables de entorno e inyección de secretos
│
├── dashboards/                     # Capa de Presentación (Frontend)
│   ├── app.py                      # Aplicación Streamlit de exploración interactiva
│   ├── Dockerfile                  # Plano de construcción para el entorno Streamlit
│   └── requirements.txt            # Dependencias de visualización gráfica
│
├── ml-service/                     # Capa Analítica y Servicios (Backend)
│   ├── app.py                      # Código de la API para servir inferencias de clústeres
│   ├── train.py                    # Script automatizado: ETL + Entrenamiento KMeans
│   ├── Dockerfile                  # Plano de construcción para el entorno analítico
│   └── requirements.txt            # Dependencias de computación científica y APIs
│
├── models/                         # Persistencia de modelos entrenados y artefactos analíticos
│   ├── metricas.json               # Métricas de evaluación de los modelos de segmentación y regresión
│   ├── modelo_arbol.pkl            # Modelo Decision Tree Regressor entrenado para la predicción
│   ├── modelo_kmeans.pkl           # Modelo K-Means utilizado para la segmentación de clientes
│   ├── modelo_regresion.pkl        # Modelo Linear Regression entrenado para la predicción
│   ├── pca.pkl                     # Modelo PCA entrenado para la reducción de dimensionalidad y visualización de los clusters
│   └── scaler.pkl                  # Transformador de escalamiento utilizado durante el entrenamiento
│
├── database/                       # Inicialización y Motor de Base de Datos
│   ├── init.sql                    # Script DDL de creación de tablas y cargas masivas
│   └── perfil_usuarios.csv         # Datos transaccionales de origen CRM corporativo
│
└── data/                           # Volumen físico compartido (Host Bind Mount)
    ├── usuarios_streaming.csv      # Dataset local con hábitos de navegación web
    ├── etl_pipeline.log            # Archivo de auditoría técnica del pipeline ETL
    ├── data_usuarios.csv           # Matriz analítica intermedia consolidada
    ├── centroides.csv              # Coordenadas matemáticas de los clústeres
    └── clientes_segmentados.csv    # Registros etiquetados con su clúster asignado
```
---

## Tecnologías Utilizadas

La solución tecnológica se ha diseñado utilizando componentes de software desacoplados y librerías especializadas del ecosistema científico de Python, distribuidas de la siguiente manera:

### Lenguaje
* Python 3.11: Lenguaje de programación de alto nivel utilizado como núcleo de desarrollo para la construcción tanto de los scripts analíticos como de la lógica de servicios y presentación.

### Backend
* FastAPI: Framework web asíncrono y de alto rendimiento utilizado para la creación y exposición de los endpoints de inferencia analítica.
* Uvicorn: Servidor ASGI rápido y ligero encargado de ejecutar y servir la aplicación FastAPI.
* Requests: Librería HTTP utilizada por la capa de visualización para comunicarse internamente con los servicios expuestos por la API analítica.

### Machine Learning
* Scikit-Learn: Kit de herramientas científicas utilizado para el preprocesamiento de datos (StandardScaler) y para el entrenamiento e instanciación de los algoritmos de aprendizaje no supervisado(Kmeans) y supervisado(LinearRegression y DecisionTreeRegressor).
* Kneed: Librería especializada empleada en la fase de optimización analítica para identificar de forma programática el punto de inflexión matemática óptimo (Elbow Method o Método del Codo).

### Datos
* Pandas: Biblioteca de estructuras de datos utilizada para la manipulación tabular de alta velocidad, transformaciones intermedias, cruces de tipos join y la limpieza de fuentes de datos.
* NumPy: Soporte matemático y de computación matricial utilizado para el manejo eficiente de vectores y operaciones numéricas complejas.
* SQLAlchemy: Toolkit SQL y Mapeador Objeto-Relacional (ORM) encargado de gestionar la abstracción de consultas y la conexión directa con el motor de base de datos relacional.
* Psycopg2-Binary: Driver analítico nativo que funciona como el adaptador de base de datos PostgreSQL para Python. Este componente es fundamental y no sobra dentro de la arquitectura, ya que permite que SQLAlchemy traduzca comandos de Python en instrucciones SQL comprensibles por el motor transaccional.

### Visualización
* Streamlit: Plataforma web interactiva utilizada para estructurar la interfaz de usuario y desplegar el dashboard interactivo de negocio.
* Plotly: Librería de gráficos dinámicos que permite al usuario final explorar visualmente las dispersiones tridimensionales y métricas analíticas.
* Matplotlib: Motor gráfico secundario utilizado para el renderizado de gráficos estadísticos estáticos internos.

### Infraestructura
* Docker: Tecnología de contenedorización utilizada para empaquetar de forma aislada el código, el entorno de ejecución, las librerías del sistema y las dependencias del proyecto.
* Docker Compose: Herramienta de orquestación utilizada para definir, configurar y sincronizar el ciclo de vida de los servicios multiservicio (base de datos, api y frontend).
* PostgreSQL 16: Motor de base de datos relacional robusto utilizado para simular el entorno transaccional CRM corporativo.

---

## Datos

El pipeline de integración y unificación de datos procesa dos fuentes independientes de información corporativa para estructurar la matriz analítica final:

### Fuente 1: Información de Consumo Local (Archivo CSV)
Insumo tabular almacenado localmente en el directorio físico del proyecto bajo el nombre de `usuarios_streaming.csv`. Contiene variables asociadas estrictamente a los patrones y hábitos de comportamiento dentro del entorno digital de streaming. Su esquema de datos se compone por los siguientes campos:

* id_cliente
* horas_consumo_mensual
* gasto_mensual
* cantidad_contenidos_vistos
* sesiones_semana
* porcentaje_finalizacion
* tiempo_promedio_sesion_min
* cantidad_generos_consumidos
* porcentaje_uso_promociones
* antiguedad_cliente_meses

### Fuente 2: Información Complementaria (Base de Datos Relacional SQL)
Datos demográficos y transaccionales históricos alojados en un motor de base de datos PostgreSQL denominado `crm_database`. El proceso de carga masiva automatizada pobla las tablas transaccionales a partir del archivo maestro de respaldo corporativo `perfil_usuarios.csv`. Este entorno provee la información de contexto de los clientes a través del siguiente esquema de variables numéricas:

* id_cliente
* edad
* dispositivos_registrados
* porcentaje_uso_app_movil
* cantidad_perfiles_creados
* interacciones_mensuales_soporte
* distancia_promedio_red_km

---

## Requisitos

Es necesario contar con las siguientes herramientas instaladas en el sistema:
* Docker
* Docker Compose

### Levantar la Solución

Desde la raíz del proyecto, ejecute el siguiente comando en su terminal:


```bash
docker compose up --build
```


Esto levantará tres servicios:


| PostgreSQL con puerto 5432 |

| FastAPI con puerto 8000 |

| Streamlit con puerto 8501 |

---
## Acceso a los servicios


### API Machine Learning

Abrir:


```
http://localhost:8000
```


Respuesta esperada:


```json
{
 "mensaje": "Servicio ML funcionando"
}
```


---

## Dashboard

Abrir:


```
http://localhost:8501
```

El dashboard de segmentación permite:

- métricas del modelo.
- visualizar los clientes segmentados con centroides.
- visualizacion de los clientes segmentados con PCA.
- cantidad de clientes por cluster. 
- visualizar metodo del codo
- perfil de los clusters
- comparacion de los centroides con cluster
- visualizacion interactiva de los datos del cliente

El dashboard predictivo permite:
- comparar entre los dos modelos predictivos desarrollados.
- visualizar métricas de regresión.
- comparar rendimiento de los modelos.
- visualizar resultados de predicciones individuales.

---
### Detener servicios


Para detener los contenedores:


```bash
docker compose down
```


Para eliminar también los volúmenes:


```bash
docker compose down -v
```

---

### Para cambios de código


Para detener los contenedores:


```bash
docker compose down
```


Para eliminar también los volúmenes:


```bash
docker compose build --no-cache
```


Luego, levantamos 


```bash
docker compose up
```

---
Para verificar el contenido de la tabla

```bash
docker exec -it crm_database psql -U admin -d crm_clientes
```

Luego, para ver las tablas, ejecutar

```bash
\dt
```
---
### Modelos supervisados
La solución incorpora dos modelos de aprendizaje supervisado orientados a tareas de regresión. Se utiliza Linear Regression y Decision Tree Regressor, permitiendo comparar el desempeño entre un modelo lineal y uno no lineal sobre este conjunto de datos.

El dashboard incorpora un panel específico en el que se comparan ambos modelos mediante métricas de desempeño y predicciones realizadas sobre un mismo dato.

---

### Objetivo del proyecto

Construir e integrar una solución analítica que combine técnicas de aprendizaje no supervisado y supervisado para apoyar la toma de decisiones dentro de una plataforma de streaming.

Mediante K-means se segmentan los usuarios según sus patrones de comportamiento, mientras que los modelos Linear Regression y Decision Tree Regressor permiten realizar predicciones sobre variables objetivos relevantes para el negocio.

Finalmente, todos los resultados son publicados mediante una API REST y consumidos por dashboards interactivos desarrollados con Streamlit, proporcionando una plataforma completa para el análisis exploratorio, la segmentación y la predicción.

El propósito de la segmentación es descubrir patrones de comportamiento y perfiles de consumo para diseñar estrategias comerciales dirigidas de recomendación de contenido, retención de clientes en riesgo de fuga y maximización del consumo digital. Mientras que el de la predicción es anticipar el valor económico que representaría un usuario a la empresa, permitiendo optimizar campañas comerciales y/o proyectar ingresos futuros.
