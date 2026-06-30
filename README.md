# Sistema End-to-End de Segmentación de Usuarios de Streaming (K-Means)

## Contexto de Negocio y Objetivos
Una plataforma internacional de entretenimiento vía streaming digital busca mitigar la pérdida de clientes y maximizar el consumo dentro de la aplicación mediante estrategias comerciales diferenciadas (campañas de retención, recomendaciones de contenido dirigidas y beneficios de fidelización). Históricamente, la organización operaba bajo reglas heurísticas generales. 

Este proyecto implementa una solución tecnológica **End-to-End (E2E)** que unifica datos analíticos aislados en múltiples entornos corporativos mediante un Pipeline ETL robusto, segmenta a los clientes de forma científica utilizando un algoritmo no supervisado **K-Means**, expone los artefactos mediante una API escalable de Machine Learning y aprovisiona un Dashboard gerencial interactivo. Toda la solución se encuentra orquestada y contenerizada bajo **Docker Compose**.

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
* Scikit-Learn: Kit de herramientas científicas utilizado para el preprocesamiento de datos (StandardScaler) y para el entrenamiento e instanciación del algoritmo de agrupamiento no supervisado KMeans.
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

### Objetivo del proyecto

Construir e integrar una solución analítica de aprendizaje no supervisado mediante el algoritmo K-Means para segmentar a los usuarios de la plataforma de streaming. El propósito es descubrir patrones de comportamiento y perfiles de consumo para diseñar estrategias comerciales dirigidas de recomendación de contenido, retención de clientes en riesgo de fuga y maximización del consumo digital.
