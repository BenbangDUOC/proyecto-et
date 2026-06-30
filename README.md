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
