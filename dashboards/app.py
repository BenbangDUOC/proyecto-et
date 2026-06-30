import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

#Configuración inicial de la página
st.set_page_config(
    page_title="Dashboard para segmentación de servicio de streaming", 
    page_icon=":bar_chart:", 
    layout="wide"
)

#Título y descripción de la página
st.title("Dashboard para segmentación de servicio de streaming")
st.markdown("En este dashboard se presentan los resultados del modelo de segmentación de clientes basado en KMeans."
            "La información a mostrar incluye los datos de los clientes, los centroides de los clusters y las métricas del modelo.")

#Función para obtener los datos del dashboard desde el servicio FastAPI
def obtener_datos():
    respuesta = requests.get("http://ml-service:8000/dashboard-data")
    respuesta.raise_for_status()  # Lanza un error si la solicitud no fue exitosa
    payload = respuesta.json()
    clientes = pd.DataFrame(payload['clientes'])
    centroides = pd.DataFrame(payload['centroides'])
    metricas = pd.DataFrame([payload['metricas']])
    return clientes, centroides, metricas

#Ejecución de la función para obtener los datos
clientes, centroides, metricas = obtener_datos()

#Métricas del modelo y justificación de la segmentación

st.subheader("Métricas del modelo")
st.markdown("En esta sección se presentan las métricas de evaluación del modelo, permitiendo entender la calidad de la segmentación realizada")

#Muestra de las métricas del modelo
col1, col2, col3, col4= st.columns(4)
with col1:
    st.metric('K óptimo', metricas['k_optimo'])
with col2:
    st.metric('Silhouette Score', round(metricas['silhouette_score'],4))
with col3:
    st.metric('Varianza PCA', round(metricas['varianza_pca'],4))
with col4:
    st.metric('Inercia', round(metricas['inercia'], 2))


#Scatter plot de PCA con centroides
st.subheader("Visualización de la segmentación de clientes con centroides")

fig_scatter = px.scatter(
    clientes,
    x='pc1',
    y='pc2',
    color='cluster',
    title='Visualización PCA de la segmentación de clientes',
    labels={'PCA1': 'Componente Principal 1', 'PCA2': 'Componente Principal 2', 'cluster': 'Cluster'}
)
fig_scatter.add_trace(go.Scatter(
    #x=centroides['PCA1'], falta guardar estas variables en el endpoint de FastAPI
    #y=centroides['PCA2'],
    mode='markers',
    marker=dict(color='black', size=12, symbol='x'),
    name='Centroides'
))
fig_scatter.update_traces(marker=dict(size=7), selector=dict(mode='markers'))
fig_scatter.update_layout(
    legend=dict(title='Cluster', x=1, y=1),
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_scatter)

#Gráfico de barras para la cantidad de clientes por cluster
st.subheader("Cantidad de clientes por cluster")
fig_bar = go.Figure()
for cluster in clientes['cluster'].unique():
    fig_bar.add_trace(go.Bar(
        x=[cluster],
        y=[len(clientes[clientes['cluster'] == cluster])],
        name=f'Cluster {cluster}'
    ))
fig_bar.update_layout(
    title='Cantidad de clientes por cluster',
    xaxis_title='Cluster',
    yaxis_title='Cantidad de clientes',
    barmode='group',
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_bar)
#Falta por agregar: método del codo(gráfico), perfil de los clusters(heatmap), comparación de los centroides(radar chart), análisis de variables específicas por clúster (boxplot), análisis de la información
st.write("Datos de métricas recibidos por Streamlit:")
st.json(metricas.to_dict()) # Esto mostrará el JSON en la pantalla