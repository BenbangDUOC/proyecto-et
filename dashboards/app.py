import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px

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

#Método del codo para determinar el número óptimo de clusters
st.subheader("Método del codo")
st.markdown("El método del codo es una técnica utilizada para determinar el número óptimo de clusters en un conjunto de datos.")
k_values = metricas['lista_k'].iloc[0]
inercia_values = metricas['lista_inercias'].iloc[0]
fig_elbow = go.Figure()
fig_elbow.add_trace(go.Scatter(
    x=k_values,
    y=inercia_values,
    mode='lines+markers',
    name='Inercia'
))
fig_elbow.update_layout(
    title='Método del codo',
    xaxis_title='Número de clusters (k)',
    yaxis_title='Inercia',
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_elbow)

#Scatter plot de PCA con centroides
st.subheader("Visualización de la segmentación de clientes con centroides")
st.subheader("Gráfico para visualizar la segmentación de clientes, utilizando los componentes principales obtenidos del análisis de PCA.")
fig_scatter = px.scatter(
    clientes,
    x='pc1',
    y='pc2',
    color='cluster',
    title='Visualización PCA de la segmentación de clientes',
    labels={'PCA1': 'Componente Principal 1', 'PCA2': 'Componente Principal 2', 'cluster': 'Cluster'}
)
#Centroides en el scatter plot
fig_scatter.add_trace(go.Scatter(
    x=centroides['pc1'],
    y=centroides['pc2'],
    mode='markers',
    marker=dict(size=12, color='black', symbol='x'),
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

#Muestra de los datos de métricas recibidos por Streamlit
st.write("Datos de métricas recibidos por Streamlit:")
st.json(metricas.to_dict()) # Esto mostrará el JSON en la pantalla

#Heatmap con el perfil de los clusters
st.subheader("Perfil de los clusters")
#Creación de la tabla de perfil de clusters
perfil_clusters = clientes.groupby('cluster').mean().reset_index()
#Creación del heatmap
fig_heatmap = px.imshow(
    perfil_clusters.drop('cluster', axis=1).set_index(perfil_clusters['cluster']),
    labels=dict(x="Variables", y="Cluster", color="Valor promedio"),
    x=perfil_clusters.drop('cluster', axis=1).columns,
    y=perfil_clusters['cluster'],
    color_continuous_scale='Viridis'
)
fig_heatmap.update_layout(
    title='Perfil de los clusters',
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_heatmap)

#Radar chart para comparar los centroides de los clusters
st.subheader("Comparación de los centroides de los clusters")

centroides['cluster'] = [str(i) for i in range(len(centroides))]
#Preparación de los datos para el radar chart
centroides_radar = centroides.set_index('cluster').T
#Creación del radar chart
fig_radar = go.Figure()
for cluster in centroides_radar.columns:
    fig_radar.add_trace(go.Scatterpolar(
        r=centroides_radar[cluster].values,
        theta=centroides_radar.index,
        fill='toself',
        name=f'Cluster {cluster}'
    ))
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True)
    ),
    title='Comparación de los centroides de los clusters',
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_radar)

#Visualización interactiva de los datos de clientes mediante boxplots
st.subheader("Visualización interactiva de los datos de clientes")
st.markdown("En esta sección se presentan boxplots interactivos para analizar la distribución de las variables de los clientes por cluster.")
#Variables numéricas para boxplots
variables = [
    col for col in clientes.select_dtypes(include=['float64', 'int64']).columns if col not in ['pc1', 'pc2', 'cluster']
]

variable = st.selectbox("Selecciona una variable para visualizar su distribución por cluster:", variables)
#Creación del boxplot
fig_box = px.box(
    clientes,
    x='cluster',
    y=variable,
    color='cluster',
    title=f'Distribución de {variable} por cluster',
    labels={'cluster': 'Cluster', variable: variable}
)
fig_box.update_layout(
    xaxis_title='Cluster',
    yaxis_title=variable,
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_box)