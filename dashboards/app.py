import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# CONFIGURACIÓN INICIAL DEL ENTORNO WEB
# ============================================================================
st.set_page_config(
    page_title="Dashboard Analítico - Streaming", 
    page_icon=":bar_chart:", 
    layout="wide"
)

st.title("Plataforma de Inteligencia Analítica - Streaming")
st.markdown("Plataforma interactiva end-to-end para la segmentación no supervisada de clientes y la estimación supervisada de su gasto mensual.")

# ============================================================================
# CONEXIÓN A LA API DE INFERENCIA EN MEMORIA (BACKEND FASTAPI)
# ============================================================================
@st.cache_data(ttl=300) # Caché analítico de 5 minutos para optimizar transferencias de red
def obtener_datos():
    respuesta = requests.get("http://ml-service:8000/dashboard-data")
    respuesta.raise_for_status()
    payload = respuesta.json()
    clientes = pd.DataFrame(payload['clientes'])
    centroides = pd.DataFrame(payload['centroides'])
    metricas = pd.DataFrame([payload['metricas']])
    return clientes, centroides, metricas

try:
    clientes, centroides, metricas = obtener_datos()
except Exception as e:
    st.error(f"Error al conectar con la API de FastAPI en Docker. Verifique la orquestación. Detalles: {e}")
    st.stop()

# ============================================================================
# ESTRUCTURA DE PESTAÑAS (TABS UNIFICADAS PARA EFT)
# ============================================================================
tab_segmentacion, tab_regresion = st.tabs(["Análisis de Segmentos (K-Means)", "Predicción de Gasto (Regresión)"])

# ----------------------------------------------------------------------------
# PESTAÑA 1: SEGMENTACIÓN DE CLIENTES (APRENDIZAJE NO SUPERVISADO)
# ----------------------------------------------------------------------------
with tab_segmentacion:
    st.header("Análisis No Supervisado: K-Means & Reducción de Dimensionalidad (PCA)")
    st.subheader("Métricas del modelo de agrupamiento")
    st.markdown("Métricas de evaluación del modelo matemático obtenidas mediante validación de varianza e índices de cohesión estructural.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('K óptimo (Codo)', int(metricas['k_optimo'].iloc[0]))
    with col2:
        st.metric('Silhouette Score Global', round(metricas['silhouette_score'].iloc[0], 4))
    with col3:
        st.metric('Varianza Explicada (PCA)', f"{round(metricas['varianza_pca'].iloc[0] * 100, 2)}%")
    with col4:
        st.metric('Inercia (WCSS)', f"{round(metricas['inercia'].iloc[0], 2):,}")

    st.markdown("---")
    
    # Grid de visualizaciones analíticas de segmentación
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Método del codo (SSE)")
        k_values = metricas['lista_k'].iloc[0]
        inercia_values = metricas['lista_inercias'].iloc[0]
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=k_values, y=inercia_values, mode='lines+markers', name='Inercia', line=dict(color='firebrick', width=2)
        ))
        fig_elbow.update_layout(
            xaxis_title='Número de clusters (k)', yaxis_title='Inercia', margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_g2:
        st.subheader("Espacio Latente de Clientes (PCA)")
        clientes['cluster'] = clientes['cluster'].astype(str)
        fig_scatter = px.scatter(
            clientes, x='pc1', y='pc2', color='cluster',
            title='Visualización PCA de la segmentación de clientes',
            labels={'pc1': 'Componente Principal 1', 'pc2': 'Componente Principal 2', 'cluster': 'Cluster'},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_scatter.add_trace(go.Scatter(
            x=centroides['pc1'], y=centroides['pc2'], mode='markers',
            marker=dict(size=14, color='black', symbol='x', line=dict(width=2)), name='Centroides'
        ))
        fig_scatter.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
        fig_scatter.update_layout(legend=dict(title='Cluster', x=1, y=1), margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    
    col_g3, col_g4 = st.columns([1, 2])
    
    with col_g3:
        st.subheader("Distribución Volumétrica de Usuarios")
        fig_bar = go.Figure()
        for cluster in sorted(clientes['cluster'].unique()):
            fig_bar.add_trace(go.Bar(
                x=[f"Cluster {cluster}"], y=[len(clientes[clientes['cluster'] == cluster])], name=f'Cluster {cluster}'
            ))
        fig_bar.update_layout(
            xaxis_title='Estructura de Cluster', yaxis_title='Cantidad de clientes', barmode='group', margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_g4:
        st.subheader("Heatmap del Perfil Operacional de Clusters")
        columnas_numericas = clientes.select_dtypes(include=['float64', 'int64']).columns
        columnas_agrupar = [col for col in columnas_numericas if col not in ['id_cliente', 'pc1', 'pc2']]
        perfil_clusters = clientes.groupby('cluster')[columnas_agrupar].mean().reset_index()
        
        fig_heatmap = px.imshow(
            perfil_clusters.set_index('cluster'),
            labels=dict(x="Variables de Comportamiento", y="Cluster", color="Valor Promedio"),
            color_continuous_scale='Viridis', aspect="auto"
        )
        fig_heatmap.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")
    
    # Herramientas de diagnóstico de variables grupales
    st.subheader("Análisis de Centroides y Distribuciones")
    variables_radar = [col for col in columnas_numericas if col not in ['id_cliente', 'pc1', 'pc2', 'cluster', 'gasto_mensual']]
    variables_seleccionadas = st.multiselect("Selecciona variables para el Radar Chart de comparación:", options=variables_radar, default=variables_radar[:5])
    
    col_rad, col_box = st.columns(2)
    
    with col_rad:
        if len(variables_seleccionadas) >= 3:
            centroides['cluster'] = [str(i) for i in range(len(centroides))]
            centroides_con_filtro = centroides[['cluster'] + variables_seleccionadas]
            centroides_radar = centroides_con_filtro.set_index('cluster').T
            
            fig_radar = go.Figure()
            for cluster in centroides_radar.columns:
                fig_radar.add_trace(go.Scatterpolar(
                    r=centroides_radar[cluster].values, theta=centroides_radar.index, fill='toself', name=f'Cluster {cluster}'
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)), margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.warning("Seleccione al menos 3 variables cuantitativas para activar el gráfico radial.")

    with col_box:
        variable_box = st.selectbox("Seleccione un atributo analítico para inspeccionar sus outliers grupales:", columnas_agrupar)
        fig_box = px.box(
            clientes, x='cluster', y=variable_box, color='cluster',
            labels={'cluster': 'Cluster', variable_box: variable_box},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_box.update_layout(xaxis_title='Cluster', yaxis_title=variable_box, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_box, use_container_width=True)


# ----------------------------------------------------------------------------
# PESTAÑA 2: PREDICCIÓN DE GASTO MENSUAL (APRENDIZAJE SUPERVISADO OBLIGATORIO)
# ----------------------------------------------------------------------------
with tab_regresion:
    st.header("Simulador Supervisado de Gasto Mensual (CLP)")
    st.markdown("""
    **Gobernanza del Modelo:** En esta sección se evalúa el comportamiento predictivo de la plataforma. 
    Se contrastan las inferencias de un enfoque de frontera lineal frente a una estructura jerárquica no lineal para mitigar el subajuste.
    """)
    
    # === MEJORA REQUERIDA: MÉTRICAS HISTÓRICAS DE FORMA FIJA ===
    st.markdown("### Rendimiento Histórico del Conjunto de Validación (Test)")
    st.markdown("Estas métricas se calculan al entrenar el Pipeline en `train.py` utilizando Cross-Validation sobre la matriz relacional consolidada.")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(
            label="Regresión Lineal - R² (Ajuste)", 
            value=f"{metricas['lr_r2'].iloc[0]:.4f}",
            help="Coeficiente de determinación. Representa la proporción de la varianza del gasto explicada por las variables numéricas."
        )
    with col_m2:
        st.metric(
            label="Regresión Lineal - MAE (Error)", 
            value=f"${metricas['lr_mae'].iloc[0]:,.0f} CLP",
            help="Error Absoluto Medio. Indica el desvío monetario promedio por cada predicción en unidades reales del negocio."
        )
    with col_m3:
        st.metric(
            label="Decision Tree - R² (Ajuste)", 
            value=f"{metricas['dt_r2'].iloc[0]:.4f}",
            help="Capacidad del árbol para capturar interacciones no lineales controlando la profundidad máxima para evitar Overfitting."
        )
    with col_m4:
        st.metric(
            label="Decision Tree - MAE (Error)", 
            value=f"${metricas['dt_mae'].iloc[0]:,.0f} CLP",
            help="Error promedio del Decision Tree Regressor sobre el conjunto de prueba (Test)."
        )
        
    st.markdown("---")

    # Layout de Inferencia: Formulario a la izquierda, resultados estables a la derecha
    col_form, col_res = st.columns([5, 4])
    
    with col_form:
        st.subheader("Parámetros Operacionales del Cliente")
        with st.form("form_regresion"):
            c1, c2 = st.columns(2)
            with c1:
                edad = st.number_input("Edad del Usuario (Años)", min_value=18, max_value=100, value=34)
                antiguedad = st.number_input("Antigüedad en la Plataforma (Meses)", min_value=0, max_value=120, value=18)
                tiempo_sesion = st.number_input("Tiempo Promedio por Sesión (Min)", min_value=0.0, value=52.3)
                interacciones = st.number_input("Interacciones de Soporte Técnicos", min_value=0, value=1)
                contenidos = st.number_input("Cantidad de Contenidos Vistos (Mes)", min_value=0, value=84)
                generos = st.number_input("Cantidad de Géneros Distintos Consumidos", min_value=1, value=5)
            
            with c2:
                porcentaje_fin = st.slider("Porcentaje de Finalización de Videos (%)", 0.0, 100.0, 82.5)
                porcentaje_promo = st.slider("Uso de Cupones y Promociones (%)", 0.0, 100.0, 10.0)
                porcentaje_movil = st.slider("Uso de Dispositivos Móviles (%)", 0.0, 100.0, 45.0)
                dispositivos = st.slider("Número de Dispositivos Registrados", 1, 5, 2)
                perfiles = st.slider("Número de Perfiles Creados en Cuenta", 1, 10, 3)
                distancia = st.number_input("Distancia Promedio al Nodo de Red (Km)", min_value=0.0, value=8.7)

            btn_predecir = st.form_submit_button("Calcular Inferencia de Gasto", use_container_width=True)

    # Inferencia en vivo llamando de manera paralela a los endpoints seguros de FastAPI
    with col_res:
        st.subheader("Estimación de Ingreso Mensual Obtenido")
        st.markdown("Al presionar el botón, el preprocesador modular imputará y escalará las variables de forma atómica protegiendo contra el *Data Leakage*.")
        
        if btn_predecir:
            payload = {
                "cantidad_contenidos_vistos": float(contenidos),
                "porcentaje_finalizacion": float(porcentaje_fin),
                "tiempo_promedio_sesion_min": float(tiempo_sesion),
                "cantidad_generos_consumidos": float(generos),
                "porcentaje_uso_promociones": float(porcentaje_promo),
                "antiguedad_cliente_meses": float(antiguedad),
                "edad": float(edad),
                "dispositivos_registrados": float(dispositivos),
                "porcentaje_uso_app_movil": float(porcentaje_movil),
                "cantidad_perfiles_creados": float(perfiles),
                "interacciones_mensuales_soporte": float(interacciones),
                "distancia_promedio_red_km": float(distancia)
            }

            with st.spinner("Consumiendo microservicios analíticos..."):
                try:
                    # Llamadas HTTP síncronas a FastAPI
                    res_lr = requests.post("http://ml-service:8000/predecir_comportamiento", json=payload)
                    res_dt = requests.post("http://ml-service:8000/predecir_comportamiento_arbol", json=payload)
                    
                    res_lr.raise_for_status()
                    res_dt.raise_for_status()
                    
                    gasto_lr = res_lr.json().get("gasto_mensual_estimado", 0.0)
                    gasto_dt = res_dt.json().get("gasto_mensual_estimado", 0.0)
                    
                    # Almacenamiento en variables de sesión para evitar borrado de pantalla por clicks accidentales
                    st.session_state['pred_lr'] = gasto_lr
                    st.session_state['pred_dt'] = gasto_dt
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Fallo en la comunicación con FastAPI de Docker: {e}")

        # Despliegue de tarjetas de resultados estables
        if 'pred_lr' in st.session_state and 'pred_dt' in st.session_state:
            st.info("**Inferencia: Regresión Lineal Ordinaria**")
            st.metric(label="Gasto Estimado (OLS)", value=f"${st.session_state['pred_lr']:,.2f} CLP")
            
            st.success("**Inferencia: Decision Tree Regressor**")
            st.metric(label="Gasto Estimado (Tree)", value=f"${st.session_state['pred_dt']:,.2f} CLP")
            
            # Justificación estadística para la defensa oral (Puntos extra en argumentación)
            st.markdown("---")
            st.markdown("**Nota de Argumentación Técnica:**")
            diff = abs(st.session_state['pred_lr'] - st.session_state['pred_dt'])
            st.caption(f"""
            La discrepancia calculada en tiempo real para este perfil es de **${diff:,.2f} CLP**. 
            Se recomienda considerar la robustez del modelo de árbol para capturar interacciones no lineales, mientras que la regresión lineal ofrece interpretabilidad y simplicidad.
            """)
        else:
            st.warning("Complete los parámetros de entrada y presione el botón para proyectar los ingresos del usuario simulado.")
