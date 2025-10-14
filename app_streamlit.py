import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(
    page_title="Segmentación de Clientes RFM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar estilo CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.6rem;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 100%;
        color: #2c3e50 !important;
    }
    .metric-card h3 {
        color: #2c3e50 !important;
        font-size: 0.9rem;
        margin: 0 0 0.5rem 0;
        opacity: 0.8;
    }
    .metric-card .value {
        color: #2c3e50 !important;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0;
    }
    .cluster-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .cluster-info h3 {
        color: white !important;
        margin: 0 0 0.5rem 0 !important;
        font-size: 1.8rem;
        font-weight: 600;
    }
    .cluster-info p {
        color: white !important;
        margin: 0 !important;
        opacity: 0.9;
        font-size: 1rem;
        line-height: 1.5;
    }
    .section-divider {
        margin: 2rem 0;
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #3498db, transparent);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
    .feature-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2ecc71;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #2c3e50 !important;
        border: 1px solid #e0e0e0;
    }
    .feature-box strong {
        color: #2c3e50 !important;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
        color: #2c3e50 !important;
        border: 1px solid #b3d9ff;
    }
    .info-box strong {
        color: #2c3e50 !important;
    }
    .strategy-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
        color: #856404 !important;
        border: 1px solid #ffeaa7;
    }
    .strategy-box strong {
        color: #856404 !important;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stat-card h4 {
        color: white;
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stat-card .value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
        color: white;
    }
    .stat-card .subvalue {
        font-size: 0.9rem;
        opacity: 0.8;
        margin: 0;
        color: white;
    }
    /* Estilos para texto dentro de las cajas */
    .box-content {
        color: #2c3e50 !important;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0;
    }
    .feature-box .box-content {
        color: #2c3e50 !important;
    }
    .strategy-box .box-content {
        color: #856404 !important;
    }
    .info-box .box-content {
        color: #2c3e50 !important;
    }
    .dataframe-container {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .comparison-table {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">Sistema de Segmentación de Clientes - Análisis RFM</h1>', unsafe_allow_html=True)
st.markdown("**Proyecto de Minería de Datos – Grupo 4**")
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Cargar modelo y datos
@st.cache_resource
def cargar_modelo():
    try:
        ruta_base = r"E:\Descargas\Proyecto de Minería de Datos – Grupo 4"
        modelo_path = os.path.join(ruta_base, "modelo_kmeans.pkl")
        if os.path.exists(modelo_path):
            modelo = joblib.load(modelo_path)
            return modelo
        else:
            st.warning("Modelo no encontrado. Continuando sin modelo.")
            return None
    except Exception as e:
        st.warning(f"Modelo no disponible: {e}")
        return None

@st.cache_data
def cargar_datos_originales():
    try:
        ruta_base = r"E:\Descargas\Proyecto de Minería de Datos – Grupo 4"
        ruta_excel = os.path.join(ruta_base, "Online Retail.xlsx")
        if os.path.exists(ruta_excel):
            df = pd.read_excel(ruta_excel)
            
            # Limpieza básica
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
            df = df.dropna(subset=["CustomerID"])
            df = df[df["Quantity"] > 0]
            
            return df
        else:
            st.error("Archivo de datos no encontrado.")
            return None
    except Exception as e:
        st.error(f"Error cargando dataset: {e}")
        return None

@st.cache_data
def cargar_resultados_rfm():
    try:
        ruta_base = r"E:\Descargas\Proyecto de Minería de Datos – Grupo 4"
        csv_path = os.path.join(ruta_base, "resultados_segmentacion.csv")
        if os.path.exists(csv_path):
            rfm = pd.read_csv(csv_path, index_col=0)
            return rfm
    except:
        pass
    
    # Si no existe el CSV, calcular RFM desde cero
    df = cargar_datos_originales()
    if df is None:
        return None
        
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "UnitPrice": lambda x: (x * df.loc[x.index, "Quantity"]).sum()
    }).rename(columns={
        "InvoiceDate": "Recency",
        "InvoiceNo": "Frequency",
        "UnitPrice": "Monetary"
    })
    
    # Aplicar clustering si el modelo está disponible
    modelo = cargar_modelo()
    if modelo is not None:
        try:
            # Escalar datos para el modelo
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
            rfm['Cluster'] = modelo.predict(rfm_scaled)
        except Exception as e:
            st.warning(f"No se pudo aplicar el modelo de clustering: {e}")
    
    return rfm

# Sidebar para navegación
st.sidebar.title("Panel de Navegación")
st.sidebar.markdown("---")
opcion = st.sidebar.radio(
    "Selecciona una sección:",
    ["Dashboard General", "Análisis de Segmentos", "Buscar Cliente", "Recomendaciones Estratégicas"]
)

# Cargar recursos
with st.spinner('Cargando datos...'):
    modelo = cargar_modelo()
    df_original = cargar_datos_originales()
    rfm_data = cargar_resultados_rfm()

if df_original is None or rfm_data is None:
    st.error("No se pudieron cargar los datos. Verifica las rutas de archivo.")
    st.stop()

# --- SECCIÓN 1: DASHBOARD GENERAL ---
if opcion == "Dashboard General":
    st.markdown('<h2 class="sub-header">Dashboard General RFM</h2>', unsafe_allow_html=True)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_clientes = len(rfm_data)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Clientes", f"{total_clientes:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        total_ventas = df_original['Quantity'].sum()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Productos Vendidos", f"{total_ventas:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        ingresos_totales = (df_original['Quantity'] * df_original['UnitPrice']).sum()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Ingresos Totales", f"${ingresos_totales:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        transacciones_totales = df_original['InvoiceNo'].nunique()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Transacciones", f"{transacciones_totales:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # Distribución de clusters
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown('<h3 class="sub-header">Distribución de Segmentos</h3>', unsafe_allow_html=True)
        
        if 'Cluster' in rfm_data.columns:
            cluster_counts = rfm_data['Cluster'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            bars = ax.bar([f'Segmento {i}' for i in cluster_counts.index], 
                         cluster_counts.values, 
                         color=colors[:len(cluster_counts)])
            
            ax.set_ylabel('Número de Clientes', fontweight='bold', color='#2c3e50')
            ax.set_xlabel('Segmentos', fontweight='bold', color='#2c3e50')
            ax.set_title('Distribución de Clientes por Segmento RFM', fontsize=14, fontweight='bold', color='#2c3e50')
            ax.tick_params(colors='#2c3e50')
            plt.xticks(rotation=45)
            
            # Agregar valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}\n({height/total_clientes*100:.1f}%)', 
                       ha='center', va='bottom', fontweight='bold', fontsize=9, color='#2c3e50')
            
            # Configurar color del gráfico
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#f8f9fa')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No hay datos de segmentación disponibles. Ejecuta el análisis de clustering primero.")
    
    with col_right:
        st.markdown('<h3 class="sub-header">Resumen RFM por Segmento</h3>', unsafe_allow_html=True)
        
        if 'Cluster' in rfm_data.columns:
            resumen_clusters = rfm_data.groupby('Cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean', 
                'Monetary': 'mean'
            }).round(2)
            
            # Formatear para mejor visualización
            resumen_display = resumen_clusters.rename(columns={
                'Recency': 'Recencia (días)',
                'Frequency': 'Frecuencia',
                'Monetary': 'Valor Monetario ($)'
            })
            
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(
                resumen_display.style.format({
                    'Valor Monetario ($)': '${:,.2f}',
                    'Recencia (días)': '{:.1f}',
                    'Frecuencia': '{:.1f}'
                }).background_gradient(cmap='Blues'), 
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Mostrar estadísticas generales si no hay clusters
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Estadísticas RFM Generales:**")
            st.write(f"- **Recencia promedio:** {rfm_data['Recency'].mean():.1f} días")
            st.write(f"- **Frecuencia promedio:** {rfm_data['Frequency'].mean():.1f} transacciones")
            st.write(f"- **Valor monetario promedio:** ${rfm_data['Monetary'].mean():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Gráficos de distribución RFM
    st.markdown('<h3 class="sub-header">Distribución de Variables RFM</h3>', unsafe_allow_html=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Recency
    axes[0].hist(rfm_data['Recency'], bins=30, alpha=0.7, color='#3498db', edgecolor='black')
    axes[0].set_title('Distribución de Recencia', fontweight='bold', color='#2c3e50')
    axes[0].set_xlabel('Días desde última compra', color='#2c3e50')
    axes[0].set_ylabel('Frecuencia', color='#2c3e50')
    axes[0].tick_params(colors='#2c3e50')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor('#f8f9fa')
    
    # Frequency
    axes[1].hist(rfm_data['Frequency'], bins=30, alpha=0.7, color='#2ecc71', edgecolor='black')
    axes[1].set_title('Distribución de Frecuencia', fontweight='bold', color='#2c3e50')
    axes[1].set_xlabel('Número de transacciones', color='#2c3e50')
    axes[1].set_ylabel('Frecuencia', color='#2c3e50')
    axes[1].tick_params(colors='#2c3e50')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor('#f8f9fa')
    
    # Monetary
    monetary_filtered = rfm_data[rfm_data['Monetary'] < rfm_data['Monetary'].quantile(0.95)]['Monetary']
    axes[2].hist(monetary_filtered, bins=30, alpha=0.7, color='#e74c3c', edgecolor='black')
    axes[2].set_title('Distribución de Valor Monetario (95% percentil)', fontweight='bold', color='#2c3e50')
    axes[2].set_xlabel('Valor total ($)', color='#2c3e50')
    axes[2].set_ylabel('Frecuencia', color='#2c3e50')
    axes[2].tick_params(colors='#2c3e50')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_facecolor('#f8f9fa')
    
    fig.patch.set_facecolor('#f8f9fa')
    plt.tight_layout()
    st.pyplot(fig)

# --- SECCIÓN 2: ANÁLISIS DE SEGMENTOS ---
elif opcion == "Análisis de Segmentos":
    st.markdown('<h2 class="sub-header">Análisis Detallado por Segmento</h2>', unsafe_allow_html=True)
    
    if 'Cluster' not in rfm_data.columns:
        st.warning("No se encontraron datos de segmentación. Ejecuta primero el análisis de clustering.")
        st.stop()
    
    # Selector de segmento
    segmentos_disponibles = sorted(rfm_data['Cluster'].unique())
    segmento_seleccionado = st.selectbox(
        "Selecciona un segmento para analizar:",
        segmentos_disponibles
    )
    
    # Datos del segmento seleccionado
    segmento_data = rfm_data[rfm_data['Cluster'] == segmento_seleccionado]
    
    # Métricas del segmento
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Clientes en Segmento", f"{len(segmento_data):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Recencia Promedio", f"{segmento_data['Recency'].mean():.1f} días")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Frecuencia Promedio", f"{segmento_data['Frequency'].mean():.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Valor Promedio", f"${segmento_data['Monetary'].mean():.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # Visualizaciones comparativas
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.markdown('<h3 class="sub-header">Comparación de Recencia</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        for cluster in segmentos_disponibles:
            data_cluster = rfm_data[rfm_data['Cluster'] == cluster]['Recency']
            color = '#e74c3c' if cluster == segmento_seleccionado else '#bdc3c7'
            alpha = 0.8 if cluster == segmento_seleccionado else 0.4
            label = f'Segmento {cluster}' if cluster == segmento_seleccionado else f'Seg {cluster}'
            ax.hist(data_cluster, bins=15, alpha=alpha, label=label, color=color)
        
        ax.set_xlabel('Recencia (días)', color='#2c3e50')
        ax.set_ylabel('Frecuencia', color='#2c3e50')
        ax.set_title('Distribución de Recencia por Segmento', fontweight='bold', color='#2c3e50')
        ax.tick_params(colors='#2c3e50')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_viz2:
        st.markdown('<h3 class="sub-header">Comparación de Valor Monetario</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        for cluster in segmentos_disponibles:
            data_cluster = rfm_data[rfm_data['Cluster'] == cluster]['Monetary']
            # Filtrar outliers para mejor visualización
            data_cluster = data_cluster[data_cluster < data_cluster.quantile(0.95)]
            color = '#27ae60' if cluster == segmento_seleccionado else '#bdc3c7'
            alpha = 0.8 if cluster == segmento_seleccionado else 0.4
            label = f'Segmento {cluster}' if cluster == segmento_seleccionado else f'Seg {cluster}'
            ax.hist(data_cluster, bins=15, alpha=alpha, label=label, color=color)
        
        ax.set_xlabel('Valor Monetario ($)', color='#2c3e50')
        ax.set_ylabel('Frecuencia', color='#2c3e50')
        ax.set_title('Distribución de Valor por Segmento', fontweight='bold', color='#2c3e50')
        ax.tick_params(colors='#2c3e50')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Top clientes del segmento
    st.markdown('<h3 class="sub-header">Top 10 Clientes del Segmento</h3>', unsafe_allow_html=True)
    
    top_clientes = segmento_data.nlargest(10, 'Monetary')[['Recency', 'Frequency', 'Monetary']]
    top_clientes_display = top_clientes.copy()
    top_clientes_display.index.name = 'CustomerID'
    
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(
        top_clientes_display.style.format({
            'Monetary': '${:,.2f}',
            'Recency': '{:.0f} días',
            'Frequency': '{:.0f}'
        }),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# --- SECCIÓN 3: BUSCAR CLIENTE ---
elif opcion == "Buscar Cliente":
    st.markdown('<h2 class="sub-header">Análisis de Cliente Específico</h2>', unsafe_allow_html=True)
    
    # Selector de cliente
    cliente_ids = rfm_data.index.astype(str).tolist()
    cliente_seleccionado = st.selectbox("Selecciona un CustomerID:", cliente_ids)
    
    if cliente_seleccionado:
        try:
            cliente_data = rfm_data.loc[float(cliente_seleccionado)]
            
            # Mostrar información del cliente
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("CustomerID", cliente_seleccionado)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Recencia", f"{cliente_data['Recency']:.0f} días")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Frecuencia", f"{cliente_data['Frequency']:.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'Cluster' in cliente_data:
                    st.metric("Segmento", f"Segmento {cliente_data['Cluster']}")
                else:
                    st.metric("Segmento", "No asignado")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            
            # Valor monetario
            col_valor, col_comparativa = st.columns([1, 2])
            
            with col_valor:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Valor Monetario Total", f"${cliente_data['Monetary']:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Comparación con el segmento
            if 'Cluster' in cliente_data:
                segmento_cliente = cliente_data['Cluster']
                segmento_promedio = rfm_data[rfm_data['Cluster'] == segmento_cliente].mean()
                
                with col_comparativa:
                    st.markdown('<h3 class="sub-header">Comparación con su Segmento</h3>', unsafe_allow_html=True)
                    
                    comparacion = pd.DataFrame({
                        'Este Cliente': [
                            cliente_data['Recency'],
                            cliente_data['Frequency'], 
                            cliente_data['Monetary']
                        ],
                        'Promedio del Segmento': [
                            segmento_promedio['Recency'],
                            segmento_promedio['Frequency'],
                            segmento_promedio['Monetary']
                        ]
                    }, index=['Recencia (días)', 'Frecuencia', 'Valor Monetario ($)'])
                    
                    # Calcular diferencias porcentuales
                    comparacion['Diferencia %'] = (
                        (comparacion['Este Cliente'] - comparacion['Promedio del Segmento']) / 
                        comparacion['Promedio del Segmento'] * 100
                    ).round(1)
                    
                    st.markdown('<div class="comparison-table">', unsafe_allow_html=True)
                    st.dataframe(comparacion.style.format({
                        'Este Cliente': '{:.2f}',
                        'Promedio del Segmento': '{:.2f}',
                        'Diferencia %': '{:.1f}%'
                    }), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        except KeyError:
            st.error("Cliente no encontrado en los datos.")

# --- SECCIÓN 4: RECOMENDACIONES ESTRATÉGICAS ---
elif opcion == "Recomendaciones Estratégicas":
    st.markdown('<h2 class="sub-header">Recomendaciones por Segmento</h2>', unsafe_allow_html=True)
    
    recomendaciones = {
        0: {
            "nombre": "Clientes Inactivos",
            "descripcion": "Alta recencia, baja frecuencia y valor monetario. Clientes que no han realizado compras recientemente.",
            "caracteristicas": ["No compran hace mucho tiempo", "Bajo historial de compras", "Bajo valor total"],
            "estrategias": [
                "Campañas de reactivación con descuentos especiales",
                "Encuestas de satisfacción para entender su partida", 
                "Programas de recuperación con beneficios exclusivos",
                "Email marketing con ofertas personalizadas"
            ],
            "objetivos": ["Recuperar al 20% de clientes", "Incrementar frecuencia de compra"]
        },
        1: {
            "nombre": "Clientes Leales", 
            "descripcion": "Recencia media, frecuencia constante y valor moderado. Base estable de la empresa.",
            "caracteristicas": ["Clientes regulares", "Fidelidad comprobada", "Valor constante"],
            "estrategias": [
                "Programas de fidelización con puntos canjeables",
                "Acceso anticipado a nuevos productos y promociones",
                "Comunicación personalizada y reconocimiento especial",
                "Ofertas de cross-selling basadas en historial"
            ],
            "objetivos": ["Mantener tasa de retención >80%", "Incrementar valor promedio en 15%"]
        },
        2: {
            "nombre": "Clientes Nuevos/Potenciales",
            "descripcion": "Recencia baja, frecuencia en crecimiento, valor variable. Oportunidad de crecimiento.", 
            "caracteristicas": ["Clientes recientes", "En proceso de fidelización", "Potencial de crecimiento"],
            "estrategias": [
                "Programas de onboarding y bienvenida",
                "Tutoriales y guías de uso de productos",
                "Ofertas de seguimiento para segunda compra", 
                "Contenido educativo sobre beneficios de la marca"
            ],
            "objetivos": ["Convertir 40% en clientes leales", "Duplicar frecuencia en 6 meses"]
        },
        3: {
            "nombre": "Clientes de Alto Valor",
            "descripcion": "Baja recencia, alta frecuencia y valor monetario elevado. Clientes más valiosos.",
            "caracteristicas": ["Clientes más valiosos", "Frecuentes y recientes", "Alto gasto total"],
            "estrategias": [
                "Atención personalizada y dedicada (account manager)",
                "Ofertas exclusivas y productos premium",
                "Eventos especiales e invitaciones privadas", 
                "Programas VIP con beneficios superiores"
            ],
            "objetivos": ["Retención del 95%", "Incrementar valor en 25% anual"]
        }
    }
    
    # Selector de segmento para recomendaciones
    if 'Cluster' in rfm_data.columns:
        segmentos_disponibles = sorted(rfm_data['Cluster'].unique())
        segmento_recomendaciones = st.selectbox(
            "Selecciona un segmento para ver recomendaciones:",
            options=segmentos_disponibles,
            format_func=lambda x: f"Segmento {x}: {recomendaciones.get(x, {}).get('nombre', 'Desconocido')}"
        )
    else:
        st.warning("No hay segmentos disponibles para mostrar recomendaciones.")
        st.stop()
    
    if segmento_recomendaciones in recomendaciones:
        info = recomendaciones[segmento_recomendaciones]
        
        # Header del segmento - CORREGIDO: Texto visible
        st.markdown(f"""
        <div class="cluster-info">
            <h3>{info["nombre"]}</h3>
            <p>{info["descripcion"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_info, col_metrics = st.columns([2, 1])
        
        with col_info:
            st.markdown("**Características Principales**")
            for caracteristica in info['caracteristicas']:
                st.markdown(f"""
                <div class="feature-box">
                    <p class="box-content">• {caracteristica}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("**Estrategias Recomendadas**")
            for estrategia in info['estrategias']:
                st.markdown(f"""
                <div class="strategy-box">
                    <p class="box-content">• {estrategia}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("**Métricas Objetivo**")
            for objetivo in info['objetivos']:
                st.markdown(f"""
                <div class="info-box">
                    <p class="box-content">• {objetivo}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_metrics:
            # Estadísticas rápidas del segmento
            if 'Cluster' in rfm_data.columns:
                segmento_stats = rfm_data[rfm_data['Cluster'] == segmento_recomendaciones]
                
                st.markdown("**Estadísticas del Segmento**")
                
                # Métricas en cards especiales
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>Clientes</h4>
                        <div class="value">{len(segmento_stats):,}</div>
                        <div class="subvalue">en segmento</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>Valor Promedio</h4>
                        <div class="value">${segmento_stats['Monetary'].mean():.2f}</div>
                        <div class="subvalue">por cliente</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>Recencia</h4>
                        <div class="value">{segmento_stats['Recency'].mean():.1f}</div>
                        <div class="subvalue">días promedio</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>Frecuencia</h4>
                        <div class="value">{segmento_stats['Frequency'].mean():.1f}</div>
                        <div class="subvalue">transacciones</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Resumen ejecutivo
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Resumen Ejecutivo de Segmentación</h3>', unsafe_allow_html=True)
    
    if 'Cluster' in rfm_data.columns:
        resumen_final = rfm_data.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum'],
            'Cluster': 'count'
        }).round(2)
        
        resumen_final.columns = ['Recencia Prom', 'Frecuencia Prom', 'Valor Prom', 'Valor Total', 'Clientes']
        
        # Formatear valores monetarios
        resumen_display = resumen_final.copy()
        resumen_display['Valor Prom'] = resumen_display['Valor Prom'].apply(lambda x: f"${x:,.2f}")
        resumen_display['Valor Total'] = resumen_display['Valor Total'].apply(lambda x: f"${x:,.2f}")
        resumen_display['Recencia Prom'] = resumen_display['Recencia Prom'].apply(lambda x: f"{x:.1f} días")
        
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(resumen_display, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("**Proyecto de Minería de Datos - Grupo 4** | *Sistema de clasificación de clientes según patrones de compra*")
st.markdown("Desarrollado con Streamlit")

# Información del sistema en sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### Información del Sistema")
    st.write(f"**Clientes cargados:** {len(rfm_data):,}")
    st.write(f"**Transacciones:** {len(df_original):,}")
    if 'Cluster' in rfm_data.columns:
        st.write(f"**Segmentos:** {rfm_data['Cluster'].nunique()}")
    st.write(f"**Última actualización:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")