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
    layout="wide"
)

# Aplicar estilo CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .cluster-card {
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🔍 Segmentación de Clientes - Análisis RFM</h1>', unsafe_allow_html=True)
st.markdown("**Proyecto de Minería de Datos – Grupo 4**")
st.markdown("---")

# Cargar modelo y datos
@st.cache_resource
def cargar_modelo():
    try:
        ruta_base = r"E:\Descargas\Proyecto de Minería de Datos – Grupo 4"
        modelo_path = os.path.join(ruta_base, "modelo_kmeans.pkl")
        modelo = joblib.load(modelo_path)
        return modelo
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

@st.cache_data
def cargar_datos_originales():
    try:
        ruta_base = r"E:\Descargas\Proyecto de Minería de Datos – Grupo 4"
        ruta_excel = os.path.join(ruta_base, "Online Retail.xlsx")
        df = pd.read_excel(ruta_excel)
        
        # Limpieza básica (igual que en tu notebook)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df = df.dropna(subset=["CustomerID"])
        df = df[df["Quantity"] > 0]
        
        return df
    except Exception as e:
        st.error(f"Error cargando dataset: {e}")
        return None

@st.cache_data
def cargar_resultados_rfm():
    try:
        ruta_base = r"E:\Descargas\Proyecto de Minería de Datos – Grupo 4"
        csv_path = os.path.join(ruta_base, "resultados_segmentacion.csv")
        rfm = pd.read_csv(csv_path, index_col=0)
        return rfm
    except:
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
        
        return rfm

# Sidebar para navegación
st.sidebar.title("🧭 Navegación")
opcion = st.sidebar.radio(
    "Selecciona una sección:",
    ["📊 Dashboard General", "👥 Análisis de Segmentos", "🔍 Buscar Cliente", "📈 Recomendaciones Estratégicas"]
)

# Cargar recursos
modelo = cargar_modelo()
df_original = cargar_datos_originales()
rfm_data = cargar_resultados_rfm()

if df_original is None or rfm_data is None:
    st.error("No se pudieron cargar los datos. Verifica las rutas de archivo.")
    st.stop()

# --- SECCIÓN 1: DASHBOARD GENERAL ---
if opcion == "📊 Dashboard General":
    st.header("📊 Dashboard General RFM")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_clientes = len(rfm_data)
        st.metric("Total Clientes", f"{total_clientes:,}")
    
    with col2:
        total_ventas = df_original['Quantity'].sum()
        st.metric("Total Productos Vendidos", f"{total_ventas:,}")
    
    with col3:
        ingresos_totales = (df_original['Quantity'] * df_original['UnitPrice']).sum()
        st.metric("Ingresos Totales", f"${ingresos_totales:,.2f}")
    
    with col4:
        transacciones_totales = df_original['InvoiceNo'].nunique()
        st.metric("Total Transacciones", f"{transacciones_totales:,}")
    
    st.markdown("---")
    
    # Distribución de clusters
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📈 Distribución de Segmentos")
        
        if 'Cluster' in rfm_data.columns:
            cluster_counts = rfm_data['Cluster'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            bars = ax.bar([f'Segmento {i}' for i in cluster_counts.index], 
                         cluster_counts.values, 
                         color=colors[:len(cluster_counts)])
            
            ax.set_ylabel('Número de Clientes')
            ax.set_title('Clientes por Segmento RFM')
            plt.xticks(rotation=45)
            
            # Agregar valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}\n({height/total_clientes*100:.1f}%)', 
                       ha='center', va='bottom')
            
            st.pyplot(fig)
    
    with col_right:
        st.subheader("📋 Resumen RFM por Segmento")
        
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
            
            st.dataframe(resumen_display.style.background_gradient(cmap='Blues'))
    
    # Gráficos de distribución RFM
    st.subheader("📊 Distribución de Variables RFM")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Recency
    axes[0].hist(rfm_data['Recency'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title('Distribución de Recencia')
    axes[0].set_xlabel('Días desde última compra')
    axes[0].set_ylabel('Frecuencia')
    
    # Frequency
    axes[1].hist(rfm_data['Frequency'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].set_title('Distribución de Frecuencia')
    axes[1].set_xlabel('Número de transacciones')
    axes[1].set_ylabel('Frecuencia')
    
    # Monetary
    axes[2].hist(rfm_data['Monetary'], bins=30, alpha=0.7, color='salmon', edgecolor='black')
    axes[2].set_title('Distribución de Valor Monetario')
    axes[2].set_xlabel('Valor total ($)')
    axes[2].set_ylabel('Frecuencia')
    
    plt.tight_layout()
    st.pyplot(fig)

# --- SECCIÓN 2: ANÁLISIS DE SEGMENTOS ---
elif opcion == "👥 Análisis de Segmentos":
    st.header("👥 Análisis Detallado por Segmento")
    
    if 'Cluster' not in rfm_data.columns:
        st.warning("No se encontraron datos de segmentación. Ejecuta primero el notebook de análisis.")
        st.stop()
    
    # Selector de segmento
    segmento_seleccionado = st.selectbox(
        "Selecciona un segmento para analizar:",
        sorted(rfm_data['Cluster'].unique())
    )
    
    # Datos del segmento seleccionado
    segmento_data = rfm_data[rfm_data['Cluster'] == segmento_seleccionado]
    
    # Métricas del segmento
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Clientes en Segmento", len(segmento_data))
    
    with col2:
        st.metric("Recencia Promedio", f"{segmento_data['Recency'].mean():.1f} días")
    
    with col3:
        st.metric("Frecuencia Promedio", f"{segmento_data['Frequency'].mean():.1f}")
    
    with col4:
        st.metric("Valor Promedio", f"${segmento_data['Monetary'].mean():.2f}")
    
    st.markdown("---")
    
    # Visualizaciones comparativas
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("Comparación de Recencia")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for cluster in sorted(rfm_data['Cluster'].unique()):
            data_cluster = rfm_data[rfm_data['Cluster'] == cluster]['Recency']
            color = 'red' if cluster == segmento_seleccionado else 'lightgray'
            alpha = 1.0 if cluster == segmento_seleccionado else 0.5
            ax.hist(data_cluster, bins=20, alpha=alpha, label=f'Segmento {cluster}', color=color)
        
        ax.set_xlabel('Recencia (días)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Recencia por Segmento')
        ax.legend()
        st.pyplot(fig)
    
    with col_viz2:
        st.subheader("Comparación de Valor Monetario")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for cluster in sorted(rfm_data['Cluster'].unique()):
            data_cluster = rfm_data[rfm_data['Cluster'] == cluster]['Monetary']
            color = 'green' if cluster == segmento_seleccionado else 'lightgray'
            alpha = 1.0 if cluster == segmento_seleccionado else 0.5
            ax.hist(data_cluster, bins=20, alpha=alpha, label=f'Segmento {cluster}', color=color)
        
        ax.set_xlabel('Valor Monetario ($)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Valor por Segmento')
        ax.legend()
        st.pyplot(fig)
    
    # Top clientes del segmento
    st.subheader(f"🏆 Top 10 Clientes - Segmento {segmento_seleccionado}")
    
    top_clientes = segmento_data.nlargest(10, 'Monetary')[['Recency', 'Frequency', 'Monetary']]
    st.dataframe(top_clientes.style.format({
        'Monetary': '${:,.2f}',
        'Recency': '{:.0f}',
        'Frequency': '{:.0f}'
    }))

# --- SECCIÓN 3: BUSCAR CLIENTE ---
elif opcion == "🔍 Buscar Cliente":
    st.header("🔍 Análisis de Cliente Específico")
    
    # Selector de cliente
    cliente_ids = rfm_data.index.astype(str).tolist()
    cliente_seleccionado = st.selectbox("Selecciona un CustomerID:", cliente_ids)
    
    if cliente_seleccionado:
        cliente_data = rfm_data.loc[float(cliente_seleccionado)]
        
        # Mostrar información del cliente
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("CustomerID", cliente_seleccionado)
        
        with col2:
            st.metric("Recencia", f"{cliente_data['Recency']:.0f} días")
        
        with col3:
            st.metric("Frecuencia", f"{cliente_data['Frequency']:.0f}")
        
        with col4:
            if 'Cluster' in cliente_data:
                st.metric("Segmento", f"Cluster {cliente_data['Cluster']}")
            else:
                st.metric("Segmento", "No asignado")
        
        st.markdown("---")
        
        # Valor monetario
        st.metric("Valor Monetario Total", f"${cliente_data['Monetary']:,.2f}")
        
        # Comparación con el segmento
        if 'Cluster' in cliente_data:
            segmento_cliente = cliente_data['Cluster']
            segmento_promedio = rfm_data[rfm_data['Cluster'] == segmento_cliente].mean()
            
            st.subheader("📊 Comparación con su Segmento")
            
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
            }, index=['Recencia', 'Frecuencia', 'Valor Monetario'])
            
            # Calcular diferencias porcentuales
            comparacion['Diferencia %'] = (
                (comparacion['Este Cliente'] - comparacion['Promedio del Segmento']) / 
                comparacion['Promedio del Segmento'] * 100
            ).round(1)
            
            st.dataframe(comparacion.style.format({
                'Este Cliente': '{:.2f}',
                'Promedio del Segmento': '{:.2f}',
                'Diferencia %': '{:.1f}%'
            }))

# --- SECCIÓN 4: RECOMENDACIONES ESTRATÉGICAS ---
elif opcion == "📈 Recomendaciones Estratégicas":
    st.header("📈 Recomendaciones por Segmento")
    
    recomendaciones = {
        0: {
            "nombre": "🚪 Clientes Inactivos",
            "descripcion": "Alta recencia, baja frecuencia y valor monetario",
            "caracteristicas": ["No compran hace mucho tiempo", "Bajo historial de compras", "Bajo valor total"],
            "estrategias": [
                "💡 Campañas de reactivación con descuentos especiales",
                "💡 Encuestas de satisfacción para entender su partida", 
                "💡 Programas de 'Te extrañamos' con beneficios exclusivos",
                "💡 Email marketing con ofertas personalizadas"
            ],
            "objetivos": ["Recuperar al 20% de clientes", "Incrementar frecuencia de compra"]
        },
        1: {
            "nombre": "⭐ Clientes Leales", 
            "descripcion": "Recencia media, frecuencia constante y valor moderado",
            "caracteristicas": ["Clientes regulares", "Fidelidad comprobada", "Valor constante"],
            "estrategias": [
                "💡 Programas de fidelización con puntos canjeables",
                "💡 Acceso anticipado a nuevos productos y promociones",
                "💡 Comunicación personalizada y reconocimiento especial",
                "💡 Ofertas de cross-selling basadas en historial"
            ],
            "objetivos": ["Mantener tasa de retención >80%", "Incrementar valor promedio en 15%"]
        },
        2: {
            "nombre": "🎯 Clientes Nuevos/Potenciales",
            "descripcion": "Recencia baja, frecuencia en crecimiento, valor variable", 
            "caracteristicas": ["Clientes recientes", "En proceso de fidelización", "Potencial de crecimiento"],
            "estrategias": [
                "💡 Programas de onboarding y bienvenida",
                "💡 Tutoriales y guías de uso de productos",
                "💡 Ofertas de seguimiento para segunda compra", 
                "💡 Contenido educativo sobre beneficios de la marca"
            ],
            "objetivos": ["Convertir 40% en clientes leales", "Duplicar frecuencia en 6 meses"]
        },
        3: {
            "nombre": "👑 Clientes de Alto Valor",
            "descripcion": "Baja recencia, alta frecuencia y valor monetario elevado",
            "caracteristicas": ["Clientes más valiosos", "Frecuentes y recientes", "Alto gasto total"],
            "estrategias": [
                "💡 Atención personalizada y dedicada (account manager)",
                "💡 Ofertas exclusivas y productos premium",
                "💡 Eventos especiales e invitaciones privadas", 
                "💡 Programas VIP con beneficios superiores"
            ],
            "objetivos": ["Retención del 95%", "Incrementar valor en 25% anual"]
        }
    }
    
    # Selector de segmento para recomendaciones
    segmento_recomendaciones = st.selectbox(
        "Selecciona un segmento para ver recomendaciones:",
        options=list(recomendaciones.keys()),
        format_func=lambda x: f"Segmento {x}: {recomendaciones[x]['nombre']}"
    )
    
    if segmento_recomendaciones in recomendaciones:
        info = recomendaciones[segmento_recomendaciones]
        
        # Header del segmento
        st.markdown(f"## {info['nombre']}")
        st.write(f"**Descripción:** {info['descripcion']}")
        
        col_info, col_metrics = st.columns([2, 1])
        
        with col_info:
            st.subheader("🎯 Características Principales")
            for caracteristica in info['caracteristicas']:
                st.write(f"• {caracteristica}")
            
            st.subheader("🚀 Estrategias Recomendadas")
            for estrategia in info['estrategias']:
                st.write(estrategia)
        
        with col_metrics:
            st.subheader("📊 Métricas Objetivo")
            for objetivo in info['objetivos']:
                st.write(f"• {objetivo}")
            
            # Estadísticas rápidas del segmento
            if 'Cluster' in rfm_data.columns:
                segmento_stats = rfm_data[rfm_data['Cluster'] == segmento_recomendaciones]
                st.metric("Clientes en segmento", len(segmento_stats))
                st.metric("Valor promedio", f"${segmento_stats['Monetary'].mean():.2f}")
                st.metric("Recencia promedio", f"{segmento_stats['Recency'].mean():.1f} días")
    
    # Resumen ejecutivo
    st.markdown("---")
    st.subheader("📋 Resumen Ejecutivo de Segmentación")
    
    if 'Cluster' in rfm_data.columns:
        resumen_final = rfm_data.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum'],
            'Cluster': 'count'
        }).round(2)
        
        resumen_final.columns = ['Recencia Prom', 'Frecuencia Prom', 'Valor Prom', 'Valor Total', 'Clientes']
        st.dataframe(resumen_final)

# Footer
st.markdown("---")
st.markdown("**Proyecto de Minería de Datos - Grupo 4** | *Clasificación de clientes según patrones de compra*")
st.markdown("Desarrollado con Streamlit 🚀")

# Información de debug (opcional - puedes comentar estas líneas)
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ Información del Sistema")
    st.write(f"Clientes cargados: {len(rfm_data)}")
    st.write(f"Transacciones: {len(df_original)}")
    if 'Cluster' in rfm_data.columns:
        st.write(f"Segmentos: {rfm_data['Cluster'].nunique()}")