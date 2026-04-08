import streamlit as st
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os
import gc
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events
import umap

# --- 0. CONFIGURACIÓN ---
st.set_page_config(layout="wide", page_title="Mapa Visual de Libros", initial_sidebar_state="expanded")

# Estilos CSS para mejorar la estética de las tarjetas y el fondo
st.markdown("""
    <style>
    .book-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #464b5d;
        margin-bottom: 20px;
    }
    .stPlotlyChart {
        background-color: #0E1117;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

PATH_RECO = "recomendador"

# --- 1. CARGA DE RECURSOS ---
@st.cache_resource
def load_resources():
    base = os.path.dirname(__file__)
    # Rutas unificadas según tu indicación
    pkl_path = os.path.join(base, PATH_RECO, "recomendador.pkl")
    index_path = os.path.join(base, PATH_RECO, "recomendador.index")
    excel_path = os.path.join(base, PATH_RECO, "recomendador.xlsx")

    # Carga de Metadatos (Excel)
    try:
        df_meta = pd.read_excel(excel_path)
    except FileNotFoundError:
        st.error(f"No se encuentra el archivo Excel en: {excel_path}")
        st.stop()
    
    # Carga de IA (PKL)
    try:
        with open(pkl_path, "rb") as f:
            df_ia = pickle.load(f)
    except EOFError:
        st.error("El archivo .pkl está corrupto o incompleto. Por favor, súbelo de nuevo.")
        st.stop()
   
    # Normalización de columna Lote
    if 'Lote' not in df_ia.columns:
        # Asumimos que la primera columna es el Lote si no tiene nombre
        df_ia.rename(columns={df_ia.columns[0]: 'Lote'}, inplace=True)
    df_ia['Lote'] = df_ia['Lote'].astype(str).str.strip()
 
    # Carga de Índice y Modelo
    index = faiss.read_index(index_path)
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    
    # Limpieza de Keywords para evitar errores en el filtro
    if 'Keywords_ES' in df_ia.columns:
        df_ia['Keywords_ES'] = df_ia['Keywords_ES'].fillna("Desconocido").astype(str)
 
    gc.collect()
    return df_meta, df_ia, index, model

df_meta, df_ia, index, model = load_resources()

# --- 2. CÁLCULO DE COORDENADAS (UMAP) ---
@st.cache_resource
def get_visual_map(_index):
    # Extraemos vectores 384D
    vectors = np.array([_index.reconstruct(i) for i in range(_index.ntotal)]).astype('float32')
    # Reducción a 2D para el gráfico (Coseno para embeddings de texto)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    coords = reducer.fit_transform(vectors)
    return coords, vectors

coords, vectors = get_visual_map(index)
# Aseguramos que las coordenadas estén sincronizadas
if len(coords) == len(df_ia):
    df_ia['x'] = coords[:, 0]
    df_ia['y'] = coords[:, 1]
else:
    st.error("Discrepancia de tamaño entre el índice FAISS y el archivo PKL.")
    st.stop()

# --- 3. PANEL LATERAL (FILTROS INTERACTIVOS) ---
with st.sidebar:
    st.title("🎯 Panel de Control")
    st.markdown("Usa estos filtros para explorar el mapa semántico de la biblioteca.")
    
    with st.form("panel_filtros"):
        # Filtro de Género
        if 'Genero_Principal_IA' in df_ia.columns:
            generos = sorted(df_ia['Genero_Principal_IA'].unique())
            sel_gens = st.multiselect("Filtrar por Géneros", options=generos, default=[])
        else:
            sel_gens = []
        
        # Filtro por Palabra Clave
        kw_search = st.text_input("Buscar concepto (Keyword)", placeholder="ej: Historia, Magia, Crimen")
        
        # Botón para ejecutar
        aplicar = st.form_submit_button("Actualizar Mapa")

# Lógica de filtrado visual para el grafo
df_plot = df_ia.copy()
df_plot['Es_Visible'] = "Normal" # Estado por defecto

# Si el usuario aplica filtros, definimos la visibilidad
if aplicar and (sel_gens or kw_search):
    mask = pd.Series([True] * len(df_plot))
    if sel_gens:
        mask &= df_plot['Genero_Principal_IA'].isin(sel_gens)
    if kw_search:
        # El filtrado es insensible a mayúsculas
        mask &= df_plot['Keywords_ES'].str.contains(kw_search, case=False)
    
    df_plot.loc[mask, 'Es_Visible'] = "Resaltado"
    df_plot.loc[~mask, 'Es_Visible'] = "Fondo"
else:
    # Si no hay filtros activos, coloreamos por género (comportamiento original)
    df_plot['Es_Visible'] = df_plot['Genero_Principal_IA'] if 'Genero_Principal_IA' in df_plot.columns else "Libro"

# --- 4. CUERPO PRINCIPAL (GRAFO + INFO) ---
col_map, col_info = st.columns([2, 1])

with col_map:
    # --- RENDERIZADO OPTIMIZADO DEL GRAFO (px.scatter normal) ---
    st.subheader("Mapa Semántico de la Biblioteca")
    
    # Definimos paleta de colores para el estado normal o los géneros
    color_col = 'Es_Visible'
    # Paleta de colores discreta y vibrante para destacar sobre fondo oscuro
    color_scale = px.colors.qualitative.Alphabet 
    
    # Mapa de colores para cuando el panel de filtros está activo
    discrete_map = {
        "Resaltado": "#00FBFF", # Cian Neón para destacar
        "Fondo": "rgba(80, 80, 80, 0.15)", # Gris muy transparente
        "Normal": None # Usa paleta automática
    }

    fig = px.scatter(
        df_plot, x='x', y='y',
        color=color_col,
        hover_name='Título',
        # Definimos el mapa de colores si estamos filtrando
        color_discrete_map=discrete_map if aplicar and (sel_gens or kw_search) else None,
        # Si no filtramos, usamos la paleta automática
        color_discrete_sequence=color_scale if not (aplicar and (sel_gens or kw_search)) else None,
        template="plotly_dark",
        height=750
    )
    
    # OPTIMIZACIÓN DE PUNTOS: Sin línea de contorno y con buena opacidad
    fig.update_traces(
        marker=dict(
            size=7, 
            opacity=0.8, 
            line=dict(width=0) # ELIMINAR CONTORNO para mejor rendimiento y visualización
        )
    )
    
    # Limpieza de diseño
    fig.update_layout(
        showlegend=False, 
        margin=dict(l=0, r=0, t=0, b=0),
        clickmode='event+select',
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False)
    )
    
    # Capturar el clic con plotly_events
    selected = plotly_events(fig, click_event=True, override_height=750)

with col_info:
    st.markdown("### 📖 Información del Lote")
    
    # Lógica al seleccionar un libro en el mapa
    if selected:
        idx = selected[0]['pointIndex']
        # Mapeamos el índice de Plotly al índice real del DataFrame
        libro = df_plot.iloc[idx]
        
        # Tarjeta visual con estilo CSS personalizado
        st.markdown(f"""
            <div class="book-card">
                <h2 style='color: #00FBFF; margin-top: 0;'>{libro['Título']}</h2>
                <p><b>Autor:</b> {libro.get('Autor', 'Desconocido')}</p>
                <p><b>Lote:</b> {libro['Lote']}</p>
                <hr style="border: 0.5px solid #464b5d;">
                <p><i>{libro.get('Keywords_ES', 'No hay palabras clave disponibles')}</i></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        # Acción: Buscar similares usando el índice FAISS
        if st.button("🔍 Ver libros similares", use_container_width=True):
            with st.spinner("Calculando similitud vectorial..."):
                # Obtenemos el vector de búsqueda y buscamos los 6 más cercanos
                distancias, indices = index.search(vectors[idx:idx+1], 6)
                
                st.write("---")
                st.success("Recomendaciones basadas en afinidad semántica:")
                # Listamos los 5 similares (saltamos el primero que es él mismo)
                for i in indices[0][1:]:
                    sim = df_ia.iloc[i]
                    st.write(f"• **{sim['Título']}** ({sim['Lote']})")
    else:
        # Mensaje inicial cuando no hay selección
        st.info("Haz clic en un punto del mapa para desplegar su ficha técnica y ver recomendaciones.")


