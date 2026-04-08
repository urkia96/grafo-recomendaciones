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
st.set_page_config(layout="wide", page_title="Recomendador Visual de Libros")

# Estilos para las fichas de recomendación
st.markdown("""
    <style>
    .main-card { background-color: #1E1E2E; padding: 20px; border-radius: 15px; border: 2px solid #00FBFF; margin-bottom: 20px; }
    .rec-card { background-color: #262730; padding: 15px; border-radius: 10px; border-left: 5px solid #FF0055; margin-bottom: 10px; }
    .keyword-badge { background-color: #3d4150; color: #00FBFF; padding: 2px 8px; border-radius: 5px; font-size: 0.8em; margin-right: 5px; }
    </style>
""", unsafe_allow_html=True)

PATH_RECO = "recomendador"

@st.cache_resource
def load_resources():
    base = os.path.dirname(__file__)
    pkl_path = os.path.join(base, PATH_RECO, "recomendador.pkl")
    index_path = os.path.join(base, PATH_RECO, "recomendador.index")
    excel_path = os.path.join(base, PATH_RECO, "recomendador.xlsx")

    df_ia = pd.read_pickle(pkl_path)
    if 'Lote' not in df_ia.columns:
        df_ia.rename(columns={df_ia.columns[0]: 'Lote'}, inplace=True)
    df_ia['Lote'] = df_ia['Lote'].astype(str).str.strip()
    
    # Limpieza de textos
    for col in ['Keywords_ES', 'Genero_Principal_IA']:
        if col in df_ia.columns:
            df_ia[col] = df_ia[col].fillna("No disponible").astype(str)

    index = faiss.read_index(index_path)
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    
    gc.collect()
    return df_ia, index, model

df_ia, index, model = load_resources()

# --- 1. PROCESAMIENTO VISUAL (UMAP) ---
@st.cache_resource
def get_map_coords(_index):
    vectors = np.array([_index.reconstruct(i) for i in range(_index.ntotal)]).astype('float32')
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    return reducer.fit_transform(vectors), vectors

coords, vectors = get_map_coords(index)
df_ia['x'], df_ia['y'] = coords[:, 0], coords[:, 1]

# --- 2. PANEL LATERAL (PASO 4: FILTROS) ---
with st.sidebar:
    st.header("⚙️ Herramientas de Control")
    st.write("Selecciona qué libros quieres ver en el mapa:")
    
    generos_disponibles = sorted(df_ia['Genero_Principal_IA'].unique())
    sel_generos = st.multiselect("Filtrar por Géneros:", options=generos_disponibles, default=generos_disponibles)
    
    kw_filtro = st.text_input("Filtrar por palabra clave (ej. 'magia'):").lower()

# Aplicar filtros de visibilidad
mask_filtros = df_ia['Genero_Principal_IA'].isin(sel_generos)
if kw_filtro:
    mask_filtros &= df_ia['Keywords_ES'].str.lower().str.contains(kw_filtro)

df_display = df_ia[mask_filtros].copy()

# --- 3. INPUT DE USUARIO (PASO 1: PEDIR LOTE) ---
st.title("🤖 Recomendador de Afinidad Semántica")
lote_usuario = st.text_input("1️⃣ Introduce el número de Lote para analizar:", placeholder="Ejemplo: 121N").strip().upper()

# --- 4. LÓGICA DE RECOMENDACIÓN Y GRAFO (PASO 2 y 3) ---
if lote_usuario:
    if lote_usuario in df_ia['Lote'].values:
        # Encontrar el libro semilla
        idx_semilla = df_ia[df_ia['Lote'] == lote_usuario].index[0]
        libro_base = df_ia.iloc[idx_semilla]
        
        # Buscar 10 similares en el índice FAISS
        distancias, indices = index.search(vectors[idx_semilla:idx_semilla+1], 11)
        indices_vecinos = indices[0]
        
        # Marcar estados para el grafo
        df_display['Relación'] = "Otros libros"
        df_display.loc[df_display.index.isin(indices_vecinos), 'Relación'] = "Similares"
        df_display.loc[idx_semilla, 'Relación'] = "LIBRO SELECCIONADO"

        # --- GRAFO INTERACTIVO ---
        fig = px.scatter(
            df_display, x='x', y='y',
            color='Relación',
            hover_name='Título',
            color_discrete_map={
                "LIBRO SELECCIONADO": "#FF0055",
                "Similares": "#00FBFF",
                "Otros libros": "rgba(100,100,100,0.15)"
            },
            template="plotly_dark", height=600
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=1, color='white')))
        st.plotly_chart(fig, use_container_width=True)

        # --- PANEL DE CONTROL Y JUSTIFICACIÓN (PASO 3) ---
        st.markdown(f"### 2️⃣ Análisis de Similitud para el Lote {lote_usuario}")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""<div class="main-card">
                <h3>{libro_base['Título']}</h3>
                <p><b>Género:</b> {libro_base['Genero_Principal_IA']}</p>
                <p><b>Keywords:</b><br>{libro_base['Keywords_ES']}</p>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.write("#### ¿Por qué estos libros son parecidos?")
            for i in indices_vecinos[1:6]: # Mostrar top 5 similares
                sim = df_ia.iloc[i]
                # Encontrar keywords comunes (intersección simple de texto)
                kw_comunes = [w for w in libro_base['Keywords_ES'].split(',') if w.strip() in sim['Keywords_ES']]
                
                st.markdown(f"""<div class="rec-card">
                    <b>{sim['Título']}</b> (Lote: {sim['Lote']})<br>
                    <small>Coincidencia en: {' '.join([f'<span class="keyword-badge">{k}</span>' for k in kw_comunes[:3]])}</small>
                </div>""", unsafe_allow_html=True)
    else:
        st.error(f"El lote {lote_usuario} no se encuentra en la base de datos.")
else:
    st.info("Introduce un código de lote arriba para empezar la exploración.")
    # Grafo por defecto (solo géneros)
    fig_def = px.scatter(df_display, x='x', y='y', color='Genero_Principal_IA', template="plotly_dark", height=600)
    st.plotly_chart(fig_def, use_container_width=True)


