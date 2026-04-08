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
st.set_page_config(layout="wide", page_title="Recomendador Semántico")

# CSS Mejorado para lectura clara
st.markdown("""
    <style>
    .main-card { 
        background-color: #1E1E2E; 
        padding: 20px; 
        border-radius: 15px; 
        border: 2px solid #00FBFF; 
        color: white !important;
        margin-bottom: 20px;
    }
    .rec-card { 
        background-color: #2D2D3F; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #FF0055; 
        color: white !important;
        margin-bottom: 10px;
    }
    .main-card h3, .rec-card b { color: #00FBFF !important; }
    .keyword-badge { 
        background-color: #4E4E6E; 
        color: #00FBFF; 
        padding: 3px 10px; 
        border-radius: 15px; 
        font-size: 0.85em; 
        margin-right: 5px;
        display: inline-block;
        margin-top: 5px;
    }
    p, span, small { color: #E0E0E0 !important; }
    </style>
""", unsafe_allow_html=True)

PATH_RECO = "recomendador"

@st.cache_resource
def load_resources():
    base = os.path.dirname(__file__)
    pkl_path = os.path.join(base, PATH_RECO, "recomendador.pkl")
    index_path = os.path.join(base, PATH_RECO, "recomendador.index")
    
    # Carga con validación
    with open(pkl_path, "rb") as f:
        df_ia = pickle.load(f)
    
    if 'Lote' not in df_ia.columns:
        df_ia.rename(columns={df_ia.columns[0]: 'Lote'}, inplace=True)
    df_ia['Lote'] = df_ia['Lote'].astype(str).str.strip()
    
    for col in ['Keywords_ES', 'Genero_Principal_IA']:
        if col in df_ia.columns:
            df_ia[col] = df_ia[col].fillna("General").astype(str)

    index = faiss.read_index(index_path)
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    return df_ia, index, model

df_ia, index, model = load_resources()

@st.cache_resource
def get_map_coords(_index):
    vectors = np.array([_index.reconstruct(i) for i in range(_index.ntotal)]).astype('float32')
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    return reducer.fit_transform(vectors), vectors

coords, vectors = get_map_coords(index)
df_ia['x'], df_ia['y'] = coords[:, 0], coords[:, 1]

# --- SIDEBAR (PASO 4) ---
with st.sidebar:
    st.header("⚙️ Filtros Globales")
    generos = sorted(df_ia['Genero_Principal_IA'].unique())
    sel_generos = st.multiselect("Mostrar géneros:", options=generos, default=generos)
    kw_filtro = st.text_input("Filtrar por palabra clave:").lower()

mask_filtros = df_ia['Genero_Principal_IA'].isin(sel_generos)
if kw_filtro:
    mask_filtros &= df_ia['Keywords_ES'].str.lower().str.contains(kw_filtro)

df_display = df_ia[mask_filtros].copy()

# --- BUSCADOR ---
st.title("🗺️ Navegador de Similitud de Lotes")
lote_usuario = st.text_input("1️⃣ Introduce un Lote (ej: 121N):").strip().upper()

if lote_usuario:
    if lote_usuario in df_ia['Lote'].values:
        idx_semilla = df_ia[df_ia['Lote'] == lote_usuario].index[0]
        libro_base = df_ia.iloc[idx_semilla]
        
        # Búsqueda FAISS (Estos son los parecidos REALES matemáticamente)
        distancias, indices = index.search(vectors[idx_semilla:idx_semilla+1], 11)
        indices_vecinos = indices[0]
        
        # Clasificación para el gráfico
        df_display['Relación'] = "Otros libros"
        df_display.loc[df_display.index.isin(indices_vecinos), 'Relación'] = "Recomendados"
        df_display.loc[idx_semilla, 'Relación'] = "SELECCIONADO"

        # --- GRAFO CON ZOOM DINÁMICO ---
        # Calculamos límites para el zoom (basado en los recomendados)
        vecinos_data = df_display[df_display.index.isin(indices_vecinos)]
        padding = 0.5
        x_range = [vecinos_data['x'].min() - padding, vecinos_data['x'].max() + padding]
        y_range = [vecinos_data['y'].min() - padding, vecinos_data['y'].max() + padding]

        fig = px.scatter(
            df_display, x='x', y='y', color='Relación',
            hover_name='Título',
            color_discrete_map={
                "SELECCIONADO": "#FF0055",
                "Recomendados": "#00FBFF",
                "Otros libros": "rgba(80,80,80,0.1)"
            },
            template="plotly_dark", height=500
        )
        
        # Aplicamos el zoom automático al área de interés
        fig.update_xaxes(range=x_range, visible=False)
        fig.update_yaxes(range=y_range, visible=False)
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
        
        st.plotly_chart(fig, use_container_width=True)

        # --- PANEL DE EXPLICACIÓN (PASO 3) ---
        st.markdown("### 2️⃣ ¿Por qué estos libros son similares?")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""<div class="main-card">
                <h3>{libro_base['Título']}</h3>
                <p><b>Lote:</b> {libro_base['Lote']}<br>
                <b>Género:</b> {libro_base['Genero_Principal_IA']}</p>
                <p><b>Conceptos clave:</b><br>
                {' '.join([f'<span class="keyword-badge">{k.strip()}</span>' for k in libro_base['Keywords_ES'].split(',')[:5]])}</p>
            </div>""", unsafe_allow_html=True)

        with col2:
            for i in indices_vecinos[1:6]:
                sim = df_ia.iloc[i]
                # Lógica de coincidencia de palabras
                k1 = set(libro_base['Keywords_ES'].lower().split(','))
                k2 = set(sim['Keywords_ES'].lower().split(','))
                comunes = list(k1.intersection(k2))
                
                st.markdown(f"""<div class="rec-card">
                    <b>{sim['Título']}</b> (Lote: {sim['Lote']})<br>
                    <p>Género: {sim['Genero_Principal_IA']}</p>
                    {"".join([f'<span class="keyword-badge">{c}</span>' for c in comunes[:3]]) if comunes else "<i>Afinidad por contexto temático</i>"}
                </div>""", unsafe_allow_html=True)
    else:
        st.error("Lote no encontrado.")
else:
    st.info("Escribe un lote para ver su galaxia de recomendaciones.")
    fig_gen = px.scatter(df_display, x='x', y='y', color='Genero_Principal_IA', template="plotly_dark", height=500)
    fig_gen.update_layout(showlegend=False)
    st.plotly_chart(fig_gen, use_container_width=True)

