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

# --- 0. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(layout="wide", page_title="Recomendador Semántico")

st.markdown("""
    <style>
    .main-card { 
        background-color: #1E1E2E; padding: 20px; border-radius: 15px; 
        border: 2px solid #00FBFF; color: white !important; margin-bottom: 20px;
    }
    .rec-card { 
        background-color: #2D2D3F; padding: 15px; border-radius: 10px; 
        border-left: 5px solid #FF0055; color: white !important; margin-bottom: 10px;
    }
    .main-card h3, .rec-card b { color: #00FBFF !important; }
    .keyword-badge { 
        background-color: #4E4E6E; color: #00FBFF; padding: 3px 10px; 
        border-radius: 15px; font-size: 0.85em; margin-right: 5px;
        display: inline-block; margin-top: 5px;
    }
    p, span, small, li { color: #E0E0E0 !important; }
    </style>
""", unsafe_allow_html=True)

PATH_RECO = "recomendador"

# --- 1. CARGA DE RECURSOS (PKL + INDEX + EXCEL) ---
@st.cache_resource
def load_resources():
    base = os.path.dirname(__file__)
    pkl_path = os.path.join(base, PATH_RECO, "recomendador.pkl")
    index_path = os.path.join(base, PATH_RECO, "recomendador.index")
    excel_path = os.path.join(base, PATH_RECO, "recomendador.xlsx")

    # Carga de archivos
    df_ia = pd.read_pickle(pkl_path)
    df_excel = pd.read_excel(excel_path)
    index = faiss.read_index(index_path)
    model = SentenceTransformer('intfloat/multilingual-e5-small')

    # Unificamos Lote como string
    if 'Lote' not in df_ia.columns:
        df_ia.rename(columns={df_ia.columns[0]: 'Lote'}, inplace=True)
    df_ia['Lote'] = df_ia['Lote'].astype(str).str.strip()
    df_excel['Lote'] = df_excel['Lote'].astype(str).str.strip()

    # Combinamos para tener toda la info del Excel en el DF de la IA
    df_final = pd.merge(df_ia, df_excel[['Lote', 'Título', 'Autor']], on='Lote', how='left', suffixes=('', '_ex'))
    
    # Limpieza de textos
    for col in ['Keywords_ES', 'Genero_Principal_IA', 'Título', 'Autor']:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna("General").astype(str)

    return df_final, index, model

df_ia, index, model = load_resources()

# --- 2. MAPA VISUAL ---
@st.cache_resource
def get_map_coords(_index):
    vectors = np.array([_index.reconstruct(i) for i in range(_index.ntotal)]).astype('float32')
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    return reducer.fit_transform(vectors), vectors

coords, vectors = get_map_coords(index)
df_ia['x'], df_ia['y'] = coords[:, 0], coords[:, 1]

# --- 3. FILTROS (PASO 4) ---
with st.sidebar:
    st.header("⚙️ Filtros")
    generos = sorted(df_ia['Genero_Principal_IA'].unique())
    sel_generos = st.multiselect("Géneros visibles:", options=generos, default=generos)
    kw_filtro = st.text_input("Buscar por palabra:").lower()

mask = df_ia['Genero_Principal_IA'].isin(sel_generos)
if kw_filtro:
    mask &= df_ia['Keywords_ES'].str.lower().str.contains(kw_filtro)
df_display = df_ia[mask].copy()

# --- 4. BUSCADOR Y LÓGICA (PASO 1, 2 y 3) ---
st.title("📚 Recomendador de Lotes por Afinidad")
lote_usuario = st.text_input("1️⃣ Introduce un Lote (ej: 121N):").strip().upper()

if lote_usuario:
    if lote_usuario in df_ia['Lote'].values:
        idx_semilla = df_ia[df_ia['Lote'] == lote_usuario].index[0]
        libro_base = df_ia.iloc[idx_semilla]
        
        # Búsqueda FAISS
        distancias, indices = index.search(vectors[idx_semilla:idx_semilla+1], 11)
        indices_vecinos = indices[0]
        
        # Coloreado
        df_display['Relación'] = "Otros"
        df_display.loc[df_display.index.isin(indices_vecinos), 'Relación'] = "Parecidos"
        df_display.loc[idx_semilla, 'Relación'] = "SELECCIONADO"

        # ZOOM DINÁMICO
        vecinos_geo = df_display[df_display.index.isin(indices_vecinos)]
        pad = 0.8
        x_range = [vecinos_geo['x'].min() - pad, vecinos_geo['x'].max() + pad]
        y_range = [vecinos_geo['y'].min() - pad, vecinos_geo['y'].max() + pad]

        fig = px.scatter(
            df_display, x='x', y='y', color='Relación',
            hover_name='Título', hover_data=['Autor', 'Lote'],
            color_discrete_map={"SELECCIONADO": "#FF0055", "Parecidos": "#00FBFF", "Otros": "rgba(80,80,80,0.1)"},
            template="plotly_dark", height=500
        )
        fig.update_xaxes(range=x_range, visible=False)
        fig.update_yaxes(range=y_range, visible=False)
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
        st.plotly_chart(fig, use_container_width=True)

        # PANEL DE JUSTIFICACIÓN
        st.markdown("### 2️⃣ Análisis de los libros más parecidos")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown(f"""<div class="main-card">
                <h3>{libro_base['Título']}</h3>
                <p><b>Autor:</b> {libro_base['Autor']}<br>
                <b>Lote:</b> {libro_base['Lote']}</p>
                <div>{' '.join([f'<span class="keyword-badge">{k.strip()}</span>' for k in libro_base['Keywords_ES'].split(',')[:6]])}</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            for i in indices_vecinos[1:6]:
                sim = df_ia.iloc[i]
                k_base = set(libro_base['Keywords_ES'].lower().split(','))
                k_sim = set(sim['Keywords_ES'].lower().split(','))
                comunes = list(k_base.intersection(k_sim))
                
                st.markdown(f"""<div class="rec-card">
                    <b>{sim['Título']}</b> - {sim['Autor']} (Lote: {sim['Lote']})<br>
                    <p style='font-size: 0.9em; margin-bottom: 5px;'>Género: {sim['Genero_Principal_IA']}</p>
                    {"".join([f'<span class="keyword-badge">{c.strip()}</span>' for c in comunes[:3]]) if comunes else "<i>Similitud por contexto narrativo</i>"}
                </div>""", unsafe_allow_html=True)
    else:
        st.error("Ese código de lote no existe.")
else:
    st.info("Introduce un lote para ver sus recomendaciones.")
    fig_def = px.scatter(df_display, x='x', y='y', color='Genero_Principal_IA', template="plotly_dark", height=500)
    fig_def.update_layout(showlegend=False)
    st.plotly_chart(fig_def, use_container_width=True)
