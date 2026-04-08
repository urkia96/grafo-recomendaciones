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

st.set_page_config(layout="wide", page_title="Prueba Visual Clúster")

PATH_RECO = "recomendador"

@st.cache_resource
def load_resources():
    pkl_path = os.path.join(PATH_RECO, "clubes_lectura_small_modelo1_keywords.pkl")
    index_path = os.path.join(PATH_RECO, "clubes_lectura_small_modelo1_keywords.index")
    excel_path = os.path.join(PATH_RECO, "Etiquetas_Normalizadas_Final (1).xlsx")

    # 1. Carga del Excel de metadatos
    df = pd.read_excel(excel_path)
    
    # 2. Tu lógica original de Carga IA
    with open(pkl_path, "rb") as f:
        df_ia_meta = pickle.load(f)
   
    if 'Lote' not in df_ia_meta.columns:
        df_ia_meta.rename(columns={df_ia_meta.columns[0]: 'Lote'}, inplace=True)
    df_ia_meta['Lote'] = df_ia_meta['Lote'].astype(str).str.strip()
 
    index = faiss.read_index(index_path)
    model = SentenceTransformer('intfloat/multilingual-e5-small')
 
    gc.collect()
    return df, df_ia_meta, index, model

# Ejecución de carga
df, df_ia_meta, index, model = load_resources()

# --- CÁLCULO DE MAPA VISUAL (UMAP) ---
@st.cache_resource
def get_coords(_index):
    # Extraemos vectores del índice
    vectors = np.array([_index.reconstruct(i) for i in range(_index.ntotal)]).astype('float32')
    # Reducción a 2D
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    return reducer.fit_transform(vectors), vectors

coords, vectors = get_coords(index)
df_ia_meta['x'] = coords[:, 0]
df_ia_meta['y'] = coords[:, 1]

# --- INTERFAZ STREAMLIT ---
st.title("🗺️ Explorador Visual de Libros")

col_mapa, col_info = st.columns([2, 1])

with col_mapa:
    fig = px.scatter(
        df_ia_meta, x='x', y='y',
        color='Genero_Principal_IA' if 'Genero_Principal_IA' in df_ia_meta.columns else None,
        hover_name='Título',
        template="plotly_dark",
        height=700
    )
    # Usamos plotly_events para capturar el clic
    selected_point = plotly_events(fig, click_event=True)

with col_info:
    if selected_point:
        idx_sel = selected_point[0]['pointIndex']
        libro = df_ia_meta.iloc[idx_sel]
        
        st.subheader(libro['Título'])
        st.write(f"**Autor:** {libro.get('Autor', 'Desconocido')}")
        st.write(f"**Lote:** {libro['Lote']}")
        
        if st.button("🔍 Buscar Similares"):
            distancias, indices = index.search(vectors[idx_sel:idx_sel+1], 6)
            st.write("### Recomendaciones:")
            for i in indices[0][1:]: # Saltamos el primero (él mismo)
                st.write(f"- {df_ia_meta.iloc[i]['Título']}")
    else:
        st.info("Haz clic en un punto del mapa para ver detalles.")
