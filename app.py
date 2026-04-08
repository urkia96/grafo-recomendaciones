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
st.set_page_config(layout="wide", page_title="Mapa Visual de Libros")

# Estilos CSS para mejorar la estética de las tarjetas
st.markdown("""
    <style>
    .book-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #464b5d;
    }
    </style>
""", unsafe_allow_html=True)

PATH_RECO = "recomendador"

# --- 1. CARGA DE RECURSOS ---
@st.cache_resource
def load_resources():
    base = os.path.dirname(__file__)
    # Nuevos nombres de archivos según tu indicación
    pkl_path = os.path.join(base, PATH_RECO, "recomendador.pkl")
    index_path = os.path.join(base, PATH_RECO, "recomendador.index")
    excel_path = os.path.join(base, PATH_RECO, "recomendador.xlsx")

    # Carga de Metadatos (Excel)
    df_meta = pd.read_excel(excel_path)
    
    # Carga de IA (PKL)
    with open(pkl_path, "rb") as f:
        df_ia = pickle.load(f)
   
    # Normalización de columna Lote
    if 'Lote' not in df_ia.columns:
        df_ia.rename(columns={df_ia.columns[0]: 'Lote'}, inplace=True)
    df_ia['Lote'] = df_ia['Lote'].astype(str).str.strip()
 
    # Carga de Índice y Modelo
    index = faiss.read_index(index_path)
    model = SentenceTransformer('intfloat/multilingual-e5-small')
 
    gc.collect()
    return df_meta, df_ia, index, model

df_meta, df_ia, index, model = load_resources()

# --- 2. CÁLCULO DE COORDENADAS (UMAP) ---
@st.cache_resource
def get_visual_map(_index):
    # Extraemos vectores 384D
    vectors = np.array([_index.reconstruct(i) for i in range(_index.ntotal)]).astype('float32')
    # Reducción a 2D para el gráfico
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    coords = reducer.fit_transform(vectors)
    return coords, vectors

coords, vectors = get_visual_map(index)
df_ia['x'] = coords[:, 0]
df_ia['y'] = coords[:, 1]

# --- 3. PANEL LATERAL (FILTROS INTERACTIVOS) ---
with st.sidebar:
    st.title("🎯 Panel de Control")
    st.markdown("Usa estos filtros para explorar el mapa")
    
    with st.form("panel_filtros"):
        # Filtro de Género
        generos = sorted(df_ia['Genero_Principal_IA'].unique()) if 'Genero_Principal_IA' in df_ia.columns else []
        sel_gens = st.multiselect("Filtrar por Géneros", options=generos)
        
        # Filtro por Palabra Clave
        kw_search = st.text_input("Buscar concepto (Keyword)")
        
        # Botón para ejecutar
        aplicar = st.form_submit_button("Actualizar Mapa")

# Lógica de filtrado visual
df_plot = df_ia.copy()
df_plot['Es_Visible'] = "Normal"

if aplicar:
    mask = pd.Series([True] * len(df_plot))
    if sel_gens:
        mask &= df_plot['Genero_Principal_IA'].isin(sel_gens)
    if kw_search:
        mask &= df_plot['Keywords_ES'].str.contains(kw_search, case=False, na=False)
    
    df_plot.loc[mask, 'Es_Visible'] = "Resaltado"
    df_plot.loc[~mask, 'Es_Visible'] = "Fondo"

# --- 4. CUERPO PRINCIPAL (GRAFO + INFO) ---
col_map, col_info = st.columns([2, 1])

with col_map:
    # Usamos scatter_gl para rendimiento máximo
    fig = px.scatter_gl(
        df_plot, x='x', y='y',
        color='Es_Visible' if aplicar else 'Genero_Principal_IA',
        hover_name='Título',
        color_discrete_map={
            "Resaltado": "#00FBFF", 
            "Fondo": "rgba(100,100,100,0.1)", 
            "Normal": None # Usa paleta por defecto
        },
        template="plotly_dark",
        height=750
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color='white')))
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    
    # Capturar el clic
    selected = plotly_events(fig, click_event=True, override_height=750)

with col_info:
    st.markdown("### 📖 Información del Lote")
    
    if selected:
        idx = selected[0]['pointIndex']
        libro = df_plot.iloc[idx]
        
        # Tarjeta visual
        with st.container():
            st.markdown(f"""
                <div class="book-card">
                    <h2 style='color: #00FBFF;'>{libro['Título']}</h2>
                    <p><b>Autor:</b> {libro.get('Autor', 'Desconocido')}</p>
                    <p><b>Lote:</b> {libro['Lote']}</p>
                    <hr>
                    <p><i>{libro.get('Keywords_ES', '')}</i></p>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            if st.button("🔍 Ver libros similares"):
                distancias, indices = index.search(vectors[idx:idx+1], 6)
                st.success("Libros parecidos en la biblioteca:")
                for i in indices[0][1:]:
                    sim = df_ia.iloc[i]
                    st.write(f"• **{sim['Título']}** ({sim['Lote']})")
    else:
        st.info("Haz clic en un punto del mapa para desplegar su ficha técnica.")


