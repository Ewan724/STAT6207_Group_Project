import streamlit as st
from pathlib import Path
import pandas as pd
import json

st.set_page_config(page_title="Autoencoder", layout="wide")
OUT = Path(__file__).resolve().parents[1]
PROC = OUT/"data"/"processed"; FIGS = OUT/"results"/"figures"; TABLES = OUT/"results"/"tables"; MODELS = OUT/"models"

st.header("Autoencoder")
if (MODELS/"autoencoder_config.json").exists():
    ae = json.load(open(MODELS/"autoencoder_config.json"))
    st.subheader("Training configuration")
    st.json(ae)
else:
    st.warning("Autoencoder config missing. Run stage 3.")

cols = st.columns(2)
with cols[0]:
    if (FIGS/"06_ae_architecture.png").exists():
        st.image(str(FIGS/"06_ae_architecture.png"), caption="Architecture")
    if (FIGS/"07_ae_loss_curves.png").exists():
        st.image(str(FIGS/"07_ae_loss_curves.png"), caption="Train/validation loss")
    if (FIGS/"08_ae_reconstruction_error.png").exists():
        st.image(str(FIGS/"08_ae_reconstruction_error.png"), caption="Reconstruction error by doc type")
with cols[1]:
    if (FIGS/"09_latent_space_pca.png").exists():
        st.image(str(FIGS/"09_latent_space_pca.png"), caption="Latent PCA with GT pair links")
    if (FIGS/"10_latent_space_umap.png").exists():
        st.image(str(FIGS/"10_latent_space_umap.png"), caption="Latent t-SNE/UMAP with GT pair links")

st.subheader("AE vs SVD/PCA")
if (TABLES/"ae_vs_svd.csv").exists():
    st.dataframe(pd.read_csv(TABLES/"ae_vs_svd.csv"), use_container_width=True)
elif (OUT/"results"/"tables"/"ae_vs_svd.csv").exists():
    st.dataframe(pd.read_csv(OUT/"results"/"tables"/"ae_vs_svd.csv"), use_container_width=True)

if (FIGS/"08b_ae_reconstruction_examples.png").exists():
    st.image(str(FIGS/"08b_ae_reconstruction_examples.png"))
st.info("AE optimizes reconstruction, not ranking. Low reconstruction error does not guarantee good retrieval.")
