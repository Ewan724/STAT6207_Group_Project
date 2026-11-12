import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="AI CV Matching — Baseline", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PROC = OUT / "data" / "processed"
FIGS = OUT / "results" / "figures"

st.title("AI CV Matching — Baseline Results")
st.caption("Local LLM keyword extraction → keyword clustering → autoencoder → top-10 retrieval on 10 labelled JD–CV pairs.")

# sidebar
with st.sidebar:
    st.header("Project")
    st.write("110 candidates (100 pool + 10 hired), 10 GT JDs, 50 unlabelled JDs.")
    if (PROC/"extraction_summary.json").exists():
        import json
        es = json.load(open(PROC/"extraction_summary.json"))
        st.metric("Extraction success", f"{es.get('json_success_rate',0)*100:.1f}%")
        st.write("Model:", es.get('model'))
    if (OUT/"models"/"autoencoder_config.json").exists():
        import json
        ae = json.load(open(OUT/"models"/"autoencoder_config.json"))
        st.write("AE latent:", ae.get('latent_dim'), "best epoch:", ae.get('best_epoch'))
    with st.expander("Metric definitions"):
        st.markdown("""- **Recall@10 / Hit@10**: hired CV in top 10 (1 hit per JD). Mean over 10 JDs = hits/10.
- **Hit@1/5**: hired CV in top 1/5.
- **MRR@10**: mean reciprocal rank for hits in top 10.
- **Precision@10**: relevant in top 10 / 10 (max 0.1 with 1 relevant). Not recall.""")
    st.info("Privacy: anonymised data, local Ollama LLM only, no external calls.")
    st.caption("Seed 42. Run pipeline scripts to refresh artifacts.")

# headline metrics
if (PROC/"metrics_summary.csv").exists():
    ms = pd.read_csv(PROC/"metrics_summary.csv")
    m3 = ms[ms.model=='M3_ae32'].iloc[0] if (ms.model=='M3_ae32').any() else None
    m0 = ms[ms.model=='M0_random'].iloc[0] if (ms.model=='M0_random').any() else None
    cols = st.columns(5)
    if m3 is not None:
        cols[0].metric("Recall@10 (AE-32)", f"{m3['recall_at_10']:.2f}", f"{int(m3['hits'])}/10")
        cols[1].metric("Hit@1", f"{m3['hit_at_1']:.2f}")
        cols[2].metric("Hit@5", f"{m3['hit_at_5']:.2f}")
        cols[3].metric("MRR@10", f"{m3['mrr_at_10']:.3f}")
    if m0 is not None:
        cols[4].metric("Random Recall@10", f"{m0['recall_at_10']:.2f}")

st.markdown("### Pipeline")
if (FIGS/"01_pipeline_diagram.png").exists():
    st.image(str(FIGS/"01_pipeline_diagram.png"), caption="Baseline pipeline stages")
if (FIGS/"02_dataset_overview.png").exists():
    st.image(str(FIGS/"02_dataset_overview.png"), caption="Dataset overview by type and role family")

st.markdown("### Key findings")
st.markdown("""
- Evaluation uses 10 sparse-labelled queries (1 hired CV each); one query changes Recall@10 by 0.10.
- Compare AE-32 (M3) against TF-IDF (M1), SVD, and heuristics; report all models, not only the best.
- Common failure modes: hard lookalikes, extraction/merge gaps, missing domain/metadata filters, reconstruction not ranking-aware.
- See **Keyword Merging**, **Autoencoder**, **Search Results**, and **Error Analysis** pages for details.
""")
st.caption("Data anonymised. Local processing only. Baseline for diagnosis, not production use.")
