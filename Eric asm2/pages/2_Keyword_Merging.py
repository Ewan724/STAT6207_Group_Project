import streamlit as st
from pathlib import Path
import pandas as pd
import json

st.set_page_config(page_title="Keyword Merging", layout="wide")
OUT = Path(__file__).resolve().parents[1]
PROC = OUT/"data"/"processed"; FIGS = OUT/"results"/"figures"

st.header("Keyword Extraction and Merging")
if (PROC/"cluster_metrics.json").exists():
    m = json.load(open(PROC/"cluster_metrics.json"))
    cols = st.columns(4)
    cols[0].metric("Clusters", m.get("cluster_count"))
    cols[1].metric("Unique train keywords", m.get("unique_train_keywords"))
    cols[2].metric("Singleton rate", f"{m.get('singleton_rate',0):.2f}")
    cols[3].metric("Mean coherence", f"{(m.get('mean_coherence') or 0):.3f}")

if (PROC/"keyword_clusters.csv").exists():
    cl = pd.read_csv(PROC/"keyword_clusters.csv")
    st.subheader("Cluster explorer")
    q = st.text_input("Search cluster label/member")
    view = cl
    if q:
        view = cl[cl['canonical_label'].str.contains(q, case=False, na=False) | cl['members'].str.contains(q, case=False, na=False)]
    st.dataframe(view[['cluster_id','canonical_label','size','doc_frequency','coherence','members']].head(200), use_container_width=True)

col1,col2 = st.columns(2)
with col1:
    if (FIGS/"03_keyword_cluster_network.png").exists():
        st.image(str(FIGS/"03_keyword_cluster_network.png"), caption="Cluster co-occurrence network")
    if (FIGS/"04_keyword_cluster_sizes.png").exists():
        st.image(str(FIGS/"04_keyword_cluster_sizes.png"), caption="Top clusters by document frequency")
with col2:
    if (FIGS/"05_keyword_merge_examples.png").exists():
        st.image(str(FIGS/"05_keyword_merge_examples.png"), caption="Merge examples")
    if (FIGS/"05b_keyword_coverage_heatmap.png").exists():
        st.image(str(FIGS/"05b_keyword_coverage_heatmap.png"), caption="Top keyword clusters coverage")

if (PROC/"keyword_aliases.csv").exists():
    st.subheader("Curated aliases")
    st.dataframe(pd.read_csv(PROC/"keyword_aliases.csv"), use_container_width=True)

st.markdown("**Guardrails:** confusable tech pairs (Java/JavaScript, C/C++, SQL/NoSQL etc.) are split; see cluster QA.")
if (PROC/"cluster_qa.csv").exists():
    with st.expander("Cluster QA samples"):
        st.dataframe(pd.read_csv(PROC/"cluster_qa.csv").head(40), use_container_width=True)
