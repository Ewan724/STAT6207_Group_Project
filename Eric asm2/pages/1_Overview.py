import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Overview", layout="wide")
OUT = Path(__file__).resolve().parents[1]
PROC = OUT/"data"/"processed"; FIGS = OUT/"results"/"figures"

st.header("Overview")
st.write("Second Talent AI CV Matching baseline: rank 110 candidate CVs for 10 labelled JDs and analyze hits/misses.")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Pool CVs", 100); c2.metric("Hired GT CVs", 10); c3.metric("GT JDs", 10); c4.metric("Unlabelled JDs", 50)
if (FIGS/"01_pipeline_diagram.png").exists():
    st.image(str(FIGS/"01_pipeline_diagram.png"))
if (PROC/"metrics_summary.csv").exists():
    st.subheader("Headline metrics")
    st.dataframe(pd.read_csv(PROC/"metrics_summary.csv"), use_container_width=True)
    if (FIGS/"12_metric_comparison.png").exists():
        st.image(str(FIGS/"12_metric_comparison.png"), caption="Model comparison across Recall@10/Hit@1/Hit@5/MRR@10")
st.markdown("**Note:** Recall@10 with one relevant item per JD is a hit rate (hits/10). Random baseline ~0.09 for 110 candidates.")
