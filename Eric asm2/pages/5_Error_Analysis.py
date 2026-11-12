import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Error Analysis", layout="wide")
OUT = Path(__file__).resolve().parents[1]
PROC = OUT/"data"/"processed"; FIGS = OUT/"results"/"figures"; RES = OUT/"results"; TABLES = OUT/"results"/"tables"

st.header("Recall and Error Analysis")
if (PROC/"metrics_summary.csv").exists():
    ms = pd.read_csv(PROC/"metrics_summary.csv")
    m3 = ms[ms.model=='M3_ae32'].iloc[0] if (ms.model=='M3_ae32').any() else None
    cols = st.columns(6)
    if m3 is not None:
        cols[0].metric("Recall@10", f"{m3['recall_at_10']:.2f}", f"{int(m3['hits'])}/10")
        cols[1].metric("Hit@1", f"{m3['hit_at_1']:.2f}")
        cols[2].metric("Hit@5", f"{m3['hit_at_5']:.2f}")
        cols[3].metric("MRR@10", f"{m3['mrr_at_10']:.3f}")
        cols[4].metric("Precision@10", f"{m3['precision_at_10']:.3f}", "max 0.1")
        cols[5].metric("95% CI", f"[{m3['wilson_ci_low']:.2f},{m3['wilson_ci_high']:.2f}]")
    with st.expander("Metric definitions"):
        st.markdown("Recall@10 = hits/10 (one relevant per JD). Precision@10 = relevant/10 (max 0.1). Wilson and bootstrap CIs reported; n=10 so intervals are wide.")

for f,cap in [
    ('11_recall_by_jd.png','Hired CV rank by JD (AE-32)'),
    ('12_metric_comparison.png','Model comparison'),
    ('13_rank_distribution.png','Rank distribution'),
    ('14_score_margin.png','Score margins'),
    ('17_error_taxonomy.png','Failure taxonomy'),
    ('16_ae_vs_tfidf_ranks.png','AE vs TF-IDF ranks'),
]:
    p = FIGS/f
    if p.exists(): st.image(str(p), caption=cap)

st.subheader("Case studies")
case = st.selectbox("Case study figure", [f'18_case_study_{x}.png' for x in ['P01','P04','P08']] + ['15_case_study_keyword_coverage.png'])
if (FIGS/case).exists():
    st.image(str(FIGS/case))

if (TABLES/"ae_vs_tfidf.csv").exists():
    st.subheader("AE vs TF-IDF per query")
    st.dataframe(pd.read_csv(TABLES/"ae_vs_tfidf.csv"), use_container_width=True)

if (PROC/"error_analysis.csv").exists():
    st.subheader("Per-query analysis table")
    st.dataframe(pd.read_csv(PROC/"error_analysis.csv"), use_container_width=True)

if (TABLES/"recommendations.csv").exists():
    st.subheader("Recommendations")
    st.dataframe(pd.read_csv(TABLES/"recommendations.csv"), use_container_width=True)
    if (FIGS/"19_recommendation_matrix.png").exists():
        st.image(str(FIGS/"19_recommendation_matrix.png"))

if (RES/"error_analysis.md").exists():
    with st.expander("Error analysis narrative (markdown)"):
        st.markdown((RES/"error_analysis.md").read_text(encoding='utf-8'))

st.subheader("Limitations")
st.markdown("""
- n=10 labelled queries, one relevant CV per JD; sparse/noisy labels, no significance claims.
- Hired candidate may not be best on paper; recruiter context missing.
- No salary/notice/location/citizenship/language hard filters in baseline.
- Extraction/merge noise, truncated PDFs, small AE training corpus, reconstruction ≠ ranking.
- Random baseline ~0.09 for 110 candidates.
""")
if (PROC/"metrics_summary.csv").exists():
    st.download_button("Download metrics_summary.csv", data=(PROC/"metrics_summary.csv").read_bytes(), file_name="metrics_summary.csv")
if (PROC/"error_analysis.csv").exists():
    st.download_button("Download error_analysis.csv", data=(PROC/"error_analysis.csv").read_bytes(), file_name="error_analysis.csv")
if (RES/"error_analysis.md").exists():
    st.download_button("Download error_analysis.md", data=(RES/"error_analysis.md").read_bytes(), file_name="error_analysis.md")
