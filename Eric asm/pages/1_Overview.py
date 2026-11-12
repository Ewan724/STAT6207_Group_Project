"""Page 1 — Overview."""
from __future__ import annotations

import streamlit as st

from app_utils import (CFG, footer, hero, image, metric_card, p, read_csv, read_json,
                       require)

hero("AI CV Matching — Baseline Results",
     "A reproducible keyword + autoencoder baseline that ranks 110 candidate CVs for each of the "
     "10 labelled job descriptions, evaluated with Recall@10 and a structured error analysis.")

require([p("rankings.csv"), p("metrics_summary.csv"), p("ground_truth.csv")],
        "py \"group assignment/run_all.py\"", "0-6")

stage0 = read_json(str(p("stage0_summary.json")))
metrics = read_csv(str(p("metrics_summary.csv")))
rb = read_csv(str(p("random_baseline_summary.csv")))
err = read_csv(str(p("error_analysis.csv")))
ext = read_json(str(p("extraction_summary.json")))
ae = read_json(str(CFG.models / "autoencoder_config.json"))
clusters = read_csv(str(p("keyword_clusters.csv")))

st.markdown("### Problem")
st.markdown(
    "Second Talent recruiters match CVs to job descriptions by hand. This baseline turns each JD and "
    "each CV into a set of merged keyword clusters, compresses the document–cluster matrix with a "
    "32-dimensional autoencoder, and returns the 10 closest candidates with a template-based reason. "
    "The single labelled relevance signal is the CV of the person actually hired for each of the 10 "
    "example JDs. That is a **sparse-label baseline**, not a production ranker.")

st.markdown("### Dataset at a glance")
cols = st.columns(5)
metric_card(cols[0], "Pool CVs", str(stage0.get("pool_cvs", 100)), "training candidates")
metric_card(cols[1], "Hired CVs", str(stage0.get("gt_cvs", 10)), "labelled relevance")
metric_card(cols[2], "Labelled JDs", str(stage0.get("gt_jds", 10)), "evaluation queries")
metric_card(cols[3], "Unlabelled JDs", str(stage0.get("unlabelled_jds", 50)), "training context only")
metric_card(cols[4], "Unique candidates", str(stage0.get("unique_candidates", 110)), "100 pool + 10 hired")

st.markdown("### Headline result (primary model M3: AE-32 cosine)")
if metrics is not None and len(metrics):
    m3 = metrics[metrics["model"] == "M3"].iloc[0]
    cols = st.columns(5)
    metric_card(cols[0], "Recall@10", f"{m3['recall_at_10']:.2f}", f"{int(m3['hits'])}/10 hired CVs in top 10")
    metric_card(cols[1], "Hit@1", f"{m3['hit_at_1']:.2f}", "exact top-1 hits")
    metric_card(cols[2], "Hit@5", f"{m3['hit_at_5']:.2f}", "hired CV in top 5")
    metric_card(cols[3], "MRR@10", f"{m3['mrr_at_10']:.3f}", "mean reciprocal rank")
    random_row = rb.iloc[0] if rb is not None and len(rb) else None
    note = f"random ≈ {float(random_row['expected_recall_at_10']):.3f}" if random_row is not None else "random ≈ 0.091"
    metric_card(cols[4], "Random baseline", "0.091", note)
    st.caption(
        f"Wilson 95% CI for Recall@10: {m3['wilson_ci_low']:.2f}–{m3['wilson_ci_high']:.2f}; "
        f"bootstrap 95% CI {m3['bootstrap_ci_low']:.2f}–{m3['bootstrap_ci_high']:.2f}. "
        f"Precision@10 is {m3['precision_at_10']:.3f} (maximum 0.1 because each query has one relevant CV).")
else:
    st.info("Metrics are not available yet.")

st.markdown("### Pipeline")
image("01_pipeline_diagram.png",
      "Figure 1. Seven checkpointed stages from the read-only DATA/ folder to this website.")

st.markdown("### Key findings")
if err is not None and len(err) and metrics is not None:
    m3 = metrics[metrics["model"] == "M3"].iloc[0]
    e3 = err[err["model"] == "M3"]
    misses = e3[e3["hired_rank"] > 10]
    cats = misses["primary_failure_category"].value_counts()
    cat_text = ", ".join(f"{k} ({v})" for k, v in cats.items()) if len(cats) else "none"
    comp = read_csv(str(CFG.tables / "ae_vs_tfidf.csv"))
    ae_wins = int((comp["winner"] == "AE").sum()) if comp is not None else 0
    st.markdown(
        f"- M3 retrieved the hired CV in the top 10 for **{int(m3['hits'])}/10** queries; "
        f"the remaining **{10 - int(m3['hits'])}** are analysed one by one.\n"
        f"- Failure codes for the misses: {cat_text or 'no misses'}.\n"
        f"- The autoencoder beat TF-IDF on **{ae_wins}/10** rank comparisons, so AE is a mixed result, "
        "not a uniform win.\n"
        f"- Extraction succeeded on **{ext.get('success_rate', 0):.1%}** of 170 documents with "
        f"`{ext.get('model', 'local Ollama model')}`; the AE uses **{clusters.shape[0] if clusters is not None else '—'}** "
        "keyword clusters and no test labels.\n"
        "- Salary, notice period and location/language hard filters are deliberately **not** modelled; "
        "this is the largest source of recruiter disagreement in the error analysis.")

st.markdown("### How to use this app")
st.markdown(
    "1. **Keyword extraction & merging** — inspect what the local LLM extracted and how variants were merged.\n"
    "2. **Autoencoder** — see the architecture, loss curves, reconstruction error and latent space.\n"
    "3. **Search results** — recruiter view: top 10 per labelled JD, with the hired CV highlighted.\n"
    "4. **Recall & error analysis** — metrics, failure taxonomy, case studies and recommendations.")

footer()
