"""Page 4 — Search results and ranking evaluation."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_utils import (CORAL, GREY, TEAL, badge, footer, hero, image, metric_card, p,
                       read_csv, require, t)

MODEL_LABELS = {
    "M0": "M0 · Random",
    "M1": "M1 · TF-IDF keyword cosine",
    "M2": "M2 · SVD-32 cosine",
    "M3": "M3 · AE-32 cosine (primary)",
    "M4": "M4 · AE + metadata",
    "M5": "M5 · Hybrid AE + coverage (exploratory)",
    "M6": "M6 · Role-family heuristic",
}

hero("Search results",
     "Ten labelled job descriptions, each ranked against 110 candidates. One candidate was actually "
     "hired per query; everything below measures how well each model retrieves that single CV.")

require([p("metrics_summary.csv"), p("metrics_per_query.csv"), p("rankings.csv"),
         p("top10.csv"), p("ground_truth.csv"), p("random_baseline_summary.csv"),
         p("must_have_coverage.csv")],
        "py \"group assignment/run_all.py\" --to-stage 04", "4")

summary = read_csv(str(p("metrics_summary.csv")))
pq = read_csv(str(p("metrics_per_query.csv")))
rankings = read_csv(str(p("rankings.csv")))
top10 = read_csv(str(p("top10.csv")))
gt = read_csv(str(p("ground_truth.csv")))
rb = read_csv(str(p("random_baseline_summary.csv")))
coverage = read_csv(str(p("must_have_coverage.csv")))

summary = summary.set_index("model").loc[[x for x in MODEL_LABELS if x in summary["model"].values]].reset_index()
primary = summary[summary["model"] == "M3"].iloc[0]
m0 = summary[summary["model"] == "M0"].iloc[0]
best = summary.sort_values(["recall_at_10", "mrr_at_10"], ascending=False).iloc[0]

st.markdown("### Headline metrics")
cols = st.columns(5)
metric_card(cols[0], "Recall@10", f"{primary['recall_at_10']:.2f}",
            f"{int(primary['hits'])}/{int(primary['n_queries'])} hired CVs retrieved")
metric_card(cols[1], "Hit@1", f"{primary['hit_at_1']:.2f}",
            f"random ≈ {1/110:.3f}")
metric_card(cols[2], "Hit@5", f"{primary['hit_at_5']:.2f}", "hired CV in top 5")
metric_card(cols[3], "MRR@10", f"{primary['mrr_at_10']:.3f}", "mean 1/rank of hits")
metric_card(cols[4], "nDCG@10", f"{primary['ndcg_at_10']:.3f}", "1/log2(rank+1)")

st.markdown(
    f"The pre-registered primary model **M3 (AE-32 cosine)** retrieves "
    f"**{int(primary['hits'])}/10** hired CVs in the top 10 "
    f"(95% Wilson CI {primary['wilson_ci_low']:.2f}–{primary['wilson_ci_high']:.2f}). "
    f"The seeded random draw M0 scores **{m0['recall_at_10']:.2f}**; the theoretical expectation "
    f"for one relevant item in a pool of 110 is **{10/110:.3f}**. "
    f"The best observed model on this sample is **{best['model']}** "
    f"(Recall@10 {best['recall_at_10']:.2f}), but with ten queries a one-query difference is 0.10, "
    "so that ordering is **not** statistically meaningful.")

st.markdown("### Model comparison")
left, right = st.columns([1.15, 1])
with left:
    show = summary[["model", "hits", "recall_at_10", "hit_at_1", "hit_at_5", "mrr_at_10",
                    "ndcg_at_10", "precision_at_10", "wilson_ci_low", "wilson_ci_high"]].copy()
    show["model"] = show["model"].map(MODEL_LABELS).fillna(show["model"])
    st.dataframe(show, hide_index=True, use_container_width=True,
                 column_config={
                     "recall_at_10": st.column_config.NumberColumn("Recall@10", format="%.2f"),
                     "hit_at_1": st.column_config.NumberColumn("Hit@1", format="%.2f"),
                     "hit_at_5": st.column_config.NumberColumn("Hit@5", format="%.2f"),
                     "mrr_at_10": st.column_config.NumberColumn("MRR@10", format="%.3f"),
                     "ndcg_at_10": st.column_config.NumberColumn("nDCG@10", format="%.3f"),
                     "precision_at_10": st.column_config.NumberColumn("P@10", format="%.3f"),
                     "wilson_ci_low": st.column_config.NumberColumn("CI low", format="%.2f"),
                     "wilson_ci_high": st.column_config.NumberColumn("CI high", format="%.2f"),
                 })
    st.caption("Precision@10 is capped at 0.1 because each query has exactly one relevant CV. "
               "Wilson intervals are for Recall@10; the bootstrap intervals are in the saved CSV.")
with right:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=summary["model"], y=summary["recall_at_10"],
                         marker_color=[TEAL if x == "M3" else "#0072B2" for x in summary["model"]],
                         error_y=dict(type="data",
                                      array=summary["wilson_ci_high"] - summary["recall_at_10"],
                                      arrayminus=summary["recall_at_10"] - summary["wilson_ci_low"],
                                      color=GREY),
                         text=[f"{v:.2f}" for v in summary["recall_at_10"]], textposition="outside"))
    fig.add_hline(y=10 / 110, line_dash="dot", line_color=CORAL,
                  annotation_text="random expectation 0.091", annotation_position="top left")
    fig.update_layout(height=380, yaxis_title="Recall@10", xaxis_title="",
                      yaxis_range=[0, 1.05], margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bars show Recall@10 with 95% Wilson intervals. The primary model is highlighted. "
               "Overlapping intervals mean the ranking of models is not resolved by ten queries.")

st.markdown("### Where each model finds the hired CV")
heat = pq.pivot_table(index="jd_id", columns="model", values="rank", aggfunc="first")
heat = heat[[x for x in MODEL_LABELS if x in heat.columns]]
fig = px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="RdYlGn_r",
                zmin=1, zmax=40, labels=dict(x="model", y="query", color="rank"))
fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("Cell values are the rank of the hired CV (1 = best). Ranks above 10 are misses. "
           "Green is good, red is a miss; 40 is the display cap.")

st.markdown("### Query explorer")
c1, c2 = st.columns([1, 2])
with c1:
    jd_id = st.selectbox("Job description", gt["jd_id"].tolist(),
                         format_func=lambda x: f"{x} · {gt[gt['jd_id'] == x]['jd_title'].iloc[0]}")
    model = st.selectbox("Model", list(MODEL_LABELS), index=list(MODEL_LABELS).index("M3"),
                         format_func=lambda x: MODEL_LABELS[x])
jd = gt[gt["jd_id"] == jd_id].iloc[0]
with c2:
    row = pq[(pq["jd_id"] == jd_id) & (pq["model"] == model)].iloc[0]
    cov = coverage[(coverage["jd_id"] == jd_id) & (coverage["cv_id"] == row["hired_cv_id"])].iloc[0]
    kind = "hit" if row["rank"] <= 10 else "miss"
    st.markdown(
        f"**{jd['jd_title']}** · {jd['role_family']} · {jd['jd_location'] if pd.notna(jd['jd_location']) else 'location not stated'}<br>"
        f"Hired CV **{row['hired_cv_id']}** ranked **{int(row['rank'])}/110** "
        f"{badge('hit', 'hit') if kind == 'hit' else badge('miss', 'miss')} · "
        f"score {row['score']:.4f} · top-1 {row['top1_cv_id']} ({row['top1_score']:.4f})<br>"
        f"Hired must-have coverage **{cov['coverage']:.2f}** · "
        f"matched: {cov['matched_must_haves'] or '—'} · missing: {cov['missing_must_haves'] or '—'}",
        unsafe_allow_html=True)

tbl = top10[(top10["model"] == model) & (top10["jd_id"] == jd_id)].sort_values("rank").copy()
tbl["relevant"] = tbl["is_relevant"].map({1: "✔ hired", 0: ""})
tbl = tbl[["rank", "cv_id", "cv_title", "cv_company", "role_family", "score",
           "relevant", "must_have_matches", "missing_must_haves", "reason"]]
st.dataframe(tbl, hide_index=True, use_container_width=True,
             column_config={"score": st.column_config.NumberColumn(format="%.4f"),
                            "reason": st.column_config.TextColumn(width="large")})
st.caption("Reasons are deterministic templates built from rank, score, role family, years, country, "
           "language, remote flag and must-have coverage. They are not LLM-written rationales.")

with st.expander("Full 110-candidate ranking for this query"):
    full = rankings[(rankings["model"] == model) & (rankings["jd_id"] == jd_id)].sort_values("rank")
    st.dataframe(full, hide_index=True, use_container_width=True)
    st.download_button("Download this ranking (CSV)", full.to_csv(index=False),
                       file_name=f"ranking_{model}_{jd_id}.csv", mime="text/csv")

st.markdown("### Random-baseline context")
if rb is not None and len(rb):
    r = rb.iloc[0]
    cols = st.columns(4)
    metric_card(cols[0], "Candidates per query", f"{int(r.get('n_candidates', 110))}", "labelled pool")
    metric_card(cols[1], "Expected Recall@10", f"{r.get('expected_recall_at_10', 10/110):.4f}", "analytic 10/110")
    metric_card(cols[2], "Mean Recall@10", f"{r.get('empirical_mean', float('nan')):.4f}", "1,000 seeded permutations")
    metric_card(cols[3], "95% interval", f"{r.get('p2_5', float('nan')):.3f}–{r.get('p97_5', float('nan')):.3f}",
                "2.5th–97.5th percentile")
    st.caption("Each permutation ranks the 110 candidates at random for all ten queries. "
               "M3 must beat this band to be useful; the page never claims significance beyond it.")

st.markdown("### Static reference figures")
c1, c2 = st.columns(2)
with c1:
    image("11_recall_by_jd.png", "Per-query Recall@10 for the primary model.")
with c2:
    image("13_rank_distribution.png", "Distribution of the hired CV's rank, by model.")
c3, c4 = st.columns(2)
with c3:
    image("12_metric_comparison.png", "All headline metrics across M0–M6.")
with c4:
    image("14_score_margin.png", "Hired-CV score minus top-1 score, per query (M3).")

st.markdown("### How to read this page")
st.markdown(
    "- **Recall@10 = Hit@10** here: there is exactly one relevant CV per query, so the two definitions coincide.\n"
    "- **Precision@10** can never exceed **0.1** with one relevant item; it is reported only because the brief asks for it.\n"
    "- **Ten queries is a tiny evaluation.** A single query moves Recall@10 by 0.10. Treat differences between "
    "models as descriptive, not as evidence of a winner.\n"
    "- The 10 labelled queries were **not** used for training, early stopping, threshold choice or prompt tuning.\n"
    "- M0 is one seeded representative draw; the 1,000-permutation band above is the fair random reference.")

footer()
