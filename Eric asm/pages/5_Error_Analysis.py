"""Page 5 — Error analysis, case studies and recommendations."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_utils import (CORAL, GROUP_DIR, TEAL, footer, hero, image, metric_card, p,
                       read_csv, read_json, read_text, require, t)

CATEGORY_LABELS = {
    "none": "No failure (hit)",
    "E1_extraction_missing_skill": "E1 · Extraction missed a skill",
    "E2_keyword_merge_error": "E2 · Keyword merge / coverage loss",
    "E3_representation_loss": "E3 · AE representation loss",
    "E4_hard_negative_lookalike": "E4 · Hard-negative lookalike",
    "E5_seniority_years_mismatch": "E5 · Seniority / years mismatch",
    "E6_domain_industry_mismatch": "E6 · Domain / industry mismatch",
    "E7_location_language_constraint": "E7 · Location / language constraint",
    "E8_label_noise_or_recruiter_context": "E8 · Label noise / recruiter context",
    "E9_duplicate_or_pool_artifact": "E9 · Duplicate / pool artefact",
    "E10_other": "E10 · Other",
}

hero("Error analysis",
     "Every miss is reviewed and tagged with a primary failure category. Case studies show which "
     "must-haves a lookalike shares with the job description, and the recommendations are tied to "
     "evidence visible in the saved tables.")

require([p("error_analysis.csv"), t("ae_vs_tfidf.csv"), t("recommendations.csv"),
         t("case_study_summary.csv"), p("ground_truth.csv")],
        "py \"group assignment/run_all.py\" --to-stage 06", "6")

err = read_csv(str(p("error_analysis.csv")))
comp = read_csv(str(t("ae_vs_tfidf.csv")))
recs = read_csv(str(t("recommendations.csv")))
cases = read_csv(str(t("case_study_summary.csv")))
gt = read_csv(str(p("ground_truth.csv")))
summary = read_json(str(p("error_analysis_summary.json")))
exq = read_csv(str(t("extraction_quality_summary.csv")))
exq_rows = read_csv(str(p("extraction_quality.csv")))

m3 = err[err["model"] == "M3"].copy().sort_values("hired_rank")
m1 = err[err["model"] == "M1"].copy()
n_miss = int((m3["hired_rank"] > 10).sum())

st.markdown("### Review summary")
cols = st.columns(5)
metric_card(cols[0], "Queries reviewed", f"{summary.get('queries_reviewed', len(m3))}", "all 10 labelled JDs")
metric_card(cols[1], "M3 misses", f"{n_miss}/10", "hired CV ranked > 10")
metric_card(cols[2], "AE vs TF-IDF", f"{summary.get('ae_wins', 0)} / {summary.get('tfidf_wins', 0)}",
            "queries where each wins")
metric_card(cols[3], "Case studies", f"{len(cases)}", "with must-have heatmaps")
metric_card(cols[4], "Recommendations", f"{len(recs)}", "prioritised P0–P2")
st.caption("Each query gets exactly one primary category, chosen by an explicit ordered rule set in "
           "`src/analysis.py` (hard constraints → role-family lookalike → years → AE-vs-TF-IDF degradation → "
           "coverage → label noise). Secondary signals are only set when no primary rule fires first.")

st.markdown("### Failure taxonomy")
counts = (m3["primary_failure_category"].value_counts()
          .rename_axis("category").reset_index(name="count"))
counts["label"] = counts["category"].map(CATEGORY_LABELS).fillna(counts["category"])
m1_counts = (m1["primary_failure_category"].value_counts()
             .rename_axis("category").reset_index(name="count_m1"))
merged = counts.merge(m1_counts, on="category", how="outer").fillna(0)
merged["label"] = merged["category"].map(CATEGORY_LABELS).fillna(merged["category"])
merged = merged.sort_values("count", ascending=False)

left, right = st.columns([1, 1.2])
with left:
    st.dataframe(merged[["label", "count", "count_m1"]].rename(
        columns={"label": "primary category", "count": "M3 (primary)", "count_m1": "M1 (TF-IDF)"}),
        hide_index=True, use_container_width=True)
    st.caption("Counts are out of 10 queries. A 'none' row counts hits, which is why the column sums to 10.")
with right:
    plot_df = merged.melt(id_vars="label", value_vars=["count", "count_m1"],
                          var_name="model", value_name="queries")
    plot_df["model"] = plot_df["model"].map({"count": "M3 · AE-32", "count_m1": "M1 · TF-IDF"})
    fig = px.bar(plot_df, x="queries", y="label", color="model", barmode="group", orientation="h",
                 color_discrete_sequence=[TEAL, "#0072B2"])
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="",
                      xaxis_title="queries", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Per-query review")
st.markdown("Select a query to read the full review. The table below is the saved `error_analysis.csv` "
            "restricted to the primary model.")
show = m3[["jd_id", "jd_title", "role_family", "hired_rank", "hired_cv_id", "top1_cv_id",
           "score_margin", "must_have_coverage_hired", "must_have_coverage_top1",
           "primary_failure_category"]].copy()
show["primary_failure_category"] = show["primary_failure_category"].map(CATEGORY_LABELS).fillna(
    show["primary_failure_category"])
st.dataframe(show, hide_index=True, use_container_width=True,
             column_config={
                 "score_margin": st.column_config.NumberColumn("score − top1", format="%.4f"),
                 "must_have_coverage_hired": st.column_config.NumberColumn("coverage hired", format="%.2f"),
                 "must_have_coverage_top1": st.column_config.NumberColumn("coverage top1", format="%.2f"),
             })

jd_id = st.selectbox("Open a query", m3["jd_id"].tolist(),
                     format_func=lambda x: f"{x} · {m3[m3['jd_id'] == x]['jd_title'].iloc[0]}")
r = m3[m3["jd_id"] == jd_id].iloc[0]
status = "hit" if int(r["hired_rank"]) <= 10 else "miss"
c1, c2, c3, c4 = st.columns(4)
metric_card(c1, "Hired rank", f"{int(r['hired_rank'])}/110", r["hired_cv_id"])
metric_card(c2, "Score margin", f"{r['score_margin']:.4f}", "hired − top-1")
metric_card(c3, "Coverage", f"{r['must_have_coverage_hired']:.2f}",
            f"top-1: {r['must_have_coverage_top1']:.2f}")
metric_card(c4, "Primary category", r["primary_failure_category"].split("_")[0],
            CATEGORY_LABELS.get(r["primary_failure_category"], ""))
st.markdown(f"**Review.** {r['comment']}")
detail = pd.DataFrame([
    ("Top-1 candidate", f"{r['top1_cv_id']} · {r['top1_title']}"),
    ("Hired candidate", f"{r['hired_cv_id']} · {m3[m3['jd_id'] == jd_id]['jd_title'].iloc[0]}"),
    ("Role-family match (hired / top-1)", f"{int(r['role_family_match_hired'])} / {int(r['role_family_match_top1'])}"),
    ("Seniority match (hired / top-1)", f"{int(r['seniority_match_hired'])} / {int(r['seniority_match_top1'])}"),
    ("Years (hired / top-1 / gap)", f"{r['years_hired']} / {r['years_top1']} / {r['years_gap']}"),
    ("Missing must-haves (hired)", str(r["missing_must_haves_hired"]).replace("|", ", ") or "—"),
    ("Missing must-haves (top-1)", str(r["missing_must_haves_top1"]).replace("|", ", ") or "—"),
    ("Secondary signal", CATEGORY_LABELS.get(str(r["secondary_failure_category"]), "—")
     if str(r["secondary_failure_category"]) not in ("", "nan") else "—"),
], columns=["field", "value"])
st.dataframe(detail, hide_index=True, use_container_width=True)
st.caption("The AE-vs-TF-IDF comparison uses M1 as the linear reference. Only M3 and M1 are reviewed "
           "because M3 is the pre-registered primary model and M1 is the strongest interpretable baseline.")

st.markdown("### AE vs TF-IDF head-to-head")
if comp is not None and len(comp):
    comp = comp.sort_values("hired_rank_ae")
    fig = go.Figure()
    for _, row in comp.iterrows():
        better = TEAL if row["delta"] > 0 else (CORAL if row["delta"] < 0 else "#6B7280")
        fig.add_trace(go.Scatter(x=[row["hired_rank_tfidf"], row["hired_rank_ae"]],
                                 y=[row["jd_id"], row["jd_id"]], mode="lines",
                                 line=dict(color=better, width=2.4), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=comp["hired_rank_tfidf"], y=comp["jd_id"], mode="markers",
                             name="M1 · TF-IDF", marker=dict(size=10, color="#0072B2")))
    fig.add_trace(go.Scatter(x=comp["hired_rank_ae"], y=comp["jd_id"], mode="markers",
                             name="M3 · AE-32", marker=dict(size=10, color=TEAL)))
    fig.add_vline(x=10.5, line_dash="dot", line_color=CORAL, annotation_text="top-10 cut-off",
                  annotation_position="top")
    fig.update_layout(height=420, xaxis_title="rank of the hired CV", yaxis_title="",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08),
                      xaxis_range=[0.5, max(41, float(comp[["hired_rank_ae", "hired_rank_tfidf"]].max().max()) + 2)])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp[["jd_id", "hired_rank_tfidf", "hired_rank_ae", "delta", "winner", "explanation"]],
                 hide_index=True, use_container_width=True)
    st.caption("Teal lines = AE improves the rank; coral = AE degrades it. The AE wins some queries and loses "
               "others, which is consistent with a representation that smooths rare must-haves.")

st.markdown("### Case studies")
st.markdown("Four queries are documented in detail (plus the query with the largest AE/TF-IDF disagreement). "
            "The heatmaps show how much of each JD's must-have list each top-10 CV actually contains.")
st.dataframe(cases.rename(columns={
    "jd_id": "query", "jd_title": "job title", "hired_rank": "hired rank",
    "primary_failure_category": "primary category", "coverage_hired": "coverage (hired)",
    "coverage_top1": "coverage (top-1)"}), hide_index=True, use_container_width=True)
for _, c in cases.iterrows():
    with st.expander(f"{c['jd_id']} · {c['jd_title']} — hired rank {int(c['hired_rank'])}"):
        st.markdown(f"**{CATEGORY_LABELS.get(c['primary_failure_category'], c['primary_failure_category'])}** — {c['comment']}")
        image(f"18_case_study_{c['jd_id']}.png",
              f"{c['jd_id']}: must-have cluster presence across the M3 top 10. The teal box marks the hired CV. "
              "Empty columns identify must-haves that even the hired CV does not show.")
        table = read_csv(str(t(f"case_study_{c['jd_id']}.csv")))
        if table is not None and len(table):
            st.dataframe(table, hide_index=True, use_container_width=True)
with st.expander("Combined case-study view"):
    image("15_case_study_keyword_coverage.png", "All case studies in one figure.")

st.markdown("### Recommendations")
st.markdown("These are proposals grounded in the observed errors, ordered by priority. Expected impact is a "
            "hypothesis; it is not a measured improvement and is never presented as one.")
prio = st.multiselect("Filter by priority", ["P0", "P1", "P2"], default=["P0", "P1", "P2"])
rec_show = recs[recs["priority"].isin(prio)].reset_index(drop=True)
rec_show.index = rec_show.index + 1
st.dataframe(rec_show, use_container_width=True,
             column_config={"issue": st.column_config.TextColumn(width="medium"),
                            "evidence": st.column_config.TextColumn(width="large"),
                            "proposed_fix": st.column_config.TextColumn(width="large"),
                            "expected_impact": st.column_config.TextColumn(width="medium")})
st.download_button("Download recommendations (CSV)", recs.to_csv(index=False),
                   file_name="recommendations.csv", mime="text/csv")
c1, c2 = st.columns([1.1, 1])
with c1:
    image("19_recommendation_matrix.png", "Effort vs priority. Numbers match the table above.")
with c2:
    image("17_error_taxonomy.png", "Primary failure categories for M3.")

st.markdown("### Extraction quality (silver standard)")
if exq is not None and len(exq):
    e = exq.iloc[0]
    cols = st.columns(4)
    metric_card(cols[0], "CVs scored", f"{int(e.get('n_documents', len(exq_rows) if exq_rows is not None else 0))}",
                "vs Excel Skills column")
    metric_card(cols[1], "Macro precision", f"{float(e.get('macro_precision', float('nan'))):.2f}", "silver standard")
    metric_card(cols[2], "Macro recall", f"{float(e.get('macro_recall', float('nan'))):.2f}", "silver standard")
    metric_card(cols[3], "Macro F1", f"{float(e.get('macro_f1', float('nan'))):.2f}", "silver standard")
    st.caption("The Excel Skills column is used only for this diagnostic. It is never a model feature, "
               "and the LLM never sees it.")
    if exq_rows is not None and len(exq_rows):
        with st.expander("Per-CV extraction quality"):
            st.dataframe(exq_rows.sort_values("f1").head(30), hide_index=True, use_container_width=True)
    image("20_extraction_quality.png", "Extraction F1 distribution and the worst eight CVs.")

st.markdown("### Limitations")
st.markdown(
    "- **Ten queries.** Each query is worth 0.10 Recall@10; confidence intervals are wide and overlap. "
    "The study cannot declare a winning model.\n"
    "- **One label per query.** There is no graded relevance, so nDCG@10 degenerates to 1/log2(rank+1).\n"
    "- **Recruiter context is missing.** Salary, availability, interview performance and internal "
    "references are not observable; some misses may not be model errors at all.\n"
    "- **The taxonomy is rule-ordered.** A query can plausibly fit two categories; the ordered rules "
    "pick the first, and the secondary column records the next-best signal.\n"
    "- **Extraction uses a small local model.** The silver-standard F1 above bounds how much of each miss "
    "can fairly be blamed on retrieval.\n"
    "- **No tuning on the test set.** Thresholds, prompts, architecture and the primary model were fixed "
    "before the 10 labelled queries were scored.")

with st.expander("Read the full error-analysis report"):
    st.markdown(read_text(str(GROUP_DIR / "results" / "error_analysis.md")) or "Report not found.")

footer()
