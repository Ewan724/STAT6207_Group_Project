"""Page 2 — Keyword extraction and merging."""
from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_utils import (CFG, CORAL, GREY, NAVY, TEAL, footer, hero, image, p, read_csv,
                       read_json, read_jsonl, require)

hero("Keyword extraction and merging",
     "What the local LLM extracted, how spelling variants were merged into canonical clusters, "
     "and where the merge rules still break.")

require([p("cv_extracted.jsonl"), p("jd_extracted.jsonl"), p("keyword_clusters.csv"),
         p("keyword_table.csv"), p("keyword_edges.csv"), p("cluster_metrics.json")],
        "py \"group assignment/run_all.py\" --to-stage 02", "1-2")

documents = read_jsonl(str(p("documents.jsonl")))
cv_recs = {r["doc_id"]: r for r in read_jsonl(str(p("cv_extracted.jsonl")))}
jd_recs = {r["doc_id"]: r for r in read_jsonl(str(p("jd_extracted.jsonl")))}
clusters = read_csv(str(p("keyword_clusters.csv")))
kwt = read_csv(str(p("keyword_table.csv")))
edges = read_csv(str(p("keyword_edges.csv")))
k2c = read_json(str(p("keyword_to_cluster.json")))
metrics = read_json(str(p("cluster_metrics.json")))

doc_by_id = {d["doc_id"]: d for d in documents}
cluster_by_id = {int(r["cluster_id"]): r for _, r in clusters.iterrows()}

CHIP_COLOURS = {
    "must-have": ("#E8F5EE", "#046C4E"),
    "nice-to-have": ("#EAF2FB", "#1F4E79"),
    "tools": ("#F3ECFA", "#5B2C83"),
    "soft": ("#FFF4E0", "#8A5A00"),
    "skills": ("#E8F5EE", "#046C4E"),
    "domains": ("#EEF2F7", "#374151"),
}


def chips(items: list[str], kind: str) -> str:
    bg, fg = CHIP_COLOURS.get(kind, ("#F3F4F6", "#374151"))
    if not items:
        return '<span style="color:#9CA3AF">none extracted</span>'
    return " ".join(
        f'<span style="display:inline-block;margin:2px 4px 2px 0;padding:3px 9px;border-radius:999px;'
        f'background:{bg};color:{fg};font-size:0.8rem;font-weight:600">{x}</span>'
        for x in items)


st.markdown("### 1. Document inspector")
c1, c2 = st.columns([1, 2])
with c1:
    kind = st.radio("Document type", ["CV", "JD"], horizontal=True)
    if kind == "CV":
        ids = sorted(cv_recs)
    else:
        ids = sorted(jd_recs)
    doc_id = st.selectbox("Document", ids, format_func=lambda x: f"{x} — {doc_by_id.get(x, {}).get('title') or 'untitled'}")
rec = (cv_recs if kind == "CV" else jd_recs).get(doc_id, {})
doc = doc_by_id.get(doc_id, {})
with c2:
    st.markdown(f"**{doc.get('title') or '—'}** · {doc.get('company') or doc.get('role_family') or '—'} · "
                f"{doc.get('location') or 'location not stated'} · split `{doc.get('split')}`")
    excerpt = (doc.get("raw_text") or "")[:1500].replace("\n", "  \n")
    with st.expander("Show source excerpt (first 1,500 characters)", expanded=False):
        st.markdown(excerpt if excerpt else "_No text available._")

if kind == "CV":
    groups = [
        ("skills", rec.get("skills") or [], "Extracted skills"),
        ("tools", rec.get("tools_technologies") or [], "Tools & technologies"),
        ("domains", rec.get("domains") or [], "Domains"),
        ("soft", rec.get("soft_skills") or [], "Soft skills"),
    ]
else:
    groups = [
        ("must-have", rec.get("must_have_keywords") or [], "Must-have keywords"),
        ("nice-to-have", rec.get("nice_to_have_keywords") or [], "Nice-to-have keywords"),
        ("tools", rec.get("tools_technologies") or [], "Tools & technologies"),
        ("soft", rec.get("soft_skills") or [], "Soft skills"),
    ]
cols = st.columns(2)
for i, (key, items, label) in enumerate(groups):
    with cols[i % 2]:
        st.markdown(f"**{label}**")
        st.markdown(chips(items, key), unsafe_allow_html=True)

st.caption(f"Extraction status: `{rec.get('extraction_status', 'unknown')}` · model `{rec.get('model', '—')}` · "
           f"{len(rec.get('dropped_keywords') or [])} unsupported keyword(s) dropped by validation.")

st.markdown("### 2. Keyword → canonical cluster")
if kwt is not None and len(kwt):
    table = kwt.copy()
    table["cluster_id"] = table["key"].map(k2c)
    table = table[table["cluster_id"].notna()].copy()
    table["cluster_id"] = table["cluster_id"].astype(int)
    table["canonical cluster"] = table["cluster_id"].map(lambda x: cluster_by_id.get(int(x), {}).get("canonical_label", "—"))
    table["cluster size"] = table["cluster_id"].map(lambda x: int(cluster_by_id.get(int(x), {}).get("size", 0)))
    table = table.rename(columns={
        "keyword": "raw term", "doc_frequency": "document frequency",
        "total_frequency": "total frequency", "canonical_seed": "alias seed",
    })
    q = st.text_input("Search terms or clusters", "")
    min_df = st.slider("Minimum document frequency", 1, int(max(1, table["document frequency"].max())), 1)
    view = table[table["document frequency"] >= min_df]
    if q:
        mask = (view["raw term"].astype(str).str.contains(q, case=False, na=False)
                | view["canonical cluster"].astype(str).str.contains(q, case=False, na=False))
        view = view[mask]
    st.dataframe(view[["raw term", "canonical cluster", "cluster size", "document frequency",
                       "total frequency", "doc_types", "alias seed"]]
                 .sort_values(["document frequency", "total frequency"], ascending=False).head(400),
                 use_container_width=True, hide_index=True, height=330)
    st.caption(f"{len(view):,} of {len(table):,} merged terms shown. Document frequency counts training documents; "
               "total frequency includes repeats within a document.")
else:
    st.info("keyword_table.csv is missing.")

st.markdown("### 3. Cluster explorer")
labels = [f"{int(r['cluster_id']):03d} · {r['canonical_label']} ({int(r['size'])} variants)" for _, r in clusters.iterrows()]
pick = st.selectbox("Cluster (sorted by size)", labels)
cid = int(pick.split(" · ")[0])
row = cluster_by_id[cid]
cc1, cc2, cc3 = st.columns(3)
cc1.metric("Variants merged", int(row["size"]))
cc2.metric("Document frequency", int(row["doc_frequency"]))
cc3.metric("Coherence", f"{float(row['coherence']):.3f}")
st.markdown(f"**Definition:** {row['definition']}")
st.markdown("**Members:** " + ", ".join(str(row["members"]).split("|")))
ex = kwt[kwt["key"].map(k2c) == cid] if kwt is not None else pd.DataFrame()
if len(ex):
    st.caption("Example documents: " + ", ".join(sorted({x for s in ex["example_doc_ids"].astype(str) for x in s.split("|") if x})[:10]))

st.markdown("### 4. Cluster network")
if edges is not None and len(edges):

    @st.cache_data(show_spinner="Computing network layout…")
    def build_network(node_payload: str, edge_payload: str):
        cl = pd.DataFrame(json.loads(node_payload))
        ed = pd.DataFrame(json.loads(edge_payload))
        G = nx.Graph()
        for _, r in cl.iterrows():
            G.add_node(int(r["cluster_id"]), label=str(r["canonical_label"]),
                       freq=int(r["doc_frequency"]), size=int(r["size"]))
        for _, r in ed.iterrows():
            G.add_edge(int(r["source"]), int(r["target"]), weight=float(r["weight"]))
        pos = nx.spring_layout(G, seed=42, k=1.8 / np.sqrt(max(1, len(G))), iterations=120)
        try:
            comms = nx.community.louvain_communities(G, seed=42)
            community = {n: i for i, c in enumerate(comms) for n in c}
        except Exception:
            community = {n: 0 for n in G.nodes}
        nodes = pd.DataFrame([{
            "x": pos[n][0], "y": pos[n][1], "label": G.nodes[n]["label"],
            "freq": G.nodes[n]["freq"], "size": G.nodes[n]["size"],
            "community": community.get(n, 0), "id": n,
        } for n in G.nodes])
        seg_x, seg_y = [], []
        for u, v in G.edges:
            seg_x += [pos[u][0], pos[v][0], None]
            seg_y += [pos[u][1], pos[v][1], None]
        return nodes, seg_x, seg_y

    nodes, seg_x, seg_y = build_network(clusters.to_json(), edges.to_json())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode="lines", line=dict(color="#D5DAE1", width=0.7),
                             hoverinfo="skip", showlegend=False))
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8C8C8C"]
    for comm, sub in nodes.groupby("community"):
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers+text",
            text=sub.apply(lambda r: r["label"] if r["freq"] >= nodes["freq"].quantile(0.85) else "", axis=1),
            textposition="top center", textfont=dict(size=8, color=NAVY),
            marker=dict(size=np.clip(np.sqrt(sub["freq"]) * 8, 7, 34),
                        color=palette[int(comm) % len(palette)], line=dict(width=0.6, color="white")),
            name=f"community {int(comm)}",
            customdata=np.stack([sub["label"], sub["freq"], sub["size"], sub["id"]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>cluster %{customdata[3]}<br>"
                          "doc frequency %{customdata[1]}<br>variants %{customdata[2]}<extra></extra>"))
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10),
                      title="Clusters linked when they co-occur in ≥2 training documents",
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.02), hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Node size = document frequency; colour = Louvain community; labels shown for the top 15% of clusters. "
               "Isolated nodes have no co-occurrence edge at the ≥2-document threshold.")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### 5. Largest clusters")
    image("04_keyword_cluster_sizes.png", "Cluster size is the number of merged variants; df is training document frequency.")
with col2:
    st.markdown("### 6. Merge examples")
    image("05_keyword_merge_examples.png", "Alias merging is curated; clustering runs only after normalisation.")

st.markdown("### 7. Merge quality")
guard = metrics.get("guardrail_mixed_clusters", {}) or {}
review = read_csv(str(p("merge_review_sample.csv")))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Canonical clusters", int(metrics.get("n_clusters", len(clusters))))
c2.metric("Singleton rate", f"{float(metrics.get('singleton_rate', 0)):.1%}")
c3.metric("Mean coherence", f"{float(metrics.get('mean_coherence', 0)):.3f}")
if review is not None and "manual_label" in review and review["manual_label"].fillna("").astype(str).str.strip().ne("").any():
    labelled = review[review["manual_label"].fillna("").astype(str).str.strip().ne("")]
    good = labelled["manual_label"].astype(str).str.lower().isin(["same", "merge", "yes", "correct", "1"])
    c4.metric("Manual merge accuracy", f"{good.mean():.0%}", f"{len(labelled)} labelled pairs")
else:
    c4.metric("Manual merge accuracy", "pending", "30-pair review sample")

violations = {k: v for k, v in guard.items() if v}
if violations:
    st.error("Guardrail violations: " + ", ".join(f"{k} = {v}" for k, v in violations.items()))
else:
    st.success("Guardrail check passed: no cluster mixes Java/JavaScript, C/C++, C#/C++, SQL/NoSQL, "
               "MySQL/SQL Server, AWS/Azure/GCP, Power BI/Tableau, WMS/OMS, UI/UX, Angular/AngularJS, "
               "TensorFlow/PyTorch, Docker/Kubernetes or ETL/ELT.")

if review is not None and len(review):
    with st.expander("30-pair manual merge review sample"):
        st.dataframe(review, use_container_width=True, hide_index=True, height=320)
        st.caption("`manual_label` is filled by a human reviewer after inspecting the pair; `cosine` is the "
                   "embedding similarity used by the clustering step.")

st.markdown("### 8. Cluster coverage by role family")
image("05b_keyword_coverage_heatmap.png",
      "Mean TF-IDF presence of the 25 largest clusters in each role family; darker = more characteristic.")

footer()
