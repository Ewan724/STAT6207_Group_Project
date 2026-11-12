"""Page 3 — Autoencoder diagnostics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import silhouette_score

from app_utils import (CFG, CORAL, GREY, NAVY, TEAL, footer, hero, image, m, p, read_csv,
                       read_json, require, t)

hero("Autoencoder",
     "A 32-dimensional bottleneck trained only to reconstruct the document–keyword matrix. "
     "Reconstruction quality is a sanity check, not evidence of ranking quality.")

require([m("autoencoder_config.json"), m("ae_training_history.csv"), p("document_embeddings.csv"),
         t("latent_coordinates.csv"), t("ae_vs_svd.csv"), p("ground_truth.csv")],
        "py \"group assignment/run_all.py\" --to-stage 03", "3")

ae = read_json(str(m("autoencoder_config.json")))
history = read_csv(str(m("ae_training_history.csv")))
emb = read_csv(str(p("document_embeddings.csv")))
coords = read_csv(str(t("latent_coordinates.csv")))
comp = read_csv(str(t("ae_vs_svd.csv")))
gt = read_csv(str(p("ground_truth.csv")))
recon = read_csv(str(t("reconstruction_error.csv")))
diag = read_json(str(p("ae_diagnostics.json")))

st.markdown("### Architecture and training setup")
image("06_ae_architecture.png",
      "Input = document–keyword TF-IDF matrix; encoder 256 → 64 → 32; decoder mirrors it. "
      "Only the frozen 32-d bottleneck is used for retrieval.")
c1, c2 = st.columns([1, 1])
with c1:
    st.dataframe(pd.DataFrame([
        ("Input dimension", ae.get("input_dim")),
        ("Hidden dimensions", " → ".join(map(str, ae.get("hidden_dims", [])))),
        ("Latent dimension", ae.get("latent_dim")),
        ("Dropout", ae.get("dropout")),
        ("Denoising", ae.get("denoising")),
        ("Training documents", ae.get("n_train")),
        ("Validation documents", ae.get("n_validation")),
    ], columns=["setting", "value"]), hide_index=True, use_container_width=True)
with c2:
    st.dataframe(pd.DataFrame([
        ("Best epoch", ae.get("best_epoch")),
        ("Epochs run", ae.get("epochs_run")),
        ("Training MSE", f"{float(ae.get('train_mse', 0)):.6f}"),
        ("Validation MSE", f"{float(ae.get('val_mse', 0)):.6f}"),
        ("Parameters", f"{int(ae.get('n_params', 0)):,}"),
        ("Clusters (input features)", ae.get("n_clusters")),
        ("Seed", ae.get("seed")),
    ], columns=["metric", "value"]), hide_index=True, use_container_width=True)
st.caption("Training protocol: pool CVs + unlabelled JDs only. The 10 labelled JDs and 10 hired CVs "
           "are held out of training and validation. Model selection never uses the 10 test queries.")

col1, col2 = st.columns([1.2, 1])
with col1:
    st.markdown("### Training and validation loss")
    if history is not None and len(history):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["epoch"], y=history["train_mse"], name="train MSE",
                                 line=dict(color="#0072B2", width=2)))
        fig.add_trace(go.Scatter(x=history["epoch"], y=history["val_mse"], name="validation MSE",
                                 line=dict(color=CORAL, width=2)))
        fig.add_vline(x=int(ae.get("best_epoch", 1)), line_dash="dash", line_color=TEAL,
                      annotation_text="best epoch", annotation_position="top")
        fig.update_layout(height=380, yaxis_title="MSE (log scale)", xaxis_title="epoch",
                          yaxis_type="log", margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Early stopping restores the best validation-loss weights. A low reconstruction loss "
                   "does not imply that the bottleneck preserves rare must-have skills.")
with col2:
    st.markdown("### Reconstruction error")
    if recon is not None and len(recon):
        fig = px.histogram(recon, x="reconstruction_mse", color="doc_type", nbins=30,
                           color_discrete_map={"pool_cv": "#0072B2", "gt_cv": CORAL,
                                               "gt_jd": TEAL, "unlabelled_jd": "#56B4E9"})
        fig.update_layout(height=380, xaxis_title="per-document reconstruction MSE",
                          yaxis_title="documents", margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Dead latent units: {diag.get('dead_units', '—')}; NaNs: {diag.get('nan_in_embeddings', '—')}; "
                   f"train–validation gap: {diag.get('train_val_gap', '—')}.")

st.markdown("### Latent space")
tab_pca, tab_tsne = st.tabs(["PCA (2D)", "t-SNE (2D)"])
for tab, method in [(tab_pca, "pca"), (tab_tsne, "tsne")]:
    with tab:
        if coords is None or not len(coords):
            st.info("Latent coordinates are missing.")
            continue
        colour_by = st.radio("Colour by", ["Document type", "Role family"], horizontal=True, key=f"colour_{method}")
        colour_col = "doc_type" if colour_by == "Document type" else "role_family"
        plot_df = coords.copy()
        plot_df[colour_col] = plot_df[colour_col].fillna("unknown")
        fig = px.scatter(plot_df, x=f"{method}_x", y=f"{method}_y", color=colour_col,
                         hover_data=["doc_id", "title", "role_family", "split"],
                         color_discrete_sequence=px.colors.qualitative.D3, opacity=0.85)
        fig.update_traces(marker=dict(size=7, line=dict(width=0.4, color="white")))
        if st.checkbox("Show the 10 labelled JD → hired-CV links", value=True, key=f"links_{method}"):
            row_of = coords.set_index("doc_id")
            for _, r in gt.iterrows():
                if r["jd_id"] in row_of.index and r["hired_cv_id"] in row_of.index:
                    a = row_of.loc[r["jd_id"]]
                    b = row_of.loc[r["hired_cv_id"]]
                    fig.add_trace(go.Scatter(x=[a[f"{method}_x"], b[f"{method}_x"]],
                                             y=[a[f"{method}_y"], b[f"{method}_y"]],
                                             mode="lines", line=dict(color="#D55E00", width=1.2),
                                             showlegend=False, hoverinfo="skip"))
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red lines link each labelled JD to its hired CV. A useful representation should make those "
                   "pairs closer than unrelated JD–CV pairs.")

st.markdown("### AE vs linear baselines")
if comp is not None and len(comp):
    st.dataframe(comp, use_container_width=True, hide_index=True)
    melted = comp.melt(id_vars="method", value_vars=["reconstruction_mse"],
                       var_name="metric", value_name="value").dropna()
    fig = px.bar(melted, x="method", y="value", color="method", text_auto=".5f",
                 color_discrete_sequence=["#0072B2", "#009E73", "#D55E00"])
    fig.update_layout(height=330, showlegend=False, yaxis_title="reconstruction MSE",
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Lower reconstruction MSE is better. The AE is non-linear; SVD/PCA are linear baselines "
               "with the same 32 dimensions.")

st.markdown("### What this means")
insights = []
if recon is not None and len(recon):
    gt_mse = recon[recon["doc_type"].isin(["gt_jd", "gt_cv"])]["reconstruction_mse"].mean()
    pool_mse = recon[recon["doc_type"] == "pool_cv"]["reconstruction_mse"].mean()
    insights.append(f"Held-out documents reconstruct at MSE **{gt_mse:.5f}** vs **{pool_mse:.5f}** for pool CVs. "
                    "Held-out documents are outside the training distribution, so a higher error is expected.")
if "dim_01" in emb.columns:
    dims = [c for c in emb.columns if c.startswith("dim_")]
    Z = emb[dims].to_numpy()
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    doc_ids = emb["doc_id"].tolist()
    jd_ids = [x for x in gt["jd_id"] if x in doc_ids]
    cv_ids = [x for x in emb["doc_id"] if x.startswith("CV-") or x.startswith("CAND-")]
    if jd_ids and cv_ids:
        jrows = [doc_ids.index(x) for x in jd_ids]
        crows = [doc_ids.index(x) for x in cv_ids]
        sims = Z[jrows] @ Z[crows].T
        percentiles = []
        for i, (_, r) in enumerate(gt.iterrows()):
            if r["hired_cv_id"] in cv_ids:
                j = cv_ids.index(r["hired_cv_id"])
                rank_pct = float((sims[i] > sims[i, j]).mean())
                percentiles.append(rank_pct)
        if percentiles:
            insights.append(f"The hired CV is, on average, in the **{np.mean(percentiles):.0%} percentile** "
                            "of AE cosine similarity within its own query (lower is better; 0% = top of 110).")
    known = emb[emb["role_family"].notna() & (emb["role_family"] != "unknown")]
    if known["role_family"].nunique() > 1:
        sil = silhouette_score(known[dims].to_numpy(), known["role_family"], metric="cosine")
        insights.append(f"Cosine silhouette by role family in the latent space is **{sil:.3f}** "
                        f"({known['role_family'].nunique()} families). Values near 0 mean the latent space "
                        "does not cleanly separate families.")
    insights.append(f"No NaNs were found in the embeddings and **{diag.get('dead_units', 0)}** latent units are dead "
                    f"(std < 1e-4). Mean latent std is {diag.get('latent_std_mean', 0):.3f}.")
insights.append("**Caveat:** the AE optimises reconstruction, not relevance. A neat latent picture can still rank "
                "the wrong candidate; see the AE-vs-TF-IDF comparison on the error-analysis page.")
for text in insights:
    st.markdown(f"- {text}")

st.markdown("### Static reference figures")
c1, c2 = st.columns(2)
with c1:
    image("07_ae_loss_curves.png", "Static training and validation loss.")
with c2:
    image("08_ae_reconstruction_error.png", "Static reconstruction-error histogram by document type.")
with st.expander("Reconstruction examples"):
    image("08b_ae_reconstruction_examples.png",
          "Top input clusters vs reconstructed values for 3 CVs and 2 JDs.")

footer()
