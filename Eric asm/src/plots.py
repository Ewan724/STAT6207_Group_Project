"""Shared plotting helpers. Every figure is saved as a PNG at >=200 dpi."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
NAVY = "#1F2A44"
TEAL = "#009E73"
CORAL = "#D55E00"
GREY = "#8C8C8C"


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, cfg, name: str, caption: str | None = None) -> Path:
    cfg.figures.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, 0.005, caption or "Source: STAT6207 group assignment baseline.",
             fontsize=7, color=GREY, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = cfg.figures / name
    fig.savefig(path, dpi=int(cfg.get("report.figure_dpi", 200)), bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #
def keyword_cluster_network(cfg, clusters: pd.DataFrame, edges: pd.DataFrame) -> Path:
    import networkx as nx

    G = nx.Graph()
    for _, row in clusters.iterrows():
        G.add_node(int(row["cluster_id"]), label=row["canonical_label"], freq=int(row["doc_frequency"]),
                   size=int(row["size"]))
    for _, row in edges.iterrows():
        if row["source"] in G and row["target"] in G:
            G.add_edge(int(row["source"]), int(row["target"]), weight=float(row["weight"]))
    # Keep the largest connected component plus isolated nodes for readability.
    pos = nx.spring_layout(G, seed=42, k=1.8 / np.sqrt(max(1, len(G))), iterations=120)
    communities = {}
    try:
        comms = nx.community.louvain_communities(G, seed=42)
        for i, comm in enumerate(comms):
            for n in comm:
                communities[n] = i
    except Exception:
        communities = {n: 0 for n in G.nodes}
    fig, ax = plt.subplots(figsize=(14, 10))
    for u, v, data in G.edges(data=True):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color="#D0D4DA", linewidth=min(3.0, 0.35 + 0.15 * data.get("weight", 1)), zorder=1)
    node_ids = list(G.nodes)
    sizes = [max(25, min(1400, G.nodes[n].get("freq", 1) * 9)) for n in node_ids]
    colors = [OKABE_ITO[communities.get(n, 0) % len(OKABE_ITO)] for n in node_ids]
    ax.scatter([pos[n][0] for n in node_ids], [pos[n][1] for n in node_ids],
               s=sizes, c=colors, alpha=0.85, edgecolors="white", linewidths=0.6, zorder=2)
    top = clusters.sort_values("doc_frequency", ascending=False).head(20)
    for _, row in top.iterrows():
        n = int(row["cluster_id"])
        if n in pos:
            ax.annotate(row["canonical_label"], pos[n], fontsize=8, color=NAVY,
                        xytext=(4, 4), textcoords="offset points", zorder=3)
    ax.set_title("Keyword cluster network (nodes sized by document frequency; colour = community)")
    ax.set_axis_off()
    return save(fig, cfg, "03_keyword_cluster_network.png",
                "Edges join clusters co-occurring in at least two training documents. Top 20 labels shown.")


def keyword_cluster_sizes(cfg, clusters: pd.DataFrame) -> Path:
    top = clusters.sort_values("size", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(top))
    ax.barh(y, top["size"], color=OKABE_ITO[0], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top["canonical_label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of merged keyword variants")
    ax.set_title("30 largest keyword clusters")
    for i, (size, freq) in enumerate(zip(top["size"], top["doc_frequency"])):
        ax.text(size + 0.2, i, f"{int(size)} variants · df {int(freq)}", va="center", fontsize=7, color=GREY)
    return save(fig, cfg, "04_keyword_cluster_sizes.png",
                "Cluster size is the number of merged variants; df is total document frequency in the training corpus.")


def keyword_merge_examples(cfg, clusters: pd.DataFrame) -> Path:
    top = clusters.sort_values("size", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_axis_off()
    ax.set_title("20 largest alias / cluster merges", loc="left")
    ax.text(0.0, 0.97, "Canonical label", fontweight="bold", transform=ax.transAxes)
    ax.text(0.34, 0.97, "Merged raw variants (first 6)", fontweight="bold", transform=ax.transAxes)
    ax.text(0.93, 0.97, "n", fontweight="bold", transform=ax.transAxes)
    for i, (_, row) in enumerate(top.iterrows()):
        y = 0.93 - i * 0.045
        members = str(row["members"]).split("|")
        ax.text(0.0, y, row["canonical_label"][:42], fontsize=8, color=NAVY, transform=ax.transAxes)
        ax.text(0.34, y, ", ".join(members[:6])[:95], fontsize=7.2, color="#333333", transform=ax.transAxes)
        ax.text(0.95, y, str(int(row["size"])), fontsize=8, transform=ax.transAxes)
        ax.plot([0, 1], [y - 0.012, y - 0.012], color="#E5E7EB", linewidth=0.6, transform=ax.transAxes)
    return save(fig, cfg, "05_keyword_merge_examples.png",
                "Alias merging is curated; agglomerative clustering is applied only after alias normalisation.")


def keyword_coverage_heatmap(cfg, clusters: pd.DataFrame, matrix, index: pd.DataFrame,
                             documents: list[dict]) -> Path:
    from scipy import sparse

    role = {d["doc_id"]: (d.get("role_family") or "unknown") for d in documents}
    top = clusters.sort_values("doc_frequency", ascending=False).head(25)
    cids = top["cluster_id"].tolist()
    families = [f for f in pd.Series([role.get(d, "unknown") for d in index["doc_id"]]).value_counts().index
                if f != "unknown"][:10]
    X = matrix.tocsc()
    data = np.zeros((len(cids), len(families)))
    for j, fam in enumerate(families):
        rows = [i for i, d in enumerate(index["doc_id"]) if role.get(d) == fam]
        if rows:
            data[:, j] = np.asarray(X[rows][:, cids].sum(axis=0)).ravel() / max(1, len(rows))
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(cids)))
    ax.set_yticklabels(top["canonical_label"], fontsize=8)
    ax.set_title("Top 25 keyword clusters by role family (mean TF-IDF presence)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="mean TF-IDF")
    return save(fig, cfg, "05b_keyword_coverage_heatmap.png",
                "Shows which skill clusters characterise each role family in the document corpus.")


# --------------------------------------------------------------------------- #
# AE / retrieval / error figures are added in later stages.
# --------------------------------------------------------------------------- #
def pipeline_diagram(cfg) -> Path:
    stages = [
        ("0", "Parse & validate\nDATA/ (170 docs)"),
        ("1", "Local LLM\nextraction"),
        ("2", "Alias merge &\nkeyword clustering"),
        ("3", "Autoencoder\n(32-d)"),
        ("4", "Rank 110 CVs\nM0-M6"),
        ("5", "Recall@10\n+ CIs"),
        ("6", "Error analysis\n& case studies"),
        ("7", "Streamlit\nwebsite"),
    ]
    fig, ax = plt.subplots(figsize=(15, 3.4))
    ax.set_axis_off()
    w, h = 0.105, 0.55
    for i, (num, label) in enumerate(stages):
        x = 0.02 + i * 0.122
        box = FancyBboxPatch((x, 0.22), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                             facecolor="#EAF2F8" if i < 4 else "#E8F6F1", edgecolor=NAVY, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + w / 2, 0.66, f"Stage {num}", ha="center", va="center", fontsize=9,
                color=NAVY, fontweight="bold")
        ax.text(x + w / 2, 0.43, label, ha="center", va="center", fontsize=8.2, color="#222222")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.004, 0.49), (x + 0.122 - 0.004, 0.49),
                                         arrowstyle="-|>", mutation_scale=12, color=GREY, linewidth=1.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("AI CV matching baseline pipeline", loc="left")
    return save(fig, cfg, "01_pipeline_diagram.png",
                "Seven stages from read-only DATA/ to the interactive website; every stage is checkpointed.")


def dataset_overview(cfg, documents: list[dict], clusters: pd.DataFrame | None = None) -> Path:
    docs = pd.DataFrame(documents)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    counts = docs["doc_type"].value_counts()
    axes[0].bar(counts.index, counts.values, color=[OKABE_ITO[i % 8] for i in range(len(counts))])
    axes[0].set_title("Documents by type")
    axes[0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 1, str(v), ha="center", fontsize=9)
    fam = docs["role_family"].fillna("unknown").value_counts()
    axes[1].barh(fam.index, fam.values, color=OKABE_ITO[2])
    axes[1].invert_yaxis()
    axes[1].set_title("Documents by role family")
    axes[2].hist(docs["char_count"], bins=20, color=OKABE_ITO[0], alpha=0.85)
    axes[2].set_title("Document length (characters)")
    axes[2].set_xlabel("characters")
    return save(fig, cfg, "02_dataset_overview.png",
                "100 pool CVs, 10 hired CVs, 10 labelled JDs and 50 unlabelled JDs; lengths are post-cleaning.")


# --------------------------------------------------------------------------- #
# Stage 3
# --------------------------------------------------------------------------- #
def ae_architecture(cfg, input_dim: int) -> Path:
    dims = [input_dim, 256, 64, 32, 64, 256, input_dim]
    labels = ["input", "256", "64", "bottleneck 32", "64", "256", "reconstruction"]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_axis_off()
    for i, (dim, lab) in enumerate(zip(dims, labels)):
        x = 0.02 + i * 0.14
        h = 0.22 + 0.5 * (math.log10(max(10, dim)) / math.log10(max(10, max(dims))))
        color = "#F6C85F" if i == 3 else ("#9BC1E8" if i < 3 else "#8FD3B6")
        ax.add_patch(FancyBboxPatch((x, 0.5 - h / 2), 0.095, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                    facecolor=color, edgecolor=NAVY, linewidth=1.2))
        ax.text(x + 0.0475, 0.5, f"{lab}\n({dim})", ha="center", va="center", fontsize=8.5, color=NAVY)
        if i < len(dims) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.1, 0.5), (x + 0.137, 0.5), arrowstyle="-|>",
                                         mutation_scale=11, color=GREY))
    ax.text(0.02, 0.16, "Encoder", fontsize=10, color=NAVY, fontweight="bold")
    ax.text(0.48, 0.16, "Latent document embedding", fontsize=10, color="#8A6D00", fontweight="bold")
    ax.text(0.70, 0.16, "Decoder", fontsize=10, color=NAVY, fontweight="bold")
    ax.set_title("Autoencoder architecture (GELU + dropout; MSE reconstruction loss)")
    return save(fig, cfg, "06_ae_architecture.png",
                "The encoder is frozen after training; only the 32-d bottleneck is used for retrieval.")


def ae_loss_curves(cfg, history: pd.DataFrame, best_epoch: int) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(history["epoch"], history["train_mse"], label="train MSE", color=OKABE_ITO[0], linewidth=2)
    ax.plot(history["epoch"], history["val_mse"], label="validation MSE", color=OKABE_ITO[1], linewidth=2)
    ax.axvline(best_epoch, color=TEAL, linestyle="--", linewidth=1.4, label=f"best epoch ({best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error")
    ax.set_yscale("log")
    ax.set_title("Autoencoder training and validation loss")
    ax.legend(frameon=False)
    return save(fig, cfg, "07_ae_loss_curves.png",
                "Early stopping restores the best validation-loss weights, not the last epoch.")


def ae_reconstruction_error(cfg, embeddings: pd.DataFrame, X: np.ndarray, rec: np.ndarray) -> Path:
    err = np.mean((rec - X) ** 2, axis=1)
    df = embeddings[["doc_id", "doc_type"]].copy()
    df["mse"] = err
    fig, ax = plt.subplots(figsize=(10, 5.5))
    types = [t for t in ["pool_cv", "gt_cv", "gt_jd", "unlabelled_jd"] if (df["doc_type"] == t).any()]
    data = [df.loc[df["doc_type"] == t, "mse"].values for t in types]
    ax.hist(data, bins=18, label=types, color=[OKABE_ITO[i % 8] for i in range(len(types))],
            alpha=0.7, stacked=False)
    ax.set_xlabel("per-document reconstruction MSE")
    ax.set_ylabel("documents")
    ax.set_title("Reconstruction error by document type")
    ax.legend(frameon=False)
    return save(fig, cfg, "08_ae_reconstruction_error.png",
                "Held-out documents (GT JDs / hired CVs) are encoded after the AE is frozen.")


def reconstruction_examples(cfg, examples: pd.DataFrame) -> Path:
    docs = examples["doc_id"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(docs), figsize=(4 * len(docs), 4.4))
    if len(docs) == 1:
        axes = [axes]
    for ax, doc in zip(axes, docs):
        sub = examples[examples["doc_id"] == doc].head(8)
        y = np.arange(len(sub))
        ax.barh(y - 0.18, sub["input_value"], height=0.36, color=OKABE_ITO[0], label="input")
        ax.barh(y + 0.18, sub["reconstructed_value"], height=0.36, color=OKABE_ITO[1], label="reconstructed")
        ax.set_yticks(y)
        ax.set_yticklabels([str(x)[:22] for x in sub["input_cluster"]], fontsize=6.5)
        ax.invert_yaxis()
        ax.set_title(doc, fontsize=9)
        ax.set_xlabel("TF-IDF value")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Top reconstructed clusters vs input (3 CVs, 2 JDs)", fontsize=12, fontweight="bold")
    return save(fig, cfg, "08b_ae_reconstruction_examples.png",
                "A reconstruction objective can preserve or smooth away rare must-have clusters.")


def latent_space(cfg, coords: pd.DataFrame, ground_truth: pd.DataFrame, method: str = "pca") -> Path:
    x, y = f"{method}_x", f"{method}_y"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
    type_colors = {"pool_cv": OKABE_ITO[0], "gt_cv": OKABE_ITO[1], "gt_jd": OKABE_ITO[2],
                   "unlabelled_jd": OKABE_ITO[5]}
    for t, color in type_colors.items():
        sub = coords[coords["doc_type"] == t]
        axes[0].scatter(sub[x], sub[y], s=38 if t != "pool_cv" else 22, c=color,
                        label=t, alpha=0.85, edgecolors="white", linewidths=0.4)
    pos = coords.set_index("doc_id")[[x, y]]
    for _, row in ground_truth.iterrows():
        if row["jd_id"] in pos.index and row["hired_cv_id"] in pos.index:
            a, b = pos.loc[row["jd_id"]], pos.loc[row["hired_cv_id"]]
            axes[0].plot([a[x], b[x]], [a[y], b[y]], color=CORAL, linewidth=0.8, alpha=0.7)
            axes[0].annotate(str(row["jd_id"]), (a[x], a[y]), fontsize=7, color=CORAL)
    axes[0].set_title(f"Latent space ({method.upper()}) by document type + GT pairs")
    axes[0].legend(frameon=False, fontsize=8)
    fams = coords["role_family"].fillna("unknown").value_counts().index.tolist()
    cmap = plt.get_cmap("tab20")
    for i, fam in enumerate(fams):
        sub = coords[coords["role_family"].fillna("unknown") == fam]
        axes[1].scatter(sub[x], sub[y], s=26, color=cmap(i % 20), label=str(fam), alpha=0.8)
    axes[1].set_title(f"Latent space ({method.upper()}) by role family")
    axes[1].legend(frameon=False, fontsize=6.5, ncol=2, loc="best")
    for ax in axes:
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
    return save(fig, cfg, f"{'09_latent_space_pca' if method == 'pca' else '10_latent_space_tsne'}.png",
                "Red lines link each labelled JD to its hired CV; a useful representation pulls pairs together.")


# --------------------------------------------------------------------------- #
# Stage 5 / 6
# --------------------------------------------------------------------------- #
def recall_by_jd(cfg, per_query: pd.DataFrame, model: str = "M3") -> Path:
    sub = per_query[per_query["model"] == model].sort_values("jd_id")
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(sub))
    colors = [TEAL if h else CORAL for h in sub["hit_at_10"]]
    ax.bar(x, sub["rank"], color=colors, alpha=0.9)
    ax.axhline(10, color=GREY, linestyle="--", linewidth=1.2)
    ax.text(len(sub) - 0.4, 10.4, "Recall@10 boundary", ha="right", fontsize=8, color=GREY)
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.text(i, min(r["rank"] + 1, 11), f"#{int(r['rank'])}", ha="center", fontsize=8,
                color=NAVY if r["hit_at_10"] else CORAL, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(sub["jd_id"])
    ax.invert_yaxis()
    ax.set_ylabel("rank of hired CV (1 = best)")
    ax.set_title(f"Hired-CV rank by labelled JD ({model}); green = hit, coral = miss")
    return save(fig, cfg, "11_recall_by_jd.png",
                "Ranks are from the primary M3 AE-32 model. The dashed line marks the top-10 boundary.")


def metric_comparison(cfg, summary: pd.DataFrame) -> Path:
    metrics = ["recall_at_10", "hit_at_1", "hit_at_5", "mrr_at_10"]
    fig, ax = plt.subplots(figsize=(12, 5.8))
    x = np.arange(len(summary)); width = 0.2
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, summary[m], width=width, label=m.replace("_", " "), color=OKABE_ITO[i])
    ax.set_xticks(x + width * 1.5); ax.set_xticklabels(summary["model"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title("Model comparison on 10 labelled queries")
    ax.legend(frameon=False, ncol=4)
    return save(fig, cfg, "12_metric_comparison.png",
                "All metrics treat the single hired CV as the only relevant item; differences with n=10 are descriptive.")


def rank_distribution(cfg, per_query: pd.DataFrame) -> Path:
    buckets = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 110)]
    labels = ["1", "2-5", "6-10", "11-20", "21-50", "51-110"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    width = 0.8 / len(per_query["model"].unique())
    models = sorted(per_query["model"].unique())
    for j, model in enumerate(models):
        sub = per_query[per_query["model"] == model]
        counts = [int(((sub["rank"] >= lo) & (sub["rank"] <= hi)).sum()) for lo, hi in buckets]
        ax.bar(np.arange(len(labels)) + j * width, counts, width=width, label=model,
               color=OKABE_ITO[j % 8])
    ax.set_xticks(np.arange(len(labels)) + 0.4 - width / 2); ax.set_xticklabels(labels)
    ax.set_xlabel("rank bucket of hired CV"); ax.set_ylabel("number of queries")
    ax.set_title("Where the hired CV lands across models")
    ax.legend(frameon=False, ncol=7, fontsize=8)
    return save(fig, cfg, "13_rank_distribution.png",
                "A good baseline puts most hired CVs in the 1-10 buckets; misses beyond 10 are the error-analysis focus.")


def score_margin(cfg, per_query: pd.DataFrame, model: str = "M3") -> Path:
    sub = per_query[per_query["model"] == model].sort_values("jd_id")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    colors = [TEAL if v >= 0 else CORAL for v in sub["score_margin"]]
    ax.bar(sub["jd_id"], sub["score_margin"], color=colors, alpha=0.9)
    ax.axhline(0, color=NAVY, linewidth=1)
    for i, (_, r) in enumerate(sub.iterrows()):
        if r["score_margin"] < 0:
            ax.annotate(f"#{int(r['rank'])}", (i, r["score_margin"]), xytext=(0, -12),
                        textcoords="offset points", ha="center", fontsize=8, color=CORAL)
    ax.set_ylabel("hired score - top-1 score")
    ax.set_title(f"Score margin of the hired CV ({model})")
    ax.tick_params(axis="x", rotation=25)
    return save(fig, cfg, "14_score_margin.png",
                "Negative bars mean a false positive outranks the hired CV; the magnitude shows how close the miss was.")


def ae_vs_tfidf_ranks(cfg, comp: pd.DataFrame) -> Path:
    comp = comp.sort_values("jd_id")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (_, r) in enumerate(comp.iterrows()):
        ax.plot([i, i], [r["hired_rank_tfidf"], r["hired_rank_ae"]], color="#CFD4DA", linewidth=2, zorder=1)
        ax.scatter(i, r["hired_rank_tfidf"], color=OKABE_ITO[0], s=55, label="TF-IDF" if i == 0 else None, zorder=2)
        ax.scatter(i, r["hired_rank_ae"], color=OKABE_ITO[1], s=55, label="AE-32" if i == 0 else None, zorder=2)
    ax.axhline(10, color=GREY, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(comp))); ax.set_xticklabels(comp["jd_id"])
    ax.invert_yaxis(); ax.set_ylabel("rank of hired CV")
    ax.set_title("AE vs TF-IDF: hired-CV rank per query")
    ax.legend(frameon=False)
    return save(fig, cfg, "16_ae_vs_tfidf_ranks.png",
                "A lower dot is better. The AE is not uniformly better; the diagnosis is the point of the baseline.")


def error_taxonomy(cfg, error_df: pd.DataFrame, model: str = "M3") -> Path:
    sub = error_df[(error_df["model"] == model) & (error_df["hired_rank"] > 10)]
    counts = sub["primary_failure_category"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.barh(counts.index[::-1], counts.values[::-1], color=CORAL, alpha=0.85)
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v + 0.05, i, str(v), va="center", fontsize=9)
    ax.set_xlabel("number of misses")
    ax.set_title(f"Primary failure category for AE-32 misses (n={len(sub)})")
    return save(fig, cfg, "17_error_taxonomy.png",
                "Categories are assigned with evidence in error_analysis.csv; a miss can have a secondary category.")


def case_study_heatmap(cfg, data: dict, jd_id: str, combined_ax=None):
    df = data["table"]
    kws = data["keywords"]
    cols = [k for k in kws if k in df.columns]
    vals = df[cols].to_numpy(dtype=float)
    if combined_ax is None:
        fig, ax = plt.subplots(figsize=(12, 5.8))
    else:
        fig, ax = None, combined_ax
    im = ax.imshow(vals, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([str(c)[:18] for c in cols], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(df)))
    labels = [f"#{int(r['rank'])} {r['cv_id']}" + (" HIRED" if r["is_relevant"] else "")
              for _, r in df.iterrows()]
    ax.set_yticklabels(labels, fontsize=7.5)
    for i, (_, r) in enumerate(df.iterrows()):
        if r["is_relevant"]:
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(cols), 1, fill=False, edgecolor=TEAL, linewidth=2.2))
    ax.set_title(f"{jd_id}: must-have cluster presence in the top 10")
    if combined_ax is None:
        fig.colorbar(im, ax=ax, shrink=0.7, label="weighted cluster count")
        return save(fig, cfg, f"18_case_study_{jd_id}.png",
                    "Rows are the model's top-10 CVs; the teal box marks the hired CV if it is inside the top 10.")


def case_study_combined(cfg, datasets: list[dict]) -> Path:
    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4.6 * n))
    if n == 1:
        axes = [axes]
    for ax, data in zip(axes, datasets):
        case_study_heatmap(cfg, data, data["jd"]["jd_id"], combined_ax=ax)
    fig.suptitle("Case studies: must-have coverage of top-10 candidates", fontsize=13, fontweight="bold")
    return save(fig, cfg, "15_case_study_keyword_coverage.png",
                "Each panel is one labelled JD; rows are the M3 top 10. Missing columns show why a lookalike can win.")


def recommendation_matrix(cfg, recs: pd.DataFrame) -> Path:
    effort = {"Low": 1, "Medium": 2, "High": 3}
    impact = {"P0": 3, "P1": 2, "P2": 1}
    fig, ax = plt.subplots(figsize=(9, 6))
    prio_color = {"P0": CORAL, "P1": OKABE_ITO[0], "P2": GREY}
    rng = np.random.default_rng(42)
    for i, (_, r) in enumerate(recs.iterrows()):
        x = effort.get(r["effort"], 2) + float(rng.uniform(-0.12, 0.12))
        y = impact.get(r["priority"], 2) + float(rng.uniform(-0.10, 0.10))
        ax.scatter(x, y, s=170, color=prio_color.get(r["priority"], GREY), alpha=0.8, edgecolors="white")
        ax.annotate(str(i + 1), (x, y), ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_yticks([1, 2, 3]); ax.set_yticklabels(["P2 / lower", "P1 / medium", "P0 / highest"])
    ax.set_xlim(0.6, 3.4); ax.set_ylim(0.6, 3.4)
    ax.set_xlabel("implementation effort"); ax.set_ylabel("priority / expected impact")
    ax.set_title("Recommendation matrix (colour = priority; numbers match recommendations.csv)")
    handles = [plt.Line2D([], [], marker="o", color="w", markerfacecolor=c, markersize=9, label=p)
               for p, c in prio_color.items()]
    ax.legend(handles=handles, frameon=False, title="priority")
    return save(fig, cfg, "19_recommendation_matrix.png",
                "Impact labels are hypotheses, not measured improvements; they are not promises of gain.")


def extraction_quality(cfg, quality: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(quality["f1"], bins=15, color=OKABE_ITO[0], alpha=0.85)
    axes[0].axvline(float(quality["f1"].mean()), color=CORAL, linestyle="--", label=f"macro mean {quality['f1'].mean():.2f}")
    axes[0].set_xlabel("per-CV F1 vs Excel Skills"); axes[0].set_ylabel("CVs")
    axes[0].set_title("Extraction quality distribution"); axes[0].legend(frameon=False)
    worst = quality.nsmallest(8, "f1")
    axes[1].barh(worst["doc_id"][::-1], worst["f1"][::-1], color=CORAL, alpha=0.85)
    axes[1].set_xlabel("F1"); axes[1].set_title("Worst 8 CVs by extraction F1")
    return save(fig, cfg, "20_extraction_quality.png",
                "Silver standard is the Excel Skills column; PDF truncation and synonym variation both lower this score.")
