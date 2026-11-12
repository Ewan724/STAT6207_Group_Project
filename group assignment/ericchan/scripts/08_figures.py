"""Step 8: figures for the report."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 170, "savefig.dpi": 170, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.22, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
    "font.family": "DejaVu Sans",
})

C = {"best": "#0f6b4f", "ae": "#1f6feb", "kw": "#c46a1a", "bm": "#8b5cf6",
     "rand": "#9aa5b1", "oracle": "#d64545", "nomic": "#0d8ea8"}


def main():
    m = json.load(open(os.path.join(DATA, "metrics.json"), encoding="utf-8"))
    ae = json.load(open(os.path.join(DATA, "ae_embeddings.json"), encoding="utf-8"))
    aem = json.load(open(os.path.join(DATA, "ae_metrics.json"), encoding="utf-8"))
    sweep = []
    for f in ("ae_sweep.json", "ae_sweep2.json", "ae_sweep3.json",
              "ae_sweep4.json", "ae_sweep5.json", "ae_sweep6.json"):
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            try:
                sweep.append((f, json.load(open(p, encoding="utf-8"))))
            except json.JSONDecodeError:
                pass
    res = {r["system"]: r for r in m["results"]}

    # ---- fig 1: AE training curve
    h = np.array(ae["history"])
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 2.9))
    ax[0].plot(h[:, 0], color=C["ae"], lw=1.5)
    ax[0].set_title("Autoencoder objective (1 − cos)", fontsize=10)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[0].set_yscale("log")
    ax[1].plot(h[:, 1], color=C["kw"], lw=1.3, label="skill block (1 − cos)")
    ax[1].plot(h[:, 2], color=C["bm"], lw=1.3, label="structural (MSE, logged only)")
    ax[1].set_xlabel("epoch"); ax[1].set_yscale("log")
    ax[1].set_title("Loss components", fontsize=10)
    ax[1].legend(frameon=False, fontsize=7.5)
    fig.savefig(os.path.join(FIG, "ae_training.png"))
    plt.close(fig)

    # ---- fig 2: what the AE is worth (the central negative result)
    bars = [
        ("chance", 0.10, C["rand"]),
        ("keyword-only AE\n(447→32)", aem.get("latent_purity_at5", 0), C["ae"]),
        ("raw keyword space\n(no model)", aem.get("input_space_purity_at5", 0),
         C["kw"]),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    names = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    cols = [b[2] for b in bars]
    ax.bar(names, vals, color=cols, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_ylabel("latent purity@5 (job-family neighbourhood)")
    ax.set_ylim(0, 0.45)
    ax.set_title("The autoencoder compresses useful geometry, but the\n"
                 "keyword space it compresses is the real limit", fontsize=9.5)
    fig.savefig(os.path.join(FIG, "ae_purity.png"))
    plt.close(fig)

    # ---- fig 3: recall@k curves
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    show = [("llm_rerank_blend", "Full pipeline (filter + fusion + LLM rerank)",
             C["best"], 2.2),
            ("hybrid_family_pred", "Fusion + predicted-family filter", C["nomic"], 1.6),
            ("hybrid_rerank", "Fusion + seniority rerank", C["ae"], 1.6),
            ("nomic", "Text embedding alone", C["bm"], 1.4),
            ("tfidf", "TF-IDF (raw text)", C["kw"], 1.4),
            ("keyword", "Merged keywords only", "#b0873a", 1.3),
            ("ae_latent", "Keyword AE alone", "#7fa7d8", 1.3),
            ("random", "Random", C["rand"], 1.2)]
    for name, lab, col, lw in show:
        if name not in res:
            continue
        cur = res[name]["recall_curve"]
        ks = sorted(int(k) for k in cur)
        ax.plot(ks, [cur[str(k)] for k in ks], marker="o", ms=3.4, lw=lw,
                color=col, label=lab)
    ax.set_xlabel("k  (number of CVs retrieved)")
    ax.set_ylabel("Recall@k  (mean over 10 JDs)")
    ax.set_title("Recall@k — the pipeline dominates from k=1 onward", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    fig.savefig(os.path.join(FIG, "recall_curve.png"))
    plt.close(fig)

    # ---- fig 4: per-JD recall for the best system
    best = res.get("llm_rerank_blend") or res.get("hybrid_family_pred")
    pj = best["per_jd_recall@10"]
    jds = sorted(pj)
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    cols = [C["best"] if pj[j] >= 0.99 else (C["nomic"] if pj[j] >= 0.5 else C["oracle"])
            for j in jds]
    ax.bar(jds, [pj[j] for j in jds], color=cols, width=0.6)
    for i, j in enumerate(jds):
        ax.text(i, pj[j] + 0.02, f"{pj[j]:.2f}", ha="center", fontsize=7.5)
    ax.axhline(best["recall@10"], color="#444", ls="--", lw=1,
               label=f"mean = {best['recall@10']:.3f}")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Recall@10")
    ax.set_title("Per-JD Recall@10 — full pipeline", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(FIG, "per_jd_recall.png"))
    plt.close(fig)

    # ---- fig 5: system comparison
    order = ["random", "ae_latent", "ae_kwtxt", "keyword", "keyword_mh", "bm25",
             "tfidf", "hybrid_equal", "rrf", "ae_text", "hybrid_tuned",
             "hybrid_rerank", "nomic", "family_oracle", "hybrid_family_pred",
             "llm_rerank_only", "llm_rerank_blend", "hybrid_family_gt"]
    order = [o for o in order if o in res]
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    x = np.arange(len(order))
    w = 0.2
    for i, (metric, col) in enumerate([("recall@10", "#1f6feb"),
                                       ("precision@10", "#0f8f5f"),
                                       ("ndcg@10", "#c46a1a"),
                                       ("map@10", "#8b5cf6")]):
        ax.bar(x + (i - 1.5) * w, [res[o][metric] for o in order], w,
               label=metric, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=6.6,
                       rotation=0)
    ax.set_ylabel("score")
    ax.set_title("All systems — 10 JDs × 100 CVs", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=4)
    ax.set_ylim(0, 1.02)
    fig.savefig(os.path.join(FIG, "system_comparison.png"))
    plt.close(fig)

    # ---- fig 6: grade distribution
    gd = m["grade_distribution"]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    labels = ["0 out of\nprofession", "1 below\nthe bar", "2 plausible\nshortlist",
              "3 at/above\nthe bar"]
    vals = [gd.get(str(g), 0) for g in range(4)]
    cols = ["#dfe4ea", "#f0c07a", "#7fbf7f", "#2f7d32"]
    ax.bar(labels, vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 12, str(v), ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("pairs (10 JDs × 100 CVs)")
    ax.set_title("Ground-truth grade distribution", fontsize=10)
    fig.savefig(os.path.join(FIG, "grade_distribution.png"))
    plt.close(fig)

    # ---- fig 7: AE variant comparison, purity vs downstream recall
    variants = [("keyword-only AE\n(447→32, the brief's model)", "ae_latent",
                 aem.get("latent_purity_at5", 0)),
                ("keyword+text AE\n(1,215→64)", "ae_kwtxt", None),
                ("text AE\n(772→32)", "ae_text", None)]
    labels, recalls = [], []
    for lab, key, _ in variants:
        if key in res:
            labels.append(lab)
            recalls.append(res[key]["recall@10"])
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    ax.bar(labels, recalls, color=[C["ae"], C["nomic"], C["best"]], width=0.5)
    for i, v in enumerate(recalls):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 0.9)
    ax.set_title("Same architecture, different input:\n"
                 "the input representation is the bottleneck", fontsize=9.5)
    fig.savefig(os.path.join(FIG, "ae_variants.png"))
    plt.close(fig)

    # ---- fig 8: fusion weight sensitivity
    grid = m.get("fusion_grid", [])
    if grid:
        pts = {}
        for g in grid:
            w = g["weights"]
            k = tuple(sorted(w.items()))
            pts[k] = (g["ndcg@10"], g["recall@10"])
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        nd = [v[0] for v in pts.values()]
        rc = [v[1] for v in pts.values()]
        ax.scatter(nd, rc, s=16, alpha=0.55, color=C["nomic"], edgecolor="none")
        bi = int(np.argmax(nd))
        ax.scatter([nd[bi]], [rc[bi]], s=70, facecolor="none",
                   edgecolor=C["oracle"], lw=1.6, label="selected weights")
        ax.set_xlabel("NDCG@10"); ax.set_ylabel("Recall@10")
        ax.set_title("Fusion grid — the two metrics broadly agree", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(FIG, "fusion_grid.png"))
        plt.close(fig)

    # ---- fig 9: 80:20 holdout vs in-sample
    ho_path = os.path.join(DATA, "holdout.json")
    if os.path.exists(ho_path):
        ho = json.load(open(ho_path, encoding="utf-8"))
        sys_order = [("keyword_mh", "keyword_mh", "merged keywords"),
                     ("ae_latent", "ae_latent", "keyword AE"),
                     ("nomic", "nomic", "text embedding"),
                     ("hybrid", "hybrid_tuned", "fusion"),
                     ("hybrid_family", "hybrid_family_pred", "fusion + family"),
                     ("hybrid_family_rerank", "llm_rerank_blend",
                      "fusion + family + rerank")]
        labels = [s[2] for s in sys_order]
        hold = [ho["holdout"][s[0]]["recall@10"] for s in sys_order]
        ins = [ho["in_sample"].get(s[1], {}).get("recall@10", 0)
               for s in sys_order]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(7.6, 3.4))
        ax.bar(x - 0.19, ins, 0.36, label="in-sample (all 10 JDs, tuned on them)",
               color="#a8c4e8")
        ax.bar(x + 0.19, hold, 0.36, label="80:20 holdout (5-fold, tuned on 8)",
               color=C["best"])
        for i, (a, b) in enumerate(zip(ins, hold)):
            ax.text(i - 0.19, a + 0.014, f"{a:.2f}", ha="center", fontsize=7)
            ax.text(i + 0.19, b + 0.014, f"{b:.2f}", ha="center", fontsize=7,
                    fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Recall@10")
        ax.set_ylim(0, 1.05)
        ax.set_title("Out-of-sample check: the gap is small (−0.03 on the full "
                     "pipeline)", fontsize=9.5)
        ax.legend(frameon=False, fontsize=7.6, loc="upper left")
        fig.savefig(os.path.join(FIG, "holdout.png"))
        plt.close(fig)

    print("figures written:")
    for f in sorted(os.listdir(FIG)):
        print("  ", f, os.path.getsize(os.path.join(FIG, f)))


if __name__ == "__main__":
    main()
