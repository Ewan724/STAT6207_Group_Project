"""Stage 5: recall metrics, confidence intervals and independent checks."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, dump_json


def wilson_ci(hits: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: list[float], resamples: int, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if not len(arr):
        return (0.0, 0.0)
    means = rng.choice(arr, size=(resamples, len(arr)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def compute_metrics(cfg: Config) -> dict[str, Any]:
    rankings = pd.read_csv(cfg.processed / "rankings.csv")
    gt = pd.read_csv(cfg.processed / "ground_truth.csv")
    coverage = pd.read_csv(cfg.processed / "must_have_coverage.csv")
    top10 = pd.read_csv(cfg.processed / "top10.csv")
    hired = dict(zip(gt["jd_id"], gt["hired_cv_id"]))
    top_k = int(cfg.get("retrieval.top_k", 10))

    per_query = []
    for (model, jd_id), sub in rankings.groupby(["model", "jd_id"]):
        sub = sub.sort_values("rank")
        jd = gt[gt["jd_id"] == jd_id].iloc[0]
        hcv = hired[jd_id]
        hrow = sub[sub["cv_id"] == hcv]
        if not len(hrow):
            raise RuntimeError(f"hired CV {hcv} missing from rankings for {model}/{jd_id}")
        hrow = hrow.iloc[0]
        rank = int(hrow["rank"])
        top1 = sub.iloc[0]
        cov_row = coverage[(coverage["jd_id"] == jd_id) & (coverage["cv_id"] == hcv)]
        cov = float(cov_row["coverage"].iloc[0]) if len(cov_row) else np.nan
        per_query.append({
            "model": model, "jd_id": jd_id, "jd_title": jd["jd_title"],
            "role_family": jd["role_family"], "hired_cv_id": hcv,
            "rank": rank, "score": round(float(hrow["score"]), 6),
            "hit_at_1": int(rank <= 1), "hit_at_5": int(rank <= 5),
            "hit_at_10": int(rank <= top_k),
            "reciprocal_rank": round(1.0 / rank if rank <= top_k else 0.0, 6),
            "ndcg_at_10": round(1.0 / np.log2(rank + 1) if rank <= top_k else 0.0, 6),
            "must_have_coverage_hired": round(cov, 6) if not np.isnan(cov) else None,
            "top1_cv_id": top1["cv_id"], "top1_score": round(float(top1["score"]), 6),
            "score_margin": round(float(hrow["score"]) - float(top1["score"]), 6),
        })
    pq = pd.DataFrame(per_query)
    pq.to_csv(cfg.processed / "metrics_per_query.csv", index=False)

    summary_rows = []
    for model in sorted(pq["model"].unique()):
        sub = pq[pq["model"] == model]
        hits = int(sub["hit_at_10"].sum())
        n = len(sub)
        wlo, whi = wilson_ci(hits, n)
        blo, bhi = bootstrap_ci(sub["hit_at_10"].tolist(), int(cfg.get("retrieval.bootstrap_resamples", 10000)), cfg.seed)
        summary_rows.append({
            "model": model, "n_queries": n, "hits": hits,
            "recall_at_10": round(hits / n, 6),
            "hit_at_1": round(float(sub["hit_at_1"].mean()), 6),
            "hit_at_5": round(float(sub["hit_at_5"].mean()), 6),
            "mrr_at_10": round(float(sub["reciprocal_rank"].mean()), 6),
            "ndcg_at_10": round(float(sub["ndcg_at_10"].mean()), 6),
            "precision_at_10": round(float(sub["hit_at_10"].mean()) / top_k, 6),
            "wilson_ci_low": round(wlo, 6), "wilson_ci_high": round(whi, 6),
            "bootstrap_ci_low": round(blo, 6), "bootstrap_ci_high": round(bhi, 6),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(cfg.processed / "metrics_summary.csv", index=False)

    notes = {
        "M0": "Uniform random permutation, seeded (representative draw); see random_baseline.csv for 1,000 permutations.",
        "M1": "Cosine on the training-fitted TF-IDF keyword-cluster matrix; no autoencoder.",
        "M2": "TruncatedSVD-32 fitted on training rows only; cosine on the 32-d projection.",
        "M3": "Pre-registered primary model: cosine on the frozen 32-d autoencoder bottleneck.",
        "M4": "AE latent concatenated with standardised metadata, metadata weight 0.3, L2-normalised.",
        "M5": "Exploratory: 0.8 * AE cosine + 0.2 * per-query min-max must-have coverage.",
        "M6": "Heuristic: same role-family candidates boosted above all others, then AE score.",
    }
    ablation = summary.copy()
    ablation["notes"] = ablation["model"].map(notes)
    ablation.to_csv(cfg.processed / "ablation_results.csv", index=False)

    # Independent recomputation from rankings.csv.
    checks = {}
    for model in sorted(pq["model"].unique()):
        sub = rankings[rankings["model"] == model]
        hits = 0
        rr = 0.0
        h1 = h5 = 0
        for jd_id, g in sub.groupby("jd_id"):
            rel = g[g["is_relevant"] == 1]
            if len(rel) != 1:
                checks[f"{model}:{jd_id}:relevance"] = len(rel)
                continue
            r = int(rel["rank"].iloc[0])
            hits += int(r <= 10)
            rr += 1.0 / r if r <= 10 else 0.0
            h1 += int(r <= 1); h5 += int(r <= 5)
        n = sub["jd_id"].nunique()
        checks[model] = {
            "recomputed_recall_at_10": round(hits / n, 6),
            "saved_recall_at_10": round(float(summary.loc[summary["model"] == model, "recall_at_10"].iloc[0]), 6),
            "recomputed_hit_at_1": round(h1 / n, 6),
            "recomputed_hit_at_5": round(h5 / n, 6),
            "recomputed_mrr_at_10": round(rr / n, 6),
            "match": bool(abs(hits / n - float(summary.loc[summary["model"] == model, "recall_at_10"].iloc[0])) < 1e-9),
        }
    checks["precision_at_10_max"] = float(summary["precision_at_10"].max())
    checks["precision_at_10_ok"] = bool(checks["precision_at_10_max"] <= 0.1 + 1e-9)
    checks["random_M0_recall"] = float(summary.loc[summary["model"] == "M0", "recall_at_10"].iloc[0])
    dump_json(cfg.processed / "metrics_validation.json", checks)

    # Random baseline contextualisation.
    rb_summary = pd.read_csv(cfg.processed / "random_baseline_summary.csv")
    rb = rb_summary.iloc[0].to_dict()
    rb["representative_M0_recall"] = checks["random_M0_recall"]
    dump_json(cfg.processed / "random_baseline_context.json", rb)

    return {"per_query": pq, "summary": summary, "checks": checks, "random": rb}
