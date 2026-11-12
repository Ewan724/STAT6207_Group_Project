"""Stage 4: candidate ranking for models M0-M6 with template-based reasons."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .config import Config, dump_json
from .data import read_jsonl
from .keywords import load_extractions, term_key

SENIORITY_RANK = {"intern": 0, "junior": 1, "mid": 2, "senior": 3, "lead": 4,
                  "director": 5, "executive": 6, "unknown": 2}

MODELS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]
MODEL_NAMES = {
    "M0": "Random",
    "M1": "TF-IDF keyword cosine",
    "M2": "SVD-32 cosine",
    "M3": "AE-32 cosine (primary)",
    "M4": "AE + metadata",
    "M5": "Hybrid AE + must-have coverage (exploratory)",
    "M6": "Role-family heuristic",
}


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def _doc_meta(documents: list[dict]) -> dict[str, dict]:
    return {d["doc_id"]: d for d in documents}


def _clean_meta_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "unknown"} else None


def document_role_family(doc: dict, rec: dict | None = None) -> str:
    """Role family from metadata, falling back to the LLM's CV-only inference.

    Pool CVs use the workbook's curated family. The hired CV must not inherit the
    JD family (that would leak the label), so it starts as "unknown" and uses the
    family inferred from its own text.
    """
    rec = rec or {}
    family = _clean_meta_text(doc.get("role_family"))
    if family is None:
        family = _clean_meta_text(rec.get("role_family"))
    return family or "unknown"


def build_metadata_matrix(cfg: Config, documents: list[dict], index: pd.DataFrame,
                          extracted: dict[str, dict]) -> pd.DataFrame:
    meta = _doc_meta(documents)
    rows = []
    for doc_id in index["doc_id"]:
        d = meta.get(doc_id, {})
        rec = extracted.get(doc_id, {})
        rows.append({
            "doc_id": doc_id,
            "seniority": rec.get("seniority", "unknown"),
            "role_family": document_role_family(d, rec),
            "years": rec.get("total_years_experience"),
            "country": _clean_meta_text(d.get("country")) or "unknown",
            "remote": _clean_meta_text(d.get("remote")) or "unknown",
        })
    raw = pd.DataFrame(rows)
    top_countries = raw["country"].value_counts().head(15).index.tolist()

    def onehot(series: pd.Series, categories: list[str], prefix: str) -> pd.DataFrame:
        out = pd.DataFrame({f"{prefix}_{c}": (series == c).astype(float) for c in categories})
        out[f"{prefix}_other"] = (~series.isin(categories)).astype(float)
        return out

    seniority = onehot(raw["seniority"], list(SENIORITY_RANK.keys()), "sen")
    family = onehot(raw["role_family"], sorted(raw["role_family"].unique()), "fam")
    country = onehot(raw["country"], top_countries, "country")
    remote = onehot(raw["remote"], ["yes", "no", "hybrid", "unknown"], "remote")
    years = pd.to_numeric(raw["years"], errors="coerce")
    years_known = years.notna().astype(float)
    years_scaled = ((years.fillna(0).clip(0, 30)) / 30.0).astype(float)
    matrix = pd.concat([
        raw[["doc_id"]], seniority, family, country, remote,
        pd.DataFrame({"years_scaled": years_scaled, "years_unknown": years_known}),
    ], axis=1)
    # Standardise numeric columns using pool CVs only (unsupervised, no labels).
    train_mask = index["doc_type"].eq("pool_cv").to_numpy()
    numeric_cols = [c for c in matrix.columns if c != "doc_id"]
    scaler = StandardScaler()
    vals = matrix[numeric_cols].to_numpy(dtype=float)
    scaler.fit(vals[train_mask] if train_mask.any() else vals)
    matrix[numeric_cols] = scaler.transform(vals)
    matrix.to_csv(cfg.processed / "metadata_matrix.csv", index=False)
    return matrix


def must_have_sets(cfg: Config, ground_truth: pd.DataFrame, extracted: dict[str, dict],
                   clusters: pd.DataFrame) -> tuple[dict[str, list[int]], dict[str, list[str]], dict[str, dict[int, str]]]:
    from .keywords import map_term
    import pickle

    with (cfg.models / "cluster_model.pkl").open("rb") as fh:
        blob = pickle.load(fh)
    centroids = blob["centroids"]
    vec, svd = blob.get("vectorizer"), blob.get("svd")
    threshold = float(cfg.get("clustering.unmatched_cosine_threshold", 0.45))
    term_to_cluster = json.loads((cfg.processed / "keyword_to_cluster.json").read_text())
    label_by_cluster = dict(zip(clusters["cluster_id"], clusters["canonical_label"]))
    jd_clusters: dict[str, list[int]] = {}
    jd_keywords: dict[str, list[str]] = {}
    for _, row in ground_truth.iterrows():
        jd_id = row["jd_id"]
        rec = extracted.get(jd_id, {})
        kws = list(rec.get("must_have_keywords") or [])
        if not kws:
            kws = list(rec.get("nice_to_have_keywords") or [])
        jd_keywords[jd_id] = kws
        cids = []
        for kw in kws:
            cid, _ = map_term(kw, term_to_cluster, centroids, vec, svd, threshold)
            if cid is not None:
                cids.append(cid)
        jd_clusters[jd_id] = sorted(set(cids))
    return jd_clusters, jd_keywords, {k: label_by_cluster for k in []}


def coverage_for_candidate(counts_row, cids: list[int], keywords: list[str],
                           label_by_cluster: dict[int, str],
                           term_to_cluster: dict[str, int], mapped_cids: list[int]) -> tuple[float, list[str], list[str]]:
    present = set(np.where(counts_row > 0)[0].tolist())
    matched, missing = [], []
    for kw, cid in zip(keywords, mapped_cids):
        if cid in present:
            matched.append(kw)
        else:
            missing.append(kw)
    denom = len(keywords)
    return (len(matched) / denom if denom else 0.0), matched, missing


def _reason(row: pd.Series, matched: list[str], missing: list[str], jd_family: str,
            cv_family: str, cv_years, jd_years, cv_sen, jd_sen, jd_country, cv_country,
            jd_langs: list[str], cv_langs: list[str], jd_remote: str) -> str:
    n_total = len(matched) + len(missing)
    top_matched = ", ".join(matched[:6]) if matched else "none"
    missing_txt = ", ".join(missing[:3]) if missing else "none"
    parts = [f"Matches {len(matched)}/{n_total} must-haves ({top_matched}); missing {missing_txt}."]
    same_family = bool(jd_family and cv_family and str(jd_family) != "unknown"
                       and str(cv_family) != "unknown" and str(cv_family) == str(jd_family))
    parts.append(f"{'Same' if same_family else 'Different'} role family ({cv_family or 'unknown'}).")
    if cv_years is not None:
        parts.append(f"{int(cv_years)} yrs experience.")
    if jd_sen and cv_sen:
        match = SENIORITY_RANK.get(str(cv_sen), 2) >= SENIORITY_RANK.get(str(jd_sen), 2)
        parts.append(f"Seniority {'meets' if match else 'below'} requirement ({cv_sen} vs {jd_sen}).")
    caveats = []
    if jd_country and cv_country and str(jd_country).lower() not in str(cv_country).lower():
        if jd_remote != "yes":
            caveats.append(f"location differs ({cv_country} vs {jd_country})")
    if jd_langs and cv_langs and not set(x.lower() for x in jd_langs) & set(x.lower() for x in cv_langs):
        caveats.append("language not evidenced")
    if caveats:
        parts.append("Filter caveat: " + "; ".join(caveats) + " (baseline does not apply filters).")
    parts.append(f"Rank {int(row['rank'])}, score {float(row['score']):.6f}.")
    text = " ".join(parts)
    if len(text) > 300:
        text = text[:296].rsplit(" ", 1)[0] + " ..."
    return text


def build_rankings(cfg: Config, force: bool = False) -> dict[str, Any]:
    documents = read_jsonl(cfg.processed / "documents.jsonl")
    extracted = load_extractions(cfg)
    gt = pd.read_csv(cfg.processed / "ground_truth.csv")
    index = pd.read_csv(cfg.processed / "document_matrix_index.csv")
    clusters = pd.read_csv(cfg.processed / "keyword_clusters.csv")
    X = sparse.load_npz(cfg.processed / "document_keyword_matrix.npz").astype(np.float32)
    counts = sparse.load_npz(cfg.processed / "document_keyword_counts.npz").astype(np.float32)
    embeddings = pd.read_csv(cfg.processed / "document_embeddings.csv")
    meta_matrix = build_metadata_matrix(cfg, documents, index, extracted)

    row_of = {doc_id: i for i, doc_id in enumerate(index["doc_id"])}
    cand_docs = [d for d in documents if d["doc_type"] in {"pool_cv", "gt_cv"}]
    cand_ids = [d["doc_id"] for d in cand_docs]
    query_ids = gt["jd_id"].tolist()
    hired = dict(zip(gt["jd_id"], gt["hired_cv_id"]))
    meta_by_id = _doc_meta(documents)

    # Dense representations.
    Xd = X.toarray()
    Xd_n = _normalise_rows(Xd)
    train_rows = index.index[index["is_training"]].tolist()
    svd = TruncatedSVD(n_components=32, random_state=cfg.seed)
    svd.fit(Xd[train_rows])
    Z_svd = _normalise_rows(svd.transform(Xd))

    emb_cols = [c for c in embeddings.columns if c.startswith("dim_")]
    emb_indexed = embeddings.set_index("doc_id").loc[index["doc_id"]]
    Z_ae = _normalise_rows(emb_indexed[emb_cols].to_numpy())

    meta_cols = [c for c in meta_matrix.columns if c != "doc_id"]
    meta_indexed = meta_matrix.set_index("doc_id").loc[index["doc_id"]]
    M = meta_indexed[meta_cols].to_numpy(dtype=float)
    M = _normalise_rows(M)
    Z_meta = _normalise_rows(np.concatenate([Z_ae, 0.3 * M], axis=1))

    jd_clusters, jd_keywords, _ = must_have_sets(cfg, gt, extracted, clusters)
    cluster_labels = dict(zip(clusters["cluster_id"], clusters["canonical_label"]))
    term_to_cluster = json.loads((cfg.processed / "keyword_to_cluster.json").read_text())
    from .keywords import map_term
    import pickle
    with (cfg.models / "cluster_model.pkl").open("rb") as fh:
        blob = pickle.load(fh)
    vec, svd_map, centroids = blob.get("vectorizer"), blob.get("svd"), blob["centroids"]
    threshold = float(cfg.get("clustering.unmatched_cosine_threshold", 0.45))

    def cluster_ids_for(kws: list[str]) -> list[int]:
        ids = []
        for kw in kws:
            cid, _ = map_term(kw, term_to_cluster, centroids, vec, svd_map, threshold)
            ids.append(cid)
        return ids

    rng = np.random.default_rng(cfg.seed)
    ranking_rows = []
    top10_rows = []
    coverage_rows = []
    reason_rows = []

    for jd_id in query_ids:
        jd_row = gt[gt["jd_id"] == jd_id].iloc[0]
        jd_rec = extracted.get(jd_id, {})
        jd_location = _clean_meta_text(jd_row.get("jd_location")) or ""
        jd_country = jd_location.split(",")[-1].strip() if jd_location else None
        jd_family = _clean_meta_text(jd_row.get("role_family")) or _clean_meta_text(jd_rec.get("role_family")) or "unknown"
        jd_min_years = jd_rec.get("min_years_experience")
        q = row_of[jd_id]
        c_rows = [row_of[c] for c in cand_ids]
        cid_refs = cluster_ids_for(jd_keywords.get(jd_id, []))
        cov = []
        cov_info: list[dict[str, list[str]]] = []
        for cid, row_idx in zip(cand_ids, c_rows):
            cov_val, matched, missing = coverage_for_candidate(
                counts.getrow(row_idx).toarray().ravel(), None, jd_keywords.get(jd_id, []),
                cluster_labels, term_to_cluster, cid_refs)
            cov.append(cov_val)
            cov_info.append({"matched": matched, "missing": missing})
            coverage_rows.append({
                "jd_id": jd_id, "cv_id": cid, "coverage": round(cov_val, 6),
                "matched_must_haves": "|".join(matched),
                "missing_must_haves": "|".join(missing),
            })
        cov = np.array(cov, dtype=float)
        cov_norm = (cov - cov.min()) / (cov.max() - cov.min()) if cov.max() > cov.min() else np.zeros_like(cov)

        scores_by_model: dict[str, np.ndarray] = {}
        # M0 random permutation (one representative, seeded).
        scores_by_model["M0"] = rng.random(len(cand_ids))
        scores_by_model["M1"] = cosine_similarity(Xd_n[[q]], Xd_n[c_rows])[0]
        scores_by_model["M2"] = cosine_similarity(Z_svd[[q]], Z_svd[c_rows])[0]
        scores_by_model["M3"] = cosine_similarity(Z_ae[[q]], Z_ae[c_rows])[0]
        scores_by_model["M4"] = cosine_similarity(Z_meta[[q]], Z_meta[c_rows])[0]
        scores_by_model["M5"] = (float(cfg.get("retrieval.hybrid_ae_weight", 0.8)) * scores_by_model["M3"]
                                 + float(cfg.get("retrieval.hybrid_coverage_weight", 0.2)) * cov_norm)
        same_family = np.array([
            1.0 if document_role_family(meta_by_id.get(cid, {}), extracted.get(cid)) == jd_family else 0.0
            for cid in cand_ids
        ])
        scores_by_model["M6"] = scores_by_model["M3"] + 1.0 * same_family

        for model in MODELS:
            scores = np.asarray(scores_by_model[model], dtype=float)
            order = sorted(range(len(cand_ids)), key=lambda i: (-round(float(scores[i]), 6), cand_ids[i]))
            for rank, i in enumerate(order, start=1):
                cid = cand_ids[i]
                ranking_rows.append({
                    "model": model, "jd_id": jd_id, "cv_id": cid,
                    "rank": rank, "score": round(float(scores[i]), 6),
                    "is_relevant": int(cid == hired[jd_id]),
                })
                if rank <= 10:
                    cand = meta_by_id.get(cid, {})
                    rec = extracted.get(cid, {})
                    info = cov_info[i]
                    matched = list(info["matched"])
                    missing = list(info["missing"])
                    reason = _reason(
                        pd.Series({"rank": rank, "score": scores[i]}), matched, missing,
                        jd_family, document_role_family(cand, rec),
                        rec.get("total_years_experience"), jd_min_years,
                        rec.get("seniority"), extracted.get(jd_id, {}).get("seniority"),
                        jd_country, _clean_meta_text(cand.get("country")),
                        extracted.get(jd_id, {}).get("languages") or [],
                        rec.get("languages") or [],
                        extracted.get(jd_id, {}).get("remote") or "unknown",
                    )
                    top10_rows.append({
                        "model": model, "jd_id": jd_id, "rank": rank, "cv_id": cid,
                        "cv_title": cand.get("title"), "cv_company": cand.get("company"),
                        "role_family": document_role_family(cand, rec),
                        "score": round(float(scores[i]), 6),
                        "is_relevant": int(cid == hired[jd_id]),
                        "must_have_matches": "|".join(matched),
                        "missing_must_haves": "|".join(missing),
                        "reason": reason,
                    })
        # Save per-query coverage for the hired CV.
        hired_cov = [c for c in coverage_rows if c["jd_id"] == jd_id and c["cv_id"] == hired[jd_id]][0]
        reason_rows.append({"jd_id": jd_id, **hired_cov})

    rankings = pd.DataFrame(ranking_rows)
    top10 = pd.DataFrame(top10_rows)
    rankings.to_csv(cfg.processed / "rankings.csv", index=False)
    top10.to_csv(cfg.processed / "top10.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(cfg.processed / "must_have_coverage.csv", index=False)

    # Integrity checks.
    integrity = {}
    for (model, jd), sub in rankings.groupby(["model", "jd_id"]):
        if len(sub) != len(cand_ids):
            integrity[f"{model}:{jd}"] = f"rows={len(sub)}"
        if sorted(sub["rank"].tolist()) != list(range(1, len(cand_ids) + 1)):
            integrity[f"{model}:{jd}"] = "ranks_not_1..N"
        if int(sub["is_relevant"].sum()) != 1:
            integrity[f"{model}:{jd}"] = "relevance_count_not_1"
    if integrity:
        raise RuntimeError(f"Ranking integrity failure: {list(integrity.items())[:5]}")

    summary = {
        "n_models": len(MODELS), "n_queries": len(query_ids),
        "n_candidates": len(cand_ids), "ranking_rows": len(rankings),
        "top10_rows": len(top10), "models": MODEL_NAMES,
    }
    dump_json(cfg.processed / "retrieval_summary.json", summary)
    return {"rankings": rankings, "top10": top10, "coverage": pd.DataFrame(coverage_rows),
            "summary": summary}


def random_baseline(cfg: Config) -> pd.DataFrame:
    gt = pd.read_csv(cfg.processed / "ground_truth.csv")
    docs = read_jsonl(cfg.processed / "documents.jsonl")
    n = sum(1 for d in docs if d["doc_type"] in {"pool_cv", "gt_cv"}) or 110
    rng = np.random.default_rng(cfg.seed)
    perms = int(cfg.get("retrieval.random_permutations", 1000))
    recalls = []
    for _ in range(perms):
        hits = 0
        for _ in range(len(gt)):
            rank = int(rng.integers(1, n + 1))
            hits += int(rank <= 10)
        recalls.append(hits / len(gt))
    df = pd.DataFrame({"permutation": np.arange(1, perms + 1), "recall_at_10": recalls})
    df.to_csv(cfg.processed / "random_baseline.csv", index=False)
    summary = pd.DataFrame([{
        "n_candidates": n, "top_k": 10, "expected_recall_at_10": round(10 / n, 6),
        "empirical_mean": round(float(np.mean(recalls)), 6),
        "p2_5": round(float(np.percentile(recalls, 2.5)), 6),
        "p97_5": round(float(np.percentile(recalls, 97.5)), 6),
    }])
    summary.to_csv(cfg.processed / "random_baseline_summary.csv", index=False)
    return summary
