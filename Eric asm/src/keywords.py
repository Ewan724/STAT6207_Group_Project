"""Stage 2: keyword normalisation, alias merging, embedding and clustering."""
from __future__ import annotations

import json
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from .aliases import ALIAS_TO_CANONICAL, BLOCKED_PAIRS, CURATED_ALIASES, canonical_for, conflicts
from .config import Config, dump_json
from .data import read_jsonl

GENERIC = {
    "experience", "skills", "skill", "work", "team", "ability", "strong", "good",
    "excellent", "communication", "leadership", "hard working", "problem solving",
    "detail oriented", "self motivated", "proactive", "responsible", "various",
    "other", "etc", "and", "the", "with", "for", "from", "using", "including",
    "tools", "technologies", "knowledge", "understanding", "familiarity",
}

PROTECTED_SINGULAR = {
    "kubernetes", "redis", "analytics", "statistics", "devops", "aws", "css", "html",
    "business", "sales", "operations", "finance", "physics", "mathematics", "economics",
    "ethics", "logistics", "robotics", "dynamics", "genomics", "series", "species",
    "success", "process", "access", "accessibility", "class", "business intelligence",
}

SECTION_WEIGHTS = {
    "jd": {
        "must_have_keywords": 3.0,
        "nice_to_have_keywords": 1.0,
        "tools_technologies": 2.0,
        "domain_knowledge": 1.5,
        "key_responsibilities": 0.5,
        "soft_skills": 0.5,
    },
    "cv": {
        "skills": 2.0,
        "tools_technologies": 2.0,
        "domains": 1.5,
        "work_history": 1.0,
        "education": 0.5,
        "certifications": 0.5,
        "soft_skills": 0.5,
    },
}
SHORT_FIELDS = {
    "jd": ["must_have_keywords", "nice_to_have_keywords", "tools_technologies",
           "domain_knowledge", "soft_skills"],
    "cv": ["skills", "tools_technologies", "domains", "soft_skills"],
}


def _nfkc(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value))


def normalise_term(term: str) -> str:
    s = _nfkc(term).casefold().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"^[\-–—•●\s]+|[\-–—•●\s]+$", "", s)
    s = re.sub(r"[/]+", " ", s)
    s = re.sub(r"[^a-z0-9+#.\- ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split()
    if len(words) > 1:
        last = words[-1]
        if (last.endswith("s") and not last.endswith(("ss", "us", "is", "ics"))
                and len(last) > 3 and last not in PROTECTED_SINGULAR
                and s not in PROTECTED_SINGULAR):
            words[-1] = last[:-1]
        s = " ".join(words)
    return s


def display_term(term: str) -> str:
    canonical = canonical_for(term)
    if canonical != term:
        return canonical
    s = normalise_term(term)
    small = {"and", "of", "for", "to", "in", "with", "the", "a", "an"}
    words = s.split()
    out = []
    for i, w in enumerate(words):
        if w in {"api", "sql", "aws", "gcp", "etl", "elt", "crm", "erp", "wms", "oms", "ui", "ux", "seo", "sem", "kpi", "sla", "b2b", "b2c", "ci/cd", "qa"}:
            out.append(w.upper())
        elif i > 0 and w in small:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def term_key(term: str) -> str:
    return normalise_term(canonical_for(term))


def _term_in_text(term: str, text: str) -> bool:
    t = str(text).casefold()
    t = re.sub(r"[\u2010-\u2015]", "-", t)
    n = normalise_term(term)
    if not n:
        return False
    pat = re.escape(n).replace(r"\ ", r"[\s\-]+")
    return bool(re.search(pat, t))


def _doc_type_group(doc_type: str) -> str:
    return "jd" if doc_type in {"gt_jd", "unlabelled_jd"} else "cv"


def _field_values(rec: dict, field: str) -> list[str]:
    value = rec.get(field)
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in re.split(r"[,;]|\n", value) if x.strip()]
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                out.extend(str(v) for v in item.values() if isinstance(v, str))
            else:
                out.append(str(item))
        return [x.strip() for x in out if x.strip()]
    return []


def load_extractions(cfg: Config) -> dict[str, dict]:
    out = {}
    for path in [cfg.processed / "jd_extracted.jsonl", cfg.processed / "cv_extracted.jsonl"]:
        for rec in read_jsonl(path):
            out[rec["doc_id"]] = rec
    return out


def _training_ids(cfg: Config) -> tuple[set[str], set[str]]:
    docs = read_jsonl(cfg.processed / "documents.jsonl")
    excluded = set()
    dedupe_path = cfg.processed / "jd_dedupe_report.csv"
    if dedupe_path.exists():
        d = pd.read_csv(dedupe_path)
        excluded = set(d.loc[d["decision"] == "exclude_from_training", "jd_id"])
    training = {d["doc_id"] for d in docs if d["doc_type"] in {"pool_cv", "unlabelled_jd"} and d["doc_id"] not in excluded}
    heldout = {d["doc_id"] for d in docs if d["doc_type"] in {"gt_jd", "gt_cv"}}
    return training, heldout


def build_term_stats(cfg: Config, docs: list[dict], extracted: dict[str, dict]) -> pd.DataFrame:
    training, _ = _training_ids(cfg)
    stats: dict[str, dict] = {}
    for doc in docs:
        if doc["doc_id"] not in training or doc["doc_id"] not in extracted:
            continue
        rec = extracted[doc["doc_id"]]
        group = _doc_type_group(doc["doc_type"])
        for field in SHORT_FIELDS[group]:
            for raw in _field_values(rec, field):
                key = term_key(raw)
                if not key or len(key) < 2:
                    continue
                words = key.split()
                if len(words) > 5 or key in GENERIC:
                    continue
                entry = stats.setdefault(key, {
                    "keyword": display_term(raw),
                    "canonical_seed": canonical_for(raw),
                    "doc_frequency": 0,
                    "total_frequency": 0,
                    "doc_types": set(),
                    "example_doc_ids": [],
                    "_docs": set(),
                    "in_must_have": False,
                    "in_nice_to_have": False,
                    "in_cv_skills": False,
                    "in_tools": False,
                })
                entry["total_frequency"] += 1
                entry["doc_types"].add(doc["doc_type"])
                if doc["doc_id"] not in entry["_docs"]:
                    entry["_docs"].add(doc["doc_id"])
                    entry["doc_frequency"] += 1
                    if len(entry["example_doc_ids"]) < 5:
                        entry["example_doc_ids"].append(doc["doc_id"])
                if field == "must_have_keywords":
                    entry["in_must_have"] = True
                if field == "nice_to_have_keywords":
                    entry["in_nice_to_have"] = True
                if field == "skills":
                    entry["in_cv_skills"] = True
                if field == "tools_technologies":
                    entry["in_tools"] = True
    rows = []
    for key, entry in stats.items():
        entry = dict(entry)
        entry.pop("_docs", None)
        entry["key"] = key
        entry["doc_types"] = "|".join(sorted(entry["doc_types"]))
        entry["example_doc_ids"] = "|".join(entry["example_doc_ids"])
        rows.append(entry)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["doc_frequency", "total_frequency"], ascending=False).reset_index(drop=True)
    return df


def embed_terms(terms: list[str], cfg: Config) -> tuple[np.ndarray, str, Any, Any]:
    """Try sentence-transformers, then char TF-IDF + SVD."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(str(cfg.get("clustering.st_model")))
        emb = model.encode(terms, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32), "sentence-transformers", model, None
    except Exception:
        pass
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(terms)
    n_comp = min(int(cfg.get("clustering.svd_components", 100)), max(2, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=cfg.seed)
    emb = svd.fit_transform(X)
    # L2 normalise the reduced embeddings.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-9)
    return emb.astype(np.float32), "char-tfidf+svd", vec, svd


def _split_conflicting_clusters(labels: np.ndarray, terms: list[str], emb: np.ndarray) -> np.ndarray:
    """Split any cluster containing a blocked technology pair."""
    labels = labels.copy()
    next_label = int(labels.max()) + 1
    for lab in sorted(set(labels.tolist())):
        idx = np.where(labels == lab)[0]
        if len(idx) < 2:
            continue
        has_conflict = False
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if conflicts(terms[idx[i]], terms[idx[j]]):
                    has_conflict = True
                    break
            if has_conflict:
                break
        if not has_conflict:
            continue
        # Greedy split: put each conflicting member into its own singleton.
        keep = []
        for i in idx:
            if any(conflicts(terms[i], terms[k]) for k in keep):
                labels[i] = next_label
                next_label += 1
            else:
                keep.append(i)
    # Renumber contiguous.
    mapping = {old: new for new, old in enumerate(sorted(set(labels.tolist())))}
    return np.array([mapping[x] for x in labels], dtype=int)


def fit_clusters(cfg: Config, terms: list[str], emb: np.ndarray) -> tuple[np.ndarray, dict]:
    grid = [float(x) for x in cfg.get("clustering.distance_thresholds", [0.4, 0.5, 0.6])]
    min_c = int(cfg.get("clustering.min_clusters", 100))
    max_c = int(cfg.get("clustering.max_clusters", 200))
    candidates = []
    for t in grid:
        if len(terms) < 2:
            labels = np.zeros(len(terms), dtype=int)
            n = 1
        else:
            model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=t, metric="cosine", linkage="average"
            )
            labels = model.fit_predict(emb)
            n = len(set(labels.tolist()))
        candidates.append((t, labels, n))
    in_range = [c for c in candidates if min_c <= c[2] <= max_c]
    if in_range:
        chosen = min(in_range, key=lambda c: abs(c[2] - 150))
    else:
        chosen = min(candidates, key=lambda c: min(abs(c[2] - min_c), abs(c[2] - max_c)))
    threshold, labels, n = chosen
    labels = _split_conflicting_clusters(labels, terms, emb)
    metrics = {
        "distance_threshold": threshold,
        "n_clusters_before_guardrail": int(n),
        "n_clusters": int(len(set(labels.tolist()))),
        "n_terms": len(terms),
        "grid": [{"threshold": t, "n_clusters": c} for t, _, c in candidates],
    }
    return labels, metrics


def _cluster_centroids(emb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n = int(labels.max()) + 1
    cent = np.zeros((n, emb.shape[1]), dtype=np.float32)
    for k in range(n):
        idx = np.where(labels == k)[0]
        if len(idx):
            cent[k] = emb[idx].mean(axis=0)
    norm = np.linalg.norm(cent, axis=1, keepdims=True)
    return cent / np.maximum(norm, 1e-9)


def _label_cluster(members: list[str], stats: pd.DataFrame) -> tuple[str, str]:
    # Prefer the alias canonical form, then the most frequent member.
    for m in members:
        if m in ALIAS_TO_CANONICAL.values():
            return m, f"Skills and terms related to {m}."
    rows = stats[stats["key"].isin([term_key(m) for m in members])].sort_values("doc_frequency", ascending=False)
    if len(rows):
        label = str(rows.iloc[0]["keyword"])
        return label, f"Skills and terms related to {label}."
    return (members[0] if members else "Unknown"), "Unlabelled keyword cluster."


def build_clusters(cfg: Config, force: bool = False) -> dict[str, Any]:
    docs = read_jsonl(cfg.processed / "documents.jsonl")
    extracted = load_extractions(cfg)
    if len(extracted) < 150:
        raise RuntimeError(f"Only {len(extracted)} extractions found; finish Stage 1 first.")
    stats = build_term_stats(cfg, docs, extracted)
    if not len(stats):
        raise RuntimeError("No keywords extracted from the training corpus.")
    stats.to_csv(cfg.processed / "keyword_table.csv", index=False)

    terms = stats["key"].tolist()
    displays = stats["keyword"].tolist()
    emb, method, fitted_vec, fitted_svd = embed_terms(displays, cfg)
    labels, cluster_metrics = fit_clusters(cfg, terms, emb)
    centroids = _cluster_centroids(emb, labels)

    # Cluster sizes and labels.
    clusters = []
    for lab in sorted(set(labels.tolist())):
        idx = np.where(labels == lab)[0]
        members = [displays[i] for i in idx]
        keys = [terms[i] for i in idx]
        sub = stats[stats["key"].isin(keys)]
        label, definition = _label_cluster(members, stats)
        if len(idx) > 1:
            sims = cosine_similarity(emb[idx])
            iu = np.triu_indices(len(idx), k=1)
            coherence = float(sims[iu].mean()) if len(iu[0]) else 1.0
        else:
            coherence = 1.0
        clusters.append({
            "cluster_id": int(lab),
            "canonical_label": label,
            "definition": definition,
            "members": "|".join(sorted(set(members))),
            "size": int(len(idx)),
            "doc_frequency": int(sub["doc_frequency"].sum()) if len(sub) else 0,
            "total_frequency": int(sub["total_frequency"].sum()) if len(sub) else 0,
            "coherence": round(coherence, 4),
            "merge_method": "alias+agglomerative_cosine",
            "confidence": round(max(0.0, min(1.0, coherence)), 4),
        })
    cluster_df = pd.DataFrame(clusters).sort_values(["size", "doc_frequency"], ascending=False).reset_index(drop=True)
    cluster_df.to_csv(cfg.processed / "keyword_clusters.csv", index=False)

    # Aliases.
    alias_rows = []
    for canonical, variants in CURATED_ALIASES.items():
        for v in variants:
            alias_rows.append({
                "raw_term": v, "canonical_term": canonical,
                "source": "curated", "confidence": 1.0,
                "notes": "hand-curated; distinct technologies never merged",
            })
    pd.DataFrame(alias_rows).to_csv(cfg.processed / "keyword_aliases.csv", index=False)

    # Mapping keyword -> cluster.
    term_to_cluster = {terms[i]: int(labels[i]) for i in range(len(terms))}
    dump_json(cfg.processed / "keyword_to_cluster.json", term_to_cluster)

    # Co-occurrence edges.
    training, _ = _training_ids(cfg)
    presence: dict[int, set[str]] = defaultdict(set)
    for doc in docs:
        if doc["doc_id"] not in training or doc["doc_id"] not in extracted:
            continue
        rec = extracted[doc["doc_id"]]
        group = _doc_type_group(doc["doc_type"])
        seen = set()
        for field in SHORT_FIELDS[group]:
            for raw in _field_values(rec, field):
                k = term_key(raw)
                if k in term_to_cluster:
                    seen.add(term_to_cluster[k])
        for c in seen:
            presence[c].add(doc["doc_id"])
    edge_counter: Counter = Counter()
    clist = sorted(presence)
    for i in range(len(clist)):
        for j in range(i + 1, len(clist)):
            n = len(presence[clist[i]] & presence[clist[j]])
            if n >= 2:
                edge_counter[(clist[i], clist[j])] = n
    label_map = dict(zip(cluster_df["cluster_id"], cluster_df["canonical_label"]))
    edges = pd.DataFrame([
        {"source": s, "target": t, "weight": w,
         "source_label": label_map.get(s, str(s)), "target_label": label_map.get(t, str(t))}
        for (s, t), w in edge_counter.most_common()
    ])
    edges.to_csv(cfg.processed / "keyword_edges.csv", index=False)

    # Cluster quality.
    sil = None
    try:
        sample_n = min(2000, len(terms))
        if len(set(labels.tolist())) > 1:
            sil = float(silhouette_score(emb[:sample_n], labels[:sample_n], metric="cosine"))
    except Exception:
        sil = None
    singleton_rate = float((cluster_df["size"] == 1).mean()) if len(cluster_df) else 0.0
    guardrails = {}
    for a, b in BLOCKED_PAIRS:
        mixed = 0
        for _, row in cluster_df.iterrows():
            members = str(row["members"]).split("|")
            if any(x != y and conflicts(x, y) for x in members for y in members):
                mixed += 1
        guardrails[f"{a}__vs__{b}"] = int(mixed)
    metrics = {
        **cluster_metrics,
        "embedding_method": method,
        "singleton_rate": round(singleton_rate, 4),
        "silhouette_cosine": round(sil, 4) if sil is not None else None,
        "mean_coherence": round(float(cluster_df["coherence"].mean()), 4),
        "guardrail_mixed_clusters": guardrails,
        "n_aliases": len(alias_rows),
        "training_documents": len(training),
    }
    dump_json(cfg.processed / "cluster_metrics.json", metrics)

    # QA table: largest and lowest-coherence clusters.
    top = cluster_df.head(20).copy()
    top["qa_reason"] = "largest_cluster"
    low = cluster_df.sort_values("coherence").head(20).copy()
    low["qa_reason"] = "lowest_coherence"
    qa = pd.concat([top, low]).drop_duplicates("cluster_id")
    qa["manual_review"] = ""
    qa["notes"] = ""
    qa.to_csv(cfg.processed / "cluster_qa.csv", index=False)

    # Reproducible merge-review sample of 30 pairs.
    rng = np.random.default_rng(cfg.seed)
    pairs = []
    all_terms = list(range(len(terms)))
    while len(pairs) < 30 and len(all_terms) > 1:
        a, b = rng.choice(all_terms, size=2, replace=False)
        pair = tuple(sorted((int(a), int(b))))
        if pair in pairs:
            continue
        pairs.append(pair)
    review = []
    for a, b in pairs:
        review.append({
            "term_a": displays[a], "term_b": displays[b],
            "same_cluster": bool(labels[a] == labels[b]),
            "cosine": round(float(cosine_similarity(emb[[a]], emb[[b]])[0, 0]), 4),
            "manual_label": "", "notes": "",
        })
    pd.DataFrame(review).to_csv(cfg.processed / "merge_review_sample.csv", index=False)

    # Persist the fitted objects and centroids.
    models_dir = cfg.models
    with (models_dir / "cluster_model.pkl").open("wb") as fh:
        pickle.dump({
            "vectorizer": fitted_vec, "svd": fitted_svd, "method": method,
            "terms": terms, "displays": displays, "labels": labels,
            "centroids": centroids, "unmatched_threshold": float(cfg.get("clustering.unmatched_cosine_threshold", 0.45)),
        }, fh)

    result = {
        "cluster_df": cluster_df,
        "labels": labels,
        "terms": terms,
        "displays": displays,
        "embeddings": emb,
        "centroids": centroids,
        "term_to_cluster": term_to_cluster,
        "metrics": metrics,
    }
    build_document_matrix(cfg, docs, extracted, result)
    return result


def map_term(term: str, term_to_cluster: dict[str, int], centroids: np.ndarray,
             vectorizer: Any, svd: Any, threshold: float) -> tuple[int | None, float]:
    key = term_key(term)
    if key in term_to_cluster:
        return term_to_cluster[key], 1.0
    try:
        if vectorizer is None or svd is None:
            return None, 0.0
        X = vectorizer.transform([display_term(term)])
        v = svd.transform(X)
        v = v / max(np.linalg.norm(v), 1e-9)
        sims = centroids @ v[0]
        best = int(np.argmax(sims))
        score = float(sims[best])
        return (best, score) if score >= threshold else (None, score)
    except Exception:
        return None, 0.0


def build_document_matrix(cfg: Config, docs: list[dict], extracted: dict[str, dict], result: dict) -> None:
    from scipy import sparse

    cluster_df = result["cluster_df"]
    term_to_cluster = result["term_to_cluster"]
    centroids = result["centroids"]
    with (cfg.models / "cluster_model.pkl").open("rb") as fh:
        model_blob = pickle.load(fh)
    vectorizer, svd = model_blob.get("vectorizer"), model_blob.get("svd")
    threshold = float(cfg.get("clustering.unmatched_cosine_threshold", 0.45))
    n_clusters = int(cluster_df["cluster_id"].max()) + 1
    rows, cols, vals = [], [], []
    coverage_rows = []
    training, heldout = _training_ids(cfg)
    doc_order = []

    for r, doc in enumerate(docs):
        doc_order.append(doc)
        rec = extracted.get(doc["doc_id"], {})
        group = _doc_type_group(doc["doc_type"])
        counts: dict[int, float] = defaultdict(float)
        total = 0
        matched = 0
        unmatched_terms = []
        for field in SHORT_FIELDS[group]:
            weight = SECTION_WEIGHTS[group].get(field, 0.0)
            for raw in _field_values(rec, field):
                total += 1
                cid, score = map_term(raw, term_to_cluster, centroids, vectorizer, svd, threshold)
                if cid is None:
                    unmatched_terms.append(raw)
                else:
                    matched += 1
                    counts[cid] += weight
        # Section weight for longer free text.
        text_fields = ["key_responsibilities"] if group == "jd" else ["work_history", "education", "certifications"]
        for field in text_fields:
            weight = SECTION_WEIGHTS[group].get(field, 0.0)
            text = " ".join(_field_values(rec, field))
            if not text:
                continue
            seen = set()
            for key, cid in term_to_cluster.items():
                if cid in seen:
                    continue
                if _term_in_text(key, text):
                    counts[cid] += weight * 0.5
                    seen.add(cid)
        for cid, value in counts.items():
            rows.append(r)
            cols.append(cid)
            vals.append(np.log1p(value))
        if doc["doc_id"] in heldout:
            coverage_rows.append({
                "doc_id": doc["doc_id"], "doc_type": doc["doc_type"],
                "total_keywords": total, "matched": matched,
                "unmatched": total - matched,
                "matched_rate": round(matched / total, 4) if total else 0.0,
                "unmatched_terms": "|".join(sorted(set(unmatched_terms))),
            })

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), n_clusters), dtype=np.float32)
    train_idx = [i for i, d in enumerate(docs) if d["doc_id"] in training]
    # `vals` already holds log1p(section-weighted count). Applying sklearn's
    # sublinear_tf on top would take log() again and drive weights below 1
    # negative; plain TF-IDF on the log-scaled counts stays non-negative.
    transformer = TfidfTransformer(sublinear_tf=False, norm="l2")
    transformer.fit(X[train_idx])
    X_tfidf = transformer.transform(X).astype(np.float32)
    sparse.save_npz(cfg.processed / "document_keyword_matrix.npz", X_tfidf)
    sparse.save_npz(cfg.processed / "document_keyword_counts.npz", X)

    index = pd.DataFrame([{
        "row": i, "doc_id": d["doc_id"], "doc_type": d["doc_type"],
        "split": d["split"], "is_training": d["doc_id"] in training,
        "is_heldout": d["doc_id"] in heldout,
        "title": d.get("title"), "role_family": d.get("role_family"),
    } for i, d in enumerate(docs)])
    index.to_csv(cfg.processed / "document_matrix_index.csv", index=False)

    dump_json(cfg.models / "vectorizer.json", {
        "cluster_ids": cluster_df["cluster_id"].tolist(),
        "cluster_labels": cluster_df["canonical_label"].tolist(),
        "idf": transformer.idf_.tolist(),
        "tfidf": {"sublinear_tf": False, "norm": "l2", "input": "log1p(section_weighted_count)"},
        "section_weights": SECTION_WEIGHTS,
        "document_order": [d["doc_id"] for d in docs],
        "n_clusters": n_clusters,
        "embedding_method": result["metrics"]["embedding_method"],
    })

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(cfg.processed / "heldout_keyword_coverage.csv", index=False)
    if len(coverage):
        overall = {
            "heldout_documents": int(len(coverage)),
            "total_keywords": int(coverage["total_keywords"].sum()),
            "matched": int(coverage["matched"].sum()),
            "unmatched": int(coverage["unmatched"].sum()),
            "matched_rate": round(float(coverage["matched"].sum() / max(1, coverage["total_keywords"].sum())), 4),
        }
    else:
        overall = {"heldout_documents": 0, "matched_rate": 0.0}
    dump_json(cfg.processed / "heldout_coverage_summary.json", overall)
    result["coverage_summary"] = overall
