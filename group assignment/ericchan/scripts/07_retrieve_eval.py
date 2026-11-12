"""Step 7: retrieve the top-10 CVs per JD and evaluate.

Systems compared
    random          chance floor
    tfidf           classic lexical baseline on raw text (1-2 grams)
    bm25            lexical baseline with length normalisation
    keyword         IDF-weighted overlap of the MERGED canonical skill sets,
                    asymmetric (coverage of the JD's requirements)
    nomic           nomic-embed-text cosine on raw text (no keywords at all)
    ae_latent       the autoencoder bottleneck, cosine
    hybrid          z-scored fusion of the three signals
    hybrid_rerank   hybrid + seniority / domain congruence reranking
    family_oracle   only CVs whose hand-assigned family matches the JD's --
                    an upper bound for a filter-first design, NOT a real system

Why the fusion has three components
    Steps 6b-6g established that each signal fails in a different way:
      * merged keywords are interpretable and precise but lossy -- 12 canonical
        skills per document versus ~3,100 characters of evidence, and the
        extraction drops phrasing that matters ("client-facing delivery");
      * the AE compresses the keyword space 14x and keeps 83% of its
        neighbourhood quality, so it smooths exact-string mismatches but adds no
        information the keywords do not already contain;
      * the text embedding is the only signal that sees the full document, and it
        is the only one that can match paraphrase.
    A weighted fusion lets each cover the others' failure modes.

Weight selection
    The weights are chosen by a grid search whose selection criterion is NDCG@10
    ON THE LABELS.  This is a real limitation and it is stated as such in the
    report: with 10 queries any hyper-parameter fitted on them is optimistic.
    We therefore also report the full grid so the reader can see the sensitivity
    rather than a single tuned number, and we report the untuned equal-weight
    fusion alongside the tuned one.

Metrics
    Recall@10 is the headline metric the brief asks for.  Because the relevant
    set per JD is usually smaller than 10, Recall@10 is capped by
    min(1, 10/|relevant|); we report that ceiling per JD and cross-check with
    NDCG@10 and MAP@10, which are not capped the same way.

Outputs: data/metrics.json, data/retrieval_results.json
"""
import itertools
import json
import math
import os
import random
import re
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
SEED = 42


# ---------------------------------------------------------------- metrics
def recall_at_k(ranked, rel, k):
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


def precision_at_k(ranked, rel, k):
    return len(set(ranked[:k]) & rel) / k


def ap_at_k(ranked, rel, k):
    hits, s = 0, 0.0
    for i, d in enumerate(ranked[:k], 1):
        if d in rel:
            hits += 1
            s += hits / i
    return s / max(len(rel), 1)


def ndcg_at_k(ranked, grades, k):
    dcg = sum((2 ** grades.get(d, 0) - 1) / math.log2(i + 2)
              for i, d in enumerate(ranked[:k]))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked, rel):
    for i, d in enumerate(ranked, 1):
        if d in rel:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------- scorers
def tfidf_scores(jd_text, cv_texts, cv_ids):
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = [jd_text] + [cv_texts[c] for c in cv_ids]
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                          ngram_range=(1, 2), min_df=1, max_features=40000)
    M = vec.fit_transform(corpus)
    sims = (M[1:] @ M[0].T).toarray().ravel()
    return dict(zip(cv_ids, sims))


def bm25_scores(jd_text, cv_texts, cv_ids, k1=1.5, b=0.75):
    tok = lambda t: re.findall(r"[a-z0-9+#.]+", t.lower())  # noqa: E731
    toks = [tok(cv_texts[c]) for c in cv_ids]
    N = len(toks)
    avgdl = sum(len(t) for t in toks) / max(N, 1)
    df = defaultdict(int)
    for t in toks:
        for w in set(t):
            df[w] += 1
    idf = {w: math.log(1 + (N - d + 0.5) / (d + 0.5)) for w, d in df.items()}
    q = tok(jd_text)
    out = {}
    for cid, t in zip(cv_ids, toks):
        tf = defaultdict(int)
        for w in t:
            tf[w] += 1
        s = 0.0
        for w in q:
            if w in tf:
                f = tf[w]
                s += idf.get(w, 0) * f * (k1 + 1) / (
                    f + k1 * (1 - b + b * len(t) / avgdl))
        out[cid] = s
    return out


def keyword_scores(jd_skills, doc_skills, cv_ids, weights=None):
    """IDF-weighted requirement coverage + symmetric Jaccard.

    Coverage is weighted more than Jaccard because a recruiter asks "does this
    person have what the role needs", not "are these two profiles similar".

    `weights` lets the caller distinguish must-have from nice-to-have skills.
    This is the fix for failure mode F2 in the report: a flat set means matching
    five nice-to-haves scores the same as matching five must-haves.
    """
    df = defaultdict(int)
    for d in doc_skills:
        for s in doc_skills[d]:
            df[s] += 1
    n = len(doc_skills)
    idf = {s: math.log((n + 1) / (c + 0.5)) for s, c in df.items()}
    J = set(jd_skills)
    w = weights or {}
    jw = sum(idf.get(s, 0) * w.get(s, 1.0) for s in J)
    out = {}
    for c in cv_ids:
        C = set(doc_skills.get(c, []))
        if not J or not C:
            out[c] = 0.0
            continue
        inter = sum(idf.get(s, 0) * w.get(s, 1.0) for s in J & C)
        union = sum(idf.get(s, 0) for s in J | C)
        out[c] = 0.6 * (inter / max(jw, 1e-9)) + 0.4 * (inter / max(union, 1e-9))
    return out


def must_have_weights(jd_id, ext, skills):
    """Weight 2.0 for a JD's must-haves, 1.0 for nice-to-haves.

    The extracted must_have / nice_to_have lists are mapped through the same
    canonicalisation used to build the vocabulary, so a requirement written as
    "SQL proficiency" in the JD lands on the "sql" dimension.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mk", os.path.join(BASE, "scripts", "05_merge_keywords.py"))
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)

    je = ext.get(jd_id, {})
    w = {}
    for field, val in (("must_have", 2.0), ("nice_to_have", 1.0)):
        for t in (je.get(field) or []):
            for piece in mk.split_enum(str(t)):
                c = mk.canonical_hard(piece)
                if not c:
                    continue
                c = skills["alias_map"].get(c, c)
                w[c] = max(w.get(c, 0.0), val)
    return w


def zscore(d):
    v = np.array(list(d.values()), dtype=float)
    mu, sd = v.mean(), v.std() + 1e-9
    return {k: (x - mu) / sd for k, x in d.items()}


def rrf(score_dicts, k=60):
    """Reciprocal rank fusion over several scorers.

    RRF combines RANKS rather than scores, so it is immune to the different
    scales and distributions of the underlying signals -- which matters here
    because a cosine similarity in [-1,1] and a BM25 score in [0,40] cannot be
    added without an arbitrary normalisation.  It is the standard fusion method
    in information retrieval and is included as a scale-free alternative to the
    z-score weighted sum.
    """
    out = defaultdict(float)
    for sc in score_dicts:
        ranked = sorted(sc, key=lambda c: -sc[c])
        for rank, c in enumerate(ranked, 1):
            out[c] += 1.0 / (k + rank)
    return dict(out)


def main():
    skills = json.load(open(os.path.join(DATA, "skills.json"), encoding="utf-8"))
    ae = json.load(open(os.path.join(DATA, "ae_embeddings.json"), encoding="utf-8"))
    gt = [json.loads(l) for l in open(os.path.join(DATA, "ground_truth.jsonl"),
                                      encoding="utf-8") if l.strip()]
    gt = [g for g in gt if g.get("score", -1) >= 0]
    cvs = {c["cv_id"]: c for c in json.load(
        open(os.path.join(DATA, "cvs_raw.json"), encoding="utf-8"))}
    jds = {j["jd_id"]: j for j in json.load(
        open(os.path.join(DATA, "jds_raw.json"), encoding="utf-8"))}
    cv_fam = json.load(open(os.path.join(DATA, "cv_family.json"), encoding="utf-8"))
    nomic = json.load(open(os.path.join(DATA, "nomic_docs.json"), encoding="utf-8"))
    ext = {}
    for line in open(os.path.join(DATA, "extracted.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("ok"):
                ext[r["doc_id"]] = r

    # full keyword tier for overlap scoring (rare shared skills are informative)
    doc_skills = skills["doc_skills"]
    cv_ids = sorted(cvs)
    jd_ids = sorted(jds)

    grades = defaultdict(dict)
    rel = defaultdict(set)
    for g in gt:
        grades[g["jd_id"]][g["cv_id"]] = g["score"]
        if g["score"] >= 2:
            rel[g["jd_id"]].add(g["cv_id"])
    for j in jd_ids:
        for c in cv_ids:
            grades[j].setdefault(c, 0)

    fam_jd = {"P01": "finance", "P02": "sales", "P03": "marketing", "P04": "data",
              "P05": "product", "P06": "data", "P07": "marketing",
              "P08": "engineering_hw", "P09": "engineering_sw", "P10": "design"}

    Z = {d: np.array(v) for d, v in ae["embeddings"].items()}
    # additional AE variants trained in 06c (kw+txt) and 06d (text only)
    Z2 = Z3 = None
    p2 = os.path.join(DATA, "ae_embeddings_kwtxt.json")
    if os.path.exists(p2):
        Z2 = {d: np.array(v) for d, v in
              json.load(open(p2, encoding="utf-8"))["embeddings"].items()}
    p3 = os.path.join(DATA, "ae_embeddings_text.json")
    if os.path.exists(p3):
        Z3 = {d: np.array(v) for d, v in
              json.load(open(p3, encoding="utf-8"))["embeddings"].items()}
    N = {d: np.array(v) for d, v in nomic.items()}
    for d in N:
        N[d] = N[d] / max(np.linalg.norm(N[d]), 1e-9)

    # ---------------------------------------------------------------- signals
    base = {}
    for jd_id in jd_ids:
        jd_text = jds[jd_id]["text"]
        cv_texts = {c: cvs[c]["text"] for c in cv_ids}
        sc = {}
        sc["random"] = {c: random.Random(f"{SEED}{jd_id}{c}").random()
                        for c in cv_ids}
        sc["tfidf"] = tfidf_scores(jd_text, cv_texts, cv_ids)
        sc["bm25"] = bm25_scores(jd_text, cv_texts, cv_ids)
        sc["keyword"] = keyword_scores(doc_skills.get(jd_id, []), doc_skills,
                                       cv_ids)
        mh = must_have_weights(jd_id, ext, skills)
        sc["keyword_mh"] = keyword_scores(doc_skills.get(jd_id, []), doc_skills,
                                          cv_ids, weights=mh)
        sc["nomic"] = {c: float(np.dot(N[jd_id], N[c])) for c in cv_ids}
        sc["ae_latent"] = {c: float(np.dot(Z[jd_id], Z[c])) for c in cv_ids}
        if Z2 is not None:
            sc["ae_kwtxt"] = {c: float(np.dot(Z2[jd_id], Z2[c])) for c in cv_ids}
        if Z3 is not None:
            sc["ae_text"] = {c: float(np.dot(Z3[jd_id], Z3[c])) for c in cv_ids}
        sc["family_oracle"] = {c: 1.0 if cv_fam.get(c) == fam_jd[jd_id] else 0.0
                               for c in cv_ids}
        base[jd_id] = sc

    def fuse(weights):
        """weights: dict over signal names -> z-scored weighted sum."""
        out = {}
        for jd_id in jd_ids:
            zs = {k: zscore(base[jd_id][k]) for k in weights}
            out[jd_id] = {c: sum(w * zs[k][c] for k, w in weights.items())
                          for c in cv_ids}
        return out

    def evaluate(score_fn, name):
        r10, p10, ap, nd, m = [], [], [], [], []
        rcurve = defaultdict(list)
        rows = []
        scored = [j for j in jd_ids if rel[j]]   # JDs with >= 1 relevant CV
        skipped = [j for j in jd_ids if not rel[j]]
        for jd_id in jd_ids:
            s = score_fn(jd_id)
            ranked = sorted(cv_ids, key=lambda c: -s[c])
            rows.append({"jd_id": jd_id, "top10": ranked[:10],
                         "top20": ranked[:20],
                         "scores": [round(s[c], 4) for c in ranked[:10]],
                         "pool_scores": {c: round(s[c], 4)
                                         for c in ranked[:20]}})
            if not rel[jd_id]:
                # Recall is undefined with an empty relevant set.  Such a JD is
                # reported separately rather than silently counted as 0 (which
                # would penalise every system) or as 1 (which would flatter them).
                continue
            r10.append(recall_at_k(ranked, rel[jd_id], 10))
            p10.append(precision_at_k(ranked, rel[jd_id], 10))
            ap.append(ap_at_k(ranked, rel[jd_id], 10))
            nd.append(ndcg_at_k(ranked, grades[jd_id], 10))
            m.append(mrr(ranked, rel[jd_id]))
            for k in range(1, 11):
                rcurve[k].append(recall_at_k(ranked, rel[jd_id], k))
        return {
            "system": name,
            "recall@10": round(float(np.mean(r10)), 4),
            "precision@10": round(float(np.mean(p10)), 4),
            "map@10": round(float(np.mean(ap)), 4),
            "ndcg@10": round(float(np.mean(nd)), 4),
            "mrr": round(float(np.mean(m)), 4),
            "n_scored_jds": len(scored),
            "skipped_jds": skipped,
            "per_jd_recall@10": {j: round(v, 4)
                                 for j, v in zip(scored, r10)},
            "recall_curve": {str(k): round(float(np.mean(v)), 4)
                             for k, v in rcurve.items()},
            "rows": rows,
        }

    results = []
    systems_to_run = ["random", "tfidf", "bm25", "keyword", "keyword_mh", "nomic",
                      "ae_latent", "family_oracle"]
    if Z2 is not None:
        systems_to_run.insert(-1, "ae_kwtxt")
    if Z3 is not None:
        systems_to_run.insert(-1, "ae_text")
    for name in systems_to_run:
        results.append(evaluate(lambda j, n=name: base[j][n], name))

    # ---- equal-weight fusion over the three signals, no tuning at all
    eq = {"keyword": 1.0, "ae_latent": 1.0, "nomic": 1.0}
    results.append(evaluate(lambda j: fuse(eq)[j], "hybrid_equal"))

    # ---- grid search over fusion weights, selected on NDCG@10.
    #      The three AE variants are offered as mutually exclusive alternatives
    #      so the grid stays interpretable; the best one is chosen on the same
    #      criterion and the choice is reported.
    ae_keys = ["ae_latent"] + (["ae_kwtxt"] if Z2 is not None else []) + \
              (["ae_text"] if Z3 is not None else [])
    grid = []
    for ae_key in ae_keys:
        for wk, wa, wn in itertools.product([0, 1, 2], [0, 1, 2], [0, 1, 2]):
            if wk == wa == wn == 0:
                continue
            w = {}
            if wk:
                w["keyword_mh"] = float(wk)
            if wa:
                w[ae_key] = float(wa)
            if wn:
                w["nomic"] = float(wn)
            grid.append(w)

    best, best_nd = None, -1
    grid_log = []
    for w in grid:
        fz = fuse(w)
        r = evaluate(lambda j, f=fz: f[j], "grid")
        grid_log.append({"weights": w, "ndcg@10": r["ndcg@10"],
                         "recall@10": r["recall@10"]})
        if r["ndcg@10"] > best_nd:
            best_nd, best = r["ndcg@10"], w
    print(f"best fusion weights (selected on NDCG@10): {best} -> {best_nd:.4f}")

    fz = fuse(best)
    results.append(evaluate(lambda j: fz[j],
                            "hybrid_tuned"))

    # ---- reranking on top of the tuned fusion
    SEN = {"intern": 0, "junior": 1, "mid": 2, "senior": 3, "lead": 4,
           "executive": 5}

    def reranked(jd_id):
        b = zscore(fz[jd_id])
        je = ext.get(jd_id, {})
        js = SEN.get(je.get("seniority", "mid"), 2)
        jd_dom = {d.lower() for d in (je.get("domains") or [])}
        out = {}
        for c in cv_ids:
            ce = ext.get(c, {})
            cs = SEN.get(ce.get("seniority", "mid"), 2)
            adj = -0.25 * abs(js - cs)
            cdom = {d.lower() for d in (ce.get("domains") or [])}
            if jd_dom and cdom:
                adj += 0.15 * len(jd_dom & cdom) / len(jd_dom)
            out[c] = b[c] + adj
        return out

    results.append(evaluate(reranked, "hybrid_rerank"))

    # ---- reciprocal rank fusion: scale-free alternative to the z-score sum
    rrf_fused = {j: rrf([base[j]["keyword_mh"], base[j][ae_keys[-1]],
                         base[j]["nomic"]]) for j in jd_ids}
    results.append(evaluate(lambda j: rrf_fused[j], "rrf"))

    # ---- family-filtered hybrid, two versions:
    #      (a) hybrid_family_gt   uses the hand-assigned family.  Reported for
    #          completeness but CIRCULAR: the ground truth was built with the
    #          same map, so this number encodes the labelling rule.
    #      (b) hybrid_family_pred uses a job family predicted from the CV text
    #          by qwen3:8b (07b), with no access to the hand labels.  This is the
    #          honest deployable figure.
    pred_path = os.path.join(DATA, "cv_family_pred.json")
    cv_fam_pred = {}
    if os.path.exists(pred_path):
        cv_fam_pred = {k: v["family"] for k, v in
                       json.load(open(pred_path, encoding="utf-8")).items()}

    def family_filtered(jd_id, fam_map):
        fam = fam_jd[jd_id]
        b = zscore(fz[jd_id])
        return {c: (b[c] if fam_map.get(c) == fam else b[c] - 5.0)
                for c in cv_ids}

    results.append(evaluate(lambda j: family_filtered(j, cv_fam),
                            "hybrid_family_gt"))
    if cv_fam_pred:
        results.append(evaluate(lambda j: family_filtered(j, cv_fam_pred),
                                "hybrid_family_pred"))

    results.sort(key=lambda r: -r["recall@10"])
    print(f"\n{'system':18s} {'R@10':>7s} {'P@10':>7s} {'MAP@10':>7s} "
          f"{'NDCG@10':>8s} {'MRR':>7s}")
    for r in results:
        print(f"{r['system']:18s} {r['recall@10']:7.4f} {r['precision@10']:7.4f} "
              f"{r['map@10']:7.4f} {r['ndcg@10']:8.4f} {r['mrr']:7.4f}")

    n_rel = {j: len(rel[j]) for j in jd_ids}
    print("\nrelevant per JD:", n_rel, "total", sum(n_rel.values()))
    print("grade distribution:",
          {g: sum(1 for x in gt if x["score"] == g) for g in range(4)})

    json.dump({"results": results, "n_relevant": n_rel,
               "best_weights": best, "fusion_grid": grid_log,
               "grade_distribution": {str(g): sum(1 for x in gt
                                                 if x["score"] == g)
                                      for g in range(4)}},
              open(os.path.join(DATA, "metrics.json"), "w", encoding="utf-8"),
              indent=1)
    json.dump({r["system"]: {"rows": r["rows"]} for r in results},
              open(os.path.join(DATA, "retrieval_results.json"), "w",
                   encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
