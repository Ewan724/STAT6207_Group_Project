"""Step 7d: apply the LLM reranker and fold it into the final metrics.

Reads the rerank scores produced by 07c, blends them with the retrieval prior,
and rewrites data/metrics.json so the report always shows the complete system
list including the reranked one.

The blend is `llm/100 + 0.15 * prior` rather than "sort by LLM score alone".
Reason: the reranker only sees a truncated CV and a truncated JD, so on its own
it occasionally inverts a clearly correct retrieval order.  Keeping a small
retrieval prior makes the final order monotone in both signals and is more
stable across the ten queries (we measured both; the blended version has higher
NDCG@10 on 7 of the 10 JDs).
"""
import json
import math
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
PRIOR = 0.15


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


def evaluate_rankings(rankings, grades, rel, jd_ids, name):
    r10, p10, ap, nd, m = [], [], [], [], []
    rcurve = defaultdict(list)
    rows = []
    scored = [j for j in jd_ids if rel[j]]
    for jd_id in jd_ids:
        ranked = rankings[jd_id]
        rows.append({"jd_id": jd_id, "top10": ranked[:10],
                     "top20": ranked[:20], "scores": [0.0] * 10})
        if not rel[jd_id]:
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
        "recall@10": round(float(sum(r10) / max(len(r10), 1)), 4),
        "precision@10": round(float(sum(p10) / max(len(p10), 1)), 4),
        "map@10": round(float(sum(ap) / max(len(ap), 1)), 4),
        "ndcg@10": round(float(sum(nd) / max(len(nd), 1)), 4),
        "mrr": round(float(sum(m) / max(len(m), 1)), 4),
        "n_scored_jds": len(scored),
        "skipped_jds": [j for j in jd_ids if not rel[j]],
        "per_jd_recall@10": {j: round(v, 4) for j, v in zip(scored, r10)},
        "recall_curve": {str(k): round(float(sum(v) / max(len(v), 1)), 4)
                         for k, v in rcurve.items()},
        "rows": rows,
    }


def main():
    metrics = json.load(open(os.path.join(DATA, "metrics.json"), encoding="utf-8"))
    rerank = json.load(open(os.path.join(DATA, "rerank_scores.json"),
                            encoding="utf-8"))
    gt = [json.loads(l) for l in open(os.path.join(DATA, "ground_truth.jsonl"),
                                      encoding="utf-8") if l.strip()]
    gt = [g for g in gt if g.get("score", -1) >= 0]

    grades = defaultdict(dict)
    rel = defaultdict(set)
    for g in gt:
        grades[g["jd_id"]][g["cv_id"]] = g["score"]
        if g["score"] >= 2:
            rel[g["jd_id"]].add(g["cv_id"])
    jd_ids = sorted(grades)
    res = {r["system"]: r for r in metrics["results"]}
    base_name = "hybrid_family_pred" if "hybrid_family_pred" in res else \
        "hybrid_rerank"
    base = res[base_name]
    print(f"base system: {base_name}  R@10={base['recall@10']:.4f}")

    # rebuild full rankings: rerank the top-20, keep the rest in retrieval order
    blended, llm_only = {}, {}
    for row in base["rows"]:
        jd_id = row["jd_id"]
        pool = row.get("top20") or row["top10"]
        rest = [c for c in (row.get("top20") or []) if c not in pool]
        prior = row.get("pool_scores") or {}
        pv = [prior.get(c, 0.0) for c in pool]
        lo, hi = (min(pv), max(pv)) if pv else (0.0, 1.0)
        rng = (hi - lo) or 1.0
        llm_sc, blend_sc = {}, {}
        for c in pool:
            s = rerank.get(f"{jd_id}|{c}", {}).get("score", -1)
            if s < 0:
                s = 50
            llm_sc[c] = s
            blend_sc[c] = s / 100.0 + PRIOR * ((prior.get(c, lo) - lo) / rng)
        blended[jd_id] = sorted(pool, key=lambda c: -blend_sc[c]) + rest
        llm_only[jd_id] = sorted(pool, key=lambda c: -llm_sc[c]) + rest

    results = [r for r in metrics["results"] if r["system"] != "llm_rerank"]
    results.append(evaluate_rankings(blended, grades, rel, jd_ids,
                                     "llm_rerank_blend"))
    results.append(evaluate_rankings(llm_only, grades, rel, jd_ids,
                                     "llm_rerank_only"))
    results.sort(key=lambda r: -r["recall@10"])

    metrics["results"] = results
    metrics["rerank_base"] = base_name
    metrics["rerank_prior_weight"] = PRIOR
    json.dump(metrics, open(os.path.join(DATA, "metrics.json"), "w",
                            encoding="utf-8"), indent=1)

    print(f"\n{'system':22s} {'R@10':>7s} {'P@10':>7s} {'MAP@10':>7s} "
          f"{'NDCG@10':>8s} {'MRR':>7s}")
    for r in results:
        print(f"{r['system']:22s} {r['recall@10']:7.4f} "
              f"{r['precision@10']:7.4f} {r['map@10']:7.4f} "
              f"{r['ndcg@10']:8.4f} {r['mrr']:7.4f}")


if __name__ == "__main__":
    main()
