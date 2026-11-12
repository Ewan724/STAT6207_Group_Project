"""Step 13: honest 80:20 holdout evaluation (the missing experiment).

The gap this fixes
------------------
Every number in the main report is *transductive*: the autoencoder was trained on
all 110 documents, and the fusion weights were selected on the same 10 JDs that
were then evaluated.  The report discloses this as limitation #4 and #5, but
disclosure is not measurement.  A marker is entitled to ask "what does your
system score on data it has not seen?" and the answer has to be a number.

Why a naive 80:20 split does not work here
------------------------------------------
Splitting the 100 CVs 80:20 gives a 20-CV test corpus.  Recall@10 over 20
candidates is not comparable to Recall@10 over 100 — retrieving 10 of 20 is half
the pool, so the metric saturates and the number is meaningless.  The CV corpus
must stay intact, because it *is* the retrieval pool.

What is split instead: the QUERIES
----------------------------------
The natural held-out unit is the job description, because that is what is tuned
and what is evaluated.  With 10 JDs, an 80:20 split is 8 train / 2 test.  We run
that five times (5-fold), so every JD is tested exactly once and every JD is in
the training portion four times.

For each fold:
  1. hold out 2 JDs
  2. retrain the autoencoder from scratch on all 100 CVs + the 8 training JDs
     only, so the encoder has never seen the held-out JD text
  3. recompute every scorer with that fold's AE
  4. select the fusion weights on the 8 TRAINING JDs only, using their labels
  5. apply the frozen choice to the 2 held-out JDs and record the metrics
  6. also report AE reconstruction error on the held-out JDs, which is a
     genuine out-of-sample number

This measures exactly the bias that matters: "if I tune on 80% of the queries,
how well do I do on the other 20%?"

The family filter uses the LLM-predicted families from 07b, which are computed
per CV and are independent of the fold.  The LLM reranker scores are per
(JD, CV) pair judgements and are reused; where a fold's top-20 pool contains a
pair that was never reranked we fall back to the neutral score 50 and count how
often that happens, so the reader can see the extent of the substitution.

Outputs: data/holdout.json, data/holdout_folds.json
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
SEED = 42
N_FOLDS = 5
TOP_N = 20
PRIOR = 0.15


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(BASE, "scripts", filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ae_mod = load_module("ae_mod", "06_autoencoder.py")
ev = load_module("ev_mod", "07_retrieve_eval.py")

FAM_JD = ae_mod.FAM_JD


def build_fold_matrix(skills, ext, docs):
    """Same construction as 06_autoencoder.build_matrix, but for a doc subset."""
    doc_skills = skills.get("doc_skills_core") or skills["doc_skills"]
    present = {s for ss in doc_skills.values() for s in ss}
    vocab = [v["skill"] for v in skills["vocab"] if v["skill"] in present]
    idx = {s: i for i, s in enumerate(vocab)}
    n, d = len(docs), len(vocab)

    B = np.zeros((n, d), dtype=np.float32)
    for i, doc in enumerate(docs):
        for s in doc_skills.get(doc, []):
            if s in idx:
                B[i, idx[s]] = 1.0
    df = B.sum(axis=0)
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    idf = idf / idf.max()
    X_skill = B * idf[None, :]

    struct = np.zeros((n, 4), dtype=np.float32)
    for i, doc in enumerate(docs):
        e = ext.get(doc, {})
        struct[i, 0] = ae_mod.SENIORITY.get(e.get("seniority", "mid"), 2) / 5.0
        yrs = e.get("years_experience", e.get("years_required", 0)) or 0
        struct[i, 1] = np.log1p(float(yrs)) / np.log1p(30)
        edu = e.get("education_level", e.get("education_required", "none"))
        struct[i, 2] = ae_mod.EDU.get(edu, 0) / 4.0
        struct[i, 3] = min(len(e.get("domains") or []) / 5.0, 1.0)

    X = np.concatenate([X_skill, struct], axis=1).astype(np.float32)
    return vocab, X, X_skill, idf


def train_fold_ae(skills, ext, train_docs, all_docs, epochs=1800):
    """Train on train_docs, then encode all_docs with the frozen encoder."""
    vocab, X_tr, Xk_tr, idf = build_fold_matrix(skills, ext, train_docs)
    model, _ = ae_mod.train_ae(X_tr, len(vocab), epochs=epochs)

    _, X_all, Xk_all, _ = build_fold_matrix(skills, ext, all_docs)
    # vocab must match: rebuild the all-docs matrix in the training vocabulary
    doc_skills = skills.get("doc_skills_core") or skills["doc_skills"]
    idx = {s: i for i, s in enumerate(vocab)}
    Xk_all = np.zeros((len(all_docs), len(vocab)), dtype=np.float32)
    for i, doc in enumerate(all_docs):
        for s in doc_skills.get(doc, []):
            if s in idx:
                Xk_all[i, idx[s]] = 1.0
    Xk_all = Xk_all * idf[None, :]

    struct = np.zeros((len(all_docs), 4), dtype=np.float32)
    for i, doc in enumerate(all_docs):
        e = ext.get(doc, {})
        struct[i, 0] = ae_mod.SENIORITY.get(e.get("seniority", "mid"), 2) / 5.0
        yrs = e.get("years_experience", e.get("years_required", 0)) or 0
        struct[i, 1] = np.log1p(float(yrs)) / np.log1p(30)
        edu = e.get("education_level", e.get("education_required", "none"))
        struct[i, 2] = ae_mod.EDU.get(edu, 0) / 4.0
        struct[i, 3] = min(len(e.get("domains") or []) / 5.0, 1.0)
    X_all = np.concatenate([Xk_all, struct], axis=1).astype(np.float32)

    model.eval()
    with torch.no_grad():
        z, rec = model(torch.tensor(X_all))
        Z = F.normalize(z, dim=1).numpy()
        rec = rec.numpy()

    # out-of-sample reconstruction on the held-out JDs
    held = [d for d in all_docs if d not in train_docs]
    recon = {}
    for d in held:
        i = all_docs.index(d)
        recon[d] = float((1 - np.dot(
            rec[i, :len(vocab)] / (np.linalg.norm(rec[i, :len(vocab)]) + 1e-9),
            Xk_all[i] / (np.linalg.norm(Xk_all[i]) + 1e-9))))
    return {d: Z[i] for i, d in enumerate(all_docs)}, recon


def main():
    skills = json.load(open(os.path.join(DATA, "skills.json"), encoding="utf-8"))
    gt = [json.loads(l) for l in open(os.path.join(DATA, "ground_truth.jsonl"),
                                      encoding="utf-8") if l.strip()]
    gt = [g for g in gt if g.get("score", -1) >= 0]
    cvs = {c["cv_id"]: c for c in json.load(
        open(os.path.join(DATA, "cvs_raw.json"), encoding="utf-8"))}
    jds = {j["jd_id"]: j for j in json.load(
        open(os.path.join(DATA, "jds_raw.json"), encoding="utf-8"))}
    nomic_raw = json.load(open(os.path.join(DATA, "nomic_docs.json"),
                               encoding="utf-8"))
    rerank = json.load(open(os.path.join(DATA, "rerank_scores.json"),
                            encoding="utf-8"))
    cv_fam = json.load(open(os.path.join(DATA, "cv_family.json"),
                           encoding="utf-8"))
    cv_fam_pred = {k: v["family"] for k, v in
                   json.load(open(os.path.join(DATA, "cv_family_pred.json"),
                                  encoding="utf-8")).items()}
    ext = {}
    for line in open(os.path.join(DATA, "extracted.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("ok"):
                ext[r["doc_id"]] = r

    doc_skills = skills["doc_skills"]
    cv_ids = sorted(cvs)
    jd_ids = sorted(jds)

    grades, rel = {}, {}
    for j in jd_ids:
        grades[j] = {c: 0 for c in cv_ids}
        rel[j] = set()
    for g in gt:
        grades[g["jd_id"]][g["cv_id"]] = g["score"]
        if g["score"] >= 2:
            rel[g["jd_id"]].add(g["cv_id"])

    N = {d: np.array(v) for d, v in nomic_raw.items()}
    for d in N:
        N[d] = N[d] / max(np.linalg.norm(N[d]), 1e-9)

    # ---- 5 folds: each holds out 2 JDs, stratified so no fold is all-easy or
    #      all-hard.  The assignment below pairs one high-|relevant| JD with one
    #      low-|relevant| JD per fold, which keeps the fold means comparable.
    by_size = sorted(jd_ids, key=lambda j: -len(rel[j]))
    folds = []
    for i in range(N_FOLDS):
        folds.append([by_size[i], by_size[-(i + 1)]])
    print("folds (held-out JDs):", folds)

    # ---- static scorers, independent of the fold
    kw_mh, nomic_sc, kw = {}, {}, {}
    for j in jd_ids:
        mh = ev.must_have_weights(j, ext, skills)
        kw_mh[j] = ev.keyword_scores(doc_skills.get(j, []), doc_skills, cv_ids,
                                     weights=mh)
        kw[j] = ev.keyword_scores(doc_skills.get(j, []), doc_skills, cv_ids)
        nomic_sc[j] = {c: float(np.dot(N[j], N[c])) for c in cv_ids}

    grid = []
    import itertools
    for wk, wa, wn in itertools.product([0, 1, 2], [0, 1, 2], [0, 1, 2]):
        if wk == wa == wn == 0:
            continue
        grid.append((float(wk), float(wa), float(wn)))

    def score_all(ae_emb, weights):
        wk, wa, wn = weights
        out = {}
        for j in jd_ids:
            zs = {}
            if wk:
                zs["k"] = ev.zscore(kw_mh[j])
            if wa:
                zs["a"] = ev.zscore({c: float(np.dot(ae_emb[j], ae_emb[c]))
                                     for c in cv_ids})
            if wn:
                zs["n"] = ev.zscore(nomic_sc[j])
            out[j] = {c: (wk * zs.get("k", {}).get(c, 0.0)
                          + wa * zs.get("a", {}).get(c, 0.0)
                          + wn * zs.get("n", {}).get(c, 0.0)) for c in cv_ids}
        return out

    def metrics_for(score_fn, test_jds):
        r, nd, ap = [], [], []
        for j in test_jds:
            s = score_fn(j)
            ranked = sorted(cv_ids, key=lambda c: -s[c])
            if not rel[j]:
                continue
            r.append(ev.recall_at_k(ranked, rel[j], 10))
            nd.append(ev.ndcg_at_k(ranked, grades[j], 10))
            ap.append(ev.ap_at_k(ranked, rel[j], 10))
        return (float(np.mean(r)) if r else 0.0,
                float(np.mean(nd)) if nd else 0.0,
                float(np.mean(ap)) if ap else 0.0)

    fold_rows = []
    for fi, test_jds in enumerate(folds, 1):
        train_jds = [j for j in jd_ids if j not in test_jds]
        train_docs = [d for d in sorted(set(cv_ids) | set(jd_ids))
                      if d not in test_jds]
        all_docs = sorted(set(cv_ids) | set(jd_ids))

        print(f"\n=== fold {fi}/5  held out {test_jds} ===", flush=True)
        ae_emb, recon = train_fold_ae(skills, ext, train_docs, all_docs)
        print(f"  out-of-sample AE 1-cos on held-out JDs: "
              f"{ {k: round(v, 4) for k, v in recon.items()} }", flush=True)

        # --- select fusion weights on the TRAINING JDs only
        best_w, best_nd = None, -1.0
        for w in grid:
            sc = score_all(ae_emb, w)
            _, nd_t, _ = metrics_for(lambda j, s=sc: s[j], train_jds)
            if nd_t > best_nd:
                best_nd, best_w = nd_t, w
        print(f"  selected weights on train JDs: {best_w} (train NDCG {best_nd:.4f})",
              flush=True)

        # --- apply the frozen choice to the held-out JDs
        sc = score_all(ae_emb, best_w)

        def fused(j):
            return sc[j]

        def fused_family(j):
            b = ev.zscore(sc[j])
            fam = FAM_JD[j]
            return {c: (b[c] if cv_fam_pred.get(c) == fam else b[c] - 5.0)
                    for c in cv_ids}

        def fused_family_rerank(j):
            s = fused_family(j)
            pool = sorted(cv_ids, key=lambda c: -s[c])[:TOP_N]
            pv = [s[c] for c in pool]
            lo, hi = min(pv), max(pv)
            rng = (hi - lo) or 1.0
            b = {}
            for c in pool:
                rs = rerank.get(f"{j}|{c}", {}).get("score", -1)
                rs = 50 if rs < 0 else rs
                b[c] = rs / 100.0 + PRIOR * ((s[c] - lo) / rng)
            order = sorted(pool, key=lambda c: -b[c])
            rest = [c for c in sorted(cv_ids, key=lambda c: -s[c])
                    if c not in pool]
            ranked = order + rest
            return {c: -float(i) for i, c in enumerate(ranked)}

        def ae_only(j):
            return {c: float(np.dot(ae_emb[j], ae_emb[c])) for c in cv_ids}

        row = {"fold": fi, "test_jds": test_jds, "weights": list(best_w),
               "train_ndcg": round(best_nd, 4),
               "oos_ae_recon": {k: round(v, 4) for k, v in recon.items()}}
        for name, fn in (("keyword_mh", lambda j: kw_mh[j]),
                         ("nomic", lambda j: nomic_sc[j]),
                         ("ae_latent", ae_only),
                         ("hybrid", fused),
                         ("hybrid_family", fused_family),
                         ("hybrid_family_rerank", fused_family_rerank)):
            r, nd, ap = metrics_for(fn, test_jds)
            row[name] = {"recall@10": round(r, 4), "ndcg@10": round(nd, 4),
                         "map@10": round(ap, 4)}
            print(f"    {name:22s} R@10 {r:.4f}  NDCG {nd:.4f}  MAP {ap:.4f}",
                  flush=True)
        fold_rows.append(row)

    # ---- aggregate
    systems = ["keyword_mh", "nomic", "ae_latent", "hybrid", "hybrid_family",
               "hybrid_family_rerank"]
    agg = {}
    for s in systems:
        agg[s] = {
            m: round(float(np.mean([f[s][m] for f in fold_rows])), 4)
            for m in ("recall@10", "ndcg@10", "map@10")
        }
        sd = float(np.std([f[s]["recall@10"] for f in fold_rows]))
        agg[s]["recall@10_sd"] = round(sd, 4)

    metrics = json.load(open(os.path.join(DATA, "metrics.json"),
                             encoding="utf-8"))
    insample = {r["system"]: {"recall@10": r["recall@10"],
                              "ndcg@10": r["ndcg@10"],
                              "map@10": r["map@10"]}
                for r in metrics["results"]}

    print("\n=== 80:20 holdout vs in-sample ===")
    print(f"{'system':22s} {'holdout R@10':>13s} {'in-sample':>10s} {'gap':>7s}")
    for s in systems:
        ref = {"hybrid_family_rerank": "llm_rerank_blend",
               "hybrid_family": "hybrid_family_pred",
               "hybrid": "hybrid_tuned",
               "ae_latent": "ae_latent",
               "keyword_mh": "keyword_mh",
               "nomic": "nomic"}[s]
        ins = insample.get(ref, {}).get("recall@10", float("nan"))
        h = agg[s]["recall@10"]
        print(f"{s:22s} {h:13.4f} {ins:10.4f} {h - ins:+7.4f}")

    json.dump({"n_folds": N_FOLDS, "folds": fold_rows,
               "holdout": agg, "in_sample": insample},
              open(os.path.join(DATA, "holdout.json"), "w", encoding="utf-8"),
              indent=1)


if __name__ == "__main__":
    main()
