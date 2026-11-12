#!/usr/bin/env python3
import argparse, json, math, random, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group assignment"))
from src.config import load_config, add_common_args

def main():
    parser = argparse.ArgumentParser(); add_common_args(parser); args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["_out_dir"]); proc = out/"data"/"processed"; figs = out/"results"/"figures"; tables = out/"results"/"tables"
    figs.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)

    # load
    emb = pd.read_csv(proc/"document_embeddings.csv")
    docs = [json.loads(l) for l in open(proc/"documents.jsonl", encoding='utf-8')]
    docs_by_id = {d['doc_id']: d for d in docs}
    gt = pd.read_csv(proc/"ground_truth.csv")
    jd_ext = {d['doc_id']: d for d in [json.loads(l) for l in open(proc/"jd_extracted.jsonl", encoding='utf-8')]}
    cv_ext = {d['doc_id']: d for d in [json.loads(l) for l in open(proc/"cv_extracted.jsonl", encoding='utf-8')]}
    clusters = pd.read_csv(proc/"keyword_clusters.csv")
    ktc = json.load(open(proc/"keyword_to_cluster.json", encoding='utf-8'))
    ktc_n = {str(k):int(v) for k,v in ktc.items()}
    ktc_n.update({str(k).lower():int(v) for k,v in ktc.items()})
    cid_to_idx = {int(r['cluster_id']):i for i,r in clusters.iterrows()}  # not needed
    # matrices
    from scipy import sparse
    X_counts = sparse.load_npz(proc/"document_keyword_matrix.npz").toarray().astype(np.float32)
    idx_df = pd.read_csv(proc/"document_matrix_index.csv")
    id_to_row = dict(zip(idx_df['doc_id'], idx_df['row']))
    # tfidf normalized (approx via sklearn)
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer(sublinear_tf=True, norm='l2')
    X_tfidf = tfidf.fit_transform(sparse.csr_matrix(X_counts)).toarray().astype(np.float32)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=min(32, X_tfidf.shape[1]-1), random_state=42)
    X_svd = svd.fit_transform(X_tfidf)
    X_svd = X_svd / (np.linalg.norm(X_svd, axis=1, keepdims=True)+1e-9)

    # candidates: all CVs
    cand_ids = [d['doc_id'] for d in docs if d['doc_type'] in ('pool_cv','gt_cv')]
    cand_ids = sorted(cand_ids)
    jd_ids = gt['jd_id'].tolist()
    N = len(cand_ids)

    # embeddings dict
    dim_cols = [c for c in emb.columns if c.startswith('dim_')]
    E = {r['doc_id']: r[dim_cols].values.astype(np.float32) for _,r in emb.iterrows()}

    def cosine(a,b):
        return float(np.dot(a,b) / ((np.linalg.norm(a)*np.linalg.norm(b))+1e-9))

    # must-have coverage
    def must_have_set(jd_id):
        ext = jd_ext.get(jd_id, {})
        must = ext.get('must_have_keywords',[]) or []
        s = set()
        for k in must:
            cid = ktc_n.get(str(k)) or ktc_n.get(str(k).lower())
            if cid is not None: s.add(int(cid))
        return s
    def cv_cluster_set(cv_id):
        ext = cv_ext.get(cv_id, {})
        s = set()
        for field in ['skills','tools_technologies','domains']:
            for k in (ext.get(field) or []):
                cid = ktc_n.get(str(k)) or ktc_n.get(str(k).lower())
                if cid is not None: s.add(int(cid))
        # include education/certs tokens? minimal
        for k in (ext.get('certifications') or []):
            cid = ktc_n.get(str(k)) or ktc_n.get(str(k).lower())
            if cid is not None: s.add(int(cid))
        return s

    must_sets = {j: must_have_set(j) for j in jd_ids}
    cv_sets = {c: cv_cluster_set(c) for c in cand_ids}

    # precompute role families
    cand_role = {c: docs_by_id[c].get('role_family') for c in cand_ids}
    jd_role = {j: docs_by_id[j].get('role_family') for j in jd_ids}

    rankings = []
    top10 = []
    models_list = ['M0_random','M1_tfidf','M2_svd32','M3_ae32','M4_ae_meta','M5_hybrid','M6_role_heuristic']

    rng = np.random.default_rng(42)
    # one representative random permutation per JD (seeded per jd for determinism)
    for model in models_list:
        for jd_id in jd_ids:
            scores = {}
            if model == 'M0_random':
                perm = rng.permutation(cand_ids)
                for rank, cid in enumerate(perm, start=1):
                    scores[cid] = (1.0/(rank), rank)  # store score proxy
                # we'll assign by perm
                order = list(perm)
            elif model == 'M1_tfidf':
                qi = id_to_row[jd_id]; qv = X_tfidf[qi]
                for cid in cand_ids:
                    ci = id_to_row[cid]; scores[cid] = float(np.dot(qv, X_tfidf[ci])/(np.linalg.norm(qv)*np.linalg.norm(X_tfidf[ci])+1e-9))
            elif model == 'M2_svd32':
                qi = id_to_row[jd_id]; qv = X_svd[qi]
                for cid in cand_ids:
                    ci = id_to_row[cid]; scores[cid] = cosine(qv, X_svd[ci])
            elif model == 'M3_ae32':
                qv = E[jd_id]
                for cid in cand_ids: scores[cid] = cosine(qv, E[cid])
            elif model == 'M4_ae_meta':
                qv = E[jd_id]
                for cid in cand_ids:
                    base = cosine(qv, E[cid])
                    meta_bonus = 0.0
                    if cand_role.get(cid) and cand_role.get(cid)==jd_role.get(jd_id): meta_bonus += 0.1
                    scores[cid] = 0.7*base + 0.3*meta_bonus
            elif model == 'M5_hybrid':
                qv = E[jd_id]; ms = must_sets.get(jd_id,set())
                for cid in cand_ids:
                    base = cosine(qv, E[cid])
                    cov = 0.0
                    if ms: cov = len(ms & cv_sets.get(cid,set()))/len(ms)
                    scores[cid] = 0.8*base + 0.2*cov
            elif model == 'M6_role_heuristic':
                qv = E[jd_id]
                for cid in cand_ids:
                    base = cosine(qv, E[cid])
                    same = 1.0 if (cand_role.get(cid) and cand_role.get(cid)==jd_role.get(jd_id)) else 0.0
                    scores[cid] = same*10 + base
            if model != 'M0_random':
                order = sorted(cand_ids, key=lambda c: (-round(scores[c],6), c))
            hired = gt.loc[gt['jd_id']==jd_id, 'hired_cv_id'].iloc[0]
            for rank, cid in enumerate(order, start=1):
                sc = scores[cid] if model!='M0_random' else 1.0/rank
                if isinstance(sc, tuple): sc = sc[0]
                rankings.append({"model":model,"jd_id":jd_id,"cv_id":cid,"rank":rank,"score":round(float(sc),6),"is_relevant": int(cid==hired)})

    rank_df = pd.DataFrame(rankings)
    rank_df.to_csv(proc/"rankings.csv", index=False)

    # top10 with reasons
    cluster_labels = dict(zip(clusters['cluster_id'].astype(int), clusters['canonical_label']))
    for model in models_list:
        for jd_id in jd_ids:
            sub = rank_df[(rank_df.model==model)&(rank_df.jd_id==jd_id)].sort_values('rank').head(10)
            hired = gt.loc[gt['jd_id']==jd_id,'hired_cv_id'].iloc[0]
            ms = must_sets.get(jd_id,set())
            for _, r in sub.iterrows():
                cid = r['cv_id']; ext = cv_ext.get(cid,{})
                cov_ids = ms & cv_sets.get(cid,set()) if ms else set()
                missing = list(ms - cov_ids)
                cov_names = [cluster_labels.get(i,str(i)) for i in list(cov_ids)[:5]]
                miss_names = [cluster_labels.get(i,str(i)) for i in missing[:3]]
                coverage_txt = f"Matches {len(cov_ids)}/{len(ms)} must-have skills" if ms else "Matches 0/0 must-have skills"
                if cov_names: coverage_txt += f" ({', '.join(cov_names)})"
                if miss_names: coverage_txt += f"; missing {', '.join(miss_names)}"
                meta_doc = docs_by_id.get(cid,{})
                reason = coverage_txt + f". Role family: {meta_doc.get('role_family','')}. Score {r['score']:.3f}."
                top10.append({
                    "model":model,"jd_id":jd_id,"rank":int(r['rank']),"cv_id":cid,
                    "cv_title":ext.get('current_title') or meta_doc.get('title'),
                    "cv_company":ext.get('current_company') or meta_doc.get('company'),
                    "role_family":meta_doc.get('role_family'),
                    "score":r['score'],"is_relevant":int(r['is_relevant']),
                    "must_have_matches":len(cov_ids),"missing_must_haves":"; ".join(miss_names), "reason":reason[:600]
                })
    top10_df = pd.DataFrame(top10)
    top10_df.to_csv(proc/"top10.csv", index=False)

    # metrics
    metrics_rows=[]; summary_rows=[]
    for model in models_list:
        for jd_id in jd_ids:
            sub = rank_df[(rank_df.model==model)&(rank_df.jd_id==jd_id)].sort_values('rank')
            rel = sub[sub.is_relevant==1]
            rank = int(rel.iloc[0]['rank']) if len(rel) else 999
            hired = gt.loc[gt['jd_id']==jd_id,'hired_cv_id'].iloc[0]
            hit1=int(rank<=1); hit5=int(rank<=5); hit10=int(rank<=10)
            rr = (1.0/rank) if rank<=10 else 0.0
            ndcg = (1.0/math.log2(rank+1)) if rank<=10 else 0.0
            top1 = sub.iloc[0]
            ms = must_sets.get(jd_id,set())
            cov = (len(ms & cv_sets.get(hired,set()))/len(ms)) if ms else 0.0
            metrics_rows.append({
                "model":model,"jd_id":jd_id,"jd_title":gt.loc[gt.jd_id==jd_id,'jd_title'].iloc[0],
                "role_family":gt.loc[gt.jd_id==jd_id,'role_family'].iloc[0],"hired_cv_id":hired,
                "rank":rank,"score":float(rel.iloc[0]['score']) if len(rel) else None,
                "hit_at_1":hit1,"hit_at_5":hit5,"hit_at_10":hit10,
                "reciprocal_rank":rr,"ndcg_at_10":ndcg,"must_have_coverage_hired":round(cov,4),
                "top1_cv_id":top1['cv_id'],"top1_score":float(top1['score']),"score_margin": (float(rel.iloc[0]['score'])-float(top1['score'])) if len(rel) else None
            })
    mdf = pd.DataFrame(metrics_rows)
    mdf.to_csv(proc/"metrics_per_query.csv", index=False)

    for model in models_list:
        sub = mdf[mdf.model==model]
        n=len(sub); hits=int(sub['hit_at_10'].sum())
        recall = hits/n if n else 0
        hit1=sub['hit_at_1'].mean(); hit5=sub['hit_at_5'].mean(); mrr=sub['reciprocal_rank'].mean(); ndcg=sub['ndcg_at_10'].mean()
        prec=sub['hit_at_10'].mean()/10.0
        # Wilson CI
        z=1.96; p=recall
        denom=1+z*z/n; center=(p+z*z/(2*n))/denom; margin=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/denom
        wlo=max(0.0, center-margin); whi=min(1.0, center+margin)
        # bootstrap
        vals=sub['hit_at_10'].values.astype(float)
        rngb=np.random.default_rng(42)
        boots=[rngb.choice(vals,size=n,replace=True).mean() for _ in range(10000)]
        blo, bhi = np.percentile(boots,[2.5,97.5])
        summary_rows.append({"model":model,"n_queries":n,"hits":hits,"recall_at_10":round(recall,6),
            "hit_at_1":round(float(hit1),6),"hit_at_5":round(float(hit5),6),"mrr_at_10":round(float(mrr),6),
            "ndcg_at_10":round(float(ndcg),6),"precision_at_10":round(float(prec),6),
            "wilson_ci_low":round(wlo,6),"wilson_ci_high":round(whi,6),"bootstrap_ci_low":round(float(blo),6),"bootstrap_ci_high":round(float(bhi),6)})
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(proc/"metrics_summary.csv", index=False)

    # ablation
    abl = sdf[['model','recall_at_10','hit_at_1','mrr_at_10']].copy()
    notes = {'M0_random':'uniform random baseline','M1_tfidf':'TF-IDF keyword cosine','M2_svd32':'SVD-32 cosine','M3_ae32':'Primary: AE-32 cosine','M4_ae_meta':'AE + metadata heuristic','M5_hybrid':'AE + must-have coverage (exploratory)','M6_role_heuristic':'Role-family heuristic'}
    abl['notes']=abl['model'].map(notes)
    abl.to_csv(proc/"ablation_results.csv", index=False)

    # random baseline empirical over 1000 perms
    gt_map = dict(zip(gt['jd_id'], gt['hired_cv_id']))
    emp=[]
    rng2=np.random.default_rng(42)
    for _ in range(1000):
        hits=0
        for jd_id in jd_ids:
            perm = rng2.permutation(cand_ids)
            rank = int(np.where(perm==gt_map[jd_id])[0][0])+1
            if rank<=10: hits+=1
        emp.append(hits/len(jd_ids))
    pd.DataFrame({"recall_at_10":emp}).to_csv(proc/"random_baseline.csv", index=False)
    print(sdf.to_string(index=False))
    print('expected_random', 10/N)

if __name__ == '__main__':
    main()
