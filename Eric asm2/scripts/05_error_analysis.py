#!/usr/bin/env python3
import argparse, json, math, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group assignment"))
from src.config import load_config, add_common_args

def main():
    parser = argparse.ArgumentParser(); add_common_args(parser); args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["_out_dir"]); proc = out/"data"/"processed"; figs = out/"results"/"figures"; tables = out/"results"/"tables"; res = out/"results"
    figs.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)

    rankings = pd.read_csv(proc/"rankings.csv")
    metrics = pd.read_csv(proc/"metrics_per_query.csv")
    top10 = pd.read_csv(proc/"top10.csv")
    gt = pd.read_csv(proc/"ground_truth.csv")
    docs = [json.loads(l) for l in open(proc/"documents.jsonl", encoding='utf-8')]
    docs_by_id = {d['doc_id']: d for d in docs}
    jd_ext = {d['doc_id']: d for d in [json.loads(l) for l in open(proc/"jd_extracted.jsonl", encoding='utf-8')]}
    cv_ext = {d['doc_id']: d for d in [json.loads(l) for l in open(proc/"cv_extracted.jsonl", encoding='utf-8')]}
    clusters = pd.read_csv(proc/"keyword_clusters.csv")
    ktc = json.load(open(proc/"keyword_to_cluster.json", encoding='utf-8'))
    ktc_n = {str(k):int(v) for k,v in ktc.items()}; ktc_n.update({str(k).lower():int(v) for k,v in ktc.items()})
    clabel = dict(zip(clusters['cluster_id'].astype(int), clusters['canonical_label']))

    def get_set(doc_id, is_jd=True):
        s=set()
        if is_jd:
            ext=jd_ext.get(doc_id,{})
            for k in (ext.get('must_have_keywords') or []):
                cid=ktc_n.get(str(k)) or ktc_n.get(str(k).lower());
                if cid is not None: s.add(int(cid))
        else:
            ext=cv_ext.get(doc_id,{})
            for field in ['skills','tools_technologies','domains','certifications']:
                for k in (ext.get(field) or []):
                    cid=ktc_n.get(str(k)) or ktc_n.get(str(k).lower());
                    if cid is not None: s.add(int(cid))
        return s

    rows=[]
    md_lines=["# Error Analysis\n\n",
              "This report reviews retrieval results for the 10 ground-truth JD–hired-CV pairs. ",
              "Primary model: **M3 AE-32 cosine** (selected a priori by validation reconstruction loss, not test recall). ",
              "Comparison: **M1 TF-IDF cosine**.\n\n"]
    # per-query analysis
    for _, g in gt.iterrows():
        jd=g['jd_id']; hired=g['hired_cv_id']
        must=get_set(jd, True)
        for model in ['M1_tfidf','M3_ae32']:
            sub = rankings[(rankings.model==model)&(rankings.jd_id==jd)].sort_values('rank')
            hired_row = sub[sub.cv_id==hired].iloc[0]
            top1 = sub.iloc[0]
            hr = int(hired_row['rank']); hs=float(hired_row['score'])
            t1 = top1['cv_id']; t1s=float(top1['score'])
            cov_h = len(must & get_set(hired,False))/len(must) if must else 0
            cov_t1 = len(must & get_set(t1,False))/len(must) if must else 0
            miss_h = [clabel.get(i,str(i)) for i in list(must - get_set(hired,False))[:5]]
            miss_t1 = [clabel.get(i,str(i)) for i in list(must - get_set(t1,False))[:5]]
            rf_h = docs_by_id.get(hired,{}).get('role_family')
            rf_t1 = docs_by_id.get(t1,{}).get('role_family')
            rf_j = docs_by_id.get(jd,{}).get('role_family')
            # simple failure category heuristic
            if hr <= 10:
                cat = 'none_hit'
            else:
                # heuristics
                if cov_t1 > cov_h + 0.2: cat='E4_hard_negative_lookalike'
                elif rf_t1==rf_j and rf_h!=rf_j: cat='E6_domain_industry_mismatch'
                elif cov_h < 0.3: cat='E1_extraction_missing_skill'
                else: cat='E8_label_noise_or_recruiter_context'
            rows.append({
                "model":model,"jd_id":jd,"jd_title":g['jd_title'],"role_family":g['role_family'],
                "hired_cv_id":hired,"hired_rank":hr,"hired_score":hs,
                "top1_cv_id":t1,"top1_title":docs_by_id.get(t1,{}).get('title'),"top1_score":t1s,
                "score_margin":hs-t1s,"must_have_coverage_hired":round(cov_h,4),"must_have_coverage_top1":round(cov_t1,4),
                "missing_must_haves_hired":"; ".join(miss_h),"missing_must_haves_top1":"; ".join(miss_t1),
                "role_family_match_hired": rf_h==rf_j, "role_family_match_top1": rf_t1==rf_j,
                "primary_failure_category":cat, "secondary_failure_category":"", "comment":""
            })

        # per-JD narrative (use M3)
        m3 = [r for r in rows if r['jd_id']==jd and r['model']=='M3_ae32'][-1]
        hit = m3['hired_rank']<=10
        status = f"**HIT** (rank {m3['hired_rank']})" if hit else f"**MISS** (rank {m3['hired_rank']})"
        md_lines.append(f"## {jd} — {g['jd_title']}\n")
        md_lines.append(f"- Ground truth: `{hired}` ({g['hired_cv_title']}, {g['hired_cv_company']}). AE-32 result: {status}, score {m3['hired_score']:.3f}.\n")
        md_lines.append(f"- Top-1: `{m3['top1_cv_id']}` {m3['top1_title']} (score {m3['top1_score']:.3f}, margin {m3['score_margin']:.3f}). Must-have coverage hired/top1: {m3['must_have_coverage_hired']:.2f}/{m3['must_have_coverage_top1']:.2f}.\n")
        if not hit:
            md_lines.append(f"- Primary failure category: `{m3['primary_failure_category']}`. Missing must-haves (hired): {m3['missing_must_haves_hired'] or 'n/a'}. ")
            md_lines.append("Likely improvement: better must-have weighting/filtering, richer extraction, or domain/metadata signals.\n")
        else:
            md_lines.append("- The hired candidate appears in the shortlist; ranking is supported by keyword/representation overlap, though recruiter context may differ.\n")
        md_lines.append("\n")

    pd.DataFrame(rows).to_csv(proc/"error_analysis.csv", index=False)

    # AE vs TF-IDF
    comp=[]
    for _, g in gt.iterrows():
        jd=g['jd_id']
        t = metrics[(metrics.jd_id==jd)&(metrics.model=='M1_tfidf')].iloc[0]
        a = metrics[(metrics.jd_id==jd)&(metrics.model=='M3_ae32')].iloc[0]
        delta = int(t['rank']) - int(a['rank'])  # positive means AE better (lower rank)
        winner = 'AE' if a['rank'] < t['rank'] else ('TFIDF' if t['rank'] < a['rank'] else 'tie')
        comp.append({"jd_id":jd,"tfidf_rank":int(t['rank']),"ae_rank":int(a['rank']),"delta":delta,"winner":winner,"explanation":""})
    pd.DataFrame(comp).to_csv(tables/"ae_vs_tfidf.csv", index=False)

    # taxonomy figure
    m3 = pd.DataFrame([r for r in rows if r['model']=='M3_ae32'])
    cats = m3['primary_failure_category'].value_counts()
    plt.figure(figsize=(8,4)); cats.plot(kind='bar', color='#E45756')
    plt.title('Primary Failure Categories (AE-32, 10 JDs)'); plt.ylabel('Count'); plt.tight_layout()
    plt.savefig(figs/'17_error_taxonomy.png', dpi=200); plt.close()

    # ae vs tfidf figure
    cdf = pd.DataFrame(comp)
    plt.figure(figsize=(8,4))
    x = np.arange(len(cdf))
    plt.plot(x, cdf['tfidf_rank'], 'o-', label='TF-IDF rank')
    plt.plot(x, cdf['ae_rank'], 'o-', label='AE rank')
    plt.axhline(10, color='gray', linestyle='--', label='Top-10 cutoff')
    plt.xticks(x, cdf['jd_id']); plt.gca().invert_yaxis()
    plt.ylabel('Rank (lower better)'); plt.title('Hired CV Rank: AE vs TF-IDF'); plt.legend(); plt.tight_layout()
    plt.savefig(figs/'16_ae_vs_tfidf_ranks.png', dpi=200); plt.close()

    # case study heatmap for 3 cases
    case_ids = ['P01','P04','P08']
    for cid in case_ids:
        try:
            plot_case_study(cid, rankings, top10, gt, jd_ext, cv_ext, ktc_n, clabel, figs)
        except Exception as e:
            print('case study failed', cid, e)
    # generic combined
    try:
        plot_case_study('P10', rankings, top10, gt, jd_ext, cv_ext, ktc_n, clabel, figs, outname='15_case_study_keyword_coverage.png')
    except Exception:
        pass

    # extraction quality placeholder fig
    plt.figure(figsize=(6,3)); plt.text(0.01,0.5,'See data/processed/extraction_quality.csv and summary in README.', fontsize=10)
    plt.axis('off'); plt.savefig(figs/'20_extraction_quality.png', dpi=200); plt.close()
    # recommendation matrix
    recs = pd.DataFrame([
        {"priority":"P0","issue":"Must-have underweighting","fix":"Hybrid/filter by must-have coverage","impact":3,"effort":2},
        {"priority":"P0","issue":"Extraction noise/missing skills","fix":"Stronger local model or second pass","impact":3,"effort":3},
        {"priority":"P1","issue":"Keyword merge errors","fix":"Ontology + manual review","impact":2,"effort":2},
        {"priority":"P1","issue":"Missing recruiter filters","fix":"Location/language/seniority filters","impact":3,"effort":2},
        {"priority":"P1","issue":"Reconstruction not ranking-aware","fix":"Sentence embeddings/cross-encoder","impact":3,"effort":3},
        {"priority":"P2","issue":"Small label set","fix":"Collect ~50 labelled pairs, learning-to-rank","impact":3,"effort":4},
        {"priority":"P2","issue":"Domain/years not modelled","fix":"Structured features and calibration","impact":2,"effort":2},
        {"priority":"P2","issue":"Score interpretability","fix":"Calibrated reasons and evidence spans","impact":2,"effort":2},
    ])
    recs.to_csv(tables/'recommendations.csv', index=False)
    plt.figure(figsize=(6,4))
    for _,r in recs.iterrows():
        plt.scatter(r['effort'], r['impact'], s=80)
        plt.text(r['effort']+0.05, r['impact']+0.02, r['priority'], fontsize=8)
    plt.xlabel('Effort'); plt.ylabel('Expected impact'); plt.title('Recommendation Matrix (Impact vs Effort)')
    plt.xlim(0.5,4.5); plt.ylim(0.5,3.5); plt.tight_layout(); plt.savefig(figs/'19_recommendation_matrix.png', dpi=200); plt.close()

    # recommendations in md
    md_lines.append("## Recommendations\n\n")
    md_lines.append("| Priority | Issue | Proposed fix | Evidence/expected impact | Effort |\n|---|---|---|---|---|\n")
    for _,r in recs.iterrows():
        md_lines.append(f"| {r['priority']} | {r['issue']} | {r['fix']} | Based on per-query misses and coverage gaps; hypothesis pending validation. | {r['effort']} |\n")
    md_lines.append("\n## Limitations\n\n")
    for lim in [
        "n=10 labelled queries with one relevant CV per JD; sparse and potentially noisy labels.",
        "Hired candidate is not necessarily best on paper; hiring includes interview, salary, availability, recruiter judgment.",
        "Baseline ignores salary, notice period, location/citizenship/language hard filters.",
        "LLM extraction noise and PDF truncation (ellipsis) may drop evidence.",
        "Keyword clustering is heuristic; over/under-merging possible.",
        "AE trained on small corpus (~150 docs) optimising reconstruction, not ranking.",
        "Held-out keyword coverage may disadvantage rare must-have skills.",
        "100-CV pool may not represent real candidate population; random Recall@10 ~0.09 for 110 candidates.",
        "50 unlabelled JDs are not ground truth; used only for training/context after dedupe.",
        "No statistical significance claims possible with n=10."
    ]:
        md_lines.append(f"- {lim}\n")
    (res/"error_analysis.md").write_text("".join(md_lines), encoding='utf-8')
    print('wrote error analysis')

def plot_case_study(jd_id, rankings, top10, gt, jd_ext, cv_ext, ktc_n, clabel, figs, outname=None):
    sub = rankings[(rankings.model=='M3_ae32')&(rankings.jd_id==jd_id)].sort_values('rank').head(10)
    ext_j = jd_ext.get(jd_id,{})
    must = []
    for k in (ext_j.get('must_have_keywords') or []):
        cid=ktc_n.get(str(k)) or ktc_n.get(str(k).lower())
        if cid is not None: must.append(int(cid))
    must = list(dict.fromkeys(must))[:12]
    if not must: must=[0]
    cvs = sub['cv_id'].tolist()
    mat = np.zeros((len(cvs), len(must)))
    for i,c in enumerate(cvs):
        ext_c = cv_ext.get(c,{})
        cset=set()
        for field in ['skills','tools_technologies','domains','certifications']:
            for k in (ext_c.get(field) or []):
                cid=ktc_n.get(str(k)) or ktc_n.get(str(k).lower())
                if cid is not None: cset.add(int(cid))
        for j,m in enumerate(must):
            mat[i,j]=1.0 if m in cset else 0.0
    plt.figure(figsize=(max(6,len(must)*0.6), 4))
    plt.imshow(mat, aspect='auto', cmap='Blues')
    plt.colorbar(label='Present')
    plt.yticks(range(len(cvs)), [f"{r.cv_id}({'H' if r.is_relevant else ''})" for r in sub.itertuples()])
    plt.xticks(range(len(must)), [clabel.get(m,str(m))[:18] for m in must], rotation=45, ha='right', fontsize=8)
    plt.title(f'Case Study {jd_id}: Top-10 CVs vs JD Must-have Clusters (AE-32)')
    plt.tight_layout()
    name = outname or f'18_case_study_{jd_id}.png'
    plt.savefig(figs/name, dpi=200); plt.close()

if __name__ == '__main__':
    main()
