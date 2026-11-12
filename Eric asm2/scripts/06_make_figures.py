#!/usr/bin/env python3
import argparse, json, sys
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
    out = Path(cfg["_out_dir"]); proc = out/"data"/"processed"; figs = out/"results"/"figures"; tables = out/"results"/"tables"
    figs.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(proc/"metrics_per_query.csv")
    summary = pd.read_csv(proc/"metrics_summary.csv")
    gt = pd.read_csv(proc/"ground_truth.csv")

    # 11 recall by jd
    m3 = metrics[metrics.model=='M3_ae32'].copy()
    m3 = m3.merge(gt[['jd_id','jd_title']], on='jd_id')
    plt.figure(figsize=(10,4))
    colors = ['#2CA02C' if h==1 else '#D62728' for h in m3['hit_at_10']]
    bars = plt.bar(m3['jd_id'], m3['rank'], color=colors)
    plt.axhline(10, color='gray', linestyle='--')
    plt.gca().invert_yaxis()
    plt.ylabel('Hired CV Rank (1 = best)'); plt.title('M3 AE-32: Hired CV Rank by JD')
    for b,h in zip(bars,m3['hit_at_10']):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+1, 'hit' if h==1 else 'miss', ha='center', fontsize=7)
    plt.tight_layout(); plt.savefig(figs/'11_recall_by_jd.png', dpi=200); plt.close()

    # 12 metric comparison
    models = ['M0_random','M1_tfidf','M2_svd32','M3_ae32','M4_ae_meta','M5_hybrid','M6_role_heuristic']
    x = np.arange(len(models)); width=0.2
    plt.figure(figsize=(12,4))
    for i, met in enumerate(['recall_at_10','hit_at_1','hit_at_5','mrr_at_10']):
        vals = [summary[summary.model==m][met].values[0] for m in models]
        plt.bar(x + i*width, vals, width, label=met)
    plt.xticks(x+1.5*width, models, rotation=30, ha='right'); plt.legend()
    plt.title('Model Comparison'); plt.tight_layout(); plt.savefig(figs/'12_metric_comparison.png', dpi=200); plt.close()

    # 13 rank distribution
    buckets = [(1,1),(2,5),(6,10),(11,20),(21,50),(51,110)]
    dist={}
    for b in buckets: dist[f'{b[0]}-{b[1]}']=0
    for _,r in m3.iterrows():
        rk=r['rank']
        for b in buckets:
            if b[0]<=rk<=b[1]: dist[f'{b[0]}-{b[1]}']+=1; break
    plt.figure(figsize=(8,4))
    plt.bar(dist.keys(), dist.values(), color='#9467BD')
    plt.title('M3 AE-32: Hired CV Rank Distribution'); plt.ylabel('Count'); plt.tight_layout()
    plt.savefig(figs/'13_rank_distribution.png', dpi=200); plt.close()

    # 14 score margin
    plt.figure(figsize=(10,4))
    margins = m3['score_margin']
    colors = ['#2CA02C' if x>=0 else '#D62728' for x in margins]
    plt.bar(m3['jd_id'], margins, color=colors)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Hired score - Top-1 score'); plt.title('Score Margin (positive = hired is top-1)')
    plt.tight_layout(); plt.savefig(figs/'14_score_margin.png', dpi=200); plt.close()

    # 02 dataset overview
    docs = [json.loads(l) for l in open(proc/"documents.jsonl", encoding='utf-8')]
    df=pd.DataFrame(docs)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
    df['doc_type'].value_counts().plot(kind='bar', ax=ax1, color=['#4C78A8','#F58518','#54A24B','#B279A2'])
    ax1.set_title('Documents by Type')
    df.groupby(['role_family'])['doc_id'].count().sort_values(ascending=False).plot(kind='bar', ax=ax2)
    ax2.set_title('Documents by Role Family')
    plt.tight_layout(); plt.savefig(figs/'02_dataset_overview.png', dpi=200); plt.close()

    # 01 pipeline diagram
    fig,ax=plt.subplots(figsize=(12,3)); ax.axis('off')
    stages = ['0.Validate','1.Extract\n(LLM)','2.Merge/Cluster','3.Autoencoder','4.Ranking','5.Evaluation','6.Error\nAnalysis','7.Website']
    for i,s in enumerate(stages):
        rect=plt.Rectangle((i*1.5,0.4),1.3,0.4,fill=False)
        ax.add_patch(rect); ax.text(i*1.5+0.65,0.6,s,ha='center',va='center',fontsize=8)
        if i<len(stages)-1: ax.arrow(i*1.5+1.3,0.6,0.2,0,head_width=0.03)
    ax.set_xlim(-0.1,i*1.5+1.7); ax.set_ylim(0,1.2); ax.set_title('Pipeline Overview')
    plt.tight_layout(); plt.savefig(figs/'01_pipeline_diagram.png', dpi=200); plt.close()

    # save tables to results/tables
    for name in ['metrics_summary.csv','metrics_per_query.csv','ablation_results.csv','rankings.csv','top10.csv']:
        if (proc/name).exists():
            df=pd.read_csv(proc/name); df.to_csv(tables/name, index=False)
    print('figures done')

if __name__ == '__main__':
    main()
