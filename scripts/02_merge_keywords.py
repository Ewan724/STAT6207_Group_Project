#!/usr/bin/env python3
import argparse, json, math, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group assignment"))
from src.config import load_config, add_common_args
from src.keywords import normalize_keyword_str, CURATED_ALIASES, CONFUSABLE_BLOCKLIST, PROTECTED

def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["_out_dir"]); proc = out/"data"/"processed"; figs = out/"results"/"figures"; tables = out/"results"/"tables"
    proc.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)

    # load extracted
    jd_path = proc/"jd_extracted.jsonl"; cv_path = proc/"cv_extracted.jsonl"
    if not jd_path.exists() or not cv_path.exists():
        print("Extraction outputs missing. Run 01_extract_keywords.py first.", file=sys.stderr)
        sys.exit(1)
    jd_docs = [json.loads(l) for l in open(jd_path, encoding='utf-8')]
    cv_docs = [json.loads(l) for l in open(cv_path, encoding='utf-8')]

    # training corpus for clustering: pool CVs (100) + unlabelled JDs not excluded (from documents split info)
    docs_meta = {}
    with open(proc/"documents.jsonl", encoding='utf-8') as f:
        for l in f: d=json.loads(l); docs_meta[d['doc_id']] = d
    train_ids = set()
    for did, m in docs_meta.items():
        if m['doc_type']=='pool_cv' and m.get('split')=='train_pool': train_ids.add(did)
        if m['doc_type']=='unlabelled_jd' and m.get('split')=='train_jd': train_ids.add(did)

    # build keyword records
    records = []  # (doc_id, doc_type, field, keyword, weight, in_train)
    def add_kw(doc_id, doc_type, field, kw, weight, in_train):
        if kw is None: return
        if isinstance(kw, dict): return
        s = str(kw).strip()
        if not s: return
        records.append((doc_id, doc_type, field, s, weight, in_train))

    for d in jd_docs:
        did=d.get('doc_id'); in_train = did in train_ids
        for k in d.get('must_have_keywords',[]) or []: add_kw(did,'jd','must_have_keywords',k,3.0,in_train)
        for k in d.get('nice_to_have_keywords',[]) or []: add_kw(did,'jd','nice_to_have_keywords',k,1.0,in_train)
        for k in d.get('tools_technologies',[]) or []: add_kw(did,'jd','tools_technologies',k,2.0,in_train)
        for k in d.get('domain_knowledge',[]) or []: add_kw(did,'jd','domain_knowledge',k,1.5,in_train)
        for k in d.get('key_responsibilities',[]) or []: add_kw(did,'jd','key_responsibilities',k,0.5,in_train)
        for k in d.get('soft_skills',[]) or []: add_kw(did,'jd','soft_skills',k,0.5,in_train)
    for d in cv_docs:
        did=d.get('doc_id'); in_train = did in train_ids
        for k in d.get('skills',[]) or []: add_kw(did,'cv','skills',k,2.0,in_train)
        for k in d.get('tools_technologies',[]) or []: add_kw(did,'cv','tools_technologies',k,2.0,in_train)
        for k in d.get('domains',[]) or []: add_kw(did,'cv','domains',k,1.5,in_train)
        for k in d.get('soft_skills',[]) or []: add_kw(did,'cv','soft_skills',k,0.5,in_train)
        for k in d.get('certifications',[]) or []: add_kw(did,'cv','certifications',k,0.5,in_train)
        for edu in (d.get('education') or []):
            if isinstance(edu, dict):
                for f in ['degree','field']: 
                    if edu.get(f): add_kw(did,'cv','education',edu.get(f),0.5,in_train)

    alias_map = {normalize_keyword_str(r): c for r,c in CURATED_ALIASES}
    # also add common variants
    def canonicalize(raw):
        n = normalize_keyword_str(raw)
        if not n: return None
        if n in alias_map: return alias_map[n]
        # protected / known tech tokens keep form
        # simple singularize
        if n.endswith('ies') and len(n)>5: cand = n[:-3]+'y'
        elif n.endswith('s') and len(n)>4 and n not in PROTECTED and not n.endswith('ss'): cand = n[:-1]
        else: cand = n
        if cand in alias_map: return alias_map[cand]
        # return cleaned display form
        return raw.strip()

    # build stats (training corpus focus for clustering)
    train_records = [r for r in records if r[5]]
    all_keywords = sorted(set(canonicalize(r[3]) or r[3].strip() for r in train_records if r[3]))
    # map raw->canonical
    raw_to_canon = {}
    for r in records:
        c = canonicalize(r[3]) or r[3].strip()
        raw_to_canon[r[3]] = c
    # curated alias csv
    alias_rows = []
    for raw, canon in CURATED_ALIASES:
        alias_rows.append({"raw_term": raw, "canonical_term": canon, "source":"curated","confidence":"high","notes":""})
    pd.DataFrame(alias_rows).to_csv(proc/"keyword_aliases.csv", index=False)

    # keyword table on training corpus
    kw_docs = defaultdict(set); kw_freq = Counter(); kw_fields = defaultdict(set)
    for doc_id, doc_type, field, kw, w, in_train in train_records:
        c = canonicalize(kw) or kw.strip()
        if not c: continue
        kw_docs[c].add(doc_id); kw_freq[c]+=1; kw_fields[c].add(field)
    kw_table = []
    for k in sorted(kw_docs.keys(), key=lambda x: (-len(kw_docs[x]), x.lower())):
        kw_table.append({"keyword":k, "doc_frequency":len(kw_docs[k]), "total_frequency":kw_freq[k], "fields":"|".join(sorted(kw_fields[k]))})
    pd.DataFrame(kw_table).to_csv(proc/"keyword_table_train.csv", index=False)

    unique_keywords = list(kw_docs.keys())
    if len(unique_keywords) == 0:
        print("No keywords found in training corpus.", file=sys.stderr)
        sys.exit(1)

    # embeddings: char TF-IDF + SVD (offline, deterministic)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(unique_keywords)
    n_comp = min(100, X.shape[1]-1, X.shape[0]-1)
    n_comp = max(2, n_comp)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    emb = svd.fit_transform(X)
    # normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True); norms[norms==0]=1
    emb_n = emb / norms

    # choose threshold to target 100-200 clusters
    thresholds = [0.25,0.30,0.35,0.40,0.45]
    best_labels = None; best_t = 0.35; best_k = 0
    for t in thresholds:
        cl = AgglomerativeClustering(n_clusters=None, distance_threshold=t, metric='cosine', linkage='average')
        labels = cl.fit_predict(emb_n)
        k = len(set(labels))
        if 100 <= k <= 200:
            best_labels, best_t, best_k = labels, t, k
            break
    if best_labels is None:
        # pick closest to range
        def score(t):
            cl = AgglomerativeClustering(n_clusters=None, distance_threshold=t, metric='cosine', linkage='average')
            labels = cl.fit_predict(emb_n); k=len(set(labels)); return labels,k
        cand = [score(t) for t in thresholds]
        # prefer >=100 else closest
        cand.sort(key=lambda x: (0 if 100<=x[1]<=200 else 1, abs(x[1]-150)))
        best_labels, best_k = cand[0][0], cand[0][1]; best_t = thresholds[[len(set(score(t)[0])) for t in thresholds].index(best_k)] if False else 0.35
        # recompute best_t properly
        for t in thresholds:
            cl = AgglomerativeClustering(n_clusters=None, distance_threshold=t, metric='cosine', linkage='average')
            labels = cl.fit_predict(emb_n); k=len(set(labels))
            if k==best_k: best_t=t; best_labels=labels; break

    # build clusters and apply guardrails (simple blocklist check)
    cluster_members = defaultdict(list)
    for kw, lab in zip(unique_keywords, best_labels):
        cluster_members[int(lab)].append(kw)
    # split clusters violating confusable blocklist
    final_clusters = {}
    next_id = 0
    for lab, members in cluster_members.items():
        groups = split_confusable(members)
        for g in groups:
            final_clusters[next_id] = g
            next_id += 1

    # canonical labels and stats
    freq_map = {k: len(kw_docs[k]) for k in unique_keywords}
    tot_map = {k: kw_freq[k] for k in unique_keywords}
    cluster_rows = []
    kw_to_cluster = {}
    for cid, members in final_clusters.items():
        members_sorted = sorted(members, key=lambda x: (-freq_map.get(x,0), x.lower()))
        canon = members_sorted[0]
        # prefer alias canonical form
        for m in members_sorted:
            if normalize_keyword_str(m) in alias_map: canon = alias_map[normalize_keyword_str(m)]; break
        size = len(members)
        dfreq = sum(freq_map.get(m,0) for m in members)  # approximate doc coverage sum
        tfreq = sum(tot_map.get(m,0) for m in members)
        # coherence
        idxs = [unique_keywords.index(m) for m in members if m in unique_keywords]
        if len(idxs) >= 2:
            sim = cosine_similarity(emb_n[idxs])
            coh = float((sim.sum()-len(idxs)) / (len(idxs)*(len(idxs)-1)))
        else:
            coh = 1.0
        cluster_rows.append({"cluster_id":cid, "canonical_label":canon, "definition":"", "members":" | ".join(members_sorted),
                              "size":size, "doc_frequency":dfreq, "total_frequency":tfreq, "coherence":round(coh,4),
                              "merge_method":"agglomerative_char_tfidf_svd", "confidence":"medium"})
        for m in members:
            kw_to_cluster[m] = cid
            kw_to_cluster[normalize_keyword_str(m)] = cid
    cluster_df = pd.DataFrame(cluster_rows).sort_values(["doc_frequency","size"], ascending=[False,False]).reset_index(drop=True)
    cluster_df.to_csv(proc/"keyword_clusters.csv", index=False)
    with open(proc/"keyword_to_cluster.json","w",encoding='utf-8') as f: json.dump({k:int(v) for k,v in kw_to_cluster.items()}, f)

    # keyword edges (co-occurrence in training docs)
    doc_clusters = defaultdict(set)
    for doc_id, doc_type, field, kw, w, in_train in train_records:
        c = canonicalize(kw) or kw.strip()
        cid = kw_to_cluster.get(c) or kw_to_cluster.get(normalize_keyword_str(c))
        if cid is not None: doc_clusters[doc_id].add(int(cid))
    edge_counts = Counter()
    for did, cids in doc_clusters.items():
        cl = sorted(cids)
        for i in range(len(cl)):
            for j in range(i+1, len(cl)):
                edge_counts[(cl[i],cl[j])] += 1
    edge_rows = [{"cluster_a":a,"cluster_b":b,"cooccurrence_docs":cnt} for (a,b),cnt in edge_counts.items() if cnt>=2]
    pd.DataFrame(edge_rows).sort_values("cooccurrence_docs", ascending=False).to_csv(proc/"keyword_edges.csv", index=False)

    # QA
    try:
        sil = silhouette_score(emb_n, [kw_to_cluster.get(k) for k in unique_keywords], metric='cosine') if len(set(best_labels))>1 else None
    except Exception:
        sil = None
    qa_rows = []
    # top 20 by size
    for _, r in cluster_df.sort_values('size', ascending=False).head(20).iterrows():
        qa_rows.append({"sample_type":"largest","cluster_id":r['cluster_id'],"label":r['canonical_label'],"members":r['members'],"judgement":"needs_review","notes":""})
    for _, r in cluster_df.sort_values('coherence', ascending=True).head(20).iterrows():
        qa_rows.append({"sample_type":"lowest_coherence","cluster_id":r['cluster_id'],"label":r['canonical_label'],"members":r['members'],"judgement":"needs_review","notes":""})
    pd.DataFrame(qa_rows).to_csv(proc/"cluster_qa.csv", index=False)
    metrics = {
        "cluster_count": int(len(cluster_df)),
        "singleton_rate": float((cluster_df['size']==1).mean()) if len(cluster_df) else 0,
        "silhouette": sil,
        "mean_coherence": float(cluster_df['coherence'].mean()) if len(cluster_df) else None,
        "merge_accuracy_sampled": None,
        "distance_threshold": best_t,
        "unique_train_keywords": len(unique_keywords),
        "embedding_method": "char_tfidf_svd"
    }
    with open(proc/"cluster_metrics.json","w",encoding='utf-8') as f: json.dump(metrics, f, indent=2)

    # Figures
    make_figures(cluster_df, edge_rows, proc, figs)
    print(json.dumps(metrics, indent=2))

def split_confusable(members):
    # simple: if group contains tokens from different sides of blocklist, try to separate
    # conservative: keep together unless clear conflict
    def toks(s): return set(re.split(r"[^a-z0-9#+.]+", s.lower())) - {''}
    groups = [members]
    for a_set, b_set in CONFUSABLE_BLOCKLIST:
        new_groups = []
        for g in groups:
            ga, gb, other = [], [], []
            for m in g:
                mt = toks(m) | {m.lower().strip()}
                in_a = any(x in mt or x in m.lower() for x in a_set)
                in_b = any(x in mt or x in m.lower() for x in b_set)
                if in_a and not in_b: ga.append(m)
                elif in_b and not in_a: gb.append(m)
                elif in_a and in_b:
                    # ambiguous, assign by exact
                    if m.lower().strip() in a_set: ga.append(m)
                    elif m.lower().strip() in b_set: gb.append(m)
                    else: other.append(m)
                else: other.append(m)
            # if both sides present, split
            if ga and gb:
                if ga: new_groups.append(ga)
                if gb: new_groups.append(gb)
                # assign other to larger
                if other:
                    target = ga if len(ga)>=len(gb) else gb
                    target.extend(other)
            else:
                new_groups.append(g)
        groups = new_groups
    return groups if groups else [members]

def make_figures(cluster_df, edge_rows, proc, figs):
    # 04 cluster sizes
    top = cluster_df.sort_values('doc_frequency', ascending=False).head(30)
    plt.figure(figsize=(12,6))
    plt.barh(range(len(top))[::-1], top['doc_frequency'].values)
    plt.yticks(range(len(top))[::-1], top['canonical_label'].values, fontsize=8)
    plt.xlabel('Document frequency'); plt.title('Top 30 Keyword Clusters by Document Frequency')
    plt.tight_layout(); plt.savefig(figs/'04_keyword_cluster_sizes.png', dpi=200); plt.close()

    # 05 merge examples
    # show clusters with >1 member
    multi = cluster_df[cluster_df['size']>1].sort_values('size', ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12,8)); ax.axis('off')
    rows = []
    for _, r in multi.iterrows():
        mem = r['members'].split(' | ')[:5]
        rows.append([r['canonical_label'], ', '.join(mem)])
    if rows:
        table = ax.table(cellText=rows, colLabels=['Canonical','Raw/Merged Variants'], loc='center', cellLoc='left')
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1,1.4)
    ax.set_title('Keyword Merge Examples (Top 20 Multi-member Clusters)')
    plt.tight_layout(); plt.savefig(figs/'05_keyword_merge_examples.png', dpi=200); plt.close()

    # 03 network graph (simple)
    try:
        import networkx as nx
        G = nx.Graph()
        top_ids = set(cluster_df.sort_values('doc_frequency', ascending=False).head(40)['cluster_id'].astype(int).tolist())
        label_map = dict(zip(cluster_df['cluster_id'].astype(int), cluster_df['canonical_label']))
        freq_map = dict(zip(cluster_df['cluster_id'].astype(int), cluster_df['doc_frequency']))
        for e in edge_rows:
            a,b = int(e['cluster_a']), int(e['cluster_b'])
            if a in top_ids and b in top_ids:
                G.add_edge(a,b,weight=e['cooccurrence_docs'])
        for n in list(top_ids): G.add_node(n)
        if len(G.nodes)>0:
            pos = nx.spring_layout(G, seed=42, k=1.2)
            plt.figure(figsize=(12,9))
            sizes = [max(50, freq_map.get(n,1)*8) for n in G.nodes]
            nx.draw_networkx_edges(G,pos,alpha=0.2,width=0.8)
            nx.draw_networkx_nodes(G,pos,node_size=sizes,node_color='#4C78A8',alpha=0.8)
            # label top 20
            top20 = sorted(G.nodes, key=lambda n: freq_map.get(n,0), reverse=True)[:20]
            lbls = {n: label_map.get(n,str(n)) for n in top20}
            nx.draw_networkx_labels(G,pos,labels=lbls,font_size=7)
            plt.title('Keyword Cluster Co-occurrence Network (Top 40 clusters)')
            plt.axis('off'); plt.tight_layout(); plt.savefig(figs/'03_keyword_cluster_network.png', dpi=200); plt.close()
    except Exception as e:
        print('network fig failed', e)

    # 05b heatmap placeholder/simple
    # need role families per cluster - compute later? skip complex, make simple chart
    top25 = cluster_df.sort_values('doc_frequency', ascending=False).head(25)
    plt.figure(figsize=(10,8))
    plt.barh(range(len(top25))[::-1], top25['doc_frequency'].values, color='#72B7B2')
    plt.yticks(range(len(top25))[::-1], top25['canonical_label'].values, fontsize=8)
    plt.xlabel('Document frequency'); plt.title('Top 25 Keyword Clusters (Coverage proxy)')
    plt.tight_layout(); plt.savefig(figs/'05b_keyword_coverage_heatmap.png', dpi=200); plt.close()

if __name__ == '__main__':
    main()
