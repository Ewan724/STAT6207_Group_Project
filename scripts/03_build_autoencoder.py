#!/usr/bin/env python3
import argparse, json, math, os, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group assignment"))
from src.config import load_config, add_common_args

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(); add_common_args(parser); args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["_out_dir"]); proc = out/"data"/"processed"; models = out/"models"; figs = out/"results"/"figures"; tables = out/"results"/"tables"
    models.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.get('seed',42))

    for p in [proc/"documents.jsonl", proc/"jd_extracted.jsonl", proc/"cv_extracted.jsonl", proc/"keyword_clusters.csv", proc/"keyword_to_cluster.json"]:
        if not p.exists():
            print(f"Missing {p}. Run previous stages.", file=sys.stderr); sys.exit(1)
    docs = [json.loads(l) for l in open(proc/"documents.jsonl", encoding='utf-8')]
    docs_by_id = {d['doc_id']: d for d in docs}
    jd_ext = {d['doc_id']: d for d in [json.loads(l) for l in open(proc/"jd_extracted.jsonl", encoding='utf-8')]}
    cv_ext = {d['doc_id']: d for d in [json.loads(l) for l in open(proc/"cv_extracted.jsonl", encoding='utf-8')]}
    clusters = pd.read_csv(proc/"keyword_clusters.csv")
    kw_to_cluster = json.load(open(proc/"keyword_to_cluster.json", encoding='utf-8'))
    ktc = {}
    for k,v in kw_to_cluster.items():
        ktc[str(k)] = int(v); ktc[str(k).lower()] = int(v)
    cluster_ids = clusters['cluster_id'].astype(int).tolist()
    cid_to_idx = {cid:i for i,cid in enumerate(cluster_ids)}
    n_clusters = len(cluster_ids)

    doc_ids = [d['doc_id'] for d in docs]
    X_counts = np.zeros((len(doc_ids), n_clusters), dtype=np.float32)
    unmatched = []
    for i, did in enumerate(doc_ids):
        meta = docs_by_id[did]
        vec = X_counts[i]
        if meta['doc_type'] in ('gt_jd','unlabelled_jd'):
            ext = jd_ext.get(did, {})
            add_weighted(vec, ext.get('must_have_keywords',[]), 3.0, ktc, cid_to_idx, unmatched, did)
            add_weighted(vec, ext.get('nice_to_have_keywords',[]), 1.0, ktc, cid_to_idx, unmatched, did)
            add_weighted(vec, ext.get('tools_technologies',[]), 2.0, ktc, cid_to_idx, unmatched, did)
            add_weighted(vec, ext.get('domain_knowledge',[]), 1.5, ktc, cid_to_idx, unmatched, did)
            add_weighted(vec, ext.get('key_responsibilities',[]), 0.5, ktc, cid_to_idx, unmatched, did)
            add_weighted(vec, ext.get('soft_skills',[]), 0.5, ktc, cid_to_idx, unmatched, did)
        else:
            ext = cv_ext.get(did, {})
            add_weighted(vec, ext.get('skills',[]), 2.0, ktc, cid_to_idx, unmatched, did)
            add_weighted(vec, ext.get('tools_technologies',[]), 2.0, ktc, cid_to_idx, unmatched, did)
            add_weighted(vec, ext.get('domains',[]), 1.5, ktc, cid_to_idx, unmatched, did)
            add_weighted(vec, ext.get('soft_skills',[]), 0.5, ktc, cid_to_idx, unmatched, did)
            add_weighted(vec, ext.get('certifications',[]), 0.5, ktc, cid_to_idx, unmatched, did)
            for edu in (ext.get('education') or []):
                if isinstance(edu, dict):
                    for f in ['degree','field']:
                        if edu.get(f): add_weighted(vec, [edu.get(f)], 0.5, ktc, cid_to_idx, unmatched, did)
    sparse_counts = sparse.csr_matrix(X_counts)
    sparse.save_npz(proc/"document_keyword_matrix.npz", sparse_counts)
    tfidf = TfidfTransformer(sublinear_tf=True, norm='l2')
    X_tfidf = tfidf.fit_transform(sparse_counts)
    X_dense = X_tfidf.toarray().astype(np.float32)
    vec_info = {"cluster_ids": cluster_ids, "idf": tfidf.idf_.tolist(), "n_clusters": n_clusters, "doc_ids": doc_ids}
    with open(models/"vectorizer.json","w",encoding='utf-8') as f: json.dump(vec_info, f)

    idx_rows = []
    for i,did in enumerate(doc_ids):
        m = docs_by_id[did]
        idx_rows.append({"row":i,"doc_id":did,"doc_type":m['doc_type'],"split":m.get('split'),"title":m.get('title'),"role_family":m.get('role_family')})
    pd.DataFrame(idx_rows).to_csv(proc/"document_matrix_index.csv", index=False)
    if unmatched:
        pd.DataFrame(unmatched[:20000]).to_csv(proc/"heldout_keyword_coverage.csv", index=False)

    train_mask = np.array([(docs_by_id[d]['doc_type']=='pool_cv') or (docs_by_id[d]['doc_type']=='unlabelled_jd' and docs_by_id[d].get('split')=='train_jd') for d in doc_ids])
    heldout_mask = np.array([(docs_by_id[d]['doc_type'] in ('gt_jd','gt_cv')) for d in doc_ids])
    train_idx = np.where(train_mask)[0]
    rng = np.random.default_rng(42)
    perm = rng.permutation(train_idx)
    n_val = max(1, int(len(perm)*0.15))
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    X_train = X_dense[tr_idx]; X_val = X_dense[val_idx]
    input_dim = X_dense.shape[1]
    latent_dim = int(cfg.get('autoencoder',{}).get('latent_dim',32))

    # sklearn MLP as autoencoder: input->256->64->32->64->256->input
    mlp = MLPRegressor(hidden_layer_sizes=(256,64,latent_dim,64,256), activation='relu', solver='adam',
                       alpha=1e-4, batch_size=min(32,len(X_train)), learning_rate_init=1e-3,
                       max_iter=500, early_stopping=True, n_iter_no_change=40, validation_fraction=0.15,
                       random_state=42, verbose=False)
    # fit on training corpus only (primary protocol)
    mlp.fit(X_train, X_train)
    # "best epoch"/loss not directly exposed same way; use loss_curve_
    history = []
    for i,(tr,va) in enumerate(zip(getattr(mlp,'loss_curve_',[]), getattr(mlp,'validation_scores_',[]))):
        history.append({"epoch":i+1,"train_mse":float(tr),"val_mse":float(1-va) if va is not None else None})
    if not history and hasattr(mlp,'loss_curve_'):
        for i,tr in enumerate(mlp.loss_curve_):
            history.append({"epoch":i+1,"train_mse":float(tr),"val_mse":None})
    pd.DataFrame(history).to_csv(models/"ae_training_history.csv", index=False)
    best_epoch = getattr(mlp,'n_iter_', len(history))

    # extract bottleneck activations (encoder part: layers 0,1 -> 256,64, latent at 2)
    Z = encode_mlp(mlp, X_dense, latent_layer_index=2)
    zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    emb_cols = [f"dim_{i+1:02d}" for i in range(Z.shape[1])]
    emb_df = pd.DataFrame(zn, columns=emb_cols)
    emb_df.insert(0,'role_family',[docs_by_id[d].get('role_family') for d in doc_ids])
    emb_df.insert(0,'title',[docs_by_id[d].get('title') for d in doc_ids])
    emb_df.insert(0,'split',[docs_by_id[d].get('split') for d in doc_ids])
    emb_df.insert(0,'doc_type',[docs_by_id[d].get('doc_type') for d in doc_ids])
    emb_df.insert(0,'doc_id',doc_ids)
    emb_df.to_csv(proc/"document_embeddings.csv", index=False)

    # save model
    import pickle
    with open(models/"autoencoder.pkl","wb") as f: pickle.dump(mlp,f)
    # also save a torch-like placeholder note? keep autoencoder.pt name not applicable; save config
    recon = mlp.predict(X_dense)
    val_recon = mlp.predict(X_val) if len(X_val) else recon[:1]
    val_mse = float(np.mean((X_val - val_recon)**2)) if len(X_val) else None
    ae_config = {
        "input_dim": int(input_dim), "latent_dim": latent_dim, "hidden_dims": [256,64],
        "dropout": 0.0, "denoising": False, "seed": 42, "best_epoch": int(best_epoch),
        "validation_mse": val_mse, "cluster_count": n_clusters, "train_docs": int(len(X_train)),
        "val_docs": int(len(X_val)), "heldout_docs": int(len(np.where(heldout_mask)[0])),
        "protocol": "primary_holdout_gt", "implementation": "sklearn_mlp_autoencoder"
    }
    with open(models/"autoencoder_config.json","w",encoding='utf-8') as f: json.dump(ae_config, f, indent=2)

    make_ae_figures(history, best_epoch, X_dense, recon, Z, docs_by_id, doc_ids, out, models, figs, tables, ae_config)
    print(json.dumps(ae_config, indent=2))

def encode_mlp(mlp, X, latent_layer_index=2):
    # forward through layers up to latent_layer_index (0-based hidden layer outputs after activation)
    # MLPRegressor: coefs_[i], intercepts_[i] for layer i; activation on hidden layers, identity on output
    hidden_activation = mlp.activation  # 'relu'
    A = X
    for i in range(latent_layer_index+1):
        A = A @ mlp.coefs_[i] + mlp.intercepts_[i]
        if hidden_activation == 'relu':
            A = np.maximum(A, 0)
        elif hidden_activation == 'tanh':
            A = np.tanh(A)
        elif hidden_activation == 'logistic':
            A = 1/(1+np.exp(-A))
    return A.astype(np.float32)

def add_weighted(vec, kws, weight, ktc, cid_to_idx, unmatched, doc_id):
    if not kws: return
    for kw in kws:
        if isinstance(kw, dict): continue
        s = str(kw).strip()
        if not s: continue
        cid = ktc.get(s) or ktc.get(s.lower())
        if cid is None:
            unmatched.append({"doc_id":doc_id, "keyword":s, "reason":"unmatched"})
            continue
        idx = cid_to_idx.get(int(cid))
        if idx is not None:
            vec[idx] += weight

def make_ae_figures(history, best_epoch, X_dense, recon, Z, docs_by_id, doc_ids, out, models, figs, tables, cfg):
    hdf = pd.DataFrame(history)
    plt.figure(figsize=(8,4))
    if 'train_mse' in hdf: plt.plot(hdf['epoch'], hdf['train_mse'], label='train loss')
    if 'val_mse' in hdf and hdf['val_mse'].notna().any(): plt.plot(hdf['epoch'], hdf['val_mse'], label='val (proxy)')
    plt.axvline(best_epoch, color='red', linestyle='--', label=f'best iter {best_epoch}')
    plt.xlabel('Iteration'); plt.ylabel('Loss/MSE'); plt.title('Autoencoder Training Curve (sklearn MLP)'); plt.legend(); plt.tight_layout()
    plt.savefig(figs/'07_ae_loss_curves.png', dpi=200); plt.close()

    mse = ((recon-X_dense)**2).mean(axis=1)
    types = [docs_by_id[d]['doc_type'] for d in doc_ids]
    plt.figure(figsize=(8,4))
    for t in sorted(set(types)):
        vals = [m for m,tt in zip(mse,types) if tt==t]
        plt.hist(vals, bins=20, alpha=0.5, label=t)
    plt.xlabel('Reconstruction MSE'); plt.ylabel('Count'); plt.title('Per-document Reconstruction Error'); plt.legend(); plt.tight_layout()
    plt.savefig(figs/'08_ae_reconstruction_error.png', dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(10,3)); ax.axis('off')
    labels = [f"Input\n{cfg['input_dim']}", "256", "64", f"Bottleneck\n{cfg['latent_dim']}", "64", "256", f"Output\n{cfg['input_dim']}"]
    for i,lab in enumerate(labels):
        rect = plt.Rectangle((i*1.3,0.3),1.0,0.6, fill=False)
        ax.add_patch(rect); ax.text(i*1.3+0.5,0.6,lab,ha='center',va='center',fontsize=9)
        if i < len(labels)-1: ax.arrow(i*1.3+1.0,0.6,0.3,0,head_width=0.05,length_includes_head=True)
    ax.set_xlim(-0.2, i*1.3+1.2); ax.set_ylim(0,1.2); ax.set_title('Autoencoder Architecture (sklearn MLP)')
    plt.tight_layout(); plt.savefig(figs/'06_ae_architecture.png', dpi=200); plt.close()

    pca = PCA(n_components=2, random_state=42); Z2 = pca.fit_transform(Z)
    plot_latent(Z2, doc_ids, docs_by_id, figs/'09_latent_space_pca.png', 'Latent Space PCA (2D)', out)
    try:
        from sklearn.manifold import TSNE
        Zt = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(doc_ids)//4))).fit_transform(Z)
        plot_latent(Zt, doc_ids, docs_by_id, figs/'10_latent_space_umap.png', 'Latent Space t-SNE (2D)', out)
    except Exception as e:
        print('tsne failed', e)

    rows=[]
    for name, comp in [('PCA-32',32),('SVD-32',32)]:
        if name.startswith('PCA'): red = PCA(n_components=min(comp, X_dense.shape[1]-1), random_state=42)
        else: red = TruncatedSVD(n_components=min(comp, X_dense.shape[1]-1), random_state=42)
        z = red.fit_transform(X_dense); rec = red.inverse_transform(z)
        mse_r = float(((X_dense-rec)**2).mean())
        ev = float(np.sum(getattr(red,'explained_variance_ratio_', [np.nan])))
        rows.append({'method':name,'mse':mse_r,'explained_variance':ev})
    ae_mse = float(((recon-X_dense)**2).mean())
    rows.append({'method':'AE-32','mse':ae_mse,'explained_variance':np.nan})
    pd.DataFrame(rows).to_csv(tables/'ae_vs_svd.csv', index=False)

    ex_rows=[]
    sample = [d for d in doc_ids if docs_by_id[d]['doc_type']=='pool_cv'][:3] + [d for d in doc_ids if docs_by_id[d]['doc_type']=='gt_jd'][:2]
    for did in sample:
        i = doc_ids.index(did)
        top_in = np.argsort(-X_dense[i])[:10]; top_out = np.argsort(-recon[i])[:10]
        ex_rows.append({'doc_id':did,'top_input_idx':','.join(map(str,top_in.tolist())),'top_recon_idx':','.join(map(str,top_out.tolist()))})
    pd.DataFrame(ex_rows).to_csv(tables/'reconstruction_examples.csv', index=False)
    plt.figure(figsize=(6,3)); plt.text(0.01,0.5,'Reconstruction examples saved to results/tables/reconstruction_examples.csv', fontsize=10)
    plt.axis('off'); plt.tight_layout(); plt.savefig(figs/'08b_ae_reconstruction_examples.png', dpi=200); plt.close()

def plot_latent(Z2, doc_ids, docs_by_id, path, title, out):
    plt.figure(figsize=(9,6))
    type_colors = {'pool_cv':'#4C78A8','gt_cv':'#F58518','gt_jd':'#54A24B','unlabelled_jd':'#B279A2'}
    for t,c in type_colors.items():
        idx = [i for i,d in enumerate(doc_ids) if docs_by_id[d]['doc_type']==t]
        if not idx: continue
        plt.scatter(Z2[idx,0], Z2[idx,1], s=18, alpha=0.7, label=t, c=c)
    try:
        gt = pd.read_csv(out/"data"/"processed"/"ground_truth.csv")
        id_to_i = {d:i for i,d in enumerate(doc_ids)}
        for _,r in gt.iterrows():
            j=r['jd_id']; c=r['hired_cv_id']
            if j in id_to_i and c in id_to_i:
                i1=id_to_i[j]; i2=id_to_i[c]
                plt.plot([Z2[i1,0],Z2[i2,0]],[Z2[i1,1],Z2[i2,1]],'k-',alpha=0.3,linewidth=0.8)
                plt.text(Z2[i1,0],Z2[i1,1],j,fontsize=7)
    except Exception:
        pass
    plt.legend(fontsize=8); plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

if __name__ == '__main__':
    main()
