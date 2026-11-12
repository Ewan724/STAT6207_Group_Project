"""Diagnostic 5: what input representation should the autoencoder compress?

Finding so far: every keyword-only system underperforms TF-IDF on raw text.  The
merged keyword space is a lossy summary -- 12 canonical skills per document on
average, versus ~3,100 characters of evidence.  So the autoencoder is being
asked to compress an already-degraded signal.

This experiment asks whether giving the AE a richer input fixes that, and how
much of the gap is due to the keyword bottleneck versus the autoencoder itself.
Four candidate inputs, all compressed to the same 32-d bottleneck:

  kw        IDF-weighted merged hard skills + 4 structural      (current)
  kw+txt    the above concatenated with a 128-d SVD of the raw TF-IDF
            matrix (a deterministic, local, label-free text summary)
  txt       the 128-d text SVD + 4 structural
  nomic     768-d nomic-embed-text embedding of the document text + 4 structural

Selection is on latent purity@5 (job-family neighbourhood quality), which uses
only hand-assigned families and never a relevance grade.
"""
import importlib.util
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

spec = importlib.util.spec_from_file_location(
    "ae", os.path.join(BASE, "scripts", "06_autoencoder.py"))
ae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ae)

spec2 = importlib.util.spec_from_file_location(
    "ae2", os.path.join(BASE, "scripts", "06d_ae_tune.py"))
ae2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ae2)

import sys  # noqa: E402
sys.path.insert(0, os.path.join(BASE, "scripts"))
import llm  # noqa: E402

FAM_JD = ae.FAM_JD
FAM_ALL = json.load(open(os.path.join(DATA, "cv_family.json"), encoding="utf-8"))
FAM_ALL.update(FAM_JD)


def text_svd(docs, texts, k=128):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                          ngram_range=(1, 2), min_df=1, max_features=30000)
    M = vec.fit_transform([texts[d] for d in docs])
    k = min(k, M.shape[1] - 1, M.shape[0] - 1)
    svd = TruncatedSVD(n_components=k, random_state=42)
    Z = svd.fit_transform(M)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    return Z.astype(np.float32), k


def norm01(Z):
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    return ((Z - lo) / np.maximum(hi - lo, 1e-9)).astype(np.float32)


def main():
    skills = json.load(open(os.path.join(DATA, "skills.json"), encoding="utf-8"))
    ext = ae.load_ext()
    docs, vocab, X, X_skill, struct, idf = ae.build_matrix(skills, ext)
    cvs = {c["cv_id"]: c["text"] for c in json.load(
        open(os.path.join(DATA, "cvs_raw.json"), encoding="utf-8"))}
    jds = {j["jd_id"]: j["text"] for j in json.load(
        open(os.path.join(DATA, "jds_raw.json"), encoding="utf-8"))}
    texts = {**cvs, **jds}

    T, k = text_svd(docs, texts, 128)
    print(f"text SVD -> {T.shape} (explained var "
          f"{0:.2f})".replace("0.00", "n/a"))

    # nomic embeddings of the document text (cached to disk; slow otherwise)
    cache = os.path.join(DATA, "nomic_docs.json")
    if os.path.exists(cache):
        N = np.array([json.load(open(cache, encoding="utf-8"))[d] for d in docs],
                     dtype=np.float32)
    else:
        print("embedding 110 documents with nomic-embed-text ...", flush=True)
        vecs = llm.embed([texts[d][:4000] for d in docs])
        json.dump({d: v for d, v in zip(docs, vecs)},
                  open(cache, "w", encoding="utf-8"))
        N = np.array(vecs, dtype=np.float32)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    print(f"nomic -> {N.shape}")

    n_sk = len(vocab)
    variants = {
        "kw": (X, n_sk),
        "kw+txt": (np.concatenate([X, norm01(T) * 0.5], axis=1), n_sk),
        "txt": (np.concatenate([norm01(T) * 0.5, struct], axis=1),
                norm01(T).shape[1]),
        "nomic": (np.concatenate([N, struct], axis=1), N.shape[1]),
    }
    out = []
    for name, (Xv, ns) in variants.items():
        model, z, loss = ae2.run(Xv, ns, 32, 192, 1, 0.20, 1e-4, 1800)
        p = ae2.purity(z, docs, FAM_ALL)
        out.append({"variant": name, "dim": int(Xv.shape[1]),
                    "purity": round(p, 4), "loss": round(loss, 5)})
        print(f"{name:8s} dim {Xv.shape[1]:4d} -> purity@5 {p:.3f}", flush=True)

    # raw (uncompressed) reference points
    for name, M in (("txt-raw", norm01(T)), ("nomic-raw", N)):
        Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
        sim = Mn @ Mn.T
        vals = []
        for j in FAM_JD:
            i = docs.index(j)
            order = [x for x in np.argsort(-sim[i]) if docs[x] != j][:5]
            vals.append(sum(1 for x in order
                            if FAM_ALL.get(docs[x]) == FAM_JD[j]) / 5)
        print(f"{name:8s} (no AE)      -> purity@5 {np.mean(vals):.3f}")
        out.append({"variant": name + "-raw", "dim": int(M.shape[1]),
                    "purity": round(float(np.mean(vals)), 4), "loss": None})

    json.dump(out, open(os.path.join(DATA, "ae_sweep5.json"), "w",
                        encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
