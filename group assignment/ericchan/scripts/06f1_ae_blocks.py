"""Diagnostic 4: does adding the domain / soft-skill blocks help the latent space?

The AE so far sees only the hard-skill block plus four scalar structural
features.  But the profession signal in this corpus lives largely in the
*domain* field ("fintech", "semiconductor", "logistics"), which the current
input does not include at all.  Adding it should sharpen the latent geometry.

We test three input variants under the best objective found so far
(cosine reconstruction, L32/H192, 1800 epochs):
  hard            - 443 IDF skill dims + 4 structural   (current)
  hard + domain   - plus IDF-weighted domain one-hot
  hard + domain + soft - plus IDF-weighted soft-skill one-hot

The risk with the third variant is that soft skills ("communication") are
near-universal and could blur the space, so we measure rather than assume.
"""
import importlib.util
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
SEED = 42

spec = importlib.util.spec_from_file_location(
    "ae", os.path.join(BASE, "scripts", "06_autoencoder.py"))
ae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ae)

spec2 = importlib.util.spec_from_file_location(
    "ae2", os.path.join(BASE, "scripts", "06d_ae_tune.py"))
ae2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ae2)

FAM_JD = ae.FAM_JD


def block(doc_sets, docs, min_df=2):
    present = {s for ss in doc_sets.values() for s in ss}
    vocab = sorted(present)
    idx = {s: i for i, s in enumerate(vocab)}
    B = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for i, d in enumerate(docs):
        for s in doc_sets.get(d, []):
            if s in idx:
                B[i, idx[s]] = 1.0
    df = B.sum(axis=0)
    idf = np.log((1.0 + len(docs)) / (1.0 + df)) + 1.0
    idf = idf / idf.max()
    return B * idf[None, :], vocab


def main():
    skills = json.load(open(os.path.join(DATA, "skills.json"), encoding="utf-8"))
    ext = ae.load_ext()
    docs, vocab, X, X_skill, struct, idf = ae.build_matrix(skills, ext)
    fam_of = dict(FAM_JD)
    fam_of.update(json.load(open(os.path.join(DATA, "cv_family.json"),
                                encoding="utf-8")))
    n_sk = len(vocab)

    dom_block, dom_vocab = block(skills.get("doc_dom", {}), docs)
    soft_block, soft_vocab = block(skills.get("doc_soft", {}), docs)
    print(f"hard {X_skill.shape}  domain {dom_block.shape}  "
          f"soft {soft_block.shape}")

    variants = {
        "hard": (X, n_sk),
        "hard+domain": (np.concatenate([X_skill, dom_block, struct], axis=1),
                        n_sk + dom_block.shape[1]),
        "hard+domain+soft": (np.concatenate([X_skill, dom_block, soft_block,
                                            struct], axis=1),
                             n_sk + dom_block.shape[1] + soft_block.shape[1]),
    }
    out = []
    for name, (Xv, n_skill) in variants.items():
        model, z, loss = ae2.run(Xv, n_skill, 32, 192, 1, 0.20, 1e-4, 1800)
        p = ae2.purity(z, docs, fam_of)
        out.append({"variant": name, "dim": int(Xv.shape[1]),
                    "purity": round(p, 4), "loss": round(loss, 5)})
        print(f"{name:20s} dim {Xv.shape[1]:4d} -> purity@5 {p:.3f} "
              f"loss {loss:.5f}", flush=True)

    json.dump(out, open(os.path.join(DATA, "ae_sweep4.json"), "w",
                        encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
