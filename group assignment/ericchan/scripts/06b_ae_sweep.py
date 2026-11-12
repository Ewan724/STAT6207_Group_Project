"""Diagnostic sweep: how good is the shared space, and what does the AE add?

Measures latent-space neighbour purity for
  (a) raw IDF-weighted skill cosine (the input space itself),
  (b) several AE configurations,
so we can tell whether the autoencoder is compressing useful structure or
destroying it.  Purity is measured with NO reference to relevance grades: for
each JD, the share of its 5 nearest documents that belong to the same hand-
assigned job family.  Chance is roughly 0.10.
"""
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
sys.path.insert(0, os.path.join(BASE, "scripts"))

SEED = 42
FAM_JD = {"P01": "finance", "P02": "sales", "P03": "marketing", "P04": "data",
          "P05": "product", "P06": "data", "P07": "marketing",
          "P08": "engineering_hw", "P09": "engineering_sw", "P10": "design"}

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "ae", os.path.join(BASE, "scripts", "06_autoencoder.py"))
ae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ae)


def purity(Z, docs, fam_of, k=5):
    sim = Z @ Z.T
    vals = []
    for j in FAM_JD:
        i = docs.index(j)
        order = [x for x in np.argsort(-sim[i]) if docs[x] != j][:k]
        vals.append(sum(1 for x in order
                        if fam_of.get(docs[x]) == FAM_JD[j]) / k)
    return float(np.mean(vals))


def fam_map():
    fam = dict(FAM_JD)
    fam.update(json.load(open(os.path.join(DATA, "cv_family.json"),
                              encoding="utf-8")))
    return fam


def norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-9)


def main():
    skills = json.load(open(os.path.join(DATA, "skills.json"), encoding="utf-8"))
    ext = ae.load_ext()
    docs, vocab, X, X_skill, struct, idf = ae.build_matrix(skills, ext)
    fam_of = fam_map()

    # (a) input space
    p_in = purity(norm_rows(X_skill), docs, fam_of)
    p_full = purity(norm_rows(X), docs, fam_of)
    print(f"input space, skill block only : purity@5 = {p_in:.3f}")
    print(f"input space, skills + struct   : purity@5 = {p_full:.3f}")

    configs = [
        dict(latent=8, hidden=64, wd=1e-4, noise=0.20, epochs=900),
        dict(latent=16, hidden=96, wd=1e-4, noise=0.20, epochs=900),
        dict(latent=32, hidden=192, wd=1e-4, noise=0.20, epochs=900),
        dict(latent=16, hidden=96, wd=1e-3, noise=0.30, epochs=1500),
        dict(latent=8, hidden=48, wd=1e-3, noise=0.30, epochs=1500),
        dict(latent=16, hidden=64, wd=5e-3, noise=0.35, epochs=2000),
    ]
    out = []
    for cfg in configs:
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        ae.LATENT, ae.HIDDEN, ae.WD, ae.NOISE = (cfg["latent"], cfg["hidden"],
                                                 cfg["wd"], cfg["noise"])
        model = ae.AE(X.shape[1], latent=cfg["latent"], hidden=cfg["hidden"])
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3,
                                weight_decay=cfg["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                          T_max=cfg["epochs"])
        xt = torch.tensor(X)
        g = torch.Generator().manual_seed(SEED)
        n_sk = len(vocab)
        for ep in range(cfg["epochs"]):
            model.train()
            keep = (torch.rand(xt[:, :n_sk].shape, generator=g) > cfg["noise"]).float()
            xin = xt.clone()
            xin[:, :n_sk] = xt[:, :n_sk] * keep
            opt.zero_grad()
            _, rec = model(xin)
            loss = (F.mse_loss(rec[:, :n_sk], xt[:, :n_sk])
                    + F.mse_loss(rec[:, n_sk:], xt[:, n_sk:]))
            loss.backward()
            opt.step()
            sched.step()
        Z, recon = ae.encode(model, X)
        p = purity(Z, docs, fam_of)
        # reconstruction F1 on the raw binary block
        pred = recon[:, :n_sk] > 0.5 * idf[None, :]
        true = X_skill > 0
        tp = float((pred & true).sum())
        fp = float((pred & ~true).sum())
        fn = float((~pred & true).sum())
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        params = sum(q.numel() for q in model.parameters())
        out.append({**cfg, "purity": round(p, 4), "f1": round(f1, 4),
                    "params": params, "loss": round(loss.item(), 5)})
        print(f"latent {cfg['latent']:3d} hidden {cfg['hidden']:4d} "
              f"wd {cfg['wd']:.0e} noise {cfg['noise']:.2f} "
              f"-> purity@5 {p:.3f}  f1 {f1:.3f}  params {params:,}")

    json.dump({"input_skill_purity": round(p_in, 4),
               "input_full_purity": round(p_full, 4),
               "configs": out},
              open(os.path.join(DATA, "ae_sweep.json"), "w", encoding="utf-8"),
              indent=2)


if __name__ == "__main__":
    main()
