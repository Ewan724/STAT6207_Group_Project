"""Diagnostic 2: align the AE objective with the retrieval metric.

The sweep in 06b showed the input space itself (IDF-weighted skill cosine) has
latent-purity 0.36 while every MSE-trained AE lands at 0.08-0.16.  Diagnosis:
MSE reconstruction optimises *magnitude*, but retrieval uses *cosine*, which
only cares about direction.  A sparse binary input has most of its mass on the
frequent skills, so MSE spends its capacity reproducing those and the
directions collapse.

This script tests reconstruction objectives that are direction-aware:
  A. MSE (baseline, as in the current model)
  B. cosine reconstruction loss   1 - cos(x, x_hat)
  C. MSE + cosine (mixed)
  D. cosine + explicit denoising at a higher rate
  E. Sparse AE with an L1 penalty on the latent code

Everything else (IDF input, denoising, full batch, seed) is held fixed so the
comparison is clean.
"""
import importlib.util
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
SEED = 42

spec = importlib.util.spec_from_file_location(
    "ae", os.path.join(BASE, "scripts", "06_autoencoder.py"))
ae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ae)

FAM_JD = ae.FAM_JD


def purity(Z, docs, fam_of, k=5):
    sim = Z @ Z.T
    vals = []
    for j in FAM_JD:
        i = docs.index(j)
        order = [x for x in np.argsort(-sim[i]) if docs[x] != j][:k]
        vals.append(sum(1 for x in order
                        if fam_of.get(docs[x]) == FAM_JD[j]) / k)
    return float(np.mean(vals))


class CosineAE(nn.Module):
    def __init__(self, n_in, latent, hidden, p=0.1):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(hidden, n_in))

    def forward(self, x):
        z = self.enc(x)
        return z, self.dec(x) if False else self.dec(z)


def run(X, n_sk, latent, hidden, loss_kind, noise, wd, epochs, l1=0.0):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    model = CosineAE(X.shape[1], latent, hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xt = torch.tensor(X)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        model.train()
        keep = (torch.rand(xt[:, :n_sk].shape, generator=g) > noise).float()
        xin = xt.clone()
        xin[:, :n_sk] = xt[:, :n_sk] * keep
        opt.zero_grad()
        z, rec = model(xin)
        a, b = rec[:, :n_sk], xt[:, :n_sk]
        if loss_kind == "mse":
            loss = F.mse_loss(a, b) + F.mse_loss(rec[:, n_sk:], xt[:, n_sk:])
        elif loss_kind == "cos":
            loss = (1 - F.cosine_similarity(a, b, dim=1)).mean()
        elif loss_kind == "mix":
            loss = (1 - F.cosine_similarity(a, b, dim=1)).mean() \
                + 0.5 * F.mse_loss(a, b) \
                + 0.5 * F.mse_loss(rec[:, n_sk:], xt[:, n_sk:])
        else:
            raise ValueError(loss_kind)
        if l1:
            loss = loss + l1 * z.abs().mean()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        z, rec = model(xt)
        z = F.normalize(z, dim=1).numpy()
        rec = rec.numpy()
    return model, z, rec, float(loss.item())


def main():
    skills = json.load(open(os.path.join(DATA, "skills.json"), encoding="utf-8"))
    ext = ae.load_ext()
    docs, vocab, X, X_skill, struct, idf = ae.build_matrix(skills, ext)
    fam_of = dict(FAM_JD)
    fam_of.update(json.load(open(os.path.join(DATA, "cv_family.json"),
                                encoding="utf-8")))
    n_sk = len(vocab)

    print(f"input-space reference purity@5 = "
          f"{purity(ae.np.asarray(X_skill) / np.maximum(np.linalg.norm(X_skill, axis=1, keepdims=True), 1e-9), docs, fam_of):.3f}")

    trials = [
        ("mse   L32 H192", dict(latent=32, hidden=192, loss_kind="mse",
                                noise=0.20, wd=1e-4, epochs=900)),
        ("cos   L32 H192", dict(latent=32, hidden=192, loss_kind="cos",
                                noise=0.20, wd=1e-4, epochs=900)),
        ("mix   L32 H192", dict(latent=32, hidden=192, loss_kind="mix",
                                noise=0.20, wd=1e-4, epochs=900)),
        ("cos   L16 H128", dict(latent=16, hidden=128, loss_kind="cos",
                                noise=0.20, wd=1e-4, epochs=900)),
        ("cos   L64 H256", dict(latent=64, hidden=256, loss_kind="cos",
                                noise=0.20, wd=1e-4, epochs=900)),
        ("cos   L32 H192 n.35", dict(latent=32, hidden=192, loss_kind="cos",
                                     noise=0.35, wd=1e-4, epochs=900)),
        ("cos   L32 H192 wd1e-2", dict(latent=32, hidden=192, loss_kind="cos",
                                       noise=0.20, wd=1e-2, epochs=900)),
        ("cos+L1 L32 H192", dict(latent=32, hidden=192, loss_kind="cos",
                                 noise=0.20, wd=1e-4, epochs=900, l1=1e-3)),
        ("cos   L32 H192 e1800", dict(latent=32, hidden=192, loss_kind="cos",
                                      noise=0.20, wd=1e-4, epochs=1800)),
    ]
    out = []
    for name, cfg in trials:
        model, z, rec, loss = run(X, n_sk, cfg["latent"], cfg["hidden"],
                                  cfg["loss_kind"], cfg["noise"], cfg["wd"],
                                  cfg["epochs"], cfg.get("l1", 0.0))
        p = purity(z, docs, fam_of)
        out.append({**cfg, "name": name, "purity": round(p, 4),
                    "loss": round(loss, 5),
                    "params": sum(q.numel() for q in model.parameters())})
        print(f"{name:24s} purity@5 {p:.3f}  loss {loss:.5f}  "
              f"params {out[-1]['params']:,}")

    out.sort(key=lambda r: -r["purity"])
    json.dump(out, open(os.path.join(DATA, "ae_sweep2.json"), "w",
                        encoding="utf-8"), indent=2)
    print("\nbest:", out[0]["name"], out[0]["purity"])


if __name__ == "__main__":
    main()
