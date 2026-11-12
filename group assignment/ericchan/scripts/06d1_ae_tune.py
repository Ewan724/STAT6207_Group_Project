"""Diagnostic 3: final AE tuning around the cosine objective.

06c showed the direction-aware (cosine) reconstruction loss lifts latent purity
from 0.14 (MSE) to 0.30, close to the 0.36 of the IDF-weighted input space.
This sweep explores capacity, depth and training length around that finding to
find the configuration that best preserves (and ideally sharpens) the input
geometry.  Selection is on latent purity, which uses only hand-assigned job
families and never the relevance grades.
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


class AE2(nn.Module):
    def __init__(self, n_in, latent, hidden, depth=1, p=0.1):
        super().__init__()
        enc = [nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(p)]
        for _ in range(depth - 1):
            enc += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p)]
        enc += [nn.Linear(hidden, latent)]
        dec = [nn.Linear(latent, hidden), nn.GELU(), nn.Dropout(p)]
        for _ in range(depth - 1):
            dec += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p)]
        dec += [nn.Linear(hidden, n_in)]
        self.enc, self.dec = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, x):
        z = self.enc(x)
        return z, self.dec(z)


def run(X, n_sk, latent, hidden, depth, noise, wd, epochs, lr=3e-3, p=0.1):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    model = AE2(X.shape[1], latent, hidden, depth, p)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xt = torch.tensor(X)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        model.train()
        keep = (torch.rand(xt[:, :n_sk].shape, generator=g) > noise).float()
        xin = xt.clone()
        xin[:, :n_sk] = xt[:, :n_sk] * keep
        opt.zero_grad()
        _, rec = model(xin)
        loss = (1 - F.cosine_similarity(rec[:, :n_sk], xt[:, :n_sk], dim=1)).mean()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        z, _ = model(xt)
        z = F.normalize(z, dim=1).numpy()
    return model, z, float(loss.item())


def main():
    skills = json.load(open(os.path.join(DATA, "skills.json"), encoding="utf-8"))
    ext = ae.load_ext()
    docs, vocab, X, X_skill, struct, idf = ae.build_matrix(skills, ext)
    fam_of = dict(FAM_JD)
    fam_of.update(json.load(open(os.path.join(DATA, "cv_family.json"),
                                encoding="utf-8")))
    n_sk = len(vocab)

    trials = []
    for latent in (16, 32, 64):
        for epochs in (1800, 3000):
            trials.append(dict(latent=latent, hidden=192, depth=1, noise=0.20,
                               wd=1e-4, epochs=epochs))
    trials += [
        dict(latent=32, hidden=192, depth=2, noise=0.20, wd=1e-4, epochs=3000),
        dict(latent=32, hidden=384, depth=1, noise=0.20, wd=1e-4, epochs=3000),
        dict(latent=32, hidden=192, depth=1, noise=0.30, wd=1e-3, epochs=3000),
        dict(latent=32, hidden=192, depth=1, noise=0.20, wd=1e-4, epochs=3000,
             lr=1e-3),
        dict(latent=32, hidden=192, depth=1, noise=0.20, wd=1e-4, epochs=3000,
             p=0.0),
        dict(latent=32, hidden=192, depth=1, noise=0.20, wd=1e-4, epochs=3000,
             p=0.3),
    ]
    out = []
    for cfg in trials:
        model, z, loss = run(X, n_sk, cfg["latent"], cfg["hidden"], cfg["depth"],
                             cfg["noise"], cfg["wd"], cfg["epochs"],
                             cfg.get("lr", 3e-3), cfg.get("p", 0.1))
        p = purity(z, docs, fam_of)
        out.append({**cfg, "purity": round(p, 4), "loss": round(loss, 5),
                    "params": sum(q.numel() for q in model.parameters())})
        print(f"L{cfg['latent']:<3d} H{cfg['hidden']:<4d} d{cfg['depth']} "
              f"n{cfg['noise']:.2f} wd{cfg['wd']:.0e} e{cfg['epochs']:<5d} "
              f"p{cfg.get('p', 0.1):.1f} -> purity {p:.3f}", flush=True)

    out.sort(key=lambda r: -r["purity"])
    json.dump(out, open(os.path.join(DATA, "ae_sweep3.json"), "w",
                        encoding="utf-8"), indent=2)
    print("\nTOP 3:")
    for r in out[:3]:
        print("  ", r)


if __name__ == "__main__":
    main()
