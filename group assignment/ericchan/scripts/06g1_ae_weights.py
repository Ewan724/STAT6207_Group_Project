"""Diagnostic 6: how to weight the blocks so the latent keeps both signals.

06f showed that a 768-d nomic block plus a 443-d keyword block compresses to
purity 0.22 when the two reconstruction losses are weighted equally: the text
block is bigger and easier to fit, so it monopolises the bottleneck and the
keyword structure is lost.

This sweep varies the loss weights and the latent width to find a configuration
that keeps the keyword geometry while still benefiting from the text block.
Selection is on latent purity@5 (job-family neighbourhood quality), which never
touches a relevance grade.
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

FAM_JD = ae.FAM_JD


def main():
    skills = json.load(open(os.path.join(DATA, "skills.json"), encoding="utf-8"))
    ext = ae.load_ext()
    doc_skills = skills.get("doc_skills_core") or skills["doc_skills"]
    docs = sorted(doc_skills)
    docs, vocab, X, X_kw, X_txt, struct, idf, blocks = ae.build_matrix(
        skills, ext, docs)
    fam_of = dict(FAM_JD)
    fam_of.update(json.load(open(os.path.join(DATA, "cv_family.json"),
                                encoding="utf-8")))
    k0, k1 = blocks["kw"]
    t0, t1 = blocks["txt"]
    s0, s1 = blocks["struct"]

    def purity(M, k=5):
        Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
        sim = Mn @ Mn.T
        vals = []
        for j in FAM_JD:
            i = docs.index(j)
            order = [x for x in np.argsort(-sim[i]) if docs[x] != j][:k]
            vals.append(sum(1 for x in order
                            if fam_of.get(docs[x]) == FAM_JD[j]) / k)
        return float(np.mean(vals))

    print(f"kw block  {purity(X_kw):.3f}   txt block {purity(X_txt):.3f}")

    trials = []
    for wk in (1.0, 3.0, 6.0):
        for latent in (32, 64):
            trials.append(dict(w_kw=wk, w_txt=1.0, latent=latent))
    trials += [dict(w_kw=3.0, w_txt=0.3, latent=32),
               dict(w_kw=6.0, w_txt=0.3, latent=64)]

    out = []
    for cfg in trials:
        torch.manual_seed(ae.SEED)
        model = ae.AE(X.shape[1], latent=cfg["latent"])
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3,
                                weight_decay=ae.WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                          T_max=ae.EPOCHS)
        xt = torch.tensor(X)
        g = torch.Generator().manual_seed(ae.SEED)
        for _ in range(ae.EPOCHS):
            model.train()
            keep = (torch.rand(xt[:, k0:k1].shape, generator=g)
                    > ae.NOISE).float()
            xin = xt.clone()
            xin[:, k0:k1] = xt[:, k0:k1] * keep
            opt.zero_grad()
            _, rec = model(xin)
            loss = (cfg["w_kw"] * (1 - F.cosine_similarity(
                        rec[:, k0:k1], xt[:, k0:k1], dim=1)).mean()
                    + cfg["w_txt"] * (1 - F.cosine_similarity(
                        rec[:, t0:t1], xt[:, t0:t1], dim=1)).mean()
                    + F.mse_loss(rec[:, s0:s1], xt[:, s0:s1]))
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            z, _ = model(xt)
            z = F.normalize(z, dim=1).numpy()
        p = purity(z)
        out.append({**cfg, "purity": round(p, 4)})
        print(f"w_kw {cfg['w_kw']:.0f} w_txt {cfg['w_txt']:.1f} "
              f"latent {cfg['latent']:3d} -> purity {p:.3f}", flush=True)

    out.sort(key=lambda r: -r["purity"])
    json.dump(out, open(os.path.join(DATA, "ae_sweep6.json"), "w",
                        encoding="utf-8"), indent=2)
    print("\nbest:", out[0])


if __name__ == "__main__":
    main()
