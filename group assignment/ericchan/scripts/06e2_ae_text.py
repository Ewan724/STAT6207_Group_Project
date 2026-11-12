"""Step 6d: third AE variant - autoencoder over the text embedding.

Why a third variant
    The keyword-only AE (step 6) is the deliverable the brief literally asks for,
    and it is the weakest retriever in the ensemble.  The diagnostics traced the
    cause: the AE compresses the merged keyword space, and that space is itself
    lossy (12 canonical skills per document versus ~3,100 characters of
    evidence).  An autoencoder cannot recover information its input never
    contained.

    The text embedding is a richer input, so this variant compresses THAT with
    the same architecture.  It answers a question the report needs: is the
    autoencoder itself the bottleneck, or the keyword representation it is fed?
    Comparing the three variants separates the two.

Variants, all the same architecture and training budget
    ae_latent   keyword block + structural          447 -> 32   (step 6)
    ae_kwtxt    keyword + text + structural       1,215 -> 64   (06c)
    ae_text     text embedding + structural         772 -> 32   (this file)

Outputs: data/ae_embeddings_text.json, data/ae_metrics_text.json
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
LATENT = 32
HIDDEN = 192
EPOCHS = 1800
NOISE = 0.10
WD = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

FAM_JD = {"P01": "finance", "P02": "sales", "P03": "marketing", "P04": "data",
          "P05": "product", "P06": "data", "P07": "marketing",
          "P08": "engineering_hw", "P09": "engineering_sw", "P10": "design"}


class AE(nn.Module):
    def __init__(self, n_in, latent=LATENT, hidden=HIDDEN):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(),
                                 nn.Dropout(0.10), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                 nn.Dropout(0.10), nn.Linear(hidden, n_in))

    def forward(self, x):
        z = self.enc(x)
        return z, self.dec(z)


def main():
    spec = importlib.util.spec_from_file_location(
        "ae", os.path.join(BASE, "scripts", "06_autoencoder.py"))
    ae = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ae)

    skills = json.load(open(os.path.join(DATA, "skills.json"), encoding="utf-8"))
    ext = ae.load_ext()
    docs, vocab, X, X_kw, struct, idf = ae.build_matrix(skills, ext)
    stored = json.load(open(os.path.join(DATA, "nomic_docs.json"), encoding="utf-8"))
    X_txt = np.array([stored[d] for d in docs], dtype=np.float32)
    X_txt = X_txt / np.maximum(np.linalg.norm(X_txt, axis=1, keepdims=True), 1e-9)

    Xin = np.concatenate([X_txt, struct], axis=1).astype(np.float32)
    n_txt = X_txt.shape[1]
    print(f"input {Xin.shape}  text[0:{n_txt}] struct[{n_txt}:]")

    torch.manual_seed(SEED)
    model = AE(Xin.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    xt = torch.tensor(Xin)
    g = torch.Generator().manual_seed(SEED)
    hist = []
    for ep in range(EPOCHS):
        model.train()
        keep = (torch.rand(xt[:, :n_txt].shape, generator=g) > NOISE).float()
        xin = xt.clone()
        xin[:, :n_txt] = xt[:, :n_txt] * keep
        opt.zero_grad()
        _, rec = model(xin)
        loss = (1 - F.cosine_similarity(rec[:, :n_txt], xt[:, :n_txt],
                                        dim=1)).mean()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(loss.item())
        if ep % 300 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:4d}  1-cos {loss.item():.5f}", flush=True)

    model.eval()
    with torch.no_grad():
        z, _ = model(xt)
        Z = F.normalize(z, dim=1).numpy()

    fam_of = dict(FAM_JD)
    fam_of.update(json.load(open(os.path.join(DATA, "cv_family.json"),
                                encoding="utf-8")))
    sim = Z @ Z.T
    vals = []
    for j in FAM_JD:
        i = docs.index(j)
        order = [x for x in np.argsort(-sim[i]) if docs[x] != j][:5]
        vals.append(sum(1 for x in order if fam_of.get(docs[x]) == FAM_JD[j]) / 5)
    pur = float(np.mean(vals))

    json.dump({"docs": docs, "vocab": [], "n_skill": n_txt, "latent": LATENT,
               "embeddings": {d: Z[i].round(6).tolist()
                              for i, d in enumerate(docs)}},
              open(os.path.join(DATA, "ae_embeddings_text.json"), "w",
                   encoding="utf-8"))
    json.dump({"variant": "text", "latent": LATENT, "hidden": HIDDEN,
               "epochs": EPOCHS, "noise": NOISE, "input_dim": int(Xin.shape[1]),
               "params": sum(q.numel() for q in model.parameters()),
               "latent_purity_at5": round(pur, 4),
               "final_loss": round(hist[-1], 5)},
              open(os.path.join(DATA, "ae_metrics_text.json"), "w",
                   encoding="utf-8"), indent=2)
    print(f"\nlatent purity@5 {pur:.3f}  params "
          f"{sum(q.numel() for q in model.parameters()):,}")


if __name__ == "__main__":
    main()
