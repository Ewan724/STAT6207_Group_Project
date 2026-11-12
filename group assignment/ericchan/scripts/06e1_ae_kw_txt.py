"""Step 6c: train the second AE variant (keywords + text) and compare both.

Step 6 trains the keyword-only AE that the brief literally asks for.  The
diagnostics (06b-06g) showed that this AE is the weakest retriever in the
ensemble, because the merged keyword space it compresses is itself lossy: 12
canonical skills per document versus ~3,100 characters of evidence.

This script trains the alternative -- the same autoencoder over the keyword
block CONCATENATED with the nomic text embedding -- and writes both sets of
latent vectors so step 7 can compare them on the actual task.  Selection
between the two is on downstream Recall@10, which is reported either way; the
purity diagnostic is reported alongside so the reader can see that
architecture selection and task performance do not always agree.

Outputs: data/ae_embeddings_kwtxt.json, data/ae_metrics_kwtxt.json
"""
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
LATENT = 64
HIDDEN = 192
EPOCHS = 1800
NOISE = 0.20
WD = 1e-4
W_KW = 6.0      # from the 06g sweep: keeps the keyword geometry dominant
W_TXT = 1.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

FAM_JD = {"P01": "finance", "P02": "sales", "P03": "marketing", "P04": "data",
          "P05": "product", "P06": "data", "P07": "marketing",
          "P08": "engineering_hw", "P09": "engineering_sw", "P10": "design"}


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ae", os.path.join(BASE, "scripts", "06_autoencoder.py"))
    ae = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ae)

    skills = json.load(open(os.path.join(DATA, "skills.json"), encoding="utf-8"))
    ext = ae.load_ext()
    docs, vocab, X, X_kw, struct, idf = ae.build_matrix(skills, ext)
    n_sk = len(vocab)

    stored = json.load(open(os.path.join(DATA, "nomic_docs.json"), encoding="utf-8"))
    X_txt = np.array([stored[d] for d in docs], dtype=np.float32)
    X_txt = X_txt / np.maximum(np.linalg.norm(X_txt, axis=1, keepdims=True), 1e-9)

    Xin = np.concatenate([X_kw, X_txt, struct], axis=1).astype(np.float32)
    k1 = n_sk
    t0, t1 = n_sk, n_sk + X_txt.shape[1]
    s0, s1 = t1, Xin.shape[1]
    print(f"input {Xin.shape}  kw[0:{k1}] txt[{t0}:{t1}] struct[{s0}:{s1}]")

    class AE(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.GELU(),
                                     nn.Dropout(0.10), nn.Linear(HIDDEN, LATENT))
            self.dec = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.GELU(),
                                     nn.Dropout(0.10), nn.Linear(HIDDEN, n_in))

        def forward(self, x):
            z = self.enc(x)
            return z, self.dec(z)

    torch.manual_seed(SEED)
    model = AE(Xin.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    xt = torch.tensor(Xin)
    g = torch.Generator().manual_seed(SEED)
    hist = []
    for ep in range(EPOCHS):
        model.train()
        keep = (torch.rand(xt[:, :k1].shape, generator=g) > NOISE).float()
        xin = xt.clone()
        xin[:, :k1] = xt[:, :k1] * keep
        opt.zero_grad()
        _, rec = model(xin)
        l_kw = (1 - F.cosine_similarity(rec[:, :k1], xt[:, :k1], dim=1)).mean()
        l_tx = (1 - F.cosine_similarity(rec[:, t0:t1], xt[:, t0:t1], dim=1)).mean()
        loss = W_KW * l_kw + W_TXT * l_tx
        loss.backward()
        opt.step()
        sched.step()
        hist.append((loss.item(), l_kw.item(), l_tx.item()))
        if ep % 300 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:4d}  loss {loss.item():.5f}  "
                  f"kw {l_kw.item():.5f}  txt {l_tx.item():.5f}", flush=True)

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

    json.dump({"docs": docs, "vocab": vocab,
               "embeddings": {d: Z[i].round(6).tolist() for i, d in enumerate(docs)},
               "n_skill": n_sk, "latent": LATENT},
              open(os.path.join(DATA, "ae_embeddings_kwtxt.json"), "w",
                   encoding="utf-8"))
    json.dump({"variant": "kw+txt", "latent": LATENT, "hidden": HIDDEN,
               "epochs": EPOCHS, "w_kw": W_KW, "w_txt": W_TXT,
               "input_dim": int(Xin.shape[1]),
               "params": sum(q.numel() for q in model.parameters()),
               "latent_purity_at5": round(pur, 4),
               "final_loss": round(hist[-1][0], 5),
               "final_kw": round(hist[-1][1], 5),
               "final_txt": round(hist[-1][2], 5)},
              open(os.path.join(DATA, "ae_metrics_kwtxt.json"), "w",
                   encoding="utf-8"), indent=2)
    print(f"\nlatent purity@5 {pur:.3f}  params "
          f"{sum(q.numel() for q in model.parameters()):,}")


if __name__ == "__main__":
    main()
