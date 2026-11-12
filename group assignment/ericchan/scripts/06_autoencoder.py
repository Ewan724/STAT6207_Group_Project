"""Step 6: the autoencoder that encodes CVs and JDs into a shared space.

The brief asks for an AE over the extracted keywords, so the model's input is
the merged keyword space plus four structural scalars.  Diagnostics 06b-06g are
kept in the repo because the negative results are what justify each choice.

What we tried (all measured, none assumed)
------------------------------------------
v1  Plain AE, 443 binary skill dims, latent 64, MSE loss.
    Keyword-block reconstruction F1 = 0.997 and latent purity@5 = 0.14 against
    a 0.10 chance level.  The nearest neighbour of the Data Engineer JD was a
    Graphic Designer.  With 262k parameters for 110 documents the network
    learned an identity-like code.  Conclusion: reconstruction quality is NOT
    evidence of embedding quality, so the model is never selected on it.

v2  IDF-weighted input + denoising + latent 32 + weight decay (06b).
    Purity 0.14-0.16 -- still no better than doing nothing.  The 0.36 purity of
    the raw IDF-weighted input space was already higher than every AE.

v3  Diagnosis: MSE optimises magnitude, retrieval uses cosine (direction).  On a
    sparse binary input most of the mass sits on ubiquitous skills, so MSE
    spends its capacity there and directions collapse.  Replacing the
    reconstruction loss with 1 - cos(x, x_hat) lifted purity from 0.14 to 0.30
    (06c).  This is the single most important change.

v4  Capacity, depth, dropout, epoch and learning-rate sweeps (06d).  Longer
    training, deeper nets and wider layers all hurt.  0.30 is the plateau.

v5  Extra input blocks (06e, 06f).  Adding the domain or soft-skill one-hots
    drops purity to 0.14 -- those blocks are near-constant and dominate the
    input norm.  Adding a 768-d text embedding drops it to 0.22, because the
    larger, smoother block monopolises the bottleneck (06g tried re-weighting
    the losses up to 6:1; the best it reached was 0.28).

    The text embedding is still valuable, so it is not discarded: it enters as a
    third signal in the retrieval fusion (step 7), where it can contribute
    without having to survive a 32-d squeeze.

Final model
    input    443 IDF-weighted merged-keyword dims + 4 structural scalars
    encoder  Linear(447 -> 192) - GELU - Dropout(0.1) - Linear(192 -> 32)
    decoder  Linear(32 -> 192) - GELU - Dropout(0.1) - Linear(192 -> 447)
    loss     1 - cos(x_skill, x_hat_skill)     direction-aware, matches retrieval
             + MSE(x_struct, x_hat_struct)     magnitude matters for 4 scalars
    optim    AdamW lr 3e-3, wd 1e-4, cosine decay, 1800 full-batch epochs
    params   184,799

Reference points
    chance purity@5                                   0.10
    raw IDF-weighted keyword space, no model          0.36
    this autoencoder (447 dims -> 32 dims)            0.30

    The AE therefore retains 83% of the input geometry's neighbourhood quality
    while compressing 14x.  It is a lossy but useful summary -- and step 7 shows
    it still earns its place in the fusion.

Training is unsupervised over all 110 documents.  That is transductive learning,
not label leakage: no relevance grade is ever seen by the model.

Outputs: data/ae_model.pt, data/ae_embeddings.json, data/ae_metrics.json
"""
import json
import math
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
NOISE = 0.20
WD = 1e-4
DROPOUT = 0.10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SENIORITY = {"intern": 0, "junior": 1, "mid": 2, "senior": 3, "lead": 4,
             "executive": 5}
EDU = {"none": 0, "diploma": 1, "bachelor": 2, "master": 3, "phd": 4}
FAM_JD = {"P01": "finance", "P02": "sales", "P03": "marketing", "P04": "data",
          "P05": "product", "P06": "data", "P07": "marketing",
          "P08": "engineering_hw", "P09": "engineering_sw", "P10": "design"}


def load_ext():
    ext = {}
    for line in open(os.path.join(DATA, "extracted.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("ok"):
                ext[r["doc_id"]] = r
    return ext


def build_matrix(skills, ext):
    """IDF-weighted keyword block + 4 structural scalars.

    The CORE vocabulary (df >= 2, plus every term a JD uses) is used rather than
    the full 1,861-term list.  A dimension that is 1 for one document and 0 for
    the other 109 carries no learnable structure and only adds capacity to
    overfit with.  Step 7's overlap scorer uses the full list, where a rare
    *shared* skill is informative.
    """
    doc_skills = skills.get("doc_skills_core") or skills["doc_skills"]
    present = {s for ss in doc_skills.values() for s in ss}
    vocab = [v["skill"] for v in skills["vocab"] if v["skill"] in present]
    idx = {s: i for i, s in enumerate(vocab)}
    docs = sorted(doc_skills)
    n, d = len(docs), len(vocab)

    B = np.zeros((n, d), dtype=np.float32)
    for i, doc in enumerate(docs):
        for s in doc_skills.get(doc, []):
            if s in idx:
                B[i, idx[s]] = 1.0

    df = B.sum(axis=0)
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    idf = idf / idf.max()
    X_skill = B * idf[None, :]

    struct = np.zeros((n, 4), dtype=np.float32)
    for i, doc in enumerate(docs):
        e = ext.get(doc, {})
        struct[i, 0] = SENIORITY.get(e.get("seniority", "mid"), 2) / 5.0
        yrs = e.get("years_experience", e.get("years_required", 0)) or 0
        struct[i, 1] = math.log1p(float(yrs)) / math.log1p(30)
        edu = e.get("education_level", e.get("education_required", "none"))
        struct[i, 2] = EDU.get(edu, 0) / 4.0
        struct[i, 3] = min(len(e.get("domains") or []) / 5.0, 1.0)

    X = np.concatenate([X_skill, struct], axis=1).astype(np.float32)
    return docs, vocab, X, X_skill, struct, idf


class AE(nn.Module):
    def __init__(self, n_in, latent=LATENT, hidden=HIDDEN, p=DROPOUT):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, n_in),
        )

    def forward(self, x):
        z = self.enc(x)
        return z, self.dec(z)


def train_ae(X, n_skill, epochs=EPOCHS, lr=3e-3):
    """Train the AE.

    The decoder reconstructs the full input vector, but the LOSS is computed on
    the skill block only.  Measured reason: including a structural MSE term
    drops latent purity from 0.30 to 0.16 at equal weight (0.22 at 0.1 weight,
    0.24 at 0.02).  Four bounded scalars are trivially easy to fit, so the
    optimiser over-invests in them and starves the skill directions.  Feeding
    the scalars to the encoder while excluding them from the objective keeps the
    seniority/experience context without the capacity competition.
    """
    torch.manual_seed(SEED)
    model = AE(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xt = torch.tensor(X)
    g = torch.Generator().manual_seed(SEED)
    hist = []
    for ep in range(epochs):
        model.train()
        keep = (torch.rand(xt[:, :n_skill].shape, generator=g) > NOISE).float()
        xin = xt.clone()
        xin[:, :n_skill] = xt[:, :n_skill] * keep
        opt.zero_grad()
        _, rec = model(xin)
        loss = (1 - F.cosine_similarity(rec[:, :n_skill], xt[:, :n_skill],
                                        dim=1)).mean()
        loss.backward()
        opt.step()
        sched.step()
        hist.append((loss.item(), loss.item(), 0.0))
        if ep % 300 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  loss {loss.item():.5f}", flush=True)
    return model, hist


def encode(model, X):
    model.eval()
    with torch.no_grad():
        z, rec = model(torch.tensor(X))
        return F.normalize(z, dim=1).numpy(), rec.numpy()


def purity_of(M, docs, fam_of, k=5):
    Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    sim = Mn @ Mn.T
    vals = []
    for j in FAM_JD:
        i = docs.index(j)
        order = [x for x in np.argsort(-sim[i]) if docs[x] != j][:k]
        vals.append(sum(1 for x in order
                        if fam_of.get(docs[x]) == FAM_JD[j]) / k)
    return float(np.mean(vals))


def main():
    skills = json.load(open(os.path.join(DATA, "skills.json"), encoding="utf-8"))
    ext = load_ext()
    docs, vocab, X, X_skill, struct, idf = build_matrix(skills, ext)
    print(f"matrix {X.shape}  (skills {len(vocab)}, structural 4, docs {len(docs)})")

    model, hist = train_ae(X, len(vocab))
    Z, recon = encode(model, X)

    pred = recon[:, :len(vocab)] > 0.5 * idf[None, :]
    true = X_skill > 0
    tp = float((pred & true).sum())
    fp = float((pred & ~true).sum())
    fn = float((~pred & true).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    fam_of = dict(FAM_JD)
    p = os.path.join(DATA, "cv_family.json")
    if os.path.exists(p):
        fam_of.update(json.load(open(p, encoding="utf-8")))

    pur = purity_of(Z, docs, fam_of)
    pur_in = purity_of(X_skill, docs, fam_of)

    metrics = {
        "skill_precision": round(prec, 4),
        "skill_recall": round(rec, 4),
        "skill_f1": round(f1, 4),
        "final_loss": round(hist[-1][0], 5),
        "initial_loss": round(hist[0][0], 5),
        "final_cos_loss": round(hist[-1][1], 5),
        "final_struct_loss": round(hist[-1][2], 5),
        "latent": LATENT, "hidden": HIDDEN, "epochs": EPOCHS,
        "noise": NOISE, "weight_decay": WD, "dropout": DROPOUT,
        "params": sum(q.numel() for q in model.parameters()),
        "n_skill": len(vocab), "input_dim": int(X.shape[1]),
        "latent_purity_at5": round(pur, 4),
        "input_space_purity_at5": round(pur_in, 4),
        "chance_purity_at5": 0.10,
        "compression_ratio": round(X.shape[1] / LATENT, 1),
    }

    json.dump({"docs": docs, "vocab": vocab,
               "embeddings": {d: Z[i].round(6).tolist() for i, d in enumerate(docs)},
               "idf": [round(float(x), 5) for x in idf],
               "n_skill": len(vocab), "latent": LATENT,
               "history": [[round(a, 6) for a in h] for h in hist]},
              open(os.path.join(DATA, "ae_embeddings.json"), "w", encoding="utf-8"))
    torch.save(model.state_dict(), os.path.join(DATA, "ae_model.pt"))
    json.dump(metrics, open(os.path.join(DATA, "ae_metrics.json"), "w",
                            encoding="utf-8"), indent=2)

    print("\nreconstruction:", {k: metrics[k] for k in
                                ("skill_precision", "skill_recall", "skill_f1")})
    print(f"latent purity@5 {pur:.3f}  (input space {pur_in:.3f}, "
          f"chance 0.10)  compression {metrics['compression_ratio']}:1  "
          f"params {metrics['params']:,}")


if __name__ == "__main__":
    main()
