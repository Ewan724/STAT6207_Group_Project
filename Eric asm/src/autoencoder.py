"""Stage 3: keyword-cluster autoencoder and latent document embeddings."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from .config import Config, dump_json
from .data import read_jsonl

LATENT = 32


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims=(256, 64), dropout: float = 0.2):
        super().__init__()
        h1, h2 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h2, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_inputs(cfg: Config):
    X = sparse.load_npz(cfg.processed / "document_keyword_matrix.npz").toarray().astype(np.float32)
    counts = sparse.load_npz(cfg.processed / "document_keyword_counts.npz").toarray().astype(np.float32)
    index = pd.read_csv(cfg.processed / "document_matrix_index.csv")
    return X, counts, index


def _split_indices(index: pd.DataFrame, val_fraction: float, seed: int):
    train_pool = index.index[index["is_training"]].tolist()
    labels = index.loc[train_pool, "doc_type"].astype(str).tolist()
    train_idx, val_idx = train_test_split(
        train_pool, test_size=val_fraction, random_state=seed, stratify=labels
    )
    return sorted(train_idx), sorted(val_idx)


def train_one(X: np.ndarray, train_idx: list[int], val_idx: list[int], latent_dim: int,
              dropout: float, denoising: bool, cfg: Config, max_epochs: int,
              patience: int, seed: int) -> dict[str, Any]:
    _seed_all(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    input_dim = X.shape[1]
    model = Autoencoder(input_dim, latent_dim, tuple(cfg.get("autoencoder.hidden_dims", [256, 64])), dropout)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("autoencoder.lr", 1e-3)),
                           weight_decay=float(cfg.get("autoencoder.weight_decay", 1e-4)))
    pos_weight = float(cfg.get("autoencoder.positive_weight", 10.0))

    def weighted_mse(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Plain MSE on a sparse TF-IDF matrix has a trivial optimum: emit the
        # per-dimension mean for every document, which collapses the latent
        # space. Upweight observed (non-zero) clusters so the model has to
        # encode which skills a document actually contains.
        weight = 1.0 + pos_weight * (target > 0).to(target.dtype)
        return (((out - target) ** 2) * weight).sum() / weight.sum()

    def val_loss() -> float:
        with torch.no_grad():
            out = model(X_val)
            return float(weighted_mse(out, X_val).item())
    X_train = torch.tensor(X[train_idx])
    X_val = torch.tensor(X[val_idx])
    batch_size = int(cfg.get("autoencoder.batch_size", 16))
    grad_clip = float(cfg.get("autoencoder.grad_clip", 1.0))
    denoise_rate = float(cfg.get("autoencoder.denoise_rate", 0.1))
    best_val = math.inf
    best_state = None
    best_epoch = 0
    history = []
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train))
        train_loss = 0.0
        for start in range(0, len(X_train), batch_size):
            batch = X_train[perm[start:start + batch_size]]
            clean = batch.clone()
            if denoising:
                mask = (torch.rand_like(batch) < denoise_rate) & (batch > 0)
                batch = batch * (~mask)
            opt.zero_grad()
            out = model(batch)
            loss = loss_fn(out, clean)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            train_loss += float(loss.item()) * len(batch)
        train_mse = train_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_mse = float(loss_fn(val_out, X_val).item())
        history.append({"epoch": epoch, "train_mse": round(train_mse, 6), "val_mse": round(val_mse, 6)})
        if val_mse < best_val - 1e-7:
            best_val = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_out = model(X_train)
        train_mse = float(loss_fn(train_out, X_train).item())
        val_out = model(X_val)
        val_mse = float(loss_fn(val_out, X_val).item())
    return {
        "model": model, "history": history, "best_epoch": best_epoch,
        "train_mse": round(train_mse, 6), "val_mse": round(val_mse, 6),
        "latent_dim": latent_dim, "dropout": dropout, "denoising": denoising,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "epochs_run": len(history),
    }


def ablation(cfg: Config, X: np.ndarray, train_idx: list[int], val_idx: list[int], seed: int) -> pd.DataFrame:
    rows = []
    max_epochs = int(cfg.get("autoencoder.ablation_max_epochs", 250))
    patience = int(cfg.get("autoencoder.ablation_patience", 25))
    for latent in cfg.get("autoencoder.ablation.latent_dims", [16, 32, 64]):
        for dropout in cfg.get("autoencoder.ablation.dropouts", [0.0, 0.2]):
            for denoise in cfg.get("autoencoder.ablation.denoising", [False, True]):
                started = time.time()
                result = train_one(X, train_idx, val_idx, int(latent), float(dropout), bool(denoise),
                                   cfg, max_epochs, patience, seed)
                rows.append({
                    "latent_dim": int(latent), "dropout": float(dropout),
                    "denoising": bool(denoise), "best_epoch": result["best_epoch"],
                    "epochs_run": result["epochs_run"], "train_mse": result["train_mse"],
                    "val_mse": result["val_mse"], "n_params": result["n_params"],
                    "elapsed_sec": round(time.time() - started, 1),
                    "selected": False,
                    "notes": "exploratory capacity/regularisation run; selection is by validation MSE",
                })
                print(f"  ablation latent={latent} dropout={dropout} denoise={denoise} -> val_mse={result['val_mse']:.6f}")
    df = pd.DataFrame(rows)
    primary = df[df["latent_dim"] == 32].sort_values("val_mse").index
    if len(primary):
        df.loc[primary[0], "selected"] = True
        df.loc[primary[0], "notes"] = "selected within the pre-registered 32-d primary architecture"
    df.to_csv(cfg.processed / "ae_ablation.csv", index=False)
    return df


def reconstruction_metrics(X: np.ndarray, train_idx: list[int], model: Autoencoder) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X)).numpy()
    mse = float(np.mean((out - X) ** 2))
    # Cosine reconstruction, row-wise.
    num = (out * X).sum(axis=1)
    den = np.linalg.norm(out, axis=1) * np.linalg.norm(X, axis=1)
    cos = float(np.mean(num / np.maximum(den, 1e-9)))
    return {"reconstruction_mse": round(mse, 6), "cosine_reconstruction": round(cos, 4), "reconstruction": out}


def compare_svd_pca(X: np.ndarray, train_idx: list[int], out_ae: np.ndarray) -> pd.DataFrame:
    rows = []
    svd = TruncatedSVD(n_components=32, random_state=42)
    svd.fit(X[train_idx])
    Xs = svd.transform(X)
    rec_svd = svd.inverse_transform(Xs)
    rows.append({
        "method": "TruncatedSVD-32",
        "reconstruction_mse": round(float(np.mean((rec_svd - X) ** 2)), 6),
        "cosine_reconstruction": round(float(np.mean((rec_svd * X).sum(1) / np.maximum(
            np.linalg.norm(rec_svd, axis=1) * np.linalg.norm(X, axis=1), 1e-9))), 4),
        "explained_variance": round(float(svd.explained_variance_ratio_.sum()), 4),
    })
    pca = PCA(n_components=32, random_state=42)
    pca.fit(X[train_idx])
    Xp = pca.transform(X)
    rec_pca = pca.inverse_transform(Xp)
    rows.append({
        "method": "PCA-32",
        "reconstruction_mse": round(float(np.mean((rec_pca - X) ** 2)), 6),
        "cosine_reconstruction": round(float(np.mean((rec_pca * X).sum(1) / np.maximum(
            np.linalg.norm(rec_pca, axis=1) * np.linalg.norm(X, axis=1), 1e-9))), 4),
        "explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
    })
    rows.append({
        "method": "AE-32",
        "reconstruction_mse": round(float(np.mean((out_ae - X) ** 2)), 6),
        "cosine_reconstruction": round(float(np.mean((out_ae * X).sum(1) / np.maximum(
            np.linalg.norm(out_ae, axis=1) * np.linalg.norm(X, axis=1), 1e-9))), 4),
        "explained_variance": None,
    })
    return pd.DataFrame(rows)


def encode_all(cfg: Config, model: Autoencoder, X: np.ndarray, index: pd.DataFrame,
               documents: list[dict], extracted: dict[str, dict]) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        Z = model.encode(torch.tensor(X)).numpy()
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    meta = {d["doc_id"]: d for d in documents}
    rows = []
    for i, row in index.iterrows():
        d = meta.get(row["doc_id"], {})
        rec = extracted.get(row["doc_id"], {})
        entry = {
            "doc_id": row["doc_id"], "doc_type": row["doc_type"], "split": row["split"],
            "title": d.get("title"), "role_family": d.get("role_family"),
            "seniority": rec.get("seniority", "unknown"),
            "company": d.get("company"),
        }
        for j in range(Z.shape[1]):
            entry[f"dim_{j + 1:02d}"] = round(float(Z[i, j]), 6)
        rows.append(entry)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.processed / "document_embeddings.csv", index=False)
    return df


def reconstruction_examples(cfg: Config, X: np.ndarray, out: np.ndarray, index: pd.DataFrame,
                            clusters: pd.DataFrame, documents: list[dict], n: int = 5) -> pd.DataFrame:
    meta = {d["doc_id"]: d for d in documents}
    # 3 CVs + 2 JDs, deterministic.
    cv_rows = index[index["doc_type"].isin(["pool_cv", "gt_cv"])].head(3).index.tolist()
    jd_rows = index[index["doc_type"].isin(["gt_jd", "unlabelled_jd"])].head(2).index.tolist()
    chosen = cv_rows + jd_rows
    labels = dict(zip(clusters["cluster_id"], clusters["canonical_label"]))
    rows = []
    for i in chosen:
        inp = np.argsort(-X[i])[:10]
        rec = np.argsort(-out[i])[:10]
        for rank in range(10):
            rows.append({
                "doc_id": index.loc[i, "doc_id"], "doc_type": index.loc[i, "doc_type"],
                "title": meta.get(index.loc[i, "doc_id"], {}).get("title"),
                "rank": rank + 1,
                "input_cluster": labels.get(int(inp[rank]), str(inp[rank])),
                "input_value": round(float(X[i, inp[rank]]), 5),
                "reconstructed_cluster": labels.get(int(rec[rank]), str(rec[rank])),
                "reconstructed_value": round(float(out[i, rec[rank]]), 5),
            })
    df = pd.DataFrame(rows)
    df.to_csv(cfg.tables / "reconstruction_examples.csv", index=False)
    return df


def latent_coordinates(cfg: Config, embeddings: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in embeddings.columns if c.startswith("dim_")]
    Z = embeddings[cols].to_numpy()
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(Z)
    out = embeddings[["doc_id", "doc_type", "split", "title", "role_family"]].copy()
    out["pca_x"], out["pca_y"] = xy[:, 0], xy[:, 1]
    perplexity = min(20, max(5, (len(Z) - 1) // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca", learning_rate="auto")
    xy2 = tsne.fit_transform(Z)
    out["tsne_x"], out["tsne_y"] = xy2[:, 0], xy2[:, 1]
    out.to_csv(cfg.tables / "latent_coordinates.csv", index=False)
    return out


def run_stage3(cfg: Config, force: bool = False) -> dict[str, Any]:
    _seed_all(cfg.seed)
    X, counts, index = load_inputs(cfg)
    train_idx, val_idx = _split_indices(index, float(cfg.get("autoencoder.val_fraction", 0.15)), cfg.seed)
    print(f"AE training documents: {len(train_idx)} train / {len(val_idx)} validation; input_dim={X.shape[1]}")

    if not force and (cfg.processed / "ae_ablation.csv").exists():
        abl = pd.read_csv(cfg.processed / "ae_ablation.csv")
    else:
        abl = ablation(cfg, X, train_idx, val_idx, cfg.seed)
    selected = abl[abl["selected"] == True]  # noqa: E712
    if not len(selected):
        selected = abl.sort_values("val_mse").head(1)
    sel = selected.iloc[0]
    print(f"Selected AE config: latent={sel['latent_dim']} dropout={sel['dropout']} denoise={sel['denoising']}")

    result = train_one(X, train_idx, val_idx, int(sel["latent_dim"]), float(sel["dropout"]),
                       bool(sel["denoising"]), cfg,
                       int(cfg.get("autoencoder.max_epochs", 500)),
                       int(cfg.get("autoencoder.patience", 40)), cfg.seed)
    model = result["model"]
    rec = reconstruction_metrics(X, train_idx, model)
    history = pd.DataFrame(result["history"])
    history.to_csv(cfg.models / "ae_training_history.csv", index=False)
    comparison = compare_svd_pca(X, train_idx, rec["reconstruction"])
    comparison.to_csv(cfg.tables / "ae_vs_svd.csv", index=False)

    clusters = pd.read_csv(cfg.processed / "keyword_clusters.csv")
    documents = read_jsonl(cfg.processed / "documents.jsonl")
    extracted = {}
    for path in [cfg.processed / "jd_extracted.jsonl", cfg.processed / "cv_extracted.jsonl"]:
        for r in read_jsonl(path):
            extracted[r["doc_id"]] = r
    # Per-document reconstruction error for the website / error analysis.
    per_doc_mse = np.mean((rec["reconstruction"] - X) ** 2, axis=1)
    recon_table = pd.DataFrame({
        "doc_id": index["doc_id"], "doc_type": index["doc_type"],
        "split": index["split"], "reconstruction_mse": per_doc_mse,
    })
    recon_table.to_csv(cfg.tables / "reconstruction_error.csv", index=False)
    embeddings = encode_all(cfg, model, X, index, documents, extracted)
    examples = reconstruction_examples(cfg, X, rec["reconstruction"], index, clusters, documents)
    coords = latent_coordinates(cfg, embeddings)

    torch.save(model.state_dict(), cfg.models / "autoencoder.pt")
    vectorizer_hash = None
    if (cfg.models / "vectorizer.json").exists():
        import hashlib
        vectorizer_hash = hashlib.sha256((cfg.models / "vectorizer.json").read_bytes()).hexdigest()
    config = {
        "input_dim": int(X.shape[1]),
        "hidden_dims": list(cfg.get("autoencoder.hidden_dims", [256, 64])),
        "latent_dim": int(result["latent_dim"]),
        "dropout": float(result["dropout"]),
        "denoising": bool(result["denoising"]),
        "seed": cfg.seed,
        "best_epoch": int(result["best_epoch"]),
        "epochs_run": int(result["epochs_run"]),
        "train_mse": result["train_mse"],
        "val_mse": result["val_mse"],
        "n_params": result["n_params"],
        "n_train": len(train_idx),
        "n_validation": len(val_idx),
        "n_clusters": int(clusters.shape[0]),
        "vectorizer_hash": vectorizer_hash,
        "primary_protocol": "trained on pool CVs + unlabelled JDs only; 10 GT JDs and 10 hired CVs held out",
    }
    dump_json(cfg.models / "autoencoder_config.json", config)

    # Diagnostics: latent variance / dead units.
    cols = [c for c in embeddings.columns if c.startswith("dim_")]
    Z = embeddings[cols].to_numpy()
    stds = Z.std(axis=0)
    diagnostics = {
        "latent_std_min": round(float(stds.min()), 6),
        "latent_std_mean": round(float(stds.mean()), 6),
        "dead_units": int((stds < 1e-4).sum()),
        "nan_in_embeddings": int(np.isnan(Z).sum()),
        "train_val_gap": round(float(result["val_mse"] - result["train_mse"]), 6),
        "best_epoch": int(result["best_epoch"]),
        "epochs_run": int(result["epochs_run"]),
    }
    dump_json(cfg.processed / "ae_diagnostics.json", diagnostics)
    return {
        "config": config, "history": history, "comparison": comparison,
        "embeddings": embeddings, "coords": coords, "examples": examples,
        "diagnostics": diagnostics, "reconstruction": rec,
    }
