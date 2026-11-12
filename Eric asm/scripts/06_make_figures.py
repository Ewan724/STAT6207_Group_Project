"""Stage 6b CLI: regenerate every static PNG from the saved artifacts.

This script never retrains or re-ranks. It is safe to run after any stage and is
used to prove that all figures in results/figures/ are reproducible from disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group assignment"))

from src.autoencoder import Autoencoder, load_inputs  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import read_jsonl  # noqa: E402
from src.plots import (ae_architecture, ae_loss_curves, ae_reconstruction_error,  # noqa: E402
                       dataset_overview, extraction_quality, keyword_cluster_network,
                       keyword_cluster_sizes, keyword_coverage_heatmap,
                       keyword_merge_examples, latent_space, metric_comparison,
                       pipeline_diagram, rank_distribution, recall_by_jd,
                       reconstruction_examples, score_margin, setup)


def _require(path: Path, stage: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run {stage} first.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup()

    _require(cfg.processed / "documents.jsonl", "scripts/00_validate_data.py")
    _require(cfg.processed / "keyword_clusters.csv", "scripts/02_merge_keywords.py")
    _require(cfg.processed / "document_embeddings.csv", "scripts/03_build_autoencoder.py")
    _require(cfg.processed / "rankings.csv", "scripts/04_rank_and_evaluate.py")

    documents = read_jsonl(cfg.processed / "documents.jsonl")
    clusters = pd.read_csv(cfg.processed / "keyword_clusters.csv")
    edges = pd.read_csv(cfg.processed / "keyword_edges.csv")
    index = pd.read_csv(cfg.processed / "document_matrix_index.csv")
    matrix = sparse.load_npz(cfg.processed / "document_keyword_matrix.npz")
    embeddings = pd.read_csv(cfg.processed / "document_embeddings.csv")
    coords = pd.read_csv(cfg.tables / "latent_coordinates.csv")
    examples = pd.read_csv(cfg.tables / "reconstruction_examples.csv")
    gt = pd.read_csv(cfg.processed / "ground_truth.csv")
    pq = pd.read_csv(cfg.processed / "metrics_per_query.csv")
    summary = pd.read_csv(cfg.processed / "metrics_summary.csv")
    history = pd.read_csv(cfg.models / "ae_training_history.csv")
    ae_cfg = json.loads((cfg.models / "autoencoder_config.json").read_text(encoding="utf-8"))
    X, _, _ = load_inputs(cfg)

    made = []
    made.append(pipeline_diagram(cfg))
    made.append(dataset_overview(cfg, documents))
    made.append(keyword_cluster_network(cfg, clusters, edges))
    made.append(keyword_cluster_sizes(cfg, clusters))
    made.append(keyword_merge_examples(cfg, clusters))
    made.append(keyword_coverage_heatmap(cfg, clusters, matrix, index, documents))

    made.append(ae_architecture(cfg, X.shape[1]))
    made.append(ae_loss_curves(cfg, history, int(ae_cfg["best_epoch"])))
    model = Autoencoder(int(ae_cfg["input_dim"]), int(ae_cfg["latent_dim"]),
                        tuple(ae_cfg["hidden_dims"]), float(ae_cfg["dropout"]))
    state = torch.load(cfg.models / "autoencoder.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        rec = model(torch.tensor(X)).numpy()
    made.append(ae_reconstruction_error(cfg, embeddings, X, rec))
    made.append(reconstruction_examples(cfg, examples))
    made.append(latent_space(cfg, coords, gt, "pca"))
    made.append(latent_space(cfg, coords, gt, "tsne"))

    made.append(recall_by_jd(cfg, pq, "M3"))
    made.append(metric_comparison(cfg, summary))
    made.append(rank_distribution(cfg, pq))
    made.append(score_margin(cfg, pq, "M3"))

    if (cfg.processed / "extraction_quality.csv").exists():
        quality = pd.read_csv(cfg.processed / "extraction_quality.csv")
        made.append(extraction_quality(cfg, quality))

    inventory = pd.DataFrame([{
        "file": p.name, "size_bytes": p.stat().st_size,
        "dpi": int(cfg.get("report.figure_dpi", 200)),
    } for p in sorted(cfg.figures.glob("*.png"))])
    inventory.to_csv(cfg.tables / "figure_inventory.csv", index=False)
    print(f"Regenerated {len(made)} figures from saved artifacts.")
    print(inventory.to_string(index=False))
    print("STAGE 6b STATUS:", "PASS" if len(inventory) >= 12 else "CHECK")


if __name__ == "__main__":
    main()
