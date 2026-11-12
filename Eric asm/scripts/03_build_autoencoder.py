"""Stage 3 CLI: train the autoencoder, encode documents, save diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group assignment"))

from src.config import load_config  # noqa: E402
from src.autoencoder import load_inputs, run_stage3  # noqa: E402
from src.plots import (ae_architecture, ae_loss_curves, ae_reconstruction_error,  # noqa: E402
                       latent_space, reconstruction_examples, setup)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup()
    result = run_stage3(cfg, force=args.force)
    X, _, _ = load_inputs(cfg)
    gt = pd.read_csv(cfg.processed / "ground_truth.csv")
    ae_architecture(cfg, X.shape[1])
    ae_loss_curves(cfg, result["history"], result["config"]["best_epoch"])
    ae_reconstruction_error(cfg, result["embeddings"], X, result["reconstruction"]["reconstruction"])
    reconstruction_examples(cfg, result["examples"])
    latent_space(cfg, result["coords"], gt, "pca")
    latent_space(cfg, result["coords"], gt, "tsne")
    print(json.dumps(result["config"], indent=2))
    print(json.dumps(result["diagnostics"], indent=2))
    print(result["comparison"].to_string(index=False))
    print("STAGE 3 STATUS:", "PASS" if result["diagnostics"]["nan_in_embeddings"] == 0 else "CHECK")


if __name__ == "__main__":
    main()
