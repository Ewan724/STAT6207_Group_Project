"""Stage 2 CLI: keyword normalisation, merging, clustering and matrix build."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group assignment"))

from src.config import load_config  # noqa: E402
from src.data import read_jsonl  # noqa: E402
from src.keywords import build_clusters  # noqa: E402
from src.plots import (dataset_overview, keyword_cluster_network, keyword_cluster_sizes,  # noqa: E402
                       keyword_coverage_heatmap, keyword_merge_examples, setup)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup()
    result = build_clusters(cfg, force=args.force)
    clusters = result["cluster_df"]
    edges = pd.read_csv(cfg.processed / "keyword_edges.csv")
    documents = read_jsonl(cfg.processed / "documents.jsonl")
    matrix = sparse.load_npz(cfg.processed / "document_keyword_matrix.npz")
    index = pd.read_csv(cfg.processed / "document_matrix_index.csv")

    keyword_cluster_network(cfg, clusters, edges)
    keyword_cluster_sizes(cfg, clusters)
    keyword_merge_examples(cfg, clusters)
    keyword_coverage_heatmap(cfg, clusters, matrix, index, documents)
    dataset_overview(cfg, documents)

    print(json.dumps(result["metrics"], indent=2))
    print(json.dumps(result.get("coverage_summary", {}), indent=2))
    n = result["metrics"]["n_clusters"]
    print("STAGE 2 STATUS:", "PASS" if 100 <= n <= 200 else "CHECK")


if __name__ == "__main__":
    main()
