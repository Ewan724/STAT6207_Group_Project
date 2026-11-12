"""Stage 4/5 CLI: rank candidates, evaluate recall and save metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group assignment"))

from src.config import load_config  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.retrieval import build_rankings, random_baseline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    result = build_rankings(cfg, force=args.force)
    print(json.dumps(result["summary"], indent=2))
    rb = random_baseline(cfg)
    metrics = compute_metrics(cfg)
    print(metrics["summary"].to_string(index=False))
    print("Random baseline:", rb.to_dict(orient="records")[0])
    print(json.dumps(metrics["checks"], indent=2))
    ok = all(v.get("match", True) for k, v in metrics["checks"].items() if isinstance(v, dict))
    print("STAGE 4/5 STATUS:", "PASS" if ok and metrics["checks"]["precision_at_10_ok"] else "CHECK")


if __name__ == "__main__":
    main()
