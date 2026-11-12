"""Stage 0 CLI: validate and parse the read-only DATA/ corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group assignment"))

from src.config import load_config  # noqa: E402
from src.data import run_stage0  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="unused in Stage 0 (full validation only)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    result = run_stage0(cfg, force=args.force)
    print(json.dumps(result.summary, indent=2))
    s = result.summary
    ok = (
        s["pool_cvs"] == 100 and s["gt_cvs"] == 10 and s["gt_jds"] == 10
        and s["unlabelled_jds"] == 50 and s["unique_candidates"] == 110
        and s["ground_truth_rows"] == 10
        and s["alignment_title_matches"] >= 90
        and s["leakage_checks"]["hired_for_in_cv_text"] == 0
        and s["leakage_checks"]["pxx_in_cv_text"] == 0
    )
    print("\nSTAGE 0 STATUS:", "PASS" if ok else "CHECK")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
