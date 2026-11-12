"""Stage 1 CLI: local-LLM keyword extraction for CVs and JDs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group assignment"))

from src.config import load_config  # noqa: E402
from src.data import read_jsonl  # noqa: E402
from src.llm import extract_all, score_extraction_quality  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--doc-type", choices=["cv", "jd"], default=None,
                    help="restrict to CVs or JDs (default: both)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg.raw.setdefault("llm", {})["model"] = args.model
    documents = read_jsonl(cfg.processed / "documents.jsonl")
    if not documents:
        raise SystemExit("documents.jsonl missing. Run scripts/00_validate_data.py first.")
    doc_types = {"pool_cv", "gt_cv"} if args.doc_type == "cv" else (
        {"gt_jd", "unlabelled_jd"} if args.doc_type == "jd" else None)
    result = extract_all(cfg, documents, force=args.force, limit=args.limit, doc_types=doc_types)
    print(json.dumps(result["summary"], indent=2))
    if not args.limit:
        extracted = read_jsonl(cfg.processed / "cv_extracted.jsonl")
        quality = score_extraction_quality(cfg, documents, extracted)
        print("Extraction quality vs Excel Skills:", json.dumps(quality, indent=2))
    print("STAGE 1 STATUS:", "PASS" if result["summary"]["success_rate"] >= 0.95 or args.limit else "CHECK")


if __name__ == "__main__":
    main()
