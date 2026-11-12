"""Stage 6 CLI: error taxonomy, per-query review, case studies and recommendations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group assignment"))

from src.analysis import build_error_analysis, case_study_data, write_error_markdown  # noqa: E402
from src.config import load_config  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.plots import (ae_vs_tfidf_ranks, case_study_combined, case_study_heatmap,  # noqa: E402
                       error_taxonomy, extraction_quality, recommendation_matrix, setup)

CASE_IDS = ["P01", "P04", "P08", "P10"]


def _pick_disagreement(cfg, comp: pd.DataFrame) -> str:
    """Choose the query with the largest absolute AE-vs-TF-IDF rank gap."""
    if not len(comp):
        return CASE_IDS[0]
    row = comp.reindex(comp["delta"].abs().sort_values(ascending=False).index).iloc[0]
    return str(row["jd_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup()

    required = ["rankings.csv", "ground_truth.csv", "metrics_summary.csv"]
    missing = [name for name in required if not (cfg.processed / name).exists()]
    if missing:
        raise SystemExit(f"Missing {missing}. Run scripts/04_rank_and_evaluate.py first.")

    metrics = compute_metrics(cfg)
    analysis = build_error_analysis(cfg)
    text = write_error_markdown(cfg, analysis, metrics)
    print(f"Wrote {cfg.group_dir / 'results' / 'error_analysis.md'} ({len(text)} chars)")

    # At least three case studies plus the largest AE/TF-IDF disagreement.
    extra = _pick_disagreement(cfg, analysis["ae_vs_tfidf"])
    case_ids = CASE_IDS + ([extra] if extra not in CASE_IDS else [])
    datasets = []
    case_rows = []
    for jd_id in case_ids:
        data = case_study_data(cfg, jd_id, model="M3")
        datasets.append(data)
        case_study_heatmap(cfg, data, jd_id)
        e = analysis["error_df"]
        m3 = e[(e["jd_id"] == jd_id) & (e["model"] == "M3")].iloc[0]
        case_rows.append({
            "jd_id": jd_id, "jd_title": m3["jd_title"], "role_family": m3["role_family"],
            "hired_cv_id": m3["hired_cv_id"], "hired_rank": int(m3["hired_rank"]),
            "primary_failure_category": m3["primary_failure_category"],
            "top1_cv_id": m3["top1_cv_id"], "top1_title": m3["top1_title"],
            "coverage_hired": m3["must_have_coverage_hired"],
            "coverage_top1": m3["must_have_coverage_top1"],
            "comment": m3["comment"],
        })
    case_study_combined(cfg, datasets)
    pd.DataFrame(case_rows).to_csv(cfg.tables / "case_study_summary.csv", index=False)

    ae_vs_tfidf_ranks(cfg, analysis["ae_vs_tfidf"])
    error_taxonomy(cfg, analysis["error_df"], "M3")
    recommendation_matrix(cfg, analysis["recommendations"])
    if (cfg.processed / "extraction_quality.csv").exists():
        extraction_quality(cfg, pd.read_csv(cfg.processed / "extraction_quality.csv"))

    summary = {
        "queries_reviewed": int(analysis["error_df"]["jd_id"].nunique()),
        "m3_misses": int((analysis["error_df"][analysis["error_df"]["model"] == "M3"]["hired_rank"] > 10).sum()),
        "case_studies": case_ids,
        "ae_wins": int((analysis["ae_vs_tfidf"]["winner"] == "AE").sum()),
        "tfidf_wins": int((analysis["ae_vs_tfidf"]["winner"] == "TF-IDF").sum()),
        "recommendations": int(len(analysis["recommendations"])),
    }
    (cfg.processed / "error_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("STAGE 6 STATUS: PASS" if summary["queries_reviewed"] == 10 and summary["recommendations"] >= 8 else "STAGE 6 STATUS: CHECK")


if __name__ == "__main__":
    main()
