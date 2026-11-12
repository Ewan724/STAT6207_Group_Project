"""Run the full baseline pipeline, stage by stage.

Examples (Windows PowerShell; quote the folder because it contains a space):

    py "group assignment/run_all.py"
    py "group assignment/run_all.py" --from-stage 2
    py "group assignment/run_all.py" --force
    py "group assignment/run_all.py" --limit 3      # LLM smoke test only

Every stage is checkpointed. Re-running without --force resumes from caches.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "group assignment"
CONFIG = GROUP / "config.yaml"

STAGES = [
    ("00", "scripts/00_validate_data.py", "Validate DATA/ and build the 170-document manifest"),
    ("01", "scripts/01_extract_keywords.py", "Local-LLM keyword extraction (cached per document)"),
    ("02", "scripts/02_merge_keywords.py", "Alias merging, clustering and document-keyword matrix"),
    ("03", "scripts/03_build_autoencoder.py", "Train the autoencoder and encode all documents"),
    ("04", "scripts/04_rank_and_evaluate.py", "Rank 110 candidates for M0-M6 and score recall"),
    ("05", "scripts/05_error_analysis.py", "Error taxonomy, 10 per-query reviews, case studies"),
    ("06", "scripts/06_make_figures.py", "Regenerate every static PNG from saved artifacts"),
]


def run_stage(script: str, force: bool, limit: int | None, extra: list[str] | None = None) -> int:
    cmd = [sys.executable, str(GROUP / script), "--config", str(CONFIG)]
    if force:
        cmd.append("--force")
    if limit is not None and script.endswith("01_extract_keywords.py"):
        cmd.extend(["--limit", str(limit)])
    if extra:
        cmd.extend(extra)
    print("\n$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-stage", default="00", choices=[s[0] for s in STAGES])
    ap.add_argument("--to-stage", default="06", choices=[s[0] for s in STAGES])
    ap.add_argument("--force", action="store_true", help="ignore caches and recompute")
    ap.add_argument("--limit", type=int, default=None,
                    help="only for Stage 1: extract this many documents (smoke test)")
    args = ap.parse_args()

    start = [s[0] for s in STAGES].index(args.from_stage)
    end = [s[0] for s in STAGES].index(args.to_stage)
    if start > end:
        raise SystemExit("--from-stage must not be after --to-stage")

    GROUP.joinpath("logs").mkdir(parents=True, exist_ok=True)
    overall = time.time()
    for i, (num, script, desc) in enumerate(STAGES):
        if i < start or i > end:
            continue
        print(f"\n{'=' * 78}\nSTAGE {num} - {desc}\n{'=' * 78}", flush=True)
        t0 = time.time()
        code = run_stage(script, args.force, args.limit)
        dt = time.time() - t0
        print(f"Stage {num} finished in {dt:.1f}s with exit code {code}", flush=True)
        if code != 0:
            raise SystemExit(f"Stage {num} failed (exit {code}). Fix it and rerun with --from-stage {num}.")
        if args.limit is not None and num == "01":
            print("Smoke test complete (--limit). Stopping before Stage 2.", flush=True)
            break
    print(f"\nPipeline complete in {time.time() - overall:.1f}s.")


if __name__ == "__main__":
    main()
