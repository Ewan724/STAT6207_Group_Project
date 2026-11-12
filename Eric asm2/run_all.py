#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "group assignment/scripts/00_validate_data.py",
    "group assignment/scripts/01_extract_keywords.py",
    "group assignment/scripts/02_merge_keywords.py",
    "group assignment/scripts/03_build_autoencoder.py",
    "group assignment/scripts/04_rank_and_evaluate.py",
    "group assignment/scripts/05_error_analysis.py",
    "group assignment/scripts/06_make_figures.py",
]
def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', default='group assignment/config.yaml'); p.add_argument('--force', action='store_true'); p.add_argument('--limit', type=int); p.add_argument('--model')
    args = p.parse_args()
    for s in SCRIPTS:
        cmd = [sys.executable, str(ROOT/s), '--config', args.config]
        if args.force: cmd.append('--force')
        if args.limit: cmd += ['--limit', str(args.limit)]
        if args.model and '01_extract' in s: cmd += ['--model', args.model]
        print('RUN', ' '.join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print('FAILED', s, r.returncode); sys.exit(r.returncode)
    print('Pipeline complete. Run: python3 -m streamlit run "group assignment/app.py"')
if __name__ == '__main__':
    main()
