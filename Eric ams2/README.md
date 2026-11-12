# AI CV Matching — Baseline (Second Talent)

Reproducible baseline pipeline for CV–JD matching: parse documents, extract structured keywords with a local LLM (Ollama), merge/cluster keywords, train an autoencoder representation, rank candidates by cosine similarity, evaluate Recall@10 on 10 labelled pairs, and present results in a Streamlit app.

## Data (read-only inputs)
- `CV/CV-001.pdf` … `CV-100.pdf`: 100 anonymised pool CVs.
- `JD + Hired CV/P01…P10 *.pdf`: 10 PDFs, each containing a JD and the hired CV (ground truth).
- `Manatal Examples – JD & CV (anonymised).xlsx`: `CV Examples` (100 rows aligned by Ref) and `JD Examples` (50 unlabelled JDs).
- `AI_CV_Matching_Project_Plan.pdf`: project brief.

**Ground truth:** 10 pairs (P01–P10). Candidate pool = 100 pool CVs + 10 hired CVs = **110** (no true duplicates found; see `data/processed/dedupe_report.csv`). The 50 Excel JDs are unlabelled (not used for recall).

## Pipeline
0. `00_validate_data.py` — parse/split PDFs, manifests, GT, dedupe/alignment checks.
1. `01_extract_keywords.py` — local LLM extraction via Ollama (`qwen3.5:4b`, JSON, cached in `data/cache/llm/`).
   `01b_extraction_quality.py` — compare pool-CV extraction vs Excel `Skills` (silver standard).
2. `02_merge_keywords.py` — normalization, curated aliases, embeddings, clustering, guardrails, figures.
3. `03_build_autoencoder.py` — document×cluster TF-IDF matrix, AE (primary holdout: train on pool CVs + unlabelled JDs; GT JDs/hired CVs held out), embeddings, diagnostics.
4. `04_rank_and_evaluate.py` — rank 110 candidates for 10 GT JDs (M0–M6), top-10 reasons, Recall@10/MRR/CI.
5. `05_error_analysis.py` — per-query review, failure taxonomy, case studies, recommendations.
6. `06_make_figures.py` — evaluation/report figures.
7. Streamlit app: `app.py` + `pages/`.

## Environment
- macOS, Python 3.13.x, PyTorch/scikit-learn/pandas/numpy/PyMuPDF/openpyxl/matplotlib/streamlit/plotly.
- Local LLM: Ollama at `http://127.0.0.1:11434`, model `qwen3.5:4b` (no external APIs).
- Seed: 42.

Install missing deps (if needed):
```bash
python3 -m pip install streamlit plotly networkx jsonschema rapidfuzz pyyaml requests umap-learn
```

## Run
From project root (`/Users/ngeric/Desktop/Deep learning/Asm4`):
```bash
# Stage 0
python3 "group assignment/scripts/00_validate_data.py" --config "group assignment/config.yaml"

# Stage 1 (LLM extraction, cached/resumable)
python3 "group assignment/scripts/01_extract_keywords.py" --config "group assignment/config.yaml"
python3 "group assignment/scripts/01b_extraction_quality.py" --config "group assignment/config.yaml"

# Stages 2-6
python3 "group assignment/scripts/02_merge_keywords.py" --config "group assignment/config.yaml"
python3 "group assignment/scripts/03_build_autoencoder.py" --config "group assignment/config.yaml"
python3 "group assignment/scripts/04_rank_and_evaluate.py" --config "group assignment/config.yaml"
python3 "group assignment/scripts/05_error_analysis.py" --config "group assignment/config.yaml"
python3 "group assignment/scripts/06_make_figures.py" --config "group assignment/config.yaml"

# Or orchestrate
python3 "group assignment/run_all.py" --config "group assignment/config.yaml"
```
Flags: `--force` (recompute), `--limit N` (smoke test), `--model <name>`.

## Website
```bash
python3 -m streamlit run "group assignment/app.py"
```
Pages: Overview, Keyword Merging, Autoencoder, Search Results (top-10 per JD, hired CV highlighted), Error Analysis.

## Metrics (definitions)
- **Recall@10 / Hit@10**: hired CV in top 10 for a JD (1 relevant per JD). Mean over 10 = hits/10.
- **Hit@1/5**, **MRR@10**, **nDCG@10** as in `data/processed/metrics_summary.csv`.
- **Precision@10** = relevant in top 10 / 10 (max 0.1 here). **Not recall.**
- Random baseline Recall@10 ≈ 10/110 ≈ 0.0909. Wilson and bootstrap 95% CIs reported (n=10, wide).

Primary model **M3 AE-32 cosine** selected a priori (validation reconstruction loss), not tuned on test recall. See `data/processed/metrics_summary.csv` and `ablation_results.csv` for all models M0–M6.

## Key findings & limitations
- n=10 labelled queries (sparse, possibly noisy); one query shifts mean Recall@10 by 0.10; no significance claims.
- Hired candidate may not be best on paper; baseline omits salary/notice/location/language hard filters.
- LLM extraction and keyword clustering introduce noise; AE optimises reconstruction, not ranking.
- Hard lookalikes and missing domain/years/metadata signals are common failure sources.
- See `results/error_analysis.md`, `data/processed/error_analysis.csv`, and Validation for details.

## Privacy
Anonymised CVs/JDs only. PII masked before LLM prompts. All processing local (Ollama). No uploads, no re-identification.

## Outputs
Key artifacts: `data/processed/{documents,ground_truth,rankings,top10,metrics_summary,error_analysis}.csv|jsonl`, `models/autoencoder.pt`, `results/figures/*.png`, `results/error_analysis.md`, `VALIDATION.md`.
