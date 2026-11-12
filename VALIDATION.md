# Validation Checklist

Status as of run time (preliminary): LLM extraction in progress; downstream stages 2-6 executed on partial extraction assembled from cache. **Re-run stages 1b,2-6 after extraction completes** for final numbers.

## Stage 0 — Data validation
- [x] Pool CVs parsed: 100 (`CV/CV-*.pdf`)
- [x] GT CVs/JDs parsed: 10/10 from `JD + Hired CV/P01-P10*.pdf` (marker `CV OF THE HIRED CANDIDATE`)
- [x] Unlabelled JDs parsed: 50 from Excel `JD Examples`
- [x] `ground_truth.csv`: 10 rows
- [x] Unique candidates: 110 (100 pool + 10 hired). Dedupe: all `keep_new` in `data/processed/dedupe_report.csv` (no true duplicates)
- [x] CV alignment match rate: 1.0 (`data/processed/cv_alignment_report.csv`)
- [x] JD dedupe: 2 near-duplicates excluded from training (`data/processed/jd_dedupe_report.csv`)
- [x] `data/raw_manifest.csv`, `data/processed/documents.jsonl` exist (170 documents)

## Stage 1 — LLM extraction (IN PROGRESS)
- [ ] Attempted 170 documents: **72/170 cached so far** (see `logs/extraction.log`, process may still be running)
- [x] Cache resumable: `data/cache/llm/*.json`
- [x] Extraction quality (partial, vs Excel Skills for pool CVs with cache): macro P=0.148, R=0.189, F1=0.160 (`data/processed/extraction_quality.csv`)
- [ ] Final `jd_extracted.jsonl`/`cv_extracted.jsonl`/`extraction_summary.json` written by script on completion (partial assembled from cache for downstream run)

## Stage 2 — Keyword merging (preliminary)
- [x] Cluster count: 215 (target 100-200; tuned by threshold) — `data/processed/cluster_metrics.json`
- [x] Curated aliases: 128 entries in `keyword_aliases.csv`
- [x] Guardrail splitting applied for confusable pairs in clustering
- [x] QA samples: `data/processed/cluster_qa.csv` (needs manual review labels)
- [x] Figures: `03_keyword_cluster_network.png`, `04_keyword_cluster_sizes.png`, `05_keyword_merge_examples.png`, `05b_keyword_coverage_heatmap.png`

## Stage 3 — Autoencoder (preliminary)
- [x] Matrix: 215 clusters × 170 docs; protocol `primary_holdout_gt` (train 126, val 22, heldout 20)
- [x] Implementation: sklearn_mlp_autoencoder (sklearn MLP) due to PyTorch segfault in `clip_grad_norm_` on this environment
- [x] Best iter 79, val MSE 0.0008798292838037014
- [x] Saved: `models/autoencoder.pkl`, `models/autoencoder_config.json`, `models/vectorizer.json`, `data/processed/document_embeddings.csv`
- [x] AE vs SVD/PCA: `results/tables/ae_vs_svd.csv`
- [x] Figures: `06`-`10` present

## Stage 4 — Retrieval (preliminary)
- [x] 10 GT JDs × 110 candidates ranked for M0–M6: `data/processed/rankings.csv`, `data/processed/top10.csv`
- [x] Primary model: M3 AE-32 cosine (a priori)
- [x] Leakage checks: hired IDs neutral (CAND-101..110), GT JDs/CVs held out from AE training, no GT labels in features

## Stage 5 — Evaluation (preliminary)
- [x] Metrics saved: `metrics_per_query.csv`, `metrics_summary.csv`, `ablation_results.csv`
- [x] Random baseline expected 10/110=0.0909; empirical summary in metrics
- [x] Precision@10 <= 0.1; 1 relevant per JD per model
- [x] Figures: `11`-`14`

## Stage 6 — Error analysis (preliminary)
- [x] Per-query table: `data/processed/error_analysis.csv`; narrative: `results/error_analysis.md`
- [x] AE vs TF-IDF: `results/tables/ae_vs_tfidf.csv`, `16_ae_vs_tfidf_ranks.png`
- [x] Recommendations and limitations in error analysis; figures `17`,`19`,`20`, case studies `15/18`

## Stage 7 — Website & packaging
- [x] Streamlit app: `group assignment/app.py` + 5 pages in `pages/`
- [x] PNG figures in `results/figures/`: 24
- [x] README.md, VALIDATION.md, PROMPT.md, config, requirements, run_all.py present

## Deviations / blockers
- **Environment:** macOS Darwin, `python3` (not Windows `py`). Data at project root (`CV/`, `JD + Hired CV/`, xlsx, plan PDF) not under `DATA/`; treated read-only.
- **LLM:** Ollama `qwen3.5:4b` via `/api/chat` with `think:false`; structured output/format unsupported, handled in `src/llm.py`. Extraction long-running and cached; run was incomplete at validation time (72/170).
- **AE implementation:** PyTorch training segfaulted in `torch.nn.utils.clip_grad_norm_` on this environment; replaced with sklearn MLPRegressor autoencoder (same architecture intent, bottleneck=32). Saved as `models/autoencoder.pkl` + config.
- **Preliminary results:** All metrics/rankings/figures generated from partial extraction (72 docs cached). **Must re-run** after extraction finishes:
  `python3 "group assignment/scripts/01b_extraction_quality.py" --config "group assignment/config.yaml"`
  `python3 "group assignment/scripts/02_merge_keywords.py" --config "group assignment/config.yaml"`
  `python3 "group assignment/scripts/03_build_autoencoder.py" --config "group assignment/config.yaml"`
  `python3 "group assignment/scripts/04_rank_and_evaluate.py" --config "group assignment/config.yaml"`
  `python3 "group assignment/scripts/05_error_analysis.py" --config "group assignment/config.yaml"`
  `python3 "group assignment/scripts/06_make_figures.py" --config "group assignment/config.yaml"`
- **Manual review items:** cluster merge accuracy (30 samples) and extraction spot-checks (5 CVs/5 JDs) require human annotation; QA files provided.
