# Plan — AI CV Matching Baseline

Ten-step execution plan for the STAT6207 group project baseline, written before
the full run and kept as a record of the intended order. Every artifact lives
under `group assignment/`; `DATA/` is read-only.

## Data manifest (observed at Stage 0)

| Item | Count / value |
| --- | --- |
| Data root | `DATA/` (read-only) |
| Pool CV PDFs | 100 (`CV-001` … `CV-100`) |
| Hired CV PDFs (one per labelled JD) | 10 (`CAND-101` … `CAND-110`) |
| Labelled JD PDFs | 10 (`P01` … `P10`) |
| Unlabelled JD PDFs | 50 |
| Unique candidates after dedupe | 110 |
| Ground-truth rows | 10 (one hired CV per JD) |
| Excel workbook | Manabat CV/JD index (title, company, role family, skills) |
| CV ↔ Excel alignment | 100/100 titles, 100/100 companies |
| JD duplicates excluded | 2 near-duplicates |
| Hired-CV duplicates merged | 10 kept as new candidates (no exact duplicates) |
| Leakage scan | `Hired for:` 0, `P01`–`P10` in CV text 0 |

Role families follow the Excel workbook: Finance & Legal, Engineering, Data &
Analytics, Sales & Marketing, HR & Admin, Operations & Supply Chain, Design &
Creative, Product, plus `unknown` for hired CVs (the JD family would leak the
label).

## Steps

1. **Stage 0 — validate data.** Parse the workbook and all PDFs, build
   `documents.jsonl`, deduplicate JDs, map pool CVs to Excel metadata, align the
   10 hired CVs, verify the 10 ground-truth rows, and run leakage checks. Save
   `stage0_summary.json`, manifests and alignment reports.
2. **Stage 1 — local-LLM extraction.** Extract structured keywords for CVs and
   JDs with the local Ollama model, cache every response by prompt+model hash,
   validate keywords against source text, and score extraction quality against
   the Excel Skills silver standard. Support `--limit` and `--force`.
3. **Stage 2 — keyword merging.** Embed extracted terms with char TF-IDF + SVD
   (documented fallback for the unavailable sentence-transformer), cluster them
   into 100–200 clusters, map every term to a cluster, run guardrail checks,
   and produce a 30-pair manual merge-review sample.
4. **Stage 3 — autoencoder.** Build the document × keyword TF-IDF matrix, train
   the pre-registered 32-d denoising/dropout AE on pool CVs and unlabelled JDs
   only, early-stop on validation MSE, run the 16/32/64-d ablation as
   exploratory work, and save embeddings, history, diagnostics and
   reconstruction errors.
5. **Stage 4 — ranking.** Score all 110 candidates against the 10 labelled JDs
   with M0–M6, save 7,700 ranking rows, top-10 tables with template reasons,
   must-have coverage, and the seeded random baseline (1,000 permutations).
6. **Stage 5 — metrics.** Compute Recall@10, Hit@1, Hit@5, MRR@10, nDCG@10 and
   Precision@10 with Wilson and bootstrap intervals, and independently
   recompute Recall@10 from `rankings.csv` as an integrity check.
7. **Stage 6 — error analysis.** Review all 10 queries, assign ordered primary
   failure categories, compare AE vs TF-IDF, write `results/error_analysis.md`,
   build at least three case studies, and produce at least eight prioritised
   recommendations with evidence.
8. **Stage 7 — figures.** Regenerate every required PNG from saved artifacts
   (no recomputation, no invented numbers) and record the inventory.
9. **Stage 8 — website.** Build a five-page offline Streamlit app (Overview,
   Keyword merging, Autoencoder, Search results, Recall & error analysis) with
   Plotly charts, graceful handling of missing files, and no network calls.
10. **Stage 9 — documentation and validation.** Write `README.md` and
    `VALIDATION.md`, run the full artifact checklist, verify a clean end-to-end
    run with `run_all.py`, and record every deviation and blocker.

## Reproduction

From the project root (Windows PowerShell or macOS shell):

```powershell
py "group assignment/run_all.py"
```

Stages can be resumed or repeated:

```powershell
py "group assignment/run_all.py" --from-stage 04 --force
py "group assignment/scripts/01_extract_keywords.py" --config "group assignment/config.yaml" --limit 3
```

Fixed seed 42; all randomness is seeded and all results are reproducible from
the saved CSV/JSON artifacts.
