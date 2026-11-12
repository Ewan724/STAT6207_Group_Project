# Error Analysis

This report reviews retrieval results for the 10 ground-truth JD–hired-CV pairs. Primary model: **M3 AE-32 cosine** (selected a priori by validation reconstruction loss, not test recall). Comparison: **M1 TF-IDF cosine**.

## P01 — Regional Finance Controller
- Ground truth: `CAND-101` (Regional Finance Manager, nan). AE-32 result: **MISS** (rank 81), score 0.932.
- Top-1: `CV-022` IT Manager (score 0.964, margin -0.032). Must-have coverage hired/top1: 0.00/0.00.
- Primary failure category: `E1_extraction_missing_skill`. Missing must-haves (hired): n/a. Likely improvement: better must-have weighting/filtering, richer extraction, or domain/metadata signals.

## P02 — Sales Manager in Indonesia
- Ground truth: `CAND-102` (Technical Sales Specialist, nan). AE-32 result: **MISS** (rank 77), score 0.971.
- Top-1: `CV-026` Senior Full Stack Engineer (score 1.000, margin -0.029). Must-have coverage hired/top1: 0.00/0.00.
- Primary failure category: `E1_extraction_missing_skill`. Missing must-haves (hired): n/a. Likely improvement: better must-have weighting/filtering, richer extraction, or domain/metadata signals.

## P03 — GTM Marketing Manager / Lead
- Ground truth: `CAND-103` (Head of Growth & Digital Transformation, nan). AE-32 result: **MISS** (rank 81), score 0.958.
- Top-1: `CAND-105` Senior Product Manager (score 0.990, margin -0.033). Must-have coverage hired/top1: 0.00/0.00.
- Primary failure category: `E1_extraction_missing_skill`. Missing must-haves (hired): n/a. Likely improvement: better must-have weighting/filtering, richer extraction, or domain/metadata signals.

## P04 — Data Engineer
- Ground truth: `CAND-104` (Data Engineer, nan). AE-32 result: **HIT** (rank 2), score 0.969.
- Top-1: `CAND-101` Regional Finance Manager (score 0.979, margin -0.009). Must-have coverage hired/top1: 1.00/0.00.
- The hired candidate appears in the shortlist; ranking is supported by keyword/representation overlap, though recruiter context may differ.

## P05 — Product Manager — WMS & Warehouse Integrations
- Ground truth: `CAND-105` (Senior Product Manager, nan). AE-32 result: **MISS** (rank 81), score 0.965.
- Top-1: `CV-022` IT Manager (score 0.978, margin -0.013). Must-have coverage hired/top1: 0.00/1.00.
- Primary failure category: `E4_hard_negative_lookalike`. Missing must-haves (hired): Project Management. Likely improvement: better must-have weighting/filtering, richer extraction, or domain/metadata signals.

## P06 — Deployment Strategist
- Ground truth: `CAND-106` (Principal, Data Science, nan). AE-32 result: **MISS** (rank 90), score 0.930.
- Top-1: `CV-021` Software Engineer (score 0.974, margin -0.044). Must-have coverage hired/top1: 0.00/0.00.
- Primary failure category: `E1_extraction_missing_skill`. Missing must-haves (hired): Project Management. Likely improvement: better must-have weighting/filtering, richer extraction, or domain/metadata signals.

## P07 — Marketing Manager (HK & SG)
- Ground truth: `CAND-107` (Marketing Manager, nan). AE-32 result: **MISS** (rank 76), score 0.979.
- Top-1: `CV-026` Senior Full Stack Engineer (score 1.000, margin -0.021). Must-have coverage hired/top1: 0.00/0.00.
- Primary failure category: `E1_extraction_missing_skill`. Missing must-haves (hired): n/a. Likely improvement: better must-have weighting/filtering, richer extraction, or domain/metadata signals.

## P08 — Field Service Engineer (FSE)
- Ground truth: `CAND-108` (Field Service Engineer, nan). AE-32 result: **MISS** (rank 82), score 0.948.
- Top-1: `CV-026` Senior Full Stack Engineer (score 1.000, margin -0.052). Must-have coverage hired/top1: 0.00/0.00.
- Primary failure category: `E1_extraction_missing_skill`. Missing must-haves (hired): n/a. Likely improvement: better must-have weighting/filtering, richer extraction, or domain/metadata signals.

## P09 — Tech Lead
- Ground truth: `CAND-109` (Tech Lead, nan). AE-32 result: **MISS** (rank 102), score 0.843.
- Top-1: `CAND-104` Data Engineer (score 0.977, margin -0.133). Must-have coverage hired/top1: 0.00/0.00.
- Primary failure category: `E1_extraction_missing_skill`. Missing must-haves (hired): Code Reviews; Technical Decisions; Project Management. Likely improvement: better must-have weighting/filtering, richer extraction, or domain/metadata signals.

## P10 — Regional Lead Designer UI/UX
- Ground truth: `CAND-110` (Lead, UI/UX, nan). AE-32 result: **MISS** (rank 76), score 0.975.
- Top-1: `CV-026` Senior Full Stack Engineer (score 0.985, margin -0.010). Must-have coverage hired/top1: 0.00/0.00.
- Primary failure category: `E1_extraction_missing_skill`. Missing must-haves (hired): n/a. Likely improvement: better must-have weighting/filtering, richer extraction, or domain/metadata signals.

## Recommendations

| Priority | Issue | Proposed fix | Evidence/expected impact | Effort |
|---|---|---|---|---|
| P0 | Must-have underweighting | Hybrid/filter by must-have coverage | Based on per-query misses and coverage gaps; hypothesis pending validation. | 2 |
| P0 | Extraction noise/missing skills | Stronger local model or second pass | Based on per-query misses and coverage gaps; hypothesis pending validation. | 3 |
| P1 | Keyword merge errors | Ontology + manual review | Based on per-query misses and coverage gaps; hypothesis pending validation. | 2 |
| P1 | Missing recruiter filters | Location/language/seniority filters | Based on per-query misses and coverage gaps; hypothesis pending validation. | 2 |
| P1 | Reconstruction not ranking-aware | Sentence embeddings/cross-encoder | Based on per-query misses and coverage gaps; hypothesis pending validation. | 3 |
| P2 | Small label set | Collect ~50 labelled pairs, learning-to-rank | Based on per-query misses and coverage gaps; hypothesis pending validation. | 4 |
| P2 | Domain/years not modelled | Structured features and calibration | Based on per-query misses and coverage gaps; hypothesis pending validation. | 2 |
| P2 | Score interpretability | Calibrated reasons and evidence spans | Based on per-query misses and coverage gaps; hypothesis pending validation. | 2 |

## Limitations

- n=10 labelled queries with one relevant CV per JD; sparse and potentially noisy labels.
- Hired candidate is not necessarily best on paper; hiring includes interview, salary, availability, recruiter judgment.
- Baseline ignores salary, notice period, location/citizenship/language hard filters.
- LLM extraction noise and PDF truncation (ellipsis) may drop evidence.
- Keyword clustering is heuristic; over/under-merging possible.
- AE trained on small corpus (~150 docs) optimising reconstruction, not ranking.
- Held-out keyword coverage may disadvantage rare must-have skills.
- 100-CV pool may not represent real candidate population; random Recall@10 ~0.09 for 110 candidates.
- 50 unlabelled JDs are not ground truth; used only for training/context after dedupe.
- No statistical significance claims possible with n=10.
