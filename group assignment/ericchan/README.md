# STAT6207 Assignment 4 — CV/JD Matching with a Local LLM and an Autoencoder

## Deliverables

| file | what it is |
|---|---|
| `A4_report.html` | the report (self-contained, figures embedded) |
| `A4_report.pdf` | the same report as PDF, 23 pages |
| `A4_prompt_transcript.md` | every LLM prompt and response, generated from the run logs |
| `A4_conversation_transcript.md` | the development conversation, including the wrong turns |
| `scripts/` | the pipeline, numbered by stage |
| `data/` | all intermediate artefacts (extraction, labels, embeddings, metrics) |
| `figures/` | report figures |

## Headline result

| system | Recall@10 | NDCG@10 | MAP@10 | MRR |
|---|---:|---:|---:|---:|
| random | 0.093 | 0.073 | 0.013 | 0.107 |
| keyword AE alone (the brief's model) | 0.316 | 0.287 | 0.173 | 0.432 |
| merged-keyword overlap | 0.506 | 0.474 | 0.359 | 0.621 |
| TF-IDF on raw text | 0.717 | 0.682 | 0.608 | 0.813 |
| text embedding | 0.781 | 0.712 | 0.622 | 0.883 |
| + job-family filter (predicted) | 0.875 | 0.906 | 0.756 | 0.950 |
| **+ LLM rerank (final)** | **0.887** | **0.932** | **0.822** | **1.000** |

Evaluated over 10 JDs × 100 CVs with 62 relevant pairs (grade ≥ 2).

## Out-of-sample check (80:20)

The table above is transductive — tuned and evaluated on the same 10 JDs. `scripts/13_holdout_eval.py`
runs a **5-fold 80:20 holdout over the queries**: 8 JDs to tune, 2 held out, every JD tested
exactly once, with the autoencoder retrained per fold without the held-out JD.

| system | holdout Recall@10 | ± SD | in-sample | gap |
|---|---:|---:|---:|---:|
| merged keywords | 0.506 | 0.172 | 0.506 | +0.000 |
| keyword AE alone | 0.511 | 0.163 | 0.316 | +0.195 |
| text embedding | 0.781 | 0.156 | 0.781 | +0.000 |
| fusion + family filter | 0.880 | 0.123 | 0.875 | +0.005 |
| **full pipeline** | **0.858** | 0.123 | **0.887** | **−0.030** |

**The headline generalises** (gap −0.030), but the holdout also exposes that the fusion
weights are unstable across folds (3 different choices in 5 folds) and that the reranker's
recall contribution is within noise — its real value is ordering and rationales. See §7.4.

## Two findings worth reading first

**1. The hired CV is not in the pool.** All 10 hired CVs have zero 8-gram overlap
with all 100 pooled CVs. So "recall" cannot mean "did we find the hired person" —
relevance labels had to be built explicitly for all 1,000 pairs. §2 of the report.

**2. The autoencoder's input is its limit, not its capacity.** The keyword AE
reconstructs its input with F1 = 0.999 and its latent space is nearly useless
(purity 0.30 vs 0.36 for the raw input). The same architecture on a text embedding
reaches 0.38 purity and 0.76 Recall@10. The AE was never the bottleneck; the
443-dimension keyword vector was. §5 of the report.

## Reproducing

Prerequisites: `ollama` with `qwen3:8b` and `nomic-embed-text` pulled, plus
`pymupdf`, `torch`, `scikit-learn`, `numpy`, `matplotlib`.

```bash
python scripts/01_extract.py             # PDFs -> text JSON
python scripts/00_cv_family.py           # job-family map
python scripts/02_extract_keywords.py    # qwen3:8b extraction (110 docs, ~6 min CPU)
python scripts/03_ground_truth.py        # deterministic gate + LLM grading (1,000 pairs)
python scripts/03b_second_annotator.py   # independent re-annotation for kappa
python scripts/04_verify.py              # dataset checks
python scripts/05_merge_keywords.py      # 3-stage keyword merge
python scripts/06_autoencoder.py         # the brief's AE (keywords)
python scripts/06e1_ae_kw_txt.py         # AE variant: keywords + text
python scripts/06e2_ae_text.py           # AE variant: text only
python scripts/07_retrieve_eval.py       # scorers, fusion, metrics
python scripts/07b_family_classifier.py  # independent family classifier
python scripts/07c_llm_rerank.py         # LLM reranking
python scripts/07d_apply_rerank.py       # fold rerank into metrics
python scripts/13_holdout_eval.py        # 5-fold 80:20 out-of-sample check
python scripts/08_figures.py             # figures
python scripts/09_agreement.py           # inter-annotator agreement
python scripts/10_build_report.py        # A4_report.html
python scripts/11_html_to_pdf.py         # A4_report.pdf
python scripts/12_transcript.py          # A4_prompt_transcript.md
```

The whole pipeline runs offline once the two models are pulled. Seeds are fixed;
every stage is resumable (completed documents are skipped on restart).

### Diagnostic scripts

`scripts/06b`–`06g1` are the experiments behind the autoencoder negative results
in §5.4 — they are included so the claims are reproducible, not just asserted:

| script | question | result |
|---|---|---|
| `06b_ae_sweep.py` | Is the AE better than doing nothing? | No — input 0.36 vs AE 0.14 |
| `06c1_ae_objective.py` | Does a direction-aware loss fix it? | Yes — 0.14 → 0.30 |
| `06d1_ae_tune.py` | Do capacity / depth / epochs help? | No — all ≤ 0.30 |
| `06f1_ae_blocks.py` | Do domain / soft-skill blocks help? | No — 0.30 → 0.14 |
| `06f2_ae_inputs.py` | Is the AE or its input the limit? | The input — text reaches 0.38 |
| `06g1_ae_weights.py` | Can loss re-weighting recover it? | Only to 0.28 |
| `99_check_refs.py` | Do the report's script paths exist? | Guards against rename drift |
| `97_verify_submission.py` | Are the holdout artefacts in the HTML, PDF and zip? | 12-point submission self-check |

`13_holdout_eval.py` is the 80:20 out-of-sample check described in §7.4 of the
report — it retrains the AE per fold and tunes the fusion weights on 8 JDs only.

## Where the honest caveats are

- **§2.3** — inter-annotator agreement: quadratic κ = 0.69 but binary κ = 0.47, so
  the relevant/not boundary is uncertain by about one grade. Differences below
  ~0.05 Recall@10 should be read as noise.
- **§6.4** — the family filter's 0.892 is *circular* with the labelling rule. The
  honest deployable figure uses an independent LLM family classifier (93% accurate)
  and reaches 0.875.
- **§7.4** — the 80:20 holdout. The pipeline scores 0.858 out-of-sample vs 0.887
  in-sample (gap −0.030), so the headline is not badly inflated. But the holdout
  also shows the fusion weights are unstable (3 choices across 5 folds) and that
  the reranker's recall gain is within noise.
- **§8.3** — three plausible hypotheses that were tested and rejected (bigger
  latent, more training, more input blocks).

## Note on the local-LLM requirement

All extraction, adjudication, classification and reranking ran against `qwen3:8b`
served by ollama on localhost, and all embeddings against `nomic-embed-text`. No
cloud API is called by the pipeline and no document leaves the machine. The
coding assistant used to write the code is a hosted model — that is the
development tool, not part of the submitted pipeline.
