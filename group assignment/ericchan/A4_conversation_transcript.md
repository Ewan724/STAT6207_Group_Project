# STAT6207 Assignment 4 — Conversation Transcript

A condensed record of how this project was built: what was asked, what came back,
and — more usefully — which answers turned out to be wrong and how that was
caught. The full literal LLM prompt/response log is in
`A4_prompt_transcript.md`; this file is the human conversation that produced the
code.

---

## Turn 1 — Scoping

**Asked.** Here is the assignment brief: unzip the dataset, use a local LLM
(ollama, `qwen3:8b`) to extract keywords from CVs and JDs, merge/cluster similar
keywords, build an autoencoder to encode CV and JD, retrieve the top 10 CVs for
each JD in the ground truth, measure recall, and do an error analysis. Deliver an
HTML report, a PDF, the prompt transcript and the code.

**Returned.** An inventory of the two archives: 100 CV PDFs, and 10 PDFs each
containing a JD followed by "CV of the hired candidate". Text extraction with
PyMuPDF; the hired-CV section splits cleanly on that marker.

**Decision taken.** Build the pipeline as numbered scripts (`01_` … `12_`) so
every stage is independently re-runnable and resumable, and log every LLM call to
JSONL so nothing is unverifiable.

---

## Turn 2 — The question that decided the project

**Asked.** Check whether each of the 10 hired CVs actually appears in the 100-CV
pool. Compare exact text, 8-gram shingles and employer strings.

**Returned.** Zero overlap for all ten. No shared 8-gram, no shared employer
string, no shared candidate id.

**Consequence.** This is the single most important finding in the assignment.
"The hired CV is the ground truth" — the interpretation the brief most naturally
suggests — would have made every system score recall 0. The pool is a *disjoint*
candidate set mirroring the same ten job families, so relevance labels had to be
constructed explicitly. Everything in §2 of the report follows from this.

---

## Turn 3 — Keyword extraction

**Asked.** One schema-constrained call per document against `qwen3:8b`, with
different schemas for CV and JD, and deterministic post-processing.

**Returned.** 110/110 documents extracted with zero parse failures, 3.57 s mean
latency on CPU. Mean 26.7 hard skills per CV, 16.0 must-haves per JD.

**Kept.** `format: "json"` (grammar-constrained decoding) was the decision that
removed parse failures entirely; it is worth more than any prompt wording.

---

## Turn 4 — Keyword merging, four iterations

**Asked.** Merge synonyms, spellings and noun-phrase variants.

**Iteration 1 returned.** `postgres → postgresql` failed, and `kubernetes`
became `kubernete`. **Diagnosis:** the plural-singularisation rule was applied
blindly, and the embedding threshold was set to a value that rejected the correct
merge.

**Iteration 2 returned.** Correct merges worked, but `data analysis ← data
management` was accepted, because the embedding space thinks they are close (0.79).
**Diagnosis:** the embedding space is too coarse to drive merging. Measured
evidence: `MySQL↔SQL` scores 0.71 and `PostgreSQL↔MySQL` 0.70, both *higher* than
the correct pair `Machine Learning↔ML` at 0.63. A threshold that accepts the good
merges also accepts the bad ones.

**Iteration 3 returned.** A head-noun guard was added: two skill names with
different head nouns are different skills unless one is an abbreviation of the
other. This rejected the category collapses.

**Iteration 4 (final).** Added noun-phrase suffix stripping ("sql proficiency" →
"sql") and enumeration splitting ("tools e.g. figma adobe xd" → `figma` +
`adobe xd`). Also decided *not* to prune rare skills, since a rare shared skill is
the most informative signal available.

**What this taught.** The merge stage needed three *different* kinds of guard
(lexical, LLM-with-negative-rules, structural) because each failure mode was
invisible to the others.

---

## Turn 5 — The autoencoder, and the result that changed the report

**Asked.** Train an autoencoder over the merged keyword space; report
reconstruction quality.

**Returned.** Reconstruction F1 = 0.997. Latent 64. 262k parameters.

**Asked (follow-up).** The reconstruction is near-perfect, but check whether the
latent space is *useful*: for each JD, are the nearest documents in the same job
family?

**Returned.** No. The nearest neighbour of the Data Engineer JD was a Graphic
Designer. Latent purity@5 = 0.14, against 0.10 chance.

**Consequence.** Reconstruction quality is not evidence of embedding quality, and
this became the organising idea of §5.4 of the report. It also changed the
selection criterion for every subsequent experiment: from reconstruction loss to
label-free latent purity.

**Asked.** Measure purity for the input space itself and for several AE
configurations.

**Returned.** The raw IDF-weighted input space scores **0.36** — better than every
autoencoder. The AE was actively destroying structure.

**Asked.** Why?

**Diagnosed.** MSE optimises *magnitude*; retrieval uses *cosine*. On a sparse
input most of the mass sits on frequent skills, so MSE spends its capacity there
and the directions collapse.

**Asked.** Replace the reconstruction loss with `1 − cos(x, x̂)`.

**Returned.** Purity 0.14 → 0.30. This is the single most valuable fix in the
model.

**Asked.** Sweep capacity, depth, dropout, epochs, and try adding domain,
soft-skill and text blocks.

**Returned.** All negative. Latent 64 < 32; 3,000 epochs < 1,800; domain blocks
0.30 → 0.14; text block 0.30 → 0.22; re-weighting the losses 6:1 only recovered
0.28. These negative results are reported in §8.3 rather than hidden, because they
are what makes the positive findings credible.

**Asked.** Train the same architecture on the text embedding instead of the
keywords, to separate "the AE is bad" from "the AE's input is bad".

**Returned.** Purity 0.38, and downstream Recall@10 0.76 vs 0.32. **The
autoencoder was never the bottleneck; its input was.** Reported as Finding 2 in
§5.6.

---

## Turn 6 — Ground truth, four iterations

This is the part where a careless pipeline produces numbers that look fine and
mean nothing, so the iterations are recorded in full.

**v1 asked.** Score the pair 0–3 on fit.

**v1 returned.** Grade 2 with reasons like *"Strong technical skills … but lacks AI
focus and customer collaboration"*. The label contradicted its own justification,
and 45 of 100 CVs came out "relevant" for P06.

**v2 asked.** Add a self-consistency rule: if the reason says a core requirement
is missing, cap the grade at 1.

**v2 returned.** Over-corrected. "Core requirements" was the first six must-haves,
which for P01 were *manufacturing costing / management reporting / internal
controls*. Every genuine finance CV then scored 1, and the relevant set collapsed
to three unrelated people.

**v3 asked.** Add an LLM profession gate, then compare against the hired CV.

**v3 returned.** The comparison stage worked well, but the gate did not: it
classified a Principal Product Manager as "engineering", and left two JDs with an
*empty* relevant set, which makes Recall undefined.

**v4 asked.** Move the gate out of the model and into deterministic code; keep the
LLM only for grading within the gate.

**v4 returned.** 884 pairs gated out with no LLM call, 116 graded, 62 relevant,
every JD non-empty. This is the submitted version.

**What this taught.** Give each decision to the tool that can make it. A
categorical decision is better made by a rule; a graded decision is better made by
a model, and is far more reliable when it is *comparative against a concrete
example* than when it is an abstract rubric.

**Asked.** Measure whether the labels reproduce under a different protocol.

**Returned.** Quadratic-weighted κ = 0.69, agreement within ±1 grade = 0.92, but
binary κ = 0.47. Reported honestly in §2.3, including the caveat that system
differences below ~0.05 Recall@10 should be read as noise.

---

## Turn 7 — Retrieval and the circularity check

**Asked.** Compare lexical, keyword, embedding and AE scorers, then fuse.

**Returned.** TF-IDF on raw text (0.72) beat every keyword-based system (0.51).
The merge itself is worth more than the autoencoder: `keyword` 0.51 vs `ae_latent`
0.32.

**Asked.** Add a job-family filter.

**Returned.** 0.89 Recall@10 — the biggest single gain.

**Asked (self-check).** Is that circular? The ground truth was built with the same
family map.

**Answer.** Yes. **Consequence:** an independent LLM family classifier was trained
(93% agreement with the hand labels), and the deployable system uses *predicted*
families, giving 0.875. The circular 0.892 is still reported but shaded and
labelled as such. This is the kind of number that would flatter the report if it
were not checked.

**Asked.** Add an LLM reranker over the top 20.

**Returned.** Recall 0.887, MRR 1.000, NDCG 0.932. Blending with a small retrieval
prior beat reranking alone (0.874), because the reranker occasionally inverts a
clearly correct order from a truncated CV.

---

## Turn 8 — Deliverables

**Asked.** Generate the HTML report, the PDF, the prompt transcript and the code.

**Returned.** `A4_report.html` (self-contained, figures base64-embedded, 23-page
PDF export), `A4_prompt_transcript.md` (every prompt and response, generated from
the run logs so it cannot drift), `scripts/` (28 scripts, numbered by pipeline
stage), `data/` (all intermediate artefacts).

**Verified.** The report was rendered in a browser and checked visually; every
number in it is injected from `data/*.json` at build time, so the prose cannot
disagree with the run.

---

## Summary of what the assistant got wrong

Recorded because the corrections are more instructive than the successes:

| what it produced | why it was wrong | how it was caught |
|---|---|---|
| "recall = did we find the hired CV" as the plan | the hired CVs are not in the pool | direct overlap test |
| AE with reconstruction F1 0.997, presented as a success | the latent space was meaningless | purity@5 against job family |
| `kubernete`, `postgres → postgresql` rejected | blind plural rule, wrong embedding threshold | reading the merge output |
| `data analysis ← data management` accepted | embedding space too coarse | reading the merge output |
| ground truth v1: grade 2 with a contradictory reason | absolute rubrics do not calibrate on an 8B model | reading the judge's reasons |
| ground truth v2: relevant set collapsed | arbitrary "core requirements" list | inspecting per-JD relevant counts |
| ground truth v3: empty relevant set for two JDs | LLM profession gate is unreliable | inspecting per-JD relevant counts |
| family filter at 0.892 presented as the headline | circular with the labelling rule | self-check, then independent classifier |

The pattern: **every error was caught by inspecting outputs, not by a metric.**
The metrics looked plausible in all eight cases.
