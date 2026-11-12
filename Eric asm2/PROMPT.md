# Build the AI CV Matching Baseline — Complete Agent Prompt

This file is the copy of the prompt provided for this assignment. The full prompt specified:
- Mission: build a complete, reproducible baseline AI CV matching system for Second Talent, package under `group assignment/`.
- Data: 100 pool CVs, 10 GT JD+HiredCV PDFs (P01-P10), 50 unlabelled JDs in xlsx, project plan PDF. Read-only.
- Pipeline: Stage 0 validate/split/dedupe, Stage 1 local LLM extraction (Ollama qwen3.5:4b), Stage 2 keyword merging/clustering, Stage 3 autoencoder, Stage 4 retrieval top-10, Stage 5 recall evaluation, Stage 6 error analysis, Stage 7 Streamlit website + figures.
- Rules: no external APIs, no label leakage, no fabricated results, no test-set tuning for primary model, cache/resume, seed 42, Windows/PowerShell commands in prompt adapted to macOS environment here.
- Deliverables: scripts, src, prompts, data artifacts, models, results figures/tables, README, VALIDATION, app with 5 pages.

See the conversation/request that initiated this run for the complete detailed specification.
