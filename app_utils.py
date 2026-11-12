"""Shared helpers for the offline Streamlit baseline website."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

GROUP_DIR = Path(__file__).resolve().parent
ROOT = GROUP_DIR.parent
if str(GROUP_DIR) not in sys.path:
    sys.path.insert(0, str(GROUP_DIR))

from src.config import load_config  # noqa: E402

CFG = load_config(GROUP_DIR / "config.yaml")

NAVY = "#1F2A44"
TEAL = "#009E73"
CORAL = "#D55E00"
GREY = "#6B7280"
LIGHT = "#F7F8FA"

CSS = f"""
<style>
  .block-container {{padding-top: 1.6rem; padding-bottom: 2rem;}}
  h1, h2, h3 {{color: {NAVY};}}
  .baseline-hero {{background: linear-gradient(90deg, #F2F6FB 0%, #EAF6F2 100%);
     border: 1px solid #DCE3EC; border-radius: 12px; padding: 18px 22px; margin-bottom: 14px;}}
  .baseline-hero h1 {{margin: 0 0 6px 0; font-size: 1.75rem;}}
  .baseline-hero p {{margin: 0; color: #374151;}}
  .metric-card {{background: white; border: 1px solid #E3E7ED; border-left: 5px solid {NAVY};
     border-radius: 10px; padding: 12px 14px; min-height: 92px;}}
  .metric-card .label {{color: {GREY}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em;}}
  .metric-card .value {{color: {NAVY}; font-size: 1.65rem; font-weight: 700; line-height: 1.2;}}
  .metric-card .note {{color: {GREY}; font-size: 0.78rem;}}
  .badge {{display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.76rem;
     font-weight: 700; color: white;}}
  .badge-hit {{background: {TEAL};}}
  .badge-miss {{background: {CORAL};}}
  .badge-neutral {{background: {GREY};}}
  .note-box {{background: #FFF8E8; border-left: 5px solid #E6A700; border-radius: 8px;
     padding: 12px 14px; color: #4B3A00; font-size: 0.9rem;}}
  .privacy {{color: {GREY}; font-size: 0.78rem; border-top: 1px solid #E5E7EB; padding-top: 10px;}}
  div[data-testid="stDataFrame"] {{font-size: 0.86rem;}}
  .stTabs [data-baseweb="tab-list"] {{gap: 8px;}}
</style>
"""


def setup_page() -> None:
    st.set_page_config(page_title="AI CV Matching — Baseline", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def prompt_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((GROUP_DIR / "prompts").glob("*.txt")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


@st.cache_data(show_spinner=False)
def read_csv(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def read_json(path_str: str) -> dict:
    path = Path(path_str)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def read_jsonl(path_str: str) -> list[dict]:
    path = Path(path_str)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@st.cache_data(show_spinner=False)
def read_text(path_str: str) -> str:
    path = Path(path_str)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def p(name: str) -> Path:
    return CFG.processed / name


def t(name: str) -> Path:
    return CFG.tables / name


def f(name: str) -> Path:
    return CFG.figures / name


def m(name: str) -> Path:
    return CFG.models / name


def require(paths: list[Path], stage: str, stage_num: str) -> None:
    missing = [str(x.relative_to(ROOT)) for x in paths if not x.exists()]
    if missing:
        st.error(f"Required file(s) missing: {', '.join(missing)}. Run `{stage}` "
                 f"(Stage {stage_num}) from the project root, then reload this page.")
        st.stop()


def metric_card(col, label: str, value: str, note: str = "") -> None:
    col.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="note">{note}</div></div>',
        unsafe_allow_html=True)


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def image(name: str, caption: str) -> None:
    path = f(name)
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure {name} is missing. Run scripts/06_make_figures.py to regenerate it.")


def sidebar() -> None:
    stage0 = read_json(str(p("stage0_summary.json")))
    ext = read_json(str(p("extraction_summary.json")))
    ae = read_json(str(m("autoencoder_config.json")))
    metrics = read_csv(str(p("metrics_summary.csv")))
    with st.sidebar:
        st.markdown(f"### AI CV Matching\n**Baseline results**")
        st.caption("Second Talent · STAT6207 group project")
        st.markdown("**Data summary**")
        st.markdown(
            f"- Pool CVs: **{stage0.get('pool_cvs', 100)}**\n"
            f"- Hired CVs: **{stage0.get('gt_cvs', 10)}**\n"
            f"- Labelled JDs: **{stage0.get('gt_jds', 10)}**\n"
            f"- Unlabelled JDs: **{stage0.get('unlabelled_jds', 50)}**\n"
            f"- Unique candidates: **{stage0.get('unique_candidates', 110)}**")
        st.markdown("**Model / pipeline**")
        st.markdown(
            f"- Ollama model: `{ext.get('model', 'not extracted yet')}`\n"
            f"- Extraction success: **{ext.get('success_rate', 0):.1%}**\n"
            f"- AE: latent **{ae.get('latent_dim', 32)}**, dropout **{ae.get('dropout', '—')}**\n"
            f"- Seed: **{CFG.seed}**")
        with st.expander("Metric definitions"):
            st.markdown(
                "**Recall@10** = share of the 10 labelled JDs whose single hired CV appears in the "
                "top 10 (hits/10). With one relevant CV per query this is the same as Hit@10.\n\n"
                "**Hit@1 / Hit@5** = share of queries where the hired CV is ranked 1 / within 5.\n\n"
                "**MRR@10** = mean of 1/rank for hits inside the top 10 (0 for misses).\n\n"
                "**nDCG@10** = mean of 1/log2(rank+1) for hits inside the top 10.\n\n"
                "**Precision@10** = hits/10 per query. Because each query has exactly one relevant "
                "CV, its maximum is **0.1**; it is not comparable to Recall@10.\n\n"
                "**Random baseline**: expected Recall@10 = 10/110 ≈ 0.091, also estimated with "
                "1,000 seeded permutations.")
        with st.expander("Privacy & reproducibility"):
            st.markdown(
                "All documents are anonymised. Extraction runs **only** through a local Ollama "
                "model; no external APIs, scraping, or uploads are used.\n\n"
                f"- Config hash: `{sha256_file(GROUP_DIR / 'config.yaml')}`\n"
                f"- Prompt hash: `{prompt_hash()}`\n"
                f"- Run date: `{datetime.fromtimestamp(p('rankings.csv').stat().st_mtime).isoformat(timespec='seconds') if p('rankings.csv').exists() else 'not run'}`")
            if metrics is not None and len(metrics):
                best = metrics.sort_values("recall_at_10", ascending=False).iloc[0]
                st.caption(f"Best model: {best['model']} · Recall@10 {best['recall_at_10']:.2f}")
        st.markdown('<div class="privacy">Read-only DATA/. Local LLM only. No external calls.</div>',
                    unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="baseline-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        '<div class="privacy">Data: Second Talent anonymised examples (read-only). '
        'Local Ollama extraction only; no external API, scraping, or upload. '
        'Every number is reproducible from the saved CSV/JSON artifacts.</div>',
        unsafe_allow_html=True)
