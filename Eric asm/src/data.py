"""Stage 0: data ingestion, cleaning, splitting, alignment and manifests.

DATA/ is treated as read-only source material. Nothing in this module writes
inside DATA/.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pymupdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Config, dump_json

ROLE_FAMILIES = [
    "Finance & Legal",
    "Sales & BD",
    "Marketing",
    "Data & AI",
    "Product",
    "Engineering",
    "Design & Creative",
    "Operations & HR",
    "Leadership/Other",
    "unknown",
]

SPLIT_MARKER = "CV OF THE HIRED CANDIDATE"

HEADER_PATTERNS = [
    r"Anonymised candidate profile\s*[·•]\s*personal information removed\s*[·•]\s*Second Talent",
    r"Anonymised example\s*[·•]\s*client and personal information removed\s*[·•]\s*Second Talent",
    r"CV-\d{3}\s*[·•]\s*page\s*\d+",
    r"P\d{2}\s*[·•]\s*page\s*\d+",
    r"P\d{2}\s*[·•]\s*JOB DESCRIPTION\s*[·•].*",
    r"P\d{2}\s*[·•]\s*CV OF THE HIRED CANDIDATE",
    r"Hired for:\s*.*",
    r"page\s+\d+\s*(of\s+\d+)?",
]

BULLET_CHARS = "●•▪◦‣⁃\u25aa\u25cf\ufffd"


def normalise_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str, *, remove_pxx: bool = True, remove_hired_line: bool = True) -> str:
    """Remove repeated artifacts, normalise bullets and collapse blank lines."""
    text = normalise_unicode(text).replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if any(re.fullmatch(p, line, flags=re.IGNORECASE) for p in HEADER_PATTERNS):
            continue
        if remove_hired_line and re.match(r"^Hired for\s*:", line, flags=re.IGNORECASE):
            continue
        line = line.translate({ord(c): "-" for c in BULLET_CHARS})
        if line[:1] in "-–—":
            line = "- " + line[1:].lstrip()
        lines.append(line)
    text = "\n".join(lines)
    if remove_pxx:
        text = re.sub(r"\bP\d{2}\b", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_hired_pdf(text: str) -> tuple[str, str] | None:
    idx = text.find(SPLIT_MARKER)
    if idx < 0:
        return None
    jd = text[:idx]
    cv = text[idx + len(SPLIT_MARKER):]
    return clean_text(jd), clean_text(cv)


def pdf_pages(path: Path) -> list[str]:
    with pymupdf.open(path) as doc:
        return [page.get_text("text") for page in doc]


def pdf_text(path: Path) -> tuple[str, int]:
    pages = pdf_pages(path)
    return "\n".join(pages), len(pages)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def find_excel(data_root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None and explicit.exists():
        return explicit
    candidates = sorted(data_root.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"No Excel workbook found in {data_root}")
    candidates.sort(key=lambda p: ("Manatal" not in p.name, p.name))
    return candidates[0]


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #
def _nonempty_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def parse_gt_jd_header(text: str) -> dict[str, Any]:
    lines = _nonempty_lines(text)
    role_family = None
    title = None
    for i, line in enumerate(lines):
        m = re.match(r"P\d{2}\s*[·•]\s*JOB DESCRIPTION\s*[·•]\s*(.+)$", line)
        if m:
            role_family = m.group(1).strip()
            if i + 1 < len(lines):
                title = lines[i + 1]
            break
    if title is None and lines:
        title = lines[0]
    # The metadata line is usually the line after the title.
    meta = ""
    if title in lines:
        j = lines.index(title)
        if j + 1 < len(lines):
            meta = lines[j + 1]
    location = None
    salary = None
    remote = "unknown"
    parts = [p.strip() for p in meta.split("·") if p.strip()]
    for part in parts:
        low = part.lower()
        if "remote" in low:
            remote = "yes"
        elif "hybrid" in low:
            remote = "hybrid"
        elif re.search(r"(myr|idr|sgd|usd|hkd|thb|php|vnd|rm|salary|/\s*(month|year|annum))", low):
            salary = part
        elif low not in {"full time", "part time", "contract", "permanent"} and not re.fullmatch(r"[\d\s,.\-–]+", part):
            location = part
    if remote == "unknown":
        if re.search(r"\bremote\b", text[:1200], flags=re.IGNORECASE):
            remote = "yes"
        elif re.search(r"\bhybrid\b", text[:1200], flags=re.IGNORECASE):
            remote = "hybrid"
    country = None
    if location:
        country = location.split(",")[-1].strip()
    return {
        "title": title,
        "role_family": role_family,
        "location": location,
        "country": country,
        "salary": salary,
        "remote": remote,
        "meta": meta,
    }


def parse_cv_header(text: str, role_family: str | None = None) -> dict[str, Any]:
    """Parse the title / company lines that head every anonymised CV."""
    lines = _nonempty_lines(text)
    lines = [ln for ln in lines if not re.fullmatch(r"CV-\d{3}", ln)]
    title = lines[0] if lines else None
    company = None
    country = None
    for line in lines[1:5]:
        if "·" in line:
            left, right = [p.strip() for p in line.split("·", 1)]
            company = left
            if right and (role_family is None or right != role_family):
                country = right
            break
    return {"title": title, "company": company, "country": country}


def parse_hired_cv_header(text: str) -> dict[str, Any]:
    return parse_cv_header(text)


# --------------------------------------------------------------------------- #
# Stage 0 build
# --------------------------------------------------------------------------- #
@dataclass
class Stage0Result:
    documents: list[dict[str, Any]]
    ground_truth: pd.DataFrame
    summary: dict[str, Any]


def _value(row: pd.Series, col: str | None):
    if col is None:
        return None
    v = row[col]
    return None if pd.isna(v) else v


def build_documents(cfg: Config, data_root: Path) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    excel = find_excel(data_root, cfg.excel_path)
    jd_sheet = pd.read_excel(excel, sheet_name="JD Examples")
    cv_sheet = pd.read_excel(excel, sheet_name="CV Examples")
    jd_cols = {str(c).strip().lower(): c for c in jd_sheet.columns}
    cv_cols = {str(c).strip().lower(): c for c in cv_sheet.columns}

    documents: list[dict[str, Any]] = []
    split_failures: list[dict[str, Any]] = []

    # ------------------------------------------------------------ pool CVs
    cv_excel: dict[str, pd.Series] = {}
    for _, row in cv_sheet.iterrows():
        ref = str(_value(row, cv_cols.get("ref")) or "").strip()
        if ref:
            cv_excel[ref] = row

    for path in sorted(cfg.pool_cv_dir.glob("CV-*.pdf")):
        doc_id = path.stem
        raw, pages = pdf_text(path)
        text = clean_text(raw)
        header = parse_cv_header(text)
        row = cv_excel.get(doc_id)
        excel_title = str(_value(row, cv_cols.get("current title")) or "") if row is not None else ""
        excel_company = str(_value(row, cv_cols.get("current company")) or "") if row is not None else ""
        excel_family = str(_value(row, cv_cols.get("role family")) or "") if row is not None else ""
        country = _value(row, cv_cols.get("country")) if row is not None else None
        documents.append(dict(
            doc_id=doc_id,
            doc_type="pool_cv",
            source_path=str(path.relative_to(cfg.root)),
            title=excel_title or header["title"],
            company=excel_company or header["company"],
            role_family=excel_family or "unknown",
            country=country,
            location=country,
            salary=None,
            remote="unknown",
            raw_text=text,
            pages=pages,
            char_count=len(text),
            split="train",
            notes="",
            pdf_title=header["title"],
            pdf_company=header["company"],
            excel_skills=str(_value(row, cv_cols.get("skills")) or "") if row is not None else "",
        ))

    # ------------------------------------------------------------ labelled P PDFs
    gt_rows: list[dict[str, Any]] = []
    for i, path in enumerate(sorted(cfg.gt_pdf_dir.glob("P*.pdf")), start=1):
        raw, pages = pdf_text(path)
        split = split_hired_pdf(raw)
        if split is None:
            split_failures.append({"source": path.name, "reason": "marker_not_found"})
            continue
        jd_text, cv_text = split
        m = re.match(r"(P\d{2})", path.stem)
        jd_id = m.group(1) if m else f"P{i:02d}"
        cand_id = f"CAND-{100 + i:03d}"
        if len(jd_text) < 500 or len(cv_text) < 400:
            split_failures.append({
                "source": path.name, "reason": "short_split",
                "jd_chars": len(jd_text), "cv_chars": len(cv_text),
            })

        jd_header = parse_gt_jd_header(jd_text)
        raw_family = re.search(r"P\d{2}\s*[·•]\s*JOB DESCRIPTION\s*[·•]\s*(.+)$", raw, flags=re.MULTILINE)
        if raw_family:
            jd_header["role_family"] = raw_family.group(1).strip()
        cv_header = parse_hired_cv_header(cv_text)
        role_family = jd_header["role_family"]
        documents.append(dict(
            doc_id=jd_id,
            doc_type="gt_jd",
            source_path=str(path.relative_to(cfg.root)),
            title=jd_header["title"],
            company=None,
            role_family=role_family,
            country=jd_header["country"],
            location=jd_header["location"],
            salary=jd_header["salary"],
            remote=jd_header["remote"],
            raw_text=jd_text,
            pages=pages,
            char_count=len(jd_text),
            split="test",
            notes="JD portion of labelled PDF; Pxx identifiers stripped.",
        ))
        documents.append(dict(
            doc_id=cand_id,
            doc_type="gt_cv",
            source_path=str(path.relative_to(cfg.root)),
            title=cv_header["title"],
            company=cv_header["company"],
            # Never copy the JD's role family onto the hired CV: that would leak
            # a label-derived attribute into the retrieval features. The LLM
            # infers a family from the CV text only (see src/retrieval.py).
            role_family="unknown",
            country=cv_header["country"],
            location=cv_header["country"],
            salary=None,
            remote="unknown",
            raw_text=cv_text,
            pages=pages,
            char_count=len(cv_text),
            split="test",
            notes=f"hired CV for {jd_id}; label stored only in ground_truth.csv.",
        ))
        gt_rows.append(dict(
            jd_id=jd_id,
            jd_title=jd_header["title"],
            role_family=role_family,
            jd_location=jd_header["location"],
            jd_salary=jd_header["salary"],
            hired_cv_id=cand_id,
            hired_cv_title=cv_header["title"],
            hired_cv_company=cv_header["company"],
            source_pdf=path.name,
            notes="one relevant CV per JD; all other candidates are non-relevant",
        ))

    # ------------------------------------------------------------ unlabelled JDs
    for i, (_, row) in enumerate(jd_sheet.iterrows(), start=1):
        def v(name: str):
            return _value(row, jd_cols.get(name.lower()))

        desc = str(v("Job description") or "").strip()
        salary_min, salary_max = v("Salary min"), v("Salary max")
        currency, freq = v("Currency"), v("Salary frequency")
        if salary_min is not None or salary_max is not None:
            salary = f"{currency or ''} {salary_min or ''}-{salary_max or ''} / {freq or ''}".strip()
        else:
            salary = None
        header = "\n".join([
            f"JOB TITLE: {v('Position') or ''}",
            f"ROLE FAMILY: {v('Role family') or ''}",
            f"COUNTRY: {v('Country') or ''}",
            f"CITY: {v('City') or ''}",
            f"REMOTE: {v('Remote') or ''}",
            f"CONTRACT TYPE: {v('Contract type') or ''}",
            f"SALARY: {salary or ''}",
            "",
        ])
        text = clean_text(header + desc, remove_pxx=False)
        remote_raw = str(v("Remote") or "").lower()
        remote = "yes" if remote_raw in {"yes", "true", "remote"} else ("no" if remote_raw in {"no", "false"} else "unknown")
        documents.append(dict(
            doc_id=f"JD-{i:03d}",
            doc_type="unlabelled_jd",
            source_path=str(excel.relative_to(cfg.root)) + "#JD Examples",
            title=str(v("Position") or "") or None,
            company=None,
            role_family=str(v("Role family") or "") or "unknown",
            country=v("Country"),
            location=", ".join(str(x) for x in [v("City"), v("Country")] if x) or None,
            salary=salary,
            remote=remote,
            raw_text=text,
            pages=1,
            char_count=len(text),
            split="train",
            notes="unlabelled; never used for recall.",
        ))

    # ------------------------------------------------------------ alignment
    align_rows = []
    for doc in documents:
        if doc["doc_type"] != "pool_cv":
            continue
        row = cv_excel.get(doc["doc_id"])
        if row is None:
            align_rows.append({
                "ref": doc["doc_id"], "found_in_excel": False,
                "title_match": False, "company_match": False,
                "excel_title": None, "pdf_title": doc.get("pdf_title"),
            })
            continue
        ex_title = str(_value(row, cv_cols.get("current title")) or "")
        ex_company = str(_value(row, cv_cols.get("current company")) or "")
        pdf_title = doc.get("pdf_title") or ""
        pdf_company = doc.get("pdf_company") or ""
        overlap = _token_overlap(pdf_title, ex_title)
        align_rows.append({
            "ref": doc["doc_id"], "found_in_excel": True,
            "excel_title": ex_title, "pdf_title": pdf_title,
            "excel_company": ex_company, "pdf_company": pdf_company,
            "title_match": bool(overlap >= 0.5 or _norm(ex_title) in _norm(pdf_title)),
            "company_match": bool(_norm(ex_company)[:12] in _norm(pdf_company)),
            "title_overlap": round(overlap, 3),
            "excel_role_family": str(_value(row, cv_cols.get("role family")) or ""),
            "excel_skills": str(_value(row, cv_cols.get("skills")) or ""),
        })
    alignment = pd.DataFrame(align_rows)

    dedupe = _dedupe_hired(documents)
    jd_dedupe = _dedupe_jds(documents)

    gt = pd.DataFrame(gt_rows)
    gt.to_csv(cfg.processed / "ground_truth.csv", index=False)
    alignment.to_csv(cfg.processed / "cv_alignment_report.csv", index=False)
    dedupe.to_csv(cfg.processed / "dedupe_report.csv", index=False)
    jd_dedupe.to_csv(cfg.processed / "jd_dedupe_report.csv", index=False)
    write_jsonl(cfg.processed / "documents.jsonl", documents)
    pd.DataFrame(split_failures).to_csv(cfg.processed / "split_failures.csv", index=False)
    return documents, gt, alignment, dedupe, jd_dedupe


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _dedupe_hired(documents: list[dict[str, Any]]) -> pd.DataFrame:
    pool = [d for d in documents if d["doc_type"] == "pool_cv"]
    hired = [d for d in documents if d["doc_type"] == "gt_cv"]
    if not pool or not hired:
        return pd.DataFrame()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vec.fit_transform([d["raw_text"] for d in pool + hired])
    sims = cosine_similarity(matrix[len(pool):], matrix[:len(pool)])
    rows = []
    for i, h in enumerate(hired):
        order = np.argsort(-sims[i])[:3]
        top = [{
            "cv_id": pool[j]["doc_id"],
            "cosine": round(float(sims[i, j]), 4),
            "title": pool[j].get("title"),
            "company": pool[j].get("company"),
        } for j in order]
        best = top[0]
        duplicate = (
            best["cosine"] >= 0.95
            and _token_overlap(h.get("title") or "", best["title"] or "") >= 0.5
            and _norm(h.get("company") or "")[:12] in _norm(best["company"] or "")
        )
        rows.append({
            "hired_cv_id": h["doc_id"],
            "hired_title": h.get("title"),
            "hired_company": h.get("company"),
            "top1_pool_cv": best["cv_id"], "top1_cosine": best["cosine"],
            "top1_title": best["title"], "top1_company": best["company"],
            "top2_pool_cv": top[1]["cv_id"], "top2_cosine": top[1]["cosine"],
            "top3_pool_cv": top[2]["cv_id"], "top3_cosine": top[2]["cosine"],
            "decision": "map_to_existing" if duplicate else "keep_new",
            "notes": "No exact duplicate" if not duplicate else "True duplicate found",
        })
    return pd.DataFrame(rows)


def _dedupe_jds(documents: list[dict[str, Any]]) -> pd.DataFrame:
    gt = [d for d in documents if d["doc_type"] == "gt_jd"]
    unlab = [d for d in documents if d["doc_type"] == "unlabelled_jd"]
    if not gt or not unlab:
        return pd.DataFrame()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vec.fit_transform([d["raw_text"] for d in gt + unlab])
    sims = cosine_similarity(matrix[len(gt):], matrix[:len(gt)])
    rows = []
    for i, d in enumerate(unlab):
        j = int(np.argmax(sims[i]))
        max_sim = float(sims[i, j])
        same_title = _norm(d.get("title") or "") == _norm(gt[j].get("title") or "")
        dup = max_sim >= 0.90 or same_title
        rows.append({
            "jd_id": d["doc_id"], "title": d.get("title"),
            "nearest_gt_jd": gt[j]["doc_id"], "nearest_gt_title": gt[j].get("title"),
            "cosine": round(max_sim, 4),
            "decision": "exclude_from_training" if dup else "keep",
        })
    return pd.DataFrame(rows)


def build_manifest(cfg: Config, data_root: Path, documents: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    doc_map: dict[str, str] = {}
    for d in documents:
        src = d["source_path"].split("#")[0]
        doc_map.setdefault(src, d["doc_id"])
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".xlsx"}:
            continue
        rel = str(path.relative_to(cfg.root))
        pages = None
        if path.suffix.lower() == ".pdf":
            try:
                pages = len(pdf_pages(path))
            except Exception:
                pages = None
        rows.append({
            "path": rel, "file_size": path.stat().st_size, "sha256": sha256(path),
            "pages": pages, "doc_id": doc_map.get(rel),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(cfg.processed.parent / "raw_manifest.csv", index=False)
    return manifest


def leakage_checks(documents: list[dict[str, Any]]) -> dict[str, Any]:
    gt_cv_text = "\n".join(d["raw_text"] for d in documents if d["doc_type"] == "gt_cv")
    all_cv_text = "\n".join(d["raw_text"] for d in documents if d["doc_type"] in {"gt_cv", "pool_cv"})
    return {
        "hired_for_in_cv_text": int(bool(re.search(r"Hired for\s*:", all_cv_text, re.IGNORECASE))),
        "pxx_in_cv_text": int(bool(re.search(r"\bP\d{2}\b", all_cv_text))),
        "pxx_in_gt_cv_text": int(bool(re.search(r"\bP\d{2}\b", gt_cv_text))),
    }


def run_stage0(cfg: Config, force: bool = False) -> Stage0Result:
    cfg.ensure_dirs()
    data_root = cfg.data_root
    required = [cfg.pool_cv_dir, cfg.gt_pdf_dir]
    if not all(p.exists() for p in required):
        raise FileNotFoundError(
            f"Data corpus is incomplete: looked for {required}. "
            "Set paths.pool_cv_dir / paths.gt_pdf_dir in config.yaml.")
    cfg.write_run_mode("real_data", data_root, "Read-only corpus: CV PDFs, labelled JD+hired-CV PDFs and the Manatal workbook.")

    docs_path = cfg.processed / "documents.jsonl"
    gt_path = cfg.processed / "ground_truth.csv"
    if docs_path.exists() and gt_path.exists() and not force:
        documents = read_jsonl(docs_path)
        gt = pd.read_csv(gt_path)
        alignment = pd.read_csv(cfg.processed / "cv_alignment_report.csv") if (cfg.processed / "cv_alignment_report.csv").exists() else pd.DataFrame()
        dedupe = pd.read_csv(cfg.processed / "dedupe_report.csv") if (cfg.processed / "dedupe_report.csv").exists() else pd.DataFrame()
        jd_dedupe = pd.read_csv(cfg.processed / "jd_dedupe_report.csv") if (cfg.processed / "jd_dedupe_report.csv").exists() else pd.DataFrame()
    else:
        documents, gt, alignment, dedupe, jd_dedupe = build_documents(cfg, data_root)

    manifest = build_manifest(cfg, data_root, documents)
    counts = pd.Series([d["doc_type"] for d in documents]).value_counts().to_dict()
    n_candidates = counts.get("pool_cv", 0) + counts.get("gt_cv", 0)
    checks = leakage_checks(documents)
    summary = {
        "data_root": str(data_root),
        "run_mode": "real_data",
        "pool_cvs": counts.get("pool_cv", 0),
        "gt_cvs": counts.get("gt_cv", 0),
        "gt_jds": counts.get("gt_jd", 0),
        "unlabelled_jds": counts.get("unlabelled_jd", 0),
        "unique_candidates": n_candidates,
        "ground_truth_rows": int(len(gt)),
        "manifest_files": int(len(manifest)),
        "alignment_rows": int(len(alignment)),
        "alignment_title_matches": int(alignment["title_match"].sum()) if len(alignment) and "title_match" in alignment else 0,
        "alignment_company_matches": int(alignment["company_match"].sum()) if len(alignment) and "company_match" in alignment else 0,
        "dedupe_keep_new": int((dedupe["decision"] == "keep_new").sum()) if len(dedupe) else 0,
        "jd_dedupe_excluded": int((jd_dedupe["decision"] == "exclude_from_training").sum()) if len(jd_dedupe) else 0,
        "leakage_checks": checks,
    }
    dump_json(cfg.processed / "stage0_summary.json", summary)
    return Stage0Result(documents=documents, ground_truth=gt, summary=summary)
