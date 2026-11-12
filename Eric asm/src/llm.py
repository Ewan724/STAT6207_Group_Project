"""Local-LLM extraction via Ollama, with caching, validation and fallbacks."""
from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .aliases import ALIAS_TO_CANONICAL, CURATED_ALIASES, canonical_for
from .config import Config, dump_json
from .data import read_jsonl, write_jsonl

GENERIC_KEYWORDS = {
    "experience", "skills", "skill", "work", "team", "ability", "strong", "good",
    "excellent", "communication", "leadership", "hard-working", "hard working",
    "problem solving", "problem-solving", "detail oriented", "detail-oriented",
    "self motivated", "self-motivated", "proactive", "responsible", "various",
    "other", "etc", "and", "the", "with", "for", "from", "using", "including",
}

SENIORITY = {"intern", "junior", "mid", "senior", "lead", "director", "executive", "unknown"}
REMOTE = {"yes", "no", "hybrid", "unknown"}

ALIAS_VARIANTS: dict[str, set[str]] = {}
for _canon, _variants in CURATED_ALIASES.items():
    ALIAS_VARIANTS.setdefault(_canon, set()).update(_variants + [_canon])


# --------------------------------------------------------------------------- #
# Privacy masking
# --------------------------------------------------------------------------- #
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+|linkedin\.com/\S+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")


def mask_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = URL_RE.sub("[URL_REDACTED]", text)
    text = PHONE_RE.sub("[PHONE_REDACTED]", text)
    return text


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _kw_norm(value: str) -> str:
    s = str(value).casefold().strip()
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"['’]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,;:")


def _text_norm(text: str) -> str:
    s = str(text).casefold()
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"['’]", "", s)
    s = re.sub(r"[\s\u00a0]+", " ", s)
    return s


def keyword_supported(keyword: str, text: str) -> bool:
    norm_text = _text_norm(text)
    norm_kw = _kw_norm(keyword)
    if not norm_kw:
        return False
    # Direct occurrence, allowing flexible whitespace/hyphen spelling.
    pattern = re.escape(norm_kw).replace(r"\ ", r"[\s\-]+")
    if re.search(pattern, norm_text):
        return True
    canonical = canonical_for(keyword)
    variants = ALIAS_VARIANTS.get(canonical, {canonical})
    for variant in variants:
        v = _kw_norm(variant)
        vpat = re.escape(v).replace(r"\ ", r"[\s\-]+")
        if v and re.search(vpat, norm_text):
            return True
    return False


def _clean_keywords(values: Any, text: str, errors: list[str], dropped: list[dict]) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = re.split(r"[,;]|\n", values)
    if not isinstance(values, (list, tuple)):
        values = [values]
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if isinstance(raw, dict):
            raw = raw.get("keyword") or raw.get("name") or ""
        kw = re.sub(r"\s+", " ", str(raw)).strip(" .,;:-•●")
        if not kw:
            continue
        words = kw.split()
        if len(words) > 4 or len(kw) < 2:
            errors.append(f"keyword_word_count:{kw}")
            continue
        if kw.casefold() in GENERIC_KEYWORDS:
            errors.append(f"generic_keyword:{kw}")
            continue
        if not keyword_supported(kw, text):
            dropped.append({"keyword": kw, "reason": "not_in_source_text"})
            continue
        key = kw.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(kw)
    return out


def _clean_scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        return value if value and value.lower() not in {"null", "none", "n/a", "unknown"} else default
    return value


def _clean_years(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 0 or v > 50:
        return None
    return int(round(v))


def _clean_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in re.split(r"[,;]|\n", value) if x.strip()]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _clean_str_list(value: Any) -> list[str]:
    out = []
    for item in _clean_list(value):
        if isinstance(item, dict):
            item = item.get("name") or item.get("title") or json.dumps(item, ensure_ascii=False)
        s = str(item).strip()
        if s:
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------------- #
class OllamaClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base_url = str(cfg.get("llm.base_url", "http://127.0.0.1:11434")).rstrip("/")
        self.primary = str(cfg.get("llm.model", "qwen3.5:4b"))
        self.fallback = str(cfg.get("llm.fallback_model", "qwen3.8:27b"))
        self.timeout = int(cfg.get("llm.timeout_sec", 300))
        self.timeout_fallback = int(cfg.get("llm.timeout_fallback_sec", 600))
        self.max_retries = int(cfg.get("llm.max_retries", 2))

    def health(self) -> dict[str, Any]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            models = [m.get("name") for m in r.json().get("models", [])]
            return {"ok": True, "models": models}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "models": []}

    def choose_model(self) -> tuple[str, list[str]]:
        health = self.health()
        models = health.get("models", [])
        if self.primary in models:
            return self.primary, models
        # allow tag-insensitive match
        for m in models:
            if m.split(":")[0] == self.primary.split(":")[0]:
                return m, models
        if bool(self.cfg.get("llm.allow_model_fallback", True)) and self.fallback in models:
            return self.fallback, models
        if models:
            return models[0], models
        raise RuntimeError("Ollama is reachable but no local model is installed. Run: ollama pull qwen3.5:4b")

    def chat(self, system: str, user: str, model: str, timeout: int | None = None) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": float(self.cfg.get("llm.temperature", 0)),
                "seed": int(self.cfg.get("llm.seed", 42)),
                "num_ctx": int(self.cfg.get("llm.num_ctx", 8192)),
                "num_predict": int(self.cfg.get("llm.num_predict", 2000)),
                "top_p": float(self.cfg.get("llm.top_p", 1.0)),
            },
        }
        # Ollama's grammar-constrained JSON mode is unavailable for some models
        # (it returns "structured output is unavailable" for qwen3.5:4b), so it
        # is opt-in. The prompt asks for bare JSON and _parse_json repairs it.
        if bool(self.cfg.get("llm.json_mode", False)):
            payload["format"] = "json"
        # Thinking models otherwise spend the token budget on reasoning and can
        # return an empty `content` field.
        think = self.cfg.get("llm.think", None)
        if think is not None:
            payload["think"] = bool(think)
        start = time.time()
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(f"{self.base_url}/api/chat", json=payload,
                                  timeout=timeout or self.timeout)
                r.raise_for_status()
                data = r.json()
                content = (data.get("message") or {}).get("content") or ""
                if not content.strip():
                    # Some thinking models put everything in `thinking`.
                    content = (data.get("message") or {}).get("thinking") or ""
                return {
                    "content": content,
                    "elapsed_sec": round(time.time() - start, 2),
                    "model": data.get("model", model),
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "eval_count": data.get("eval_count"),
                    "attempt": attempt + 1,
                }
            except Exception as exc:
                last_error = str(exc)
                time.sleep(min(2 ** attempt, 8))
        return {"content": "", "elapsed_sec": round(time.time() - start, 2), "model": model,
                "error": last_error, "attempt": self.max_retries + 1}


def _parse_json(content: str) -> tuple[dict | None, str | None]:
    if not content:
        return None, "empty_response"
    text = content.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text, strict=False)
        return (obj, None) if isinstance(obj, dict) else (None, "json_not_object")
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1], strict=False)
            return (obj, None) if isinstance(obj, dict) else (None, "json_not_object")
        except Exception as exc:
            return None, f"json_parse_error:{exc}"
    return None, "no_json_object"


def _validate_and_clean(parsed: dict, doc: dict, text: str, dropped: list[dict]) -> tuple[dict, list[str]]:
    errors: list[str] = []
    doc_type = "jd" if doc["doc_type"] in {"gt_jd", "unlabelled_jd"} else "cv"
    out: dict[str, Any] = {
        "doc_id": doc["doc_id"],
        "doc_type": doc_type,
    }
    if doc_type == "jd":
        out.update({
            "job_title": _clean_scalar(parsed.get("job_title")) or doc.get("title"),
            "role_family": _clean_scalar(parsed.get("role_family")) or doc.get("role_family"),
            "seniority": _clean_scalar(parsed.get("seniority"), "unknown").lower()
            if str(parsed.get("seniority", "unknown")).lower() in SENIORITY else "unknown",
            "location": _clean_scalar(parsed.get("location")) or doc.get("location"),
            "remote": _clean_scalar(parsed.get("remote"), "unknown").lower()
            if str(parsed.get("remote", "unknown")).lower() in REMOTE else "unknown",
            "min_years_experience": _clean_years(parsed.get("min_years_experience")),
            "must_have_keywords": _clean_keywords(parsed.get("must_have_keywords"), text, errors, dropped),
            "nice_to_have_keywords": _clean_keywords(parsed.get("nice_to_have_keywords"), text, errors, dropped),
            "tools_technologies": _clean_keywords(parsed.get("tools_technologies"), text, errors, dropped),
            "domain_knowledge": _clean_str_list(parsed.get("domain_knowledge")),
            "education_requirements": _clean_str_list(parsed.get("education_requirements")),
            "certifications": _clean_str_list(parsed.get("certifications")),
            "languages": _clean_str_list(parsed.get("languages")),
            "soft_skills": _clean_str_list(parsed.get("soft_skills")),
            "key_responsibilities": _clean_str_list(parsed.get("key_responsibilities")),
        })
        for field in ["must_have_keywords", "nice_to_have_keywords"]:
            if not out[field]:
                errors.append(f"empty_{field}")
    else:
        years = _clean_years(parsed.get("total_years_experience"))
        out.update({
            "current_title": _clean_scalar(parsed.get("current_title")) or doc.get("title"),
            "current_company": _clean_scalar(parsed.get("current_company")) or doc.get("company"),
            "role_family": _clean_scalar(parsed.get("role_family")) or doc.get("role_family"),
            "seniority": _clean_scalar(parsed.get("seniority"), "unknown").lower()
            if str(parsed.get("seniority", "unknown")).lower() in SENIORITY else "unknown",
            "total_years_experience": years,
            "skills": _clean_keywords(parsed.get("skills"), text, errors, dropped),
            "tools_technologies": _clean_keywords(parsed.get("tools_technologies"), text, errors, dropped),
            "domains": _clean_str_list(parsed.get("domains")),
            "education": _clean_list(parsed.get("education")),
            "certifications": _clean_str_list(parsed.get("certifications")),
            "languages": _clean_str_list(parsed.get("languages")),
            "locations": _clean_str_list(parsed.get("locations")),
            "soft_skills": _clean_str_list(parsed.get("soft_skills")),
            "work_history": _clean_list(parsed.get("work_history")),
        })
        if not out["skills"] and not out["tools_technologies"]:
            errors.append("empty_skills")
    return out, errors


def _merge_chunks(parts: list[dict], doc: dict, text: str, dropped: list[dict]) -> tuple[dict, list[str]]:
    if len(parts) == 1:
        return _validate_and_clean(parts[0], doc, text, dropped)
    errors: list[str] = []
    merged: dict[str, Any] = {}
    list_fields = {
        "must_have_keywords", "nice_to_have_keywords", "tools_technologies", "domain_knowledge",
        "education_requirements", "certifications", "languages", "soft_skills", "key_responsibilities",
        "skills", "domains", "education", "locations", "work_history",
    }
    for part in parts:
        for k, v in part.items():
            if k in list_fields:
                merged.setdefault(k, []).extend(_clean_list(v))
            elif k in {"total_years_experience", "min_years_experience"}:
                val = _clean_years(v)
                if val is not None:
                    merged[k] = max(merged.get(k) or 0, val)
            elif k not in merged or merged.get(k) in (None, "", "unknown"):
                merged[k] = v
    out, errs = _validate_and_clean(merged, doc, text, dropped)
    errors.extend(errs)
    return out, errors


# --------------------------------------------------------------------------- #
# Deterministic fallback
# --------------------------------------------------------------------------- #
SECTION_RE = re.compile(
    r"^(skills|technical skills|core skills|requirements|qualifications|essential skills.*|"
    r"must have.*|nice to have.*|bonus.*|preferred.*|experience|education|work experience|"
    r"certifications?|languages?)\s*:?\s*$",
    re.IGNORECASE,
)


def _fallback_extract(doc: dict, text: str) -> dict[str, Any]:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    sections: dict[str, list[str]] = {}
    current = None
    for line in lines:
        m = SECTION_RE.match(line)
        if m:
            current = m.group(1).lower()
            sections.setdefault(current, [])
            continue
        if current:
            if len(line.split()) <= 12 and not line.endswith("."):
                sections[current].append(line)
            elif line.startswith("-"):
                sections[current].append(line.lstrip("- ").strip())
    def collect(keys: list[str]) -> list[str]:
        items = []
        for key, vals in sections.items():
            if any(k in key for k in keys):
                for val in vals:
                    items.extend(re.split(r"[,;]", val))
        return [x.strip(" .,;:-") for x in items if x.strip()]
    doc_type = "jd" if doc["doc_type"] in {"gt_jd", "unlabelled_jd"} else "cv"
    if doc_type == "jd":
        must = collect(["requirement", "qualification", "essential", "must have"])
        nice = collect(["nice", "bonus", "preferred"])
        if not must:
            must = [ln.lstrip("- ").strip() for ln in lines if ln.startswith("-")][:20]
        parsed = {
            "doc_id": doc["doc_id"], "doc_type": "jd",
            "job_title": doc.get("title"), "role_family": doc.get("role_family"),
            "seniority": "unknown", "location": doc.get("location"),
            "remote": doc.get("remote", "unknown"), "min_years_experience": None,
            "must_have_keywords": must, "nice_to_have_keywords": nice,
            "tools_technologies": [], "domain_knowledge": [],
            "education_requirements": collect(["education"]),
            "certifications": collect(["certification"]), "languages": collect(["language"]),
            "soft_skills": [], "key_responsibilities": [],
        }
    else:
        skills = collect(["skills", "technical"])
        parsed = {
            "doc_id": doc["doc_id"], "doc_type": "cv",
            "current_title": doc.get("title"), "current_company": doc.get("company"),
            "role_family": doc.get("role_family"), "seniority": "unknown",
            "total_years_experience": None, "skills": skills, "tools_technologies": [],
            "domains": [], "education": collect(["education"]), "certifications": collect(["certification"]),
            "languages": collect(["language"]), "locations": [], "soft_skills": [],
            "work_history": [],
        }
    return parsed


# --------------------------------------------------------------------------- #
# Single-document extraction
# --------------------------------------------------------------------------- #
def _prompt_file(cfg: Config, name: str) -> str:
    return (cfg.group_dir / "prompts" / name).read_text(encoding="utf-8")


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind("\n", start + int(size * 0.6), end)
            if cut > start:
                end = cut
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def extract_document(cfg: Config, client: OllamaClient, doc: dict, model: str,
                     force: bool = False, system: str | None = None) -> dict[str, Any]:
    doc_type = "jd" if doc["doc_type"] in {"gt_jd", "unlabelled_jd"} else "cv"
    template_name = "jd_extraction.txt" if doc_type == "jd" else "cv_extraction.txt"
    system = system or _prompt_file(cfg, "system_extraction.txt")
    template = _prompt_file(cfg, template_name)
    source_text = mask_pii(doc["raw_text"])
    chunks = _chunk_text(source_text, int(cfg.get("extraction.chunk_chars", 8000)),
                         int(cfg.get("extraction.chunk_overlap", 200)))
    prompt_user = template.replace("__DOC_ID__", doc["doc_id"]).replace("__DOCUMENT_TEXT__", source_text)
    prompt_hash = hashlib.sha256(
        (system + "\n" + prompt_user + "\n" + model + "\nv2").encode("utf-8")
    ).hexdigest()

    cache_path = cfg.cache / "llm" / f"{doc['doc_id']}.json"
    if cache_path.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("prompt_hash") == prompt_hash and cached.get("status") in {"ok", "fallback"}:
                return cached
        except Exception:
            pass

    started = time.time()
    dropped: list[dict[str, Any]] = []
    parts: list[dict] = []
    raw_responses: list[str] = []
    validation_errors: list[str] = []
    used_model = model
    status = "ok"
    extraction_status = "ok"

    for idx, chunk in enumerate(chunks):
        user = template.replace("__DOC_ID__", doc["doc_id"]).replace("__DOCUMENT_TEXT__", chunk)
        response = client.chat(system, user, model)
        content = response.get("content", "")
        raw_responses.append(content)
        parsed, err = _parse_json(content)
        if parsed is None:
            validation_errors.append(f"chunk{idx}:{err}")
            # Correction attempt with the validation error.
            correction = (
                user
                + "\n\nYour previous response was invalid ("
                + str(err)
                + "). Return one valid JSON object matching the schema. No markdown."
            )
            response = client.chat(system, correction, model)
            content = response.get("content", "")
            raw_responses.append(content)
            parsed, err = _parse_json(content)
        if parsed is None:
            status = "failed"
            validation_errors.append(f"chunk{idx}:{err or 'parse_failed'}")
            continue
        parts.append(parsed)

    if parts:
        parsed, clean_errors = _merge_chunks(parts, doc, source_text, dropped)
        validation_errors.extend(clean_errors)
    else:
        parsed, clean_errors = None, []

    if parsed is None or (doc_type == "jd" and not parsed.get("must_have_keywords")):
        parsed = _fallback_extract(doc, source_text)
        parsed, clean_errors = _validate_and_clean(parsed, doc, source_text, dropped)
        status = "fallback"
        extraction_status = "fallback"
        validation_errors.append("rule_based_fallback_used")

    record = {
        "doc_id": doc["doc_id"],
        "doc_type": doc_type,
        "model": model,
        "prompt_hash": prompt_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - started, 2),
        "status": status,
        "extraction_status": extraction_status,
        "chunk_count": len(chunks),
        "merged_from_chunks": len(chunks) > 1,
        "raw_response": "\n---CHUNK---\n".join(raw_responses),
        "parsed": parsed,
        "validation_errors": validation_errors,
        "dropped_keywords": dropped,
    }
    dump_json(cache_path, record)
    return record


# --------------------------------------------------------------------------- #
# Batch extraction
# --------------------------------------------------------------------------- #
def select_documents(documents: list[dict], limit: int | None = None,
                     doc_types: set[str] | None = None) -> list[dict]:
    docs = documents
    if doc_types:
        docs = [d for d in docs if d["doc_type"] in doc_types]
    if limit is not None:
        docs = docs[:limit]
    return docs


def extract_all(cfg: Config, documents: list[dict], *, force: bool = False,
                limit: int | None = None, doc_types: set[str] | None = None) -> dict[str, Any]:
    health = OllamaClient(cfg).health()
    if not health["ok"]:
        raise RuntimeError(f"Ollama is not reachable at {cfg.get('llm.base_url')}: {health.get('error')}")
    client = OllamaClient(cfg)
    model, models = client.choose_model()
    system = _prompt_file(cfg, "system_extraction.txt")
    docs = select_documents(documents, limit, doc_types)
    workers = max(1, int(cfg.get("llm.workers", 1)))
    records: list[dict] = []
    failures: list[dict] = []
    unsupported: list[dict] = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(extract_document, cfg, client, d, model, force, system): d
            for d in docs
        }
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {
                    "doc_id": d["doc_id"], "doc_type": d["doc_type"],
                    "status": "failed", "error": str(exc), "validation_errors": [str(exc)],
                    "parsed": {}, "model": model, "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            records.append(rec)
            if rec.get("status") == "failed":
                failures.append({
                    "doc_id": rec["doc_id"], "doc_type": rec["doc_type"],
                    "status": rec["status"], "error": "; ".join(rec.get("validation_errors", []))[:500],
                    "cache_path": str(cfg.cache / "llm" / f"{rec['doc_id']}.json"),
                })
            for item in rec.get("dropped_keywords", []):
                unsupported.append({
                    "doc_id": rec["doc_id"], "doc_type": rec["doc_type"],
                    "keyword": item.get("keyword"), "reason": item.get("reason"),
                })

    records.sort(key=lambda r: r["doc_id"])
    jd_rows = [r["parsed"] for r in records if r.get("doc_type") == "jd" and r.get("parsed")]
    cv_rows = [r["parsed"] for r in records if r.get("doc_type") == "cv" and r.get("parsed")]
    # Merge with existing outputs for a resumable run.
    existing_jd = {r["doc_id"]: r for r in read_jsonl(cfg.processed / "jd_extracted.jsonl")}
    existing_cv = {r["doc_id"]: r for r in read_jsonl(cfg.processed / "cv_extracted.jsonl")}
    for r in jd_rows:
        existing_jd[r["doc_id"]] = r
    for r in cv_rows:
        existing_cv[r["doc_id"]] = r
    write_jsonl(cfg.processed / "jd_extracted.jsonl", [existing_jd[k] for k in sorted(existing_jd)])
    write_jsonl(cfg.processed / "cv_extracted.jsonl", [existing_cv[k] for k in sorted(existing_cv)])

    import pandas as pd
    pd.DataFrame(failures, columns=["doc_id", "doc_type", "status", "error", "cache_path"]).to_csv(
        cfg.processed / "extraction_failures.csv", index=False)
    pd.DataFrame(unsupported, columns=["doc_id", "doc_type", "keyword", "reason"]).to_csv(
        cfg.processed / "unsupported_keywords.csv", index=False)

    n = len(records)
    ok = sum(1 for r in records if r.get("status") in {"ok", "fallback"})
    fallback = sum(1 for r in records if r.get("status") == "fallback")
    elapsed = [float(r.get("elapsed_sec") or 0) for r in records]
    summary = {
        "requested_documents": n,
        "success": ok,
        "success_rate": round(ok / n, 4) if n else 0.0,
        "failed": n - ok,
        "fallback": fallback,
        "mean_elapsed_sec": round(sum(elapsed) / n, 2) if n else 0.0,
        "total_elapsed_sec": round(time.time() - started, 1),
        "model": model,
        "primary_model_configured": cfg.get("llm.model"),
        "installed_models": models,
        "workers": workers,
    }
    dump_json(cfg.processed / "extraction_summary.json", summary)
    return {"records": records, "summary": summary}


# --------------------------------------------------------------------------- #
# Extraction quality (silver standard = Excel Skills)
# --------------------------------------------------------------------------- #
def _normalise_skill_set(values: list[str]) -> set[str]:
    out = set()
    for v in values:
        c = canonical_for(v)
        out.add(_kw_norm(c))
    return out


def score_extraction_quality(cfg: Config, documents: list[dict], extracted: list[dict]) -> "Any":
    import pandas as pd
    from rapidfuzz import fuzz

    extracted_by_id = {e["doc_id"]: e for e in extracted if e.get("doc_type") == "cv"}
    rows = []
    for doc in documents:
        if doc["doc_type"] != "pool_cv":
            continue
        rec = extracted_by_id.get(doc["doc_id"])
        if not rec:
            continue
        gold = [x.strip() for x in re.split(r"[,;]", doc.get("excel_skills") or "") if x.strip()]
        pred = list(rec.get("skills") or []) + list(rec.get("tools_technologies") or [])
        gold_set = _normalise_skill_set(gold)
        pred_set = _normalise_skill_set(pred)
        # Fuzzy match: a predicted skill counts if it closely matches a gold skill.
        matched_gold = set()
        matched_pred = set()
        for g in gold_set:
            for p in pred_set:
                if g == p or fuzz.token_set_ratio(g, p) >= 88:
                    matched_gold.add(g)
                    matched_pred.add(p)
                    break
        tp = len(matched_gold)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gold_set) if gold_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "doc_id": doc["doc_id"], "n_gold": len(gold_set), "n_pred": len(pred_set),
            "n_matched": tp, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "extraction_status": rec.get("extraction_status", "ok"),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(cfg.processed / "extraction_quality.csv", index=False)
        summary = pd.DataFrame([{
            "n_documents": int(len(df)),
            "macro_precision": round(float(df["precision"].mean()), 4),
            "macro_recall": round(float(df["recall"].mean()), 4),
            "macro_f1": round(float(df["f1"].mean()), 4),
            "worst_5_doc_ids": ", ".join(df.nsmallest(5, "f1")["doc_id"].tolist()),
        }])
        summary.to_csv(cfg.tables / "extraction_quality_summary.csv", index=False)
        return summary.iloc[0].to_dict()
    return {}
