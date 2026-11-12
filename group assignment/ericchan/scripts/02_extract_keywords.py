"""Step 2: local-LLM keyword extraction (ollama / qwen3:8b).

Reads data/cvs_raw.json, data/jds_raw.json and writes one JSON record per
document to data/extracted.jsonl.  Every call is logged verbatim (prompt +
raw response + latency) to data/llm_calls.jsonl so the run is auditable and
resumable.

Usage:
    python scripts/02_extract_keywords.py [--limit N]
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "extracted.jsonl")
LOG = os.path.join(DATA, "llm_calls.jsonl")

CV_SCHEMA = """{
  "current_title": "string - the person's current or most recent job title",
  "seniority": "one of: intern, junior, mid, senior, lead, executive",
  "years_experience": integer - total professional years, 0 if unknown,
  "hard_skills": ["technical / functional skills, tools, languages, frameworks, platforms, methodologies"],
  "soft_skills": ["interpersonal skills only"],
  "domains": ["industry or business domain, e.g. fintech, semiconductor, logistics, e-commerce"],
  "education_level": "one of: none, diploma, bachelor, master, phd",
  "education_field": "string",
  "spoken_languages": ["natural languages only, e.g. English, Mandarin"],
  "locations": ["countries or cities mentioned"]
}"""

JD_SCHEMA = """{
  "job_title": "string",
  "seniority": "one of: intern, junior, mid, senior, lead, executive",
  "years_required": integer - minimum years of experience demanded, 0 if unspecified,
  "must_have": ["non-negotiable technical / functional skills, tools, languages, frameworks"],
  "nice_to_have": ["preferred but optional technical / functional skills"],
  "soft_skills": ["interpersonal or leadership requirements"],
  "domains": ["industry or business domain the role sits in"],
  "education_required": "one of: none, diploma, bachelor, master, phd",
  "spoken_languages": ["required natural languages"],
  "location": "string",
  "employment_type": "string - e.g. full time, contract, remote"
}"""

CV_SYSTEM = (
    "You are a recruitment data analyst. You read one candidate CV and return "
    "strict JSON only. Never invent skills that are not stated or directly "
    "implied by the text. Use short lowercase-ish canonical names "
    "('PostgreSQL' not 'postgres db experience'), one skill per list item, no "
    "duplicates, max 40 items per list."
)

JD_SYSTEM = (
    "You are a technical recruiter. You read one job description and return "
    "strict JSON only. Separate genuine hard requirements from preferences. "
    "Use short canonical skill names, one per list item, no duplicates, "
    "max 40 items per list. Never invent requirements absent from the text."
)

CV_PROMPT = """Extract structured keywords from the CV below.

Rules:
- hard_skills: every concrete technology, tool, framework, platform, language, certification or measurable methodology that appears. Include soft-skill-free items only.
- Put pure interpersonal traits (communication, teamwork, leadership) in soft_skills.
- years_experience: sum of professional (non-internship) employment, rounded down.
- seniority: judge from titles and tenure, not from the words in the skills list.
- Return ONLY the JSON object, no explanation.

JSON schema:
{schema}

CV:
\"\"\"{doc}\"\"\"
"""

JD_PROMPT = """Extract structured keywords from the job description below.

Rules:
- must_have: skills the text presents as required, essential or "must have".
- nice_to_have: skills presented as preferred, bonus, advantage or "good to have". If the JD does not distinguish, put core technical requirements in must_have and leave nice_to_have empty.
- Ignore company perks, benefits, and application-question boilerplate.
- Return ONLY the JSON object, no explanation.

JSON schema:
{schema}

JOB DESCRIPTION:
\"\"\"{doc}\"\"\"
"""


def norm_list(v, cap=40):
    if not isinstance(v, list):
        return []
    out, seen = [], set()
    for x in v:
        if not isinstance(x, str):
            continue
        s = re.sub(r"\s+", " ", x).strip(" .,;:-")
        if not s or len(s) > 60:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def coerce(rec, kind):
    """Force the LLM payload into the expected shape."""
    if kind == "cv":
        schema_keys = ["current_title", "seniority", "years_experience",
                       "hard_skills", "soft_skills", "domains",
                       "education_level", "education_field",
                       "spoken_languages", "locations"]
    else:
        schema_keys = ["job_title", "seniority", "years_required", "must_have",
                       "nice_to_have", "soft_skills", "domains",
                       "education_required", "spoken_languages", "location",
                       "employment_type"]
    out = {k: rec.get(k) for k in schema_keys}
    list_keys = ["hard_skills", "soft_skills", "domains", "spoken_languages",
                 "locations", "must_have", "nice_to_have"]
    for k in list_keys:
        if k in out:
            out[k] = norm_list(out[k])
    for k in ("years_experience", "years_required"):
        if k in out:
            try:
                out[k] = max(0, min(60, int(float(out[k] or 0))))
            except (TypeError, ValueError):
                out[k] = 0
    sen = (out.get("seniority") or "mid")
    sen = str(sen).strip().lower()
    out["seniority"] = sen if sen in (
        "intern", "junior", "mid", "senior", "lead", "executive") else "mid"
    for k in ("current_title", "job_title", "education_field", "location",
              "employment_type"):
        if k in out and out[k] is not None:
            out[k] = str(out[k]).strip()[:120]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    cvs = json.load(open(os.path.join(DATA, "cvs_raw.json"), encoding="utf-8"))
    jds = json.load(open(os.path.join(DATA, "jds_raw.json"), encoding="utf-8"))

    done = {r["doc_id"] for r in llm.read_jsonl(OUT) if r.get("ok")}
    jobs = [("cv", c) for c in cvs] + [("jd", j) for j in jds]
    todo = [j for j in jobs if j[1].get("cv_id") or j[1].get("jd_id")]
    todo = [(k, d) for k, d in todo
            if (d.get("cv_id") or d.get("jd_id")) not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(done)} already done, {len(todo)} to extract", flush=True)
    for i, (kind, doc) in enumerate(todo, 1):
        doc_id = doc.get("cv_id") or doc.get("jd_id")
        text = doc["text"][:6000]
        if kind == "cv":
            prompt = CV_PROMPT.format(schema=CV_SCHEMA, doc=text)
            system = CV_SYSTEM
        else:
            prompt = JD_PROMPT.format(schema=JD_SCHEMA, doc=text)
            system = JD_SYSTEM

        t0 = time.time()
        try:
            raw = llm.chat(prompt, system=system, num_predict=900, num_ctx=8192)
            parsed = llm.parse_json(raw)
            rec = coerce(parsed, kind)
            ok, err = True, None
        except Exception as e:  # noqa: BLE001
            raw, rec, ok, err = locals().get("raw", ""), {}, False, repr(e)[:300]
        dt = time.time() - t0

        llm.append_jsonl(LOG, {"doc_id": doc_id, "kind": kind, "prompt": prompt,
                               "system": system, "raw": raw, "seconds": round(dt, 2),
                               "ok": ok, "error": err})
        llm.append_jsonl(OUT, {"doc_id": doc_id, "kind": kind, "ok": ok,
                               "seconds": round(dt, 2), **rec})
        flag = "ok " if ok else "ERR"
        print(f"[{i}/{len(todo)}] {flag} {doc_id} {dt:5.1f}s "
              f"{len(rec.get('hard_skills') or rec.get('must_have') or [])} skills",
              flush=True)

    rows = llm.read_jsonl(OUT)
    good = [r for r in rows if r.get("ok")]
    print(f"\ndone: {len(good)}/{len(cvs) + len(jds)} successful")
    if len(good) < len(cvs) + len(jds):
        bad = [r["doc_id"] for r in rows if not r.get("ok")]
        print("failed:", bad)


if __name__ == "__main__":
    main()
