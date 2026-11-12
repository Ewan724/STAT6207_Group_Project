#!/usr/bin/env python3
import argparse, json, os, re, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group assignment"))
from src.config import load_config, add_common_args
from src.data import mask_pii
from src.llm import prompt_hash, call_ollama, extract_json, normalize_keyword, valid_keyword_phrase, keyword_in_text

JD_FIELDS = ["must_have_keywords","nice_to_have_keywords","tools_technologies","domain_knowledge","education_requirements","certifications","languages","soft_skills","key_responsibilities"]
CV_FIELDS = ["skills","tools_technologies","domains","education","certifications","languages","locations","soft_skills","work_history"]


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["_project_root"])
    out = Path(cfg["_out_dir"])
    proc = out / "data" / "processed"
    cache_dir = out / "data" / "cache" / "llm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out / "prompts"

    with open(prompts_dir/"system_extraction.txt", encoding="utf-8") as f: system_prompt = f.read()
    with open(prompts_dir/"jd_extraction.txt", encoding="utf-8") as f: jd_tmpl = f.read()
    with open(prompts_dir/"cv_extraction.txt", encoding="utf-8") as f: cv_tmpl = f.read()

    # load documents
    docs = []
    with open(proc/"documents.jsonl", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    # select extractable: all CVs and JDs (gt + unlabelled)
    targets = [d for d in docs if d["doc_type"] in ("pool_cv","gt_cv","gt_jd","unlabelled_jd")]
    # exclude nothing except maybe keep duplicates? they are marked but still docs; include all
    targets.sort(key=lambda d: (d["doc_type"], d["doc_id"]))
    if args.limit:
        targets = targets[:args.limit]

    model = args.model or cfg["ollama"]["model"]
    base_url = cfg["ollama"]["base_url"]
    options = {
        "temperature": cfg["ollama"]["temperature"],
        "seed": cfg["ollama"]["seed"],
        "num_ctx": cfg["ollama"]["num_ctx"],
        "num_predict": cfg["ollama"]["num_predict"],
        "top_p": cfg["ollama"]["top_p"],
    }
    timeout = cfg["ollama"].get("timeout", 300)

    failures = []
    unsupported = []
    results_jd = []
    results_cv = []
    times = []
    success = 0
    fallback_count = 0

    for doc in targets:
        doc_id = doc["doc_id"]
        doc_type = doc["doc_type"]
        is_jd = doc_type in ("gt_jd","unlabelled_jd")
        cache_path = cache_dir / f"{doc_id}.json"
        text = mask_pii(doc.get("raw_text","") or "")
        if len(text) > cfg["extraction"].get("chunk_threshold", 8000):
            # simple truncation/chunk note; basic chunking: split in half
            text_for_prompt = text[:7500] + "\n[...truncated for extraction...]"
            chunked = True
        else:
            text_for_prompt = text
            chunked = False
        tmpl = jd_tmpl if is_jd else cv_tmpl
        user_prompt = tmpl.replace("{DOC_ID}", doc_id).replace("{DOCUMENT_TEXT}", text_for_prompt)
        ph = prompt_hash(system_prompt, user_prompt, model)

        if cache_path.exists() and not args.force:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("prompt_hash") == ph and cached.get("status") in ("ok","fallback"):
                    parsed = cached.get("parsed")
                    if parsed:
                        if is_jd: results_jd.append(parsed)
                        else: results_cv.append(parsed)
                        success += 1
                        if cached.get("status")=="fallback": fallback_count += 1
                        continue
            except Exception:
                pass

        t0 = time.time()
        status = "ok"
        raw = ""
        parsed = None
        errs = []
        try:
            raw, meta = call_ollama(base_url, model, system_prompt, user_prompt, options, timeout=timeout, think=cfg["ollama"].get("think", False))
            parsed = extract_json(raw)
            if parsed is None:
                raise ValueError("JSON parse failed")
            parsed, verrs, uns = validate_and_clean(parsed, doc, text)
            errs.extend(verrs)
            unsupported.extend(uns)
            if not parsed:
                raise ValueError("Validation failed")
        except Exception as e:
            errs.append(str(e))
            # retry once
            try:
                corr_prompt = user_prompt + f"\n\nYour previous response was invalid ({e}). Return only one valid JSON object matching the schema."
                raw, meta = call_ollama(base_url, model, system_prompt, corr_prompt, options, timeout=timeout, think=cfg["ollama"].get("think", False))
                parsed = extract_json(raw)
                if parsed is None: raise ValueError("JSON parse failed after retry")
                parsed, verrs, uns = validate_and_clean(parsed, doc, text)
                errs.extend(verrs); unsupported.extend(uns)
            except Exception as e2:
                errs.append(str(e2))
                parsed = deterministic_fallback(doc, text, is_jd)
                status = "fallback"
                fallback_count += 1
        elapsed = time.time() - t0
        times.append(elapsed)
        if parsed:
            parsed["doc_id"] = doc_id
            parsed["doc_type"] = "jd" if is_jd else "cv"
            if chunked:
                parsed["merged_from_chunks"] = True
                parsed["chunk_count"] = 1
        cache_obj = {
            "doc_id": doc_id, "doc_type": doc_type, "model": model, "prompt_hash": ph,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_sec": round(elapsed,3),
            "status": status, "raw_response": raw[:200000] if raw else "", "parsed": parsed, "validation_errors": errs
        }
        cache_path.write_text(json.dumps(cache_obj, ensure_ascii=False), encoding="utf-8")
        if parsed and status in ("ok","fallback"):
            success += 1
            if is_jd: results_jd.append(parsed)
            else: results_cv.append(parsed)
        else:
            failures.append({"doc_id": doc_id, "doc_type": doc_type, "error": "; ".join(errs[-3:])})

        if len(times) % 10 == 0:
            print(f"processed {len(times)}/{len(targets)} (ok/fallback so far {success})", flush=True)

    # save outputs
    write_jsonl(proc/"jd_extracted.jsonl", sorted(results_jd, key=lambda x: x.get("doc_id","")))
    write_jsonl(proc/"cv_extracted.jsonl", sorted(results_cv, key=lambda x: x.get("doc_id","")))
    pd.DataFrame(failures).to_csv(proc/"extraction_failures.csv", index=False)
    pd.DataFrame(unsupported).to_csv(proc/"unsupported_keywords.csv", index=False)
    summary = {
        "attempted": len(targets), "success_or_fallback": success, "failures": len(failures),
        "fallback_count": fallback_count, "json_success_rate": (success/len(targets)) if targets else 0,
        "mean_elapsed_sec": sum(times)/len(times) if times else None, "model": model
    }
    (proc/"extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validate_and_clean(parsed, doc, source_text):
    errs = []
    uns = []
    if not isinstance(parsed, dict):
        return None, ["not a dict"], uns
    is_jd = doc["doc_type"] in ("gt_jd","unlabelled_jd")
    # ensure doc id/type
    parsed["doc_id"] = doc["doc_id"]
    parsed["doc_type"] = "jd" if is_jd else "cv"
    # clean keyword lists
    kw_fields = JD_FIELDS if is_jd else ["skills","tools_technologies","domains","soft_skills"]
    for field in kw_fields:
        val = parsed.get(field)
        if val is None:
            if field in ("must_have_keywords","nice_to_have_keywords","tools_technologies","domain_knowledge","certifications","languages","soft_skills","key_responsibilities","skills","domains","locations"):
                parsed[field] = [] if field != "education" else []
            continue
        if not isinstance(val, list):
            # try split
            if isinstance(val, str):
                val = re.split(r",|;|\n", val)
            else:
                val = []
        cleaned = []
        seen = set()
        for item in val:
            if isinstance(item, dict):
                # keep dicts for education/work_history
                cleaned.append(item)
                continue
            k = normalize_keyword(item)
            if not k: continue
            # split accidental commas
            parts = [normalize_keyword(p) for p in re.split(r",|;", k)] if ("," in k or ";" in k) else [k]
            for p in parts:
                if not p: continue
                if not valid_keyword_phrase(p):
                    # allow some longer domain phrases in domain fields? keep if <=5 words
                    if len(p.split())>5: continue
                if not keyword_in_text(p, source_text):
                    uns.append({"doc_id": doc["doc_id"], "field": field, "keyword": p, "reason": "not_in_source"})
                    continue
                key = p.lower()
                if key in seen: continue
                seen.add(key)
                cleaned.append(p)
        parsed[field] = cleaned
    # required defaults
    if is_jd:
        parsed.setdefault("job_title", doc.get("title"))
        parsed.setdefault("role_family", doc.get("role_family"))
        parsed.setdefault("seniority", "unknown")
        parsed.setdefault("remote", "unknown")
        for f in JD_FIELDS:
            parsed.setdefault(f, [])
    else:
        parsed.setdefault("current_title", doc.get("title"))
        parsed.setdefault("current_company", doc.get("company"))
        parsed.setdefault("role_family", doc.get("role_family"))
        parsed.setdefault("seniority", "unknown")
        for f in CV_FIELDS:
            parsed.setdefault(f, [] if f!="education" and f!="work_history" else [])
    return parsed, errs, uns


def deterministic_fallback(doc, text, is_jd):
    # simple rule-based extraction
    lines = [l.strip("- •\t ") for l in (text or "").splitlines()]
    skills = []
    in_skills = False
    for l in lines:
        if not l: continue
        low = l.lower()
        if any(h in low for h in ["skills", "requirements", "qualifications", "nice to have", "bonus", "technical"]):
            in_skills = True
            # also capture after colon
            if ":" in l:
                tail = l.split(":",1)[1]
                skills.extend([p.strip() for p in re.split(r",|;|\|", tail) if p.strip()])
            continue
        if in_skills:
            if len(l) < 80 and ("," in l or ";" in l or l.startswith("-")):
                skills.extend([p.strip() for p in re.split(r",|;|\|", l) if p.strip()])
            elif len(l.split()) <= 4:
                skills.append(l)
            if len(skills) > 60: break
    skills = [s for s in skills if 1 <= len(s.split()) <= 4][:40]
    if is_jd:
        return {
            "doc_id": doc["doc_id"], "doc_type": "jd", "job_title": doc.get("title"), "role_family": doc.get("role_family"),
            "seniority": "unknown", "location": None, "remote": "unknown", "min_years_experience": None,
            "must_have_keywords": skills[:25], "nice_to_have_keywords": skills[25:35], "tools_technologies": [],
            "domain_knowledge": [], "education_requirements": [], "certifications": [], "languages": [], "soft_skills": [], "key_responsibilities": []
        }
    else:
        return {
            "doc_id": doc["doc_id"], "doc_type": "cv", "current_title": doc.get("title"), "current_company": doc.get("company"),
            "role_family": doc.get("role_family"), "seniority": "unknown", "total_years_experience": None,
            "skills": skills[:30], "tools_technologies": [], "domains": [], "education": [], "certifications": [], "languages": [], "locations": [], "soft_skills": [], "work_history": []
        }

if __name__ == "__main__":
    main()
