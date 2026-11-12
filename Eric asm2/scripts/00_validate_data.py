#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "group assignment"))
from src.config import load_config, add_common_args
from src.data import sha256_file, extract_pdf_text, split_gt_pdf, clean_cv_text, clean_jd_text, normalize_text, load_excel


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["_project_root"])
    out = Path(cfg["_out_dir"])
    proc = out / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    cv_dir = root / cfg["paths"]["cv_dir"]
    gt_dir = root / cfg["paths"]["gt_dir"]
    xlsx_path = root / cfg["paths"]["xlsx"]

    # load excel
    jd_xls, cv_xls = load_excel(xlsx_path)
    cv_xls["Ref"] = cv_xls["Ref"].astype(str).str.strip()

    documents = []
    manifest = []

    # Pool CVs
    cv_files = sorted(cv_dir.glob("CV-*.pdf"))
    pool_cvs = {}
    for p in cv_files:
        raw, pages = extract_pdf_text(str(p))
        text = clean_cv_text(raw)
        doc_id = p.stem
        # try to get title/company/role from excel
        row = cv_xls[cv_xls["Ref"] == doc_id]
        title = str(row.iloc[0]["Current title"]) if len(row) else ""
        role_family = str(row.iloc[0]["Role family"]) if len(row) else ""
        company = str(row.iloc[0]["Current company"]) if len(row) else ""
        rec = {
            "doc_id": doc_id,
            "doc_type": "pool_cv",
            "source_path": str(p.relative_to(root)),
            "title": title,
            "role_family": role_family,
            "company": company,
            "raw_text": text,
            "pages": pages,
            "char_count": len(text),
            "split": "train_pool",
            "notes": ""
        }
        documents.append(rec)
        pool_cvs[doc_id] = rec
        manifest.append({"path": str(p.relative_to(root)), "file_size": p.stat().st_size, "sha256": sha256_file(p), "pages": pages, "doc_id": doc_id})

    # GT PDFs P01-P10
    gt_files = sorted(gt_dir.glob("P*.pdf"))
    gt_jds = {}
    gt_cvs = {}
    gt_rows = []
    # metadata from known titles/roles (will also parse from text)
    for i, p in enumerate(gt_files, start=101):
        raw, pages = extract_pdf_text(str(p))
        # detect P id from filename
        m = re.match(r"(P\d{2})", p.name)
        jd_id = m.group(1) if m else f"P{i-100:02d}"
        try:
            jd_text, cv_text = split_gt_pdf(raw)
            split_ok = True
            split_note = ""
        except Exception as e:
            jd_text, cv_text = clean_jd_text(raw), ""
            split_ok = False
            split_note = str(e)
        cand_id = f"CAND-{i}"
        # parse title/company from cv_text
        cv_title, cv_company = parse_cv_title_company(cv_text)
        jd_title, jd_role, jd_loc, jd_salary = parse_jd_meta(jd_text, jd_id)
        # P05 role family Product
        if jd_id == "P05":
            jd_role = "Product"
        jd_rec = {
            "doc_id": jd_id,
            "doc_type": "gt_jd",
            "source_path": str(p.relative_to(root)),
            "title": jd_title,
            "role_family": jd_role,
            "raw_text": jd_text,
            "pages": pages,
            "char_count": len(jd_text),
            "split": "heldout",
            "notes": split_note
        }
        cv_rec = {
            "doc_id": cand_id,
            "doc_type": "gt_cv",
            "source_path": str(p.relative_to(root)),
            "title": cv_title,
            "role_family": jd_role,
            "company": cv_company,
            "raw_text": cv_text,
            "pages": pages,
            "char_count": len(cv_text),
            "split": "heldout",
            "notes": f"from {jd_id}; {split_note}".strip("; ")
        }
        documents.append(jd_rec)
        documents.append(cv_rec)
        gt_jds[jd_id] = jd_rec
        gt_cvs[cand_id] = cv_rec
        gt_rows.append({
            "jd_id": jd_id,
            "jd_title": jd_title,
            "role_family": jd_role,
            "jd_location": jd_loc,
            "jd_salary": jd_salary,
            "hired_cv_id": cand_id,
            "hired_cv_title": cv_title,
            "hired_cv_company": cv_company,
            "source_pdf": str(p.relative_to(root)),
            "notes": split_note
        })
        manifest.append({"path": str(p.relative_to(root)), "file_size": p.stat().st_size, "sha256": sha256_file(p), "pages": pages, "doc_id": jd_id})

    gt_df = pd.DataFrame(gt_rows).sort_values("jd_id")
    gt_df.to_csv(proc / "ground_truth.csv", index=False)

    # Unlabelled JDs from excel
    jd_xls = jd_xls.copy()
    unlabelled = []
    for idx, row in jd_xls.iterrows():
        doc_id = f"JD-{idx+1:03d}"
        title = str(row.get("Position", "") or "")
        role = str(row.get("Role family", "") or "")
        country = row.get("Country", "")
        city = row.get("City", "")
        remote = row.get("Remote", "")
        contract = row.get("Contract type", "")
        sal_min = row.get("Salary min", "")
        sal_max = row.get("Salary max", "")
        sal_cur = row.get("Salary currency", "")
        desc = str(row.get("Job description", "") or "")
        header = f"Title: {title}\nRole family: {role}\nCountry: {country}\nCity: {city}\nRemote: {remote}\nContract type: {contract}\nSalary: {sal_min}-{sal_max} {sal_cur}\n"
        raw_text = normalize_text(header + "\n" + desc)
        rec = {
            "doc_id": doc_id,
            "doc_type": "unlabelled_jd",
            "source_path": str(xlsx_path.relative_to(root)) + "#JD Examples",
            "title": title,
            "role_family": role,
            "raw_text": raw_text,
            "pages": 1,
            "char_count": len(raw_text),
            "split": "train_jd",
            "notes": ""
        }
        documents.append(rec)
        unlabelled.append(rec)
    manifest.append({"path": str(xlsx_path.relative_to(root)), "file_size": xlsx_path.stat().st_size, "sha256": sha256_file(xlsx_path), "pages": None, "doc_id": "XLSX"})

    # Dedupe hired CVs vs pool CVs
    dedupe_rows = dedupe_hired_vs_pool(gt_cvs, pool_cvs, gt_df)
    pd.DataFrame(dedupe_rows).to_csv(proc / "dedupe_report.csv", index=False)

    # JD dedupe: unlabelled vs GT JDs
    jd_dedupe = dedupe_jds(unlabelled, gt_jds)
    pd.DataFrame(jd_dedupe).to_csv(proc / "jd_dedupe_report.csv", index=False)
    # mark excluded in documents
    exclude_ids = {r["unlabelled_jd_id"] for r in jd_dedupe if r.get("exclude_from_training")}
    for d in documents:
        if d["doc_id"] in exclude_ids:
            d["split"] = "excluded_duplicate_jd"
            d["notes"] = (d.get("notes","") + "; near-duplicate of GT JD").strip("; ")

    # CV alignment report
    align_rows = []
    for _, row in cv_xls.iterrows():
        ref = str(row.get("Ref",""))
        pdf = pool_cvs.get(ref)
        pdf_title = pdf["title"] if pdf else ""
        pdf_text = pdf["raw_text"][:2000] if pdf else ""
        x_title = str(row.get("Current title","") or "")
        x_company = str(row.get("Current company","") or "")
        title_match = norm_cmp(x_title) in norm_cmp(pdf_text) or norm_cmp(x_title) == norm_cmp(pdf_title)
        company_match = norm_cmp(x_company) in norm_cmp(pdf_text) if x_company else True
        align_rows.append({
            "ref": ref,
            "pdf_exists": ref in pool_cvs,
            "excel_title": x_title,
            "excel_company": x_company,
            "pdf_title": pdf_title,
            "title_in_pdf_text": norm_cmp(x_title) in norm_cmp(pdf_text) if x_title else False,
            "company_in_pdf_text": norm_cmp(x_company) in norm_cmp(pdf_text) if x_company else False,
            "match_ok": bool(title_match or company_match)
        })
    align_df = pd.DataFrame(align_rows)
    align_df.to_csv(proc / "cv_alignment_report.csv", index=False)

    # Save documents.jsonl and manifest
    with open(proc / "documents.jsonl", "w", encoding="utf-8") as f:
        for d in documents:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    pd.DataFrame(manifest).drop_duplicates(subset=["path","doc_id"]).to_csv(out / "data" / "raw_manifest.csv", index=False)

    # Summary
    n_pool = sum(1 for d in documents if d["doc_type"]=="pool_cv")
    n_gt_cv = sum(1 for d in documents if d["doc_type"]=="gt_cv")
    n_gt_jd = sum(1 for d in documents if d["doc_type"]=="gt_jd")
    n_unlab = sum(1 for d in documents if d["doc_type"]=="unlabelled_jd")
    unique_candidates = n_pool + n_gt_cv
    summary = {
        "pool_cvs": n_pool,
        "gt_cvs": n_gt_cv,
        "gt_jds": n_gt_jd,
        "unlabelled_jds": n_unlab,
        "unique_candidates": unique_candidates,
        "alignment_match_rate": float(align_df["match_ok"].mean()) if len(align_df) else None,
        "jd_duplicates_excluded": int(len(exclude_ids)),
    }
    with open(proc / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def norm_cmp(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def parse_cv_title_company(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # try to find "Current" section
    title = ""
    company = ""
    for i, l in enumerate(lines[:40]):
        if l.lower().startswith("current"):
            # look ahead
            for j in range(i+1, min(i+6, len(lines))):
                cand = lines[j]
                if cand.lower() in {"work history","education","skills","languages"}:
                    break
                if not title and len(cand) > 2:
                    title = cand
                    continue
                if title and not company and len(cand) > 1:
                    # heuristic: company often follows title
                    if not re.match(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4}|Present)", cand, re.I):
                        company = cand
                    break
            if title:
                break
    if not title and lines:
        title = lines[0][:120]
    return title, company


def parse_jd_meta(text, jd_id):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = ""
    role = ""
    loc = None
    salary = None
    # title often first meaningful line
    for l in lines[:10]:
        if l in {"Finance & Legal","Sales & BD","Marketing","Data & AI","Product","Engineering","Design & Creative"}:
            role = l
            continue
        if not title and len(l) > 3 and "page" not in l.lower():
            # skip Pxx artifacts
            if re.match(r"^P\d{2}", l):
                continue
            title = l
            # don't break immediately, maybe role line appears
    # location/salary lines heuristics
    for l in lines[:30]:
        if "$" in l or "SGD" in l or "MYR" in l or "IDR" in l or "RM" in l:
            salary = l
        if re.search(r"\b(Malaysia|Singapore|Indonesia|Thailand|Hong Kong|Vietnam|Remote)\b", l, re.I):
            # avoid long responsibility lines
            if len(l) < 120:
                loc = l
    # known mappings from prompt
    known = {
        "P01": ("Regional Finance Controller","Finance & Legal","Kuala Lumpur, Malaysia","~$5,550"),
        "P02": ("Sales Manager in Indonesia","Sales & BD","Indonesia","~$6,800"),
        "P03": ("GTM Marketing Manager / Lead","Marketing",None,None),
        "P04": ("Data Engineer","Data & AI","Singapore","S$6,500 - S$9,000"),
        "P05": ("Product Manager - WMS & Warehouse Integrations","Product","Remote","$3,500 - $4,500"),
        "P06": ("Deployment Strategist","Data & AI","Singapore","$7,000 - $10,000"),
        "P07": ("Marketing Manager (HK & SG)","Marketing","Hong Kong / Singapore",None),
        "P08": ("Field Service Engineer (FSE)","Engineering","Singapore",None),
        "P09": ("Tech Lead","Engineering","Kuala Lumpur, Malaysia",None),
        "P10": ("Regional Lead Designer UI/UX","Design & Creative",None,"$4,000 - $5,000"),
    }
    if jd_id in known:
        kt, kr, kl, ks = known[jd_id]
        title = title or kt
        role = kr
        loc = loc if loc else kl
        salary = salary if salary else ks
    return title, role, loc, salary


def dedupe_hired_vs_pool(gt_cvs, pool_cvs, gt_df):
    pool_ids = list(pool_cvs.keys())
    pool_texts = [pool_cvs[i]["raw_text"] for i in pool_ids]
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    if not pool_texts:
        return []
    pool_mat = vec.fit_transform(pool_texts)
    rows = []
    for _, g in gt_df.iterrows():
        cid = g["hired_cv_id"]
        ctext = gt_cvs[cid]["raw_text"]
        cvec = vec.transform([ctext])
        sims = cosine_similarity(cvec, pool_mat)[0]
        top_idx = np.argsort(-sims)[:3]
        top = []
        for ti in top_idx:
            top.append((pool_ids[ti], float(sims[ti])))
        best_id, best_sim = top[0]
        best = pool_cvs[best_id]
        title_match = norm_cmp(g["hired_cv_title"]) == norm_cmp(best.get("title",""))
        comp_match = norm_cmp(g["hired_cv_company"]) == norm_cmp(best.get("company",""))
        is_dup = (best_sim >= 0.95 and title_match and comp_match)
        rows.append({
            "hired_cv_id": cid,
            "jd_id": g["jd_id"],
            "top1_pool_cv": top[0][0], "top1_cosine": round(top[0][1],4),
            "top2_pool_cv": top[1][0], "top2_cosine": round(top[1][1],4),
            "top3_pool_cv": top[2][0], "top3_cosine": round(top[2][1],4),
            "top1_title": best.get("title",""), "top1_company": best.get("company",""),
            "title_match": title_match, "company_match": comp_match,
            "decision": "map_to_existing" if is_dup else "keep_new"
        })
    return rows


def dedupe_jds(unlabelled, gt_jds):
    if not unlabelled or not gt_jds:
        return []
    gt_ids = list(gt_jds.keys())
    gt_texts = [gt_jds[i]["raw_text"] for i in gt_ids]
    un_ids = [u["doc_id"] for u in unlabelled]
    un_texts = [u["raw_text"] for u in unlabelled]
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    all_mat = vec.fit_transform(gt_texts + un_texts)
    gt_mat = all_mat[:len(gt_texts)]
    un_mat = all_mat[len(gt_texts):]
    sims = cosine_similarity(un_mat, gt_mat)
    rows = []
    for i, uid in enumerate(un_ids):
        j = int(np.argmax(sims[i]))
        sim = float(sims[i, j])
        exclude = sim >= 0.90
        rows.append({
            "unlabelled_jd_id": uid,
            "nearest_gt_jd": gt_ids[j],
            "cosine": round(sim,4),
            "exclude_from_training": bool(exclude),
            "reason": "cosine>=0.90" if exclude else "ok"
        })
    return rows


if __name__ == "__main__":
    main()
