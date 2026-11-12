"""Stage 6: structured error analysis, AE-vs-TF-IDF comparison and recommendations."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .config import Config, dump_json
from .data import read_jsonl
from .keywords import load_extractions, map_term, term_key
from .retrieval import SENIORITY_RANK, document_role_family


def _title_overlap(a: str, b: str) -> float:
    ta = set(str(a or "").lower().split())
    tb = set(str(b or "").lower().split())
    return len(ta & tb) / max(1, len(ta | tb))


def _canonical_terms_in_text(text: str, aliases: pd.DataFrame) -> set[str]:
    t = str(text).lower().replace("&", " and ")
    found = set()
    for _, row in aliases.iterrows():
        raw = str(row["raw_term"]).lower()
        if len(raw) < 2:
            continue
        if raw in t:
            found.add(str(row["canonical_term"]))
    return found


def build_error_analysis(cfg: Config) -> dict[str, Any]:
    rankings = pd.read_csv(cfg.processed / "rankings.csv")
    gt = pd.read_csv(cfg.processed / "ground_truth.csv")
    coverage = pd.read_csv(cfg.processed / "must_have_coverage.csv")
    extracted = load_extractions(cfg)
    documents = read_jsonl(cfg.processed / "documents.jsonl")
    aliases = pd.read_csv(cfg.processed / "keyword_aliases.csv")
    clusters = pd.read_csv(cfg.processed / "keyword_clusters.csv")
    heldout_cov = pd.read_csv(cfg.processed / "heldout_keyword_coverage.csv")
    unmatched_by_doc: dict[str, set[str]] = {}
    for _, r in heldout_cov.iterrows():
        terms = str(r.get("unmatched_terms") or "")
        unmatched_by_doc[str(r["doc_id"])] = {
            x.strip().lower() for x in terms.split("|") if x.strip() and x.strip().lower() != "nan"
        }
    meta = {d["doc_id"]: d for d in documents}
    hired = dict(zip(gt["jd_id"], gt["hired_cv_id"]))

    must_terms = {}
    raw_missing = {}
    for _, row in gt.iterrows():
        jd_id = row["jd_id"]
        rec = extracted.get(jd_id, {})
        kws = list(rec.get("must_have_keywords") or []) + list(rec.get("tools_technologies") or [])
        must_terms[jd_id] = set(kws)
        raw_terms = _canonical_terms_in_text(meta.get(jd_id, {}).get("raw_text", ""), aliases)
        extracted_terms = set()
        for kw in kws:
            extracted_terms.add(kw)
        raw_missing[jd_id] = sorted(raw_terms - {str(x) for x in extracted_terms})

    rows = []
    for model in ["M1", "M3"]:
        sub_all = rankings[rankings["model"] == model]
        for jd_id in gt["jd_id"]:
            jd = gt[gt["jd_id"] == jd_id].iloc[0]
            sub = sub_all[sub_all["jd_id"] == jd_id].sort_values("rank")
            hcv = hired[jd_id]
            hrow = sub[sub["cv_id"] == hcv].iloc[0]
            top1 = sub.iloc[0]
            top1_meta = meta.get(top1["cv_id"], {})
            h_meta = meta.get(hcv, {})
            top1_rec = extracted.get(top1["cv_id"], {})
            h_rec = extracted.get(hcv, {})
            cov_h = coverage[(coverage["jd_id"] == jd_id) & (coverage["cv_id"] == hcv)]
            cov_t = coverage[(coverage["jd_id"] == jd_id) & (coverage["cv_id"] == top1["cv_id"])]
            cov_h = float(cov_h["coverage"].iloc[0]) if len(cov_h) else np.nan
            cov_t = float(cov_t["coverage"].iloc[0]) if len(cov_t) else np.nan
            miss_h = str(coverage[(coverage["jd_id"] == jd_id) & (coverage["cv_id"] == hcv)]["missing_must_haves"].iloc[0]).split("|")
            miss_t = str(coverage[(coverage["jd_id"] == jd_id) & (coverage["cv_id"] == top1["cv_id"])]["missing_must_haves"].iloc[0]).split("|")
            miss_h = [x for x in miss_h if x and x != "nan"]
            miss_t = [x for x in miss_t if x and x != "nan"]
            rank = int(hrow["rank"])
            jd_rec = extracted.get(jd_id, {})
            jd_loc = jd.get("jd_location")
            jd_loc = "" if pd.isna(jd_loc) else str(jd_loc)
            jd_country = jd_loc.split(",")[-1].strip() if jd_loc else ""
            jd_family = document_role_family(meta.get(jd_id, {}), jd_rec)
            jd_langs = set(x.lower() for x in (jd_rec.get("languages") or []))
            h_langs = set(x.lower() for x in (h_rec.get("languages") or []))
            t_langs = set(x.lower() for x in (top1_rec.get("languages") or []))
            jd_years = jd_rec.get("min_years_experience")
            h_years = h_rec.get("total_years_experience")
            t_years = top1_rec.get("total_years_experience")
            role_match_h = int(document_role_family(h_meta, h_rec) == jd_family and jd_family != "unknown")
            role_match_t = int(document_role_family(top1_meta, top1_rec) == jd_family and jd_family != "unknown")
            sen_h = h_rec.get("seniority", "unknown")
            sen_t = top1_rec.get("seniority", "unknown")
            sen_match_h = int(SENIORITY_RANK.get(sen_h, 2) >= SENIORITY_RANK.get(jd_rec.get("seniority", "unknown"), 2))
            sen_match_t = int(SENIORITY_RANK.get(sen_t, 2) >= SENIORITY_RANK.get(jd_rec.get("seniority", "unknown"), 2))

            primary, secondary, comment = "none", "", ""
            if rank > 10:
                # E7 location / language hard filter.
                loc_mismatch = (bool(jd_country) and pd.notna(h_meta.get("country"))
                                and pd.notna(top1_meta.get("country"))
                                and str(h_meta.get("country")).lower() != jd_country.lower()
                                and str(top1_meta.get("country")).lower() == jd_country.lower())
                lang_mismatch = bool(jd_langs) and bool(t_langs) and not (jd_langs & t_langs) and bool(h_langs) and bool(jd_langs & h_langs)
                if loc_mismatch or lang_mismatch:
                    primary = "E7_location_language_constraint"
                    secondary = "E4_hard_negative_lookalike"
                    comment = (f"Hired CV ranked {rank}; top-1 {top1['cv_id']} ({top1_meta.get('title')}) "
                               f"matches the JD's location/language constraint better. "
                               f"JD country={jd_country or 'n/a'}, hired country={h_meta.get('country') or 'n/a'}.")
                elif role_match_t and not role_match_h:
                    primary = "E4_hard_negative_lookalike"
                    comment = (f"Hired CV ranked {rank}; top-1 {top1['cv_id']} ({top1_meta.get('title')} at "
                               f"{top1_meta.get('company')}) is in the JD role family while the hired CV is "
                               f"{h_meta.get('role_family') or 'unknown'}.")
                elif t_years is not None and h_years is not None and t_years >= h_years + 4 and jd_years:
                    primary = "E5_seniority_years_mismatch"
                    comment = (f"Hired CV ranked {rank}; top-1 reports {t_years} years vs {h_years} for the hired CV "
                               f"(JD minimum {jd_years}).")
                elif model == "M3" and int(rankings[(rankings["model"] == "M1") & (rankings["jd_id"] == jd_id) &
                                                    (rankings["cv_id"] == hcv)]["rank"].iloc[0]) <= 10:
                    primary = "E3_representation_loss"
                    comment = (f"AE rank {rank} vs TF-IDF rank "
                               f"{int(rankings[(rankings['model'] == 'M1') & (rankings['jd_id'] == jd_id) & (rankings['cv_id'] == hcv)]['rank'].iloc[0])}; "
                               f"the 32-d bottleneck appears to smooth a discriminative must-have.")
                elif miss_h and (set(x.lower() for x in miss_h) & unmatched_by_doc.get(jd_id, set())):
                    primary = "E2_keyword_merge_error"
                    comment = f"Hired CV is missing must-have clusters ({', '.join(miss_h[:3])}); at least one is an unmatched held-out keyword (taxonomy/coverage gap)."
                elif cov_t is not None and cov_h is not None and cov_t >= cov_h:
                    primary = "E4_hard_negative_lookalike"
                    comment = (f"Hired CV ranked {rank}; top-1 {top1['cv_id']} has equal/higher must-have coverage "
                               f"({cov_t:.2f} vs {cov_h:.2f}).")
                else:
                    primary = "E8_label_noise_or_recruiter_context"
                    comment = (f"Hired CV ranked {rank}; its must-have coverage ({cov_h:.2f}) is not clearly worse "
                               f"than top-1 ({cov_t:.2f}). Hiring likely used interview/availability context.")
                if secondary == "":
                    # Identify extraction misses as a secondary signal.
                    missing_ext = [x for x in raw_missing.get(jd_id, []) if x.lower() in str(h_rec.get("skills", [])).lower()]
                    if missing_ext:
                        secondary = "E1_extraction_missing_skill"
            else:
                comment = f"Hit at rank {rank}; top-1 was {top1['cv_id']} with score {top1['score']:.4f}."

            rows.append({
                "model": model, "jd_id": jd_id, "jd_title": jd["jd_title"],
                "role_family": jd["role_family"], "hired_cv_id": hcv, "hired_rank": rank,
                "hired_score": round(float(hrow["score"]), 6),
                "top1_cv_id": top1["cv_id"], "top1_title": top1_meta.get("title"),
                "top1_score": round(float(top1["score"]), 6),
                "score_margin": round(float(hrow["score"]) - float(top1["score"]), 6),
                "must_have_coverage_hired": round(float(cov_h), 4) if cov_h is not None else None,
                "must_have_coverage_top1": round(float(cov_t), 4) if cov_t is not None else None,
                "missing_must_haves_hired": "|".join(miss_h[:5]),
                "missing_must_haves_top1": "|".join(miss_t[:5]),
                "role_family_match_hired": role_match_h, "role_family_match_top1": role_match_t,
                "seniority_match_hired": sen_match_h, "seniority_match_top1": sen_match_t,
                "years_hired": h_years, "years_top1": t_years,
                "years_gap": (int(t_years) - int(h_years)) if (t_years is not None and h_years is not None) else None,
                "primary_failure_category": primary, "secondary_failure_category": secondary,
                "comment": comment,
            })
    df = pd.DataFrame(rows)
    df.to_csv(cfg.processed / "error_analysis.csv", index=False)

    # AE vs TF-IDF dumbbell data.
    ae = df[df["model"] == "M3"][["jd_id", "hired_rank", "hired_score", "primary_failure_category"]]
    tf = df[df["model"] == "M1"][["jd_id", "hired_rank", "hired_score"]]
    comp = ae.merge(tf, on="jd_id", suffixes=("_ae", "_tfidf"))
    comp["delta"] = comp["hired_rank_tfidf"] - comp["hired_rank_ae"]
    comp["winner"] = np.where(comp["delta"] > 0, "AE", np.where(comp["delta"] < 0, "TF-IDF", "tie"))
    expl = []
    for _, r in comp.iterrows():
        if r["winner"] == "AE":
            expl.append(f"AE improves rank {int(r['hired_rank_tfidf'])} -> {int(r['hired_rank_ae'])}; smoother latent matching helped.")
        elif r["winner"] == "TF-IDF":
            expl.append(f"AE hurts rank {int(r['hired_rank_tfidf'])} -> {int(r['hired_rank_ae'])}; a rare must-have is likely smoothed away.")
        else:
            expl.append("Same rank under both representations.")
    comp["explanation"] = expl
    comp.to_csv(cfg.tables / "ae_vs_tfidf.csv", index=False)

    # Recommendations (evidence is filled with actual current counts).
    m3 = df[df["model"] == "M3"]
    n_miss_ae = int((m3["hired_rank"] > 10).sum())
    cat_counts = m3["primary_failure_category"].value_counts().to_dict()
    n_cov_incomplete = int((m3["must_have_coverage_hired"] < 1.0).sum())
    m5 = rankings[rankings["model"] == "M5"].sort_values("rank")
    m5_rank = {
        jd: int(m5[(m5["jd_id"] == jd) & (m5["is_relevant"] == 1)]["rank"].iloc[0])
        for jd in m3["jd_id"]
    }
    n_m5_changes = int(sum(1 for _, r in m3.iterrows() if m5_rank.get(r["jd_id"]) != r["hired_rank"]))
    try:
        exq = pd.read_csv(cfg.tables / "extraction_quality_summary.csv").iloc[0]
        extraction_note = (f"macro recall {float(exq['macro_recall']):.2f}, macro F1 "
                           f"{float(exq['macro_f1']):.2f} against the Excel silver standard.")
    except Exception:
        extraction_note = "see results/tables/extraction_quality_summary.csv."
    n_unmatched_terms = int(sum(len(v) for v in unmatched_by_doc.values()))
    recs = pd.DataFrame([
        {"priority": "P0", "issue": "Must-have coverage is not applied as a filter",
         "evidence": f"{n_cov_incomplete}/10 M3 rows have hired-CV coverage below 1.0; M5 (exploratory hybrid) changes the hired-CV rank on {n_m5_changes}/10 queries.",
         "proposed_fix": "Use coverage as a hard filter or calibrate the hybrid weight on more labels.",
         "expected_impact": "Likely improves precision for rare must-haves; hypothesis only.", "effort": "Low", "owner": "ML"},
        {"priority": "P0", "issue": "Extraction misses skills present in source text",
         "evidence": f"CV extraction {extraction_note} Raw JD terms absent from the LLM output are logged per query.",
         "proposed_fix": "Evidence-span prompting plus a second pass with a stronger local model.",
         "expected_impact": "Better recall on rare skills; must be measured on new labels.", "effort": "Medium", "owner": "ML"},
        {"priority": "P1", "issue": "Keyword merge errors split or blur must-haves",
         "evidence": f"{n_unmatched_terms} held-out keyword mentions could not be mapped to a training cluster; see heldout_keyword_coverage.csv.",
         "proposed_fix": "Domain ontology + manual review of borderline merges.",
         "expected_impact": "Cleaner cluster coverage, fewer false negatives.", "effort": "Medium", "owner": "Data"},
        {"priority": "P0", "issue": "No recruiter-style hard filters",
         "evidence": f"{cat_counts.get('E7_location_language_constraint', 0)}/10 M3 queries were tagged E7 (location/language constraint ignored by every model).",
         "proposed_fix": "Add country, work authorisation, language and notice-period filters.",
         "expected_impact": "Removes visibly ineligible candidates; high recruiter value.", "effort": "Medium", "owner": "Product"},
        {"priority": "P1", "issue": "Reconstruction objective is not ranking-aware",
         "evidence": f"AE wins {int((comp['winner'] == 'AE').sum())}/10 and TF-IDF wins {int((comp['winner'] == 'TF-IDF').sum())}/10 rank comparisons.",
         "proposed_fix": "Contrastive/triplet fine-tuning or a cross-encoder re-ranker.",
         "expected_impact": "Should align the latent space with relevance; needs labels.", "effort": "High", "owner": "ML"},
        {"priority": "P1", "issue": "Only 10 labelled pairs",
         "evidence": "One query changes Recall@10 by 0.10; CIs are wide.",
         "proposed_fix": "Collect the target ~50 recruiter-approved pairs and move to learning-to-rank.",
         "expected_impact": "Enables statistically meaningful comparison.", "effort": "High", "owner": "Recruiting"},
        {"priority": "P2", "issue": "Hard lookalike profiles dominate the top ranks",
         "evidence": f"{cat_counts.get('E4_hard_negative_lookalike', 0)}/10 M3 queries were tagged E4 (a role-similar CV outranked the hired CV).",
         "proposed_fix": "Rare-skill weighting, domain features and section-aware scoring.",
         "expected_impact": "Better discrimination among similar titles.", "effort": "Medium", "owner": "ML"},
        {"priority": "P2", "issue": "Reasons are template-based, not recruiter-written",
         "evidence": "All top-10 reasons are deterministic templates with no evidence spans.",
         "proposed_fix": "Add evidence spans and calibrated confidence bands.",
         "expected_impact": "Better recruiter trust and auditability.", "effort": "Low", "owner": "Product"},
        {"priority": "P2", "issue": "Salary and notice period are out of scope",
         "evidence": f"{int(gt['jd_salary'].notna().sum())}/10 labelled JDs carry a salary field, but it is unused by every model.",
         "proposed_fix": "Ingest structured salary/notice data and add budget-fit features.",
         "expected_impact": "Closer to the brief's recruiter priorities.", "effort": "Medium", "owner": "Product"},
        {"priority": "P1", "issue": "Small candidate pool limits realism",
         "evidence": f"{int(len(meta))} documents and {len(rankings[rankings['model'] == 'M0']['jd_id'].unique())} labelled queries; random Recall@10 is 0.091.",
         "proposed_fix": "Evaluate on a larger, temporally held-out pool.",
         "expected_impact": "More credible generalisation estimates.", "effort": "High", "owner": "Data"},
    ])
    recs.to_csv(cfg.tables / "recommendations.csv", index=False)

    return {"error_df": df, "ae_vs_tfidf": comp, "recommendations": recs}


def case_study_data(cfg: Config, jd_id: str, model: str = "M3") -> dict[str, Any]:
    rankings = pd.read_csv(cfg.processed / "rankings.csv")
    top10 = pd.read_csv(cfg.processed / "top10.csv")
    coverage = pd.read_csv(cfg.processed / "must_have_coverage.csv")
    gt = pd.read_csv(cfg.processed / "ground_truth.csv")
    clusters = pd.read_csv(cfg.processed / "keyword_clusters.csv")
    counts = sparse.load_npz(cfg.processed / "document_keyword_counts.npz")
    index = pd.read_csv(cfg.processed / "document_matrix_index.csv")
    extracted = load_extractions(cfg)
    row_of = {d: i for i, d in enumerate(index["doc_id"])}
    jd_row = gt[gt["jd_id"] == jd_id].iloc[0]
    kws = list(extracted.get(jd_id, {}).get("must_have_keywords") or [])[:10]
    from .keywords import map_term
    import pickle
    with (cfg.models / "cluster_model.pkl").open("rb") as fh:
        blob = pickle.load(fh)
    term_to_cluster = json.loads((cfg.processed / "keyword_to_cluster.json").read_text())
    cids = []
    for kw in kws:
        cid, _ = map_term(kw, term_to_cluster, blob["centroids"], blob.get("vectorizer"), blob.get("svd"),
                          float(cfg.get("clustering.unmatched_cosine_threshold", 0.45)))
        cids.append(cid)
    t10 = top10[(top10["model"] == model) & (top10["jd_id"] == jd_id)].sort_values("rank")
    matrix_rows = []
    for _, r in t10.iterrows():
        row = counts.getrow(row_of[r["cv_id"]]).toarray().ravel()
        entry = {
            "rank": int(r["rank"]), "cv_id": r["cv_id"], "title": r["cv_title"],
            "company": r["cv_company"], "score": r["score"], "is_relevant": int(r["is_relevant"]),
            "coverage": float(coverage[(coverage["jd_id"] == jd_id) & (coverage["cv_id"] == r["cv_id"])]["coverage"].iloc[0]),
        }
        for kw, cid in zip(kws, cids):
            entry[kw] = float(row[cid]) if cid is not None else 0.0
        matrix_rows.append(entry)
    df = pd.DataFrame(matrix_rows)
    df.to_csv(cfg.tables / f"case_study_{jd_id}.csv", index=False)
    return {"jd": jd_row.to_dict(), "keywords": kws, "cids": cids, "table": df, "top10": t10}


CATEGORY_TEXT = {
    "none": "No failure: the hired CV was retrieved in the top 10.",
    "E1_extraction_missing_skill": "A skill present in the source document was missing from the LLM extraction.",
    "E2_keyword_merge_error": "Keyword merging or cluster coverage lost a discriminative must-have.",
    "E3_representation_loss": "The AE bottleneck smoothed away a discriminative must-have that TF-IDF kept.",
    "E4_hard_negative_lookalike": "A role-similar CV outranked the hired CV.",
    "E5_seniority_years_mismatch": "Seniority or years of experience was the main discriminator.",
    "E6_domain_industry_mismatch": "Industry/domain background was the missing signal.",
    "E7_location_language_constraint": "A hard location, citizenship or language constraint is not modelled.",
    "E8_label_noise_or_recruiter_context": "The hired person is not clearly strongest on paper; recruiter context likely mattered.",
    "E9_duplicate_or_pool_artifact": "Near-duplicate profiles or a pool artefact crowded the ranking.",
    "E10_other": "Other; see the comment.",
}


def write_error_markdown(cfg: Config, analysis: dict, metrics: dict) -> str:
    err = analysis["error_df"]
    comp = analysis["ae_vs_tfidf"]
    recs = analysis["recommendations"]
    pq = metrics["per_query"]
    summary = metrics["summary"]
    m3 = summary[summary["model"] == "M3"].iloc[0]
    lines = []
    lines.append("# Error analysis — AI CV matching baseline\n")
    lines.append("## Headline\n")
    lines.append(
        f"The pre-registered primary model (M3, AE-32 cosine) retrieved the hired CV in the top 10 for "
        f"**{int(m3['hits'])}/10 queries** (Recall@10 = {m3['recall_at_10']:.2f}; Wilson 95% CI "
        f"{m3['wilson_ci_low']:.2f}–{m3['wilson_ci_high']:.2f}). Hit@1 was {m3['hit_at_1']:.2f}, Hit@5 "
        f"{m3['hit_at_5']:.2f}, MRR@10 {m3['mrr_at_10']:.3f}. With one relevant CV per JD, Precision@10 is at most "
        f"0.1; it is reported separately and is not recall.\n")
    lines.append("All 10 labelled JDs are reviewed below. Failure codes are assigned in `data/processed/error_analysis.csv` "
                 "with the evidence that triggered them. Categories are diagnostic, not mutually exclusive.\n")

    lines.append("## Per-query review\n")
    for jd_id in sorted(err["jd_id"].unique()):
        e = err[(err["jd_id"] == jd_id) & (err["model"] == "M3")].iloc[0]
        q = pq[(pq["jd_id"] == jd_id) & (pq["model"] == "M3")].iloc[0]
        lines.append(f"### {jd_id} — {e['jd_title']} ({e['role_family']})\n")
        lines.append(
            f"The hired CV **{e['hired_cv_id']}** ranked **#{int(e['hired_rank'])}** under M3 "
            f"(score {e['hired_score']:.3f}); the top-1 was {e['top1_cv_id']} — {e['top1_title']} "
            f"(score {e['top1_score']:.3f}). Must-have coverage was {e['must_have_coverage_hired']} for the hired CV "
            f"vs {e['must_have_coverage_top1']} for the top-1; the hired CV was missing "
            f"{e['missing_must_haves_hired'] or 'no extracted must-haves'}.\n")
        lines.append(f"**Primary category:** `{e['primary_failure_category']}` — {CATEGORY_TEXT.get(e['primary_failure_category'], '')} "
                     f"{e['comment']}\n")
        if e["secondary_failure_category"]:
            lines.append(f"**Secondary signal:** `{e['secondary_failure_category']}`.\n")
        if e["hired_rank"] <= 10:
            improve = ("The representation already retrieves the pair; the next gain would come from reordering "
                       "within the top 10 using must-have coverage or seniority.")
        elif e["primary_failure_category"] == "E7_location_language_constraint":
            improve = ("Adding a country/work-authorisation/language filter before ranking is the single highest-value "
                       "change for this query; the baseline deliberately does not model it.")
        elif e["primary_failure_category"] == "E4_hard_negative_lookalike":
            improve = ("Down-weighting generic role vocabulary and up-weighting rare must-have clusters is the most direct "
                       "fix; a domain feature would also help separate lookalikes.")
        elif e["primary_failure_category"] == "E3_representation_loss":
            improve = ("Keeping a sparse TF-IDF score alongside the AE (or using a ranking-aware objective) would recover "
                       "the rare skill that the bottleneck smoothed away.")
        elif e["primary_failure_category"] == "E5_seniority_years_mismatch":
            improve = ("Adding years/seniority as an explicit feature or filter would address the observed gap.")
        else:
            improve = ("A larger labelled set is needed before changing the model; the current evidence is not enough to "
                       "separate representation error from recruiter context.")
        lines.append(f"**What would most improve this query:** {improve}\n")

    lines.append("## AE vs TF-IDF\n")
    ae_wins = comp[comp["winner"] == "AE"]
    tf_wins = comp[comp["winner"] == "TF-IDF"]
    lines.append(
        f"The AE improved the hired-CV rank on {len(ae_wins)}/10 queries, TF-IDF was better on {len(tf_wins)}/10, "
        f"and {10 - len(ae_wins) - len(tf_wins)} were ties. The AE is therefore a mixed result, not a uniform win. "
        f"Where the AE loses, the usual pattern is a rare must-have being averaged into a broad role-family direction; "
        f"where it wins, the smoother latent space appears to bridge vocabulary that TF-IDF treats as unrelated.\n")
    lines.append(comp[["jd_id", "hired_rank_tfidf", "hired_rank_ae", "delta", "winner", "explanation"]]
                 .to_markdown(index=False) + "\n")

    lines.append("## Extraction and merge quality\n")
    lines.append(
        "Extraction quality is measured only as a silver standard against the Excel `Skills` column for the 100 pool CVs; "
        "the 10 hired CVs have no Excel row, so their extraction cannot be scored the same way. The macro F1 and worst "
        "documents are in `results/tables/extraction_quality_summary.csv` and `data/processed/extraction_quality.csv`. "
        "Held-out keyword coverage (the share of GT JD/hired-CV keywords that map to a training cluster) is reported in "
        "`data/processed/heldout_keyword_coverage.csv`. Unmatched keywords are a real failure mode: if a rare must-have has "
        "no cluster, no candidate can earn coverage for it.\n")

    lines.append("## Recommendations\n")
    lines.append(recs[["priority", "issue", "evidence", "proposed_fix", "expected_impact", "effort"]]
                 .to_markdown(index=False) + "\n")

    lines.append("## Limitations\n")
    lines.extend([
        "- **n = 10 labelled queries, one relevant CV each.** This is a sparse-label baseline; one query changes Recall@10 by 0.10 and the confidence intervals are wide. No statistical significance is claimed.",
        "- **The hired candidate is not necessarily the best on paper.** Hiring includes interview performance, salary, availability and recruiter judgement, none of which is observed.",
        "- **Salary, notice period, location, citizenship and language filters are not modelled.** E7 cases show this directly; the omission is deliberate and is not hidden in the metrics.",
        "- **LLM extraction is noisy and the source PDFs are partly truncated with `…`.** A missing mention is not proof that a candidate lacks a skill.",
        "- **Keyword clustering is heuristic.** Some skills are over- or under-merged even after curated aliases and guardrails.",
        "- **The AE is trained on ~148 documents and optimises reconstruction, not ranking.** A low reconstruction error does not imply good retrieval.",
        "- **Held-out keyword coverage can disadvantage rare skills.** Unmatched keywords are logged but cannot contribute to the score.",
        "- **The 110-CV pool is small and may not represent the real candidate population.** Random Recall@10 is 0.091, so the absolute metric is coarse.",
        "- **The 50 unlabelled JDs are not ground truth.** They are used only for training/context after near-duplicate removal.",
        "- **No significance tests are possible with this sample size.** Model differences are descriptive.",
    ])
    text = "\n".join(lines) + "\n"
    (cfg.group_dir / "results" / "error_analysis.md").write_text(text, encoding="utf-8")
    return text
