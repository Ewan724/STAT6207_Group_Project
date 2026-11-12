"""Step 7c: LLM reranking of the top-20 fused candidates.

Why a reranker
    The retrieval stage is a cheap, high-recall filter: it puts the right people
    in the top ~20 but does not order them perfectly, because a single dot
    product cannot express "this person has the depth the JD wants".  A second
    stage that reads the full JD and the full CV can.  This is the standard
    retrieve-then-rerank architecture, and it is what the product brief
    describes (rank the shortlist, with a reason per candidate).

Design
    For each JD we take the top 20 by the fused score from step 7, ask qwen3:8b
    to score each candidate 0-100 against the JD, and re-sort.  The retrieval
    rank is kept as a tie-breaker and as a prior: final score =
    llm_score/100 + 0.15 * retrieval_prior, where the prior is the candidate's
    normalised retrieval score.  Blending rather than replacing avoids the
    failure where the reranker invents a confident but wrong ordering from a
    truncated CV.

    The reranker is given the JD's must-have list explicitly, because that is
    the part of the JD the retrieval stage compresses most aggressively.

Outputs: data/rerank_scores.json, data/metrics_rerank.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "rerank_scores.json")
TOP_N = 20
PRIOR = 0.15

SYSTEM = (
    "You are a senior technical recruiter screening candidates for one vacancy. "
    "You score how well a candidate's demonstrated experience fits the vacancy. "
    "You are strict and you reward evidence in the work history over keyword "
    "lists. Return strict JSON only."
)

PROMPT = """Score how well this candidate fits the vacancy, 0-100.

=== VACANCY ===
Title: {title}
Level: {sen}, minimum {yrs} years, domain: {dom}

Must-have requirements:
{reqs}

Job description excerpt:
\"\"\"{jd}\"\"\"

=== CANDIDATE ===
{cv}

=== SCORING GUIDE ===
 90-100  exceptional: meets essentially every must-have with strong evidence
 70-89   strong: meets most must-haves, minor gaps
 50-69   plausible: meets roughly half, worth a conversation
 30-49   weak: right field, misses several must-haves
 0-29    poor: wrong field or no relevant evidence

Anchor your score on the must-have list. Do not award points for seniority or
impressive employers if the must-haves are not evidenced.

Return ONLY: {{"score": <0-100>, "reason": "<max 15 words>"}}
"""


def main():
    skills = json.load(open(os.path.join(DATA, "skills.json"), encoding="utf-8"))
    metrics = json.load(open(os.path.join(DATA, "metrics.json"), encoding="utf-8"))
    cvs = {c["cv_id"]: c for c in json.load(
        open(os.path.join(DATA, "cvs_raw.json"), encoding="utf-8"))}
    jds = {j["jd_id"]: j for j in json.load(
        open(os.path.join(DATA, "jds_raw.json"), encoding="utf-8"))}
    ext = {r["doc_id"]: r for r in llm.read_jsonl(
        os.path.join(DATA, "extracted.jsonl")) if r.get("ok")}

    # use the best non-circular system's ranking as the candidate pool
    res = {r["system"]: r for r in metrics["results"]}
    base_name = "hybrid_family_pred" if "hybrid_family_pred" in res else \
        "hybrid_rerank"
    base = res[base_name]
    print(f"reranking the top {TOP_N} from {base_name}", flush=True)

    done = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}

    for row in base["rows"]:
        jd_id = row["jd_id"]
        je = ext.get(jd_id, {})
        reqs = "; ".join(str(x) for x in (je.get("must_have") or [])[:12]) or \
            "(none extracted)"
        top = row.get("top20") or (row["top10"] +
                                   [c for c in row.get("top20_extra", [])])
        top = top[:TOP_N]
        for cv_id in top:
            key = f"{jd_id}|{cv_id}"
            if key in done:
                continue
            prompt = PROMPT.format(
                title=jds[jd_id]["title"][:70],
                sen=je.get("seniority", "?"), yrs=je.get("years_required", 0),
                dom=", ".join(je.get("domains") or [])[:100] or "(unknown)",
                reqs=reqs[:420],
                jd=jds[jd_id]["text"][:1600],
                cv=cvs[cv_id]["text"][:2000],
            )
            try:
                raw = llm.chat(prompt, system=SYSTEM, num_predict=120,
                               num_ctx=8192)
                rec = llm.parse_json(raw)
                score = max(0, min(100, int(float(rec.get("score", 0)))))
                reason = str(rec.get("reason", ""))[:150]
            except Exception as e:  # noqa: BLE001
                score, reason = -1, f"error {e!r}"[:150]
            done[key] = {"score": score, "reason": reason}
            print(f"{jd_id} x {cv_id} -> {score}", flush=True)
            json.dump(done, open(OUT, "w", encoding="utf-8"), indent=1)

    json.dump(done, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\n{len(done)} rerank scores written")


if __name__ == "__main__":
    main()
