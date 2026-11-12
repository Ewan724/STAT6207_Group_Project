"""Ground truth construction - v4, deterministic gate + LLM graded fit.

The ground-truth problem
    Each of the 10 source PDFs contains a JD plus the CV of the person actually
    hired.  That hired CV is NOT in the 100-CV pool: zero exact overlap, zero
    8-gram shingle overlap and no shared employer string (scripts/04_verify.py).
    The pool is a *disjoint* candidate set that mirrors the same ten job
    families.  So retrieval cannot be scored by "did we find the hired CV", and
    explicit relevance labels must be built for all 10 x 100 = 1,000 pairs.

Why there are four versions, and what each one taught us
    v1  Absolute 0-3 fit score.  The judge returned grade 2 with reasons like
        "but lacks AI focus and customer collaboration" -- the label contradicted
        its own justification.  Lesson: an 8B model cannot calibrate an absolute
        rubric.
    v2  Added a self-consistency cap using "the first six must-haves" as core
        requirements.  Over-corrected: for P01 the first six were
        "manufacturing costing / management reporting / internal controls", so
        every genuine finance CV scored 1 and the relevant set collapsed.
        Lesson: an arbitrary requirement list is not a bar.
    v3  Two-stage: an LLM profession gate, then a reference-anchored comparison
        against the hired CV.  The comparison works well, but the gate does not:
        it graded a Principal Product Manager as "engineering" and left two JDs
        with an empty relevant set, which makes Recall undefined.
        Lesson: do not ask a small model to make a categorical decision that a
        deterministic rule can make.

    v4 (this file) splits the two decisions and gives each to the right tool:

    GATE (deterministic, in code)
        A pair is in scope iff the CV's hand-assigned job family equals the JD's
        job family.  The families are read off the ten JD titles and the 100 CV
        titles (see CV_FAMILY, taken from data/cv_family.json, and
        data/profiles.txt for the evidence).  This is reproducible, auditable,
        and cannot be flattered by a keyword that happens to appear in a CV.
        It is coarse on purpose: it separates professions, not seniority.

    GRADE (LLM, reference-anchored, within family only)
        For in-scope pairs, qwen3:8b compares the candidate with the person the
        client actually hired and grades 1-3.  A comparative judgement against a
        concrete example is something a small model does reliably; an absolute
        rubric is not.  The model sees the hired CV, the JD, the JD's must-haves
        and the candidate.

    Out-of-scope pairs are assigned 0 without an LLM call.  They are kept in the
    file so every metric is computed over the same 1,000-pair universe.

Grade semantics
    3  at or above the hired bar       -> relevant
    2  plausibly shortlistable         -> relevant
    1  in the profession, below the bar -> not relevant
    0  out of profession               -> not relevant

Outputs: data/ground_truth.jsonl, data/gt_calls.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

JD_FAMILY = {
    "P01": "finance", "P02": "sales", "P03": "marketing", "P04": "data",
    "P05": "product", "P06": "data", "P07": "marketing",
    "P08": "engineering_hw", "P09": "engineering_sw", "P10": "design",
}

SYSTEM = (
    "You are a senior technical recruiter. You compare one candidate with the "
    "person the client actually hired for the same vacancy, and you place the "
    "candidate on the client's bar. You judge only what the work history "
    "demonstrates, and you do not let a job title substitute for evidence. "
    "Return strict JSON only."
)

PROMPT = """Compare this candidate with the person the client actually hired for
this vacancy, then place the candidate relative to that bar.

=== THE VACANCY ===
Title: {title}
Level: {sen}, minimum {yrs} years, domain: {dom}

Requirements the client emphasised:
{reqs}

Job description excerpt:
\"\"\"{jd}\"\"\"

=== THE PERSON THE CLIENT HIRED (this is the bar) ===
{hire}

=== THIS CANDIDATE ===
{cv}

=== TASK ===
Both people are in the same line of work, so do not judge profession. Judge
RELATIVE STRENGTH for this specific vacancy, using the reference above as the
standard:

  3 = at or above the bar. Evidence on the key requirements is comparable to or
      stronger than the reference. Would shortlist first.
  2 = below the bar but plausibly shortlistable. Meaningfully weaker evidence
      than the reference, yet a recruiter would still want a conversation.
  1 = clearly below the bar. Same line of work, but substantially less relevant
      evidence than the reference; would not shortlist while better candidates
      are available.

Be discriminating. Reserve 2 for people you would genuinely want to interview
for THIS vacancy, and 3 for people who would beat the person hired.

Return ONLY:
{{"score": 1|2|3,
  "evidence": ["specific things in this CV that support the grade"],
  "gap": ["specific things the reference has that this CV lacks"],
  "reason": "<max 20 words>"}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cvs = {c["cv_id"]: c for c in json.load(
        open(os.path.join(DATA, "cvs_raw.json"), encoding="utf-8"))}
    jds = {j["jd_id"]: j for j in json.load(
        open(os.path.join(DATA, "jds_raw.json"), encoding="utf-8"))}
    hired = {h["jd_id"]: h for h in json.load(
        open(os.path.join(DATA, "hired_raw.json"), encoding="utf-8"))}
    ext = {r["doc_id"]: r for r in llm.read_jsonl(
        os.path.join(DATA, "extracted.jsonl")) if r.get("ok")}
    cv_fam = json.load(open(os.path.join(DATA, "cv_family.json"),
                            encoding="utf-8"))

    out_path = os.path.join(DATA, "ground_truth.jsonl")
    if os.path.exists(out_path) and not os.path.exists(
            os.path.join(DATA, "ground_truth_v3_backup.jsonl")):
        os.replace(out_path, os.path.join(DATA, "ground_truth_v3_backup.jsonl"))
    done = {(r["jd_id"], r["cv_id"]) for r in llm.read_jsonl(out_path)
            if r.get("score", -1) >= 0}

    # ---- stage 1: deterministic gate, no LLM
    gate_rows = []
    for jd_id in sorted(jds):
        for cv_id in sorted(cvs):
            fam = cv_fam.get(cv_id, "?")
            gate_rows.append((jd_id, cv_id, fam == JD_FAMILY[jd_id], fam))
    n_in = sum(1 for r in gate_rows if r[2])
    print(f"gate: {n_in} in-scope / {len(gate_rows)} pairs "
          f"({len(gate_rows) - n_in} assigned grade 0 without an LLM call)",
          flush=True)

    # write the out-of-scope zeros immediately so the file is complete even if
    # the graded stage is interrupted
    for jd_id, cv_id, in_scope, fam in gate_rows:
        if not in_scope and (jd_id, cv_id) not in done:
            llm.append_jsonl(out_path, {
                "jd_id": jd_id, "cv_id": cv_id, "score": 0,
                "reason": "out of scope: different job family",
                "evidence": [], "gap": [],
                "jd_family": JD_FAMILY[jd_id], "cv_family": fam,
                "gated": True})
            done.add((jd_id, cv_id))

    # ---- stage 2: LLM grading of in-scope pairs
    todo = [(j, c) for j, c, ok, _ in gate_rows if ok and (j, c) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"grading {len(todo)} in-scope pairs", flush=True)

    for i, (jd_id, cv_id) in enumerate(todo, 1):
        je = ext.get(jd_id, {})
        reqs = "; ".join(str(x) for x in (je.get("must_have") or [])[:10]) or \
            "(none extracted)"
        prompt = PROMPT.format(
            title=jds[jd_id]["title"][:70],
            sen=je.get("seniority", "?"), yrs=je.get("years_required", 0),
            dom=", ".join(je.get("domains") or [])[:110] or "(unknown)",
            reqs=reqs[:400],
            jd=jds[jd_id]["text"][:1800],
            hire=hired[jd_id]["text"][:2000],
            cv=cvs[cv_id]["text"][:2000],
        )
        raw, score, reason, ev, gap = "", -1, "", [], []
        try:
            raw = llm.chat(prompt, system=SYSTEM, num_predict=300, num_ctx=8192)
            rec = llm.parse_json(raw)
            score = max(1, min(3, int(rec.get("score", 1))))
            reason = str(rec.get("reason", ""))[:200]
            ev = [str(x)[:70] for x in (rec.get("evidence") or [])][:4]
            gap = [str(x)[:70] for x in (rec.get("gap") or [])][:4]
        except Exception as e:  # noqa: BLE001
            reason = f"error {e!r}"[:200]

        llm.append_jsonl(os.path.join(DATA, "gt_calls.jsonl"), {
            "jd_id": jd_id, "cv_id": cv_id, "prompt": prompt,
            "system": SYSTEM, "raw": raw})
        llm.append_jsonl(out_path, {
            "jd_id": jd_id, "cv_id": cv_id, "score": score, "reason": reason,
            "evidence": ev, "gap": gap,
            "jd_family": JD_FAMILY[jd_id], "cv_family": cv_fam.get(cv_id, "?"),
            "gated": False})
        print(f"[{i}/{len(todo)}] {jd_id} x {cv_id} -> {score}  {reason[:56]}",
              flush=True)


if __name__ == "__main__":
    main()
