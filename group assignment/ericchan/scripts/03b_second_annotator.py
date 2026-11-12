"""Step 3b: independent second annotation -> inter-annotator agreement.

Why a second annotator at all
    The primary labels in step 3 are LLM-generated.  That is the biggest threat
    to validity in this project, and asserting "the labels are correct" is not a
    defence.  What we can do is measure how stable the labels are under a
    *deliberately different* protocol, and report the agreement.  A high kappa
    does not prove the labels match a human recruiter, but a low one would prove
    they do not even reproduce, and that would invalidate every number in §7.

How this differs from step 3
    * different rubric wording, written independently of the primary prompt
    * a different sample: all same-profession pairs plus a random 100 pairs
      across professions (so the 0 end of the scale is exercised too)
    * a different seed and a lower token budget, so the calls are not identical
    * the profession gate is kept (it is part of the task definition, not a
      tuning knob) but expressed as a plain question

Reported in data/agreement.json:
    exact agreement, agreement within +-1 grade, Cohen's quadratic-weighted
    kappa on the 0-3 scale, and Cohen's kappa on the binary relevant/not cut.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "ground_truth_v2.jsonl")

SYSTEM_V2 = (
    "You are a hiring manager reviewing applications for one vacancy. You "
    "decide which candidates deserve a first interview. Answer with JSON only."
)

PROMPT_V2 = """Read the vacancy and the candidate CV, then decide how this
candidate compares with the person the company hired for this vacancy last time.

VACANCY
\"\"\"
{jd}
\"\"\"

PERSON HIRED (the bar the company actually applied)
\"\"\"
{hire}
\"\"\"

CANDIDATE
\"\"\"
{cv}
\"\"\"

First decide whether the candidate works in the same line of work as the
vacancy. If not, the answer is 0. If they do, grade them against the hired
person:

0 = different line of work
1 = same line of work, but clearly weaker than the person hired
2 = same line of work, close enough to be worth an interview
3 = same line of work, as strong as or stronger than the person hired

Return ONLY: {{"score": <0|1|2|3>, "reason": "<max 15 words>"}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=100)
    args = ap.parse_args()

    cvs = {c["cv_id"]: c for c in json.load(
        open(os.path.join(DATA, "cvs_raw.json"), encoding="utf-8"))}
    jds = {j["jd_id"]: j for j in json.load(
        open(os.path.join(DATA, "jds_raw.json"), encoding="utf-8"))}
    hired = {h["jd_id"]: h for h in json.load(
        open(os.path.join(DATA, "hired_raw.json"), encoding="utf-8"))}
    gt = [json.loads(l) for l in open(os.path.join(DATA, "ground_truth.jsonl"),
                                      encoding="utf-8") if l.strip()]
    gt = [g for g in gt if g.get("score", -1) >= 0]
    if not gt:
        raise SystemExit("run 03_ground_truth.py first")

    same = [(g["jd_id"], g["cv_id"]) for g in gt
            if g["jd_family"] == g["cv_family"]]
    cross = [(g["jd_id"], g["cv_id"]) for g in gt
             if g["jd_family"] != g["cv_family"]]
    rng = random.Random(7)
    n_cross = min(args.n_random, len(cross))
    sample = same + rng.sample(cross, n_cross)
    print(f"sampling {len(same)} same-profession + {n_cross} cross = "
          f"{len(sample)} pairs", flush=True)

    done = {(r["jd_id"], r["cv_id"]) for r in llm.read_jsonl(OUT)
            if r.get("score", -1) >= 0}
    todo = [(j, c) for j, c in sample if (j, c) not in done]
    print(f"{len(todo)} to annotate", flush=True)

    for i, (jd_id, cv_id) in enumerate(todo, 1):
        prompt = PROMPT_V2.format(jd=jds[jd_id]["text"][:2200],
                                  hire=hired[jd_id]["text"][:2000],
                                  cv=cvs[cv_id]["text"][:2200])
        raw, score, reason = "", -1, ""
        try:
            raw = llm.chat(prompt, system=SYSTEM_V2, num_predict=200,
                           num_ctx=8192, temperature=0.0, seed=7)
            rec = llm.parse_json(raw)
            score = max(0, min(3, int(rec.get("score", 0))))
            reason = str(rec.get("reason", ""))[:160]
        except Exception as e:  # noqa: BLE001
            reason = f"error {e!r}"[:160]
        llm.append_jsonl(OUT, {"jd_id": jd_id, "cv_id": cv_id, "score": score,
                               "reason": reason})
        print(f"[{i}/{len(todo)}] {jd_id} x {cv_id} -> {score}", flush=True)


if __name__ == "__main__":
    main()
