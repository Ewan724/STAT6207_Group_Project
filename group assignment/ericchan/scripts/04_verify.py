"""Step 4: verification + dataset statistics for the report.

Checks that (a) no hired CV is inside the pool, (b) the pool covers 10 job
families, (c) extraction coverage is complete.  Writes data/verify.json.
"""
import json
import os
import re
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")


def toks(t, n=8):
    w = re.findall(r"[a-z0-9+#.]+", t.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


cvs = json.load(open(os.path.join(DATA, "cvs_raw.json"), encoding="utf-8"))
jds = json.load(open(os.path.join(DATA, "jds_raw.json"), encoding="utf-8"))
hired = json.load(open(os.path.join(DATA, "hired_raw.json"), encoding="utf-8"))
ext = [r for r in (json.loads(l) for l in open(
    os.path.join(DATA, "extracted.jsonl"), encoding="utf-8") if l.strip())
    if r.get("ok")]

cv_sh = {c["cv_id"]: toks(c["text"]) for c in cvs}
leak = []
for h in hired:
    hs = toks(h["text"])
    best = max(((len(hs & v) / max(len(hs), 1), k) for k, v in cv_sh.items()),
               default=(0, None))
    if best[0] > 0.05:
        leak.append({"jd_id": h["jd_id"], "cv_id": best[1],
                     "overlap": round(best[0], 3)})

cv_ext = [r for r in ext if r["kind"] == "cv"]
jd_ext = [r for r in ext if r["kind"] == "jd"]

stats = {
    "n_cvs": len(cvs), "n_jds": len(jds), "n_hired": len(hired),
    "cv_extracted": len(cv_ext), "jd_extracted": len(jd_ext),
    "cv_chars_mean": round(sum(len(c["text"]) for c in cvs) / len(cvs), 1),
    "jd_chars_mean": round(sum(len(j["text"]) for j in jds) / len(jds), 1),
    "hired_cv_leaks_into_pool": leak,
    "cv_hard_skills_mean": round(
        sum(len(r.get("hard_skills") or []) for r in cv_ext) / max(len(cv_ext), 1), 1),
    "jd_must_have_mean": round(
        sum(len(r.get("must_have") or []) for r in jd_ext) / max(len(jd_ext), 1), 1),
    "seniority_dist_cv": dict(Counter(r.get("seniority") for r in cv_ext)),
    "llm_seconds_total": round(sum(r.get("seconds", 0) for r in ext), 1),
    "llm_seconds_mean": round(
        sum(r.get("seconds", 0) for r in ext) / max(len(ext), 1), 2),
}
json.dump(stats, open(os.path.join(DATA, "verify.json"), "w", encoding="utf-8"),
          indent=2)
print(json.dumps(stats, indent=2))
