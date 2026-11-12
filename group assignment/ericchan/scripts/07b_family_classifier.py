"""Step 7b: LLM-based job-family classifier as a deployable first-stage filter.

The problem with hybrid_family
    Step 7's `hybrid_family` system filters candidates by hand-assigned job
    family and reaches Recall@10 = 0.89.  But the ground truth was built with
    that same family map, so the filter is partly circular: it encodes the
    labelling rule rather than discovering it.  Reporting 0.89 as the headline
    number would be misleading.

What this script does instead
    It asks qwen3:8b to assign each CV to one of the ten job families using only
    the CV text, with no access to the family map.  That gives an INDEPENDENT
    family classifier.  We then measure how well it agrees with the hand labels
    and how the filter performs when it uses the predicted family instead of the
    true one.  That number is the honest deployable figure.

    The classifier is a single short call per CV with a closed label set, which
    is the kind of task an 8B model handles well.

Outputs: data/cv_family_pred.json, data/family_eval.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "cv_family_pred.json")

FAMILIES = {
    "finance": "corporate finance, accounting, audit, tax, financial control",
    "sales": "sales, business development, account management, commercial",
    "marketing": "marketing, growth, brand, communications, digital marketing",
    "data": "data engineering, data science, analytics, machine learning, AI",
    "product": "product management, product ownership, product strategy",
    "design": "UI/UX design, visual design, graphic design, creative",
    "engineering_sw": "software engineering, backend/frontend, DevOps, QA, IT",
    "engineering_hw": "hardware, semiconductor, field service, embedded, firmware",
    "operations": "operations, HR, administration, supply chain, warehouse, PMO",
}

SYSTEM = ("You classify CVs into job families. Return strict JSON only.")

PROMPT = """Assign this CV to exactly one job family.

Families:
{fams}

Rules:
- Judge by the WORK HISTORY, not by the most technical word in the skills list.
- A person whose titles are marketing roles belongs to "marketing" even if they
  list SQL and Python as tools.
- A person whose titles are engineering roles belongs to a software or hardware
  engineering family even if they mention marketing or sales.
- "operations" covers HR, admin, supply chain, warehouse and PMO.

CV:
\"\"\"{cv}\"\"\"

Return ONLY: {{"family": "<one of the family keys>", "why": "<max 12 words>"}}
"""


def main():
    cvs = {c["cv_id"]: c for c in json.load(
        open(os.path.join(DATA, "cvs_raw.json"), encoding="utf-8"))}
    fams = "\n".join(f"- {k}: {v}" for k, v in FAMILIES.items())
    done = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}

    todo = [c for c in sorted(cvs) if c not in done]
    print(f"{len(todo)} CVs to classify ({len(done)} cached)", flush=True)
    for i, cid in enumerate(todo, 1):
        prompt = PROMPT.format(fams=fams, cv=cvs[cid]["text"][:2400])
        try:
            raw = llm.chat(prompt, system=SYSTEM, num_predict=80, num_ctx=4096)
            rec = llm.parse_json(raw)
            fam = str(rec.get("family", "")).strip().lower()
            if fam not in FAMILIES:
                fam = "unknown"
            why = str(rec.get("why", ""))[:80]
        except Exception as e:  # noqa: BLE001
            fam, why = "unknown", f"error {e!r}"[:80]
        done[cid] = {"family": fam, "why": why}
        print(f"[{i}/{len(todo)}] {cid} -> {fam}", flush=True)
        if i % 10 == 0:
            json.dump(done, open(OUT, "w", encoding="utf-8"), indent=1)

    json.dump(done, open(OUT, "w", encoding="utf-8"), indent=1)

    truth = json.load(open(os.path.join(DATA, "cv_family.json"),
                           encoding="utf-8"))
    pairs = [(c, truth[c], done[c]["family"]) for c in truth
             if done.get(c, {}).get("family")]
    acc = sum(1 for _, t, p in pairs if t == p) / max(len(pairs), 1)
    # confusion at the family level
    conf = {}
    for _, t, p in pairs:
        conf.setdefault(t, {}).setdefault(p, 0)
        conf[t][p] += 1
    print(f"\nfamily accuracy: {acc:.3f} over {len(pairs)} CVs")
    for t in sorted(conf):
        row = conf[t]
        top = sorted(row.items(), key=lambda x: -x[1])[:3]
        print(f"  {t:15s} -> {top}")

    json.dump({"accuracy": round(acc, 4), "n": len(pairs),
               "confusion": conf}, open(os.path.join(DATA, "family_eval.json"),
                                        "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
