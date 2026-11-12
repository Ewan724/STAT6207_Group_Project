import json
import re

cvs = json.load(open("data/cvs_raw.json", encoding="utf-8"))
for c in cvs:
    lines = [l.strip() for l in c["text"].split("\n") if l.strip()]
    title = lines[2] if len(lines) > 2 else ""
    org = lines[3] if len(lines) > 3 else ""
    # skills block
    try:
        i = lines.index("Skills")
        skills = " ".join(lines[i + 1:i + 4])
    except ValueError:
        skills = ""
    skills = re.sub(r"\s+", " ", skills)[:260]
    # first work line
    exp = ""
    if "Work experience" in lines:
        j = lines.index("Work experience")
        exp = " / ".join(lines[j + 1:j + 3])[:90]
    print(f"{c['cv_id']} | {title[:52]} | {org[:46]}")
    print(f"    SK: {skills}")
    print(f"    EX: {exp}")
