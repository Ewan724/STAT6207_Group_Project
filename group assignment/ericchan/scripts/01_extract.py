"""Step 1: extract raw text from the PDFs into JSON.

- 100 anonymised CVs -> data/cvs_raw.json
- 10 JD PDFs, split into the JD part and the "CV OF THE HIRED CANDIDATE" part
  -> data/jds_raw.json  and  data/hired_raw.json
"""
import glob
import json
import os
import re

import pymupdf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

HEADER_RE = re.compile(
    r"Anonymised example · client and personal information removed · Second Talent\s*\n"
    r"P\d+ · page \d+\s*\n",
    re.I,
)


def pdf_pages(path):
    doc = pymupdf.open(path)
    pages = [p.get_text() for p in doc]
    doc.close()
    return pages


def clean(t):
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ---------------------------------------------------------------- CVs
cvs = []
for path in sorted(glob.glob(os.path.join(BASE, "cvs", "Anonymised CVs", "*.pdf"))):
    cid = os.path.splitext(os.path.basename(path))[0]
    pages = pdf_pages(path)
    text = "\n".join(pages)
    text = HEADER_RE.sub("", text)
    cvs.append({"cv_id": cid, "file": os.path.basename(path), "n_pages": len(pages),
                "text": clean(text)})

with open(os.path.join(DATA, "cvs_raw.json"), "w", encoding="utf-8") as f:
    json.dump(cvs, f, ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- JDs
jds, hired = [], []
for path in sorted(glob.glob(os.path.join(BASE, "jd", "JD + Hired CV examples", "*.pdf"))):
    name = os.path.splitext(os.path.basename(path))[0]
    pid = name.split(" - ")[0]
    pages = pdf_pages(path)
    text = HEADER_RE.sub("", "\n".join(pages))

    marker = "CV OF THE HIRED CANDIDATE"
    idx = text.upper().find(marker)
    if idx == -1:
        jd_text, cv_text = text, ""
    else:
        # the JD part ends where the hired-CV page starts; back up to the last
        # "Pxx · CV OF THE HIRED CANDIDATE" line
        line_start = text.rfind("\n", 0, idx)
        jd_text = text[:line_start]
        cv_text = text[line_start + 1:]
        cv_text = cv_text.split("\n", 1)[1] if "\n" in cv_text else cv_text

    title = name.split(" - ", 1)[1] if " - " in name else name
    jds.append({"jd_id": pid, "file": os.path.basename(path), "title": title.strip(),
                "n_pages": len(pages), "text": clean(jd_text)})
    hired.append({"jd_id": pid, "title": title.strip(), "text": clean(cv_text)})

with open(os.path.join(DATA, "jds_raw.json"), "w", encoding="utf-8") as f:
    json.dump(jds, f, ensure_ascii=False, indent=1)
with open(os.path.join(DATA, "hired_raw.json"), "w", encoding="utf-8") as f:
    json.dump(hired, f, ensure_ascii=False, indent=1)

print(f"CVs: {len(cvs)}  JDs: {len(jds)}  hired: {len(hired)}")
print("CV chars:", [len(c["text"]) for c in cvs][:10], "...")
print("JD chars:", [(j["jd_id"], len(j["text"])) for j in jds])
print("hired chars:", [(h["jd_id"], len(h["text"])) for h in hired])
