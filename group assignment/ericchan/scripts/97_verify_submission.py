"""Confirm the holdout work is actually present in every submitted artefact."""
import json
import os
import zipfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

checks = []

# 1. the script exists and is non-trivial
p = os.path.join(BASE, "scripts", "13_holdout_eval.py")
checks.append(("scripts/13_holdout_eval.py exists",
               os.path.exists(p), f"{os.path.getsize(p):,} bytes" if os.path.exists(p) else "MISSING"))

# 2. the holdout results file exists and has 5 folds
p = os.path.join(DATA, "holdout.json")
if os.path.exists(p):
    ho = json.load(open(p, encoding="utf-8"))
    checks.append(("data/holdout.json", True,
                   f"{ho['n_folds']} folds, systems={len(ho['holdout'])}"))
else:
    ho = None
    checks.append(("data/holdout.json", False, "MISSING"))

# 3. the report HTML contains the section
p = os.path.join(BASE, "A4_report.html")
html = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
checks.append(("HTML has §7.4 heading",
               "7.4 · Out-of-sample check" in html, ""))
checks.append(("HTML has holdout table rows",
               "holdout Recall@10" in html, ""))
checks.append(("HTML has holdout figure",
               'alt="holdout vs in-sample"' in html, "base64-embedded"))

# 4. the PDF contains the section text
try:
    import pymupdf
    d = pymupdf.open(os.path.join(BASE, "A4_report.pdf"))
    txt = "".join(pg.get_text() for pg in d)
    checks.append(("PDF has holdout section", "Out-of-sample check" in txt,
                   f"{len(d)} pages"))
    checks.append(("PDF has 0.858 holdout number", "0.858" in txt, ""))
except Exception as e:  # noqa: BLE001
    checks.append(("PDF check", False, repr(e)))

# 5. the zip contains the updated files
zp = os.path.join(BASE, "A4_submission.zip")
if os.path.exists(zp):
    z = zipfile.ZipFile(zp)
    names = z.namelist()
    checks.append(("zip has 13_holdout_eval.py",
                   "scripts/13_holdout_eval.py" in names, ""))
    checks.append(("zip has holdout.json",
                   "data/holdout.json" in names, ""))
    checks.append(("zip has holdout.png",
                   "figures/holdout.png" in names, ""))
    # does the zipped HTML contain the section?
    zh = z.read("A4_report.html").decode("utf-8")
    checks.append(("zipped HTML has §7.4",
                   "Out-of-sample check" in zh, ""))
    zpdf = z.read("A4_report.pdf")
    import io
    import pymupdf
    zd = pymupdf.open(stream=zpdf, filetype="pdf")
    zt = "".join(pg.get_text() for pg in zd)
    checks.append(("zipped PDF has holdout section",
                   "Out-of-sample check" in zt, f"{len(zd)} pages"))

print(f"{'check':45s} {'pass':>6s}  detail")
print("-" * 80)
for name, ok, detail in checks:
    print(f"{name:45s} {'OK' if ok else 'FAIL':>6s}  {detail}")

print("\n--- timestamps (newest last) ---")
for f in ("scripts/13_holdout_eval.py", "data/holdout.json",
          "A4_report.html", "A4_report.pdf", "A4_submission.zip"):
    p = os.path.join(BASE, f)
    if os.path.exists(p):
        import datetime
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(p))
        print(f"  {f:32s} {ts:%Y-%m-%d %H:%M:%S}  {os.path.getsize(p):>10,} B")
