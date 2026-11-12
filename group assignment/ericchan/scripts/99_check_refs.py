"""Verify that every script path referenced in the report and README exists."""
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
refs = set()
for f in ("A4_report.html", "README.md"):
    p = os.path.join(BASE, f)
    if not os.path.exists(p):
        continue
    txt = open(p, encoding="utf-8").read()
    refs |= set(re.findall(r"scripts/([0-9A-Za-z_]+\.py)", txt))

missing = []
for r in sorted(refs):
    if not os.path.exists(os.path.join(BASE, "scripts", r)):
        missing.append(r)

on_disk = sorted(x for x in os.listdir(os.path.join(BASE, "scripts"))
                 if x.endswith(".py"))
print(f"{len(refs)} script paths referenced")
print("missing:", missing or "none")

# also accept a bare filename mention (the diagnostic table lists them without
# the scripts/ prefix)
alltext = ""
for f in ("A4_report.html", "README.md"):
    p = os.path.join(BASE, f)
    if os.path.exists(p):
        alltext += open(p, encoding="utf-8").read()
undocumented = [x for x in on_disk
                if x not in refs and x not in alltext]
print("undocumented:", undocumented or "none")
print("\nreferenced:", sorted(refs))
print("\non disk   :", on_disk)
unref = [x for x in on_disk if x not in refs]
print("\non disk but never referenced:", unref or "none")
