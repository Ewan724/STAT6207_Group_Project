"""Step 11: HTML -> PDF via headless Microsoft Edge (Chromium).

WeasyPrint is installed but Edge gives better fidelity for the CSS grid/flex
layout used in the report, and needs no extra native dependencies.
"""
import os
import shutil
import subprocess
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EDGE = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def find_browser():
    for p in EDGE:
        if os.path.exists(p):
            return p
    for name in ("msedge", "chrome", "chromium"):
        p = shutil.which(name)
        if p:
            return p
    return None


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "A4_report.html")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "A4_report.pdf")
    if not os.path.exists(src):
        raise SystemExit(f"missing {src}")
    browser = find_browser()
    if not browser:
        raise SystemExit("no Chromium-family browser found")

    profile = tempfile.mkdtemp(prefix="edge_pdf_")
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile}",
        "--virtual-time-budget=12000",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={os.path.abspath(dst)}",
        "--print-to-pdf-no-header",
        "file:///" + os.path.abspath(src).replace(os.sep, "/"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not os.path.exists(dst):
        raise SystemExit("PDF not produced:\n" + (r.stderr or r.stdout or "")[:3000])
    shutil.rmtree(profile, ignore_errors=True)
    print(f"PDF written: {dst} ({os.path.getsize(dst):,} bytes)")


if __name__ == "__main__":
    main()
