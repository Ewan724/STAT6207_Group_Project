import hashlib
import json
import re
from pathlib import Path
import pandas as pd

CV_HEADER_RE = re.compile(r"Anonymised candidate profile.*?Second Talent", re.IGNORECASE)
GT_HEADER_RE = re.compile(r"Anonymised example.*?Second Talent", re.IGNORECASE)
CV_PAGE_RE = re.compile(r"^CV-\d{3}\s*·\s*page\s*\d+\s*$", re.MULTILINE)
P_PAGE_RE = re.compile(r"^P\d{2}\s*·\s*page\s*\d+\s*$", re.MULTILINE)
P_JD_HEADER_RE = re.compile(r"^P\d{2}\s*·\s*JOB DESCRIPTION\s*·.*$", re.MULTILINE)
P_CV_HEADER_RE = re.compile(r"^P\d{2}\s*·\s*CV OF THE HIRED CANDIDATE\s*$", re.MULTILINE)
HIRED_FOR_RE = re.compile(r"^Hired for:.*$", re.MULTILINE)

MARKER = "CV OF THE HIRED CANDIDATE"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(text):
    if text is None:
        return ""
    # normalize bullets
    text = text.replace("●", "-").replace("•", "-").replace("�", "-")
    # remove headers/footers
    text = CV_HEADER_RE.sub("", text)
    text = GT_HEADER_RE.sub("", text)
    text = CV_PAGE_RE.sub("", text)
    text = P_PAGE_RE.sub("", text)
    text = P_JD_HEADER_RE.sub("", text)
    # keep marker removal separate in split logic
    # collapse spaces/tabs but preserve newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_cv_text(text):
    text = normalize_text(text)
    text = HIRED_FOR_RE.sub("", text)
    text = P_CV_HEADER_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_jd_text(text):
    text = normalize_text(text)
    # remove marker if accidentally included
    text = text.replace(MARKER, "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(path):
    import pymupdf
    doc = pymupdf.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    text = "\n".join(pages)
    return text, len(pages)


def split_gt_pdf(text):
    # find exact marker
    idx = text.find(MARKER)
    if idx == -1:
        raise ValueError(f"Marker '{MARKER}' not found")
    jd_raw = text[:idx]
    cv_raw = text[idx + len(MARKER):]
    return clean_jd_text(jd_raw), clean_cv_text(cv_raw)


def load_excel(xlsx_path):
    xls = pd.ExcelFile(xlsx_path)
    jd_df = pd.read_excel(xls, "JD Examples")
    cv_df = pd.read_excel(xls, "CV Examples")
    return jd_df, cv_df


def mask_pii(text):
    if not text:
        return text
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"\+?\d[\d\s\-()]{7,}\d", "[PHONE]", text)
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    return text
