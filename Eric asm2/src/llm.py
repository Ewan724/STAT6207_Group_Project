import hashlib
import json
import os
import re
import time
from pathlib import Path
import requests

GENERIC_STOPWORDS = {"experience","experiences","skills","skill","work","team","ability","strong","good","excellent","responsible","develop","manage","build","support","knowledge","understanding"}


def prompt_hash(system_prompt, user_prompt, model):
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\0")
    h.update(system_prompt.encode())
    h.update(b"\0")
    h.update(user_prompt.encode())
    return h.hexdigest()


def call_ollama(base_url, model, system_prompt, user_prompt, options, timeout=300, think=False):
    url = base_url.rstrip('/') + '/api/chat'
    base_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": dict(options),
    }
    # try combinations: prefer no-thinking and plain json content
    attempts = []
    p1 = json.loads(json.dumps(base_payload))
    p1["think"] = False
    attempts.append(p1)
    p2 = json.loads(json.dumps(base_payload))
    attempts.append(p2)
    last_err = None
    for payload in attempts:
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code >= 400:
                last_err = RuntimeError(f"{r.status_code} {r.reason}: {r.text[:300]}")
                continue
            data = r.json()
            msg = data.get("message", {})
            content = msg.get("content", "") or ""
            if not content.strip():
                # some models put answer in thinking; try to extract json from it
                thinking = msg.get("thinking", "") or ""
                content = thinking
            return content, data
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError("ollama call failed")


def extract_json(text):
    if text is None:
        return None
    t = text.strip()
    # strip fences
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    # try direct
    try:
        return json.loads(t, strict=False)
    except Exception:
        pass
    # find balanced-ish json objects by trying from each {
    starts = [m.start() for m in re.finditer(r"\{", t)]
    for s in starts:
        # try to parse progressively to end
        for e in range(len(t), s+1, -1):
            if t[e-1] != '}':
                continue
            chunk = t[s:e]
            try:
                return json.loads(chunk, strict=False)
            except Exception:
                continue
    return None


def normalize_keyword(kw):
    if kw is None: return None
    s = str(kw).strip()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" -•,.;:/")
    return s


def valid_keyword_phrase(kw):
    if not kw: return False
    words = [w for w in re.split(r"\s+", kw) if w]
    if len(words) < 1 or len(words) > 5: return False
    low = kw.lower()
    if low in GENERIC_STOPWORDS: return False
    if len(kw) < 2: return False
    if any(p in kw for p in [". ", "!", "?"]): return False
    return True


def keyword_in_text(kw, text, aliases=None):
    aliases = aliases or {}
    low_text = (text or "").lower()
    k = kw.lower().strip()
    if not k: return False
    if k in low_text:
        return True
    abbr = {
        'gcp':'google cloud','aws':'amazon web services','k8s':'kubernetes','js':'javascript','ts':'typescript',
        'ml':'machine learning','ai':'artificial intelligence','nlp':'natural language processing','sql':'structured query language',
        'ui':'user interface','ux':'user experience','ci/cd':'continuous integration','cicd':'ci/cd','ms excel':'excel','postgres':'postgresql',
        'nodejs':'node.js','reactjs':'react','vuejs':'vue','nextjs':'next.js','powerbi':'power bi','.net':'dotnet','c#':'c#','c++':'c++',
        'p&l':'profit and loss','fp&a':'financial planning'
    }
    if k in abbr and abbr[k] in low_text:
        return True
    toks = [t for t in re.split(r"[^a-z0-9+#./&]+", k) if t]
    if toks and all(t in low_text for t in toks):
        return True
    if k.endswith('s') and len(k)>3 and k[:-1] in low_text:
        return True
    return False
