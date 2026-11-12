"""Shared Ollama client for the STAT6207 A4 CV<->JD matching project."""
import json
import os
import time
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"


def chat(prompt, system=None, model=MODEL, num_predict=700, num_ctx=4096,
         temperature=0.0, seed=42, json_mode=True, think=False, retries=4):
    """Single non-streaming chat call. Returns the assistant text."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "seed": seed,
                    "num_predict": num_predict, "num_ctx": num_ctx,
                    "top_p": 0.9, "repeat_penalty": 1.05},
        "messages": msgs,
    }
    if json_mode:
        payload["format"] = "json"
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{HOST}/api/chat", data=data,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                out = json.load(r)
            return out["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"ollama chat failed after {retries} tries: {last}")


def parse_json(text):
    """Tolerant JSON parse: strips code fences and trailing prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.lower().startswith("json"):
            t = t[4:]
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j != -1:
        t = t[i:j + 1]
    return json.loads(t)


def embed(texts, model=EMBED_MODEL, batch=32):
    """Return a list of embedding vectors for `texts`."""
    vecs = []
    for k in range(0, len(texts), batch):
        chunk = texts[k:k + batch]
        req = urllib.request.Request(
            f"{HOST}/api/embed",
            data=json.dumps({"model": model, "input": chunk}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as r:
            vecs.extend(json.load(r)["embeddings"])
    return vecs


def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
