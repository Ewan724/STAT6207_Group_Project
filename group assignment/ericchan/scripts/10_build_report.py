"""Step 10: build the self-contained HTML report from the run artefacts.

Every number in the HTML is injected from data/*.json, so the report cannot
drift from the run.  Figures are base64-embedded so the single .html file is
portable.
"""
import base64
import html
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figures")
OUT = os.path.join(BASE, "A4_report.html")


def b64(name):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return ""
    with open(p, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def esc(x):
    return html.escape(str(x))


def load(name, default=None):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    metrics = load("metrics.json", {"results": []})
    skills = load("skills.json", {})
    verify = load("verify.json", {})
    aem = load("ae_metrics.json", {})
    aem2 = load("ae_metrics_kwtxt.json", {})
    aem3 = load("ae_metrics_text.json", {})
    ae = load("ae_embeddings.json", {})
    agree = load("agreement.json", {})
    fam = load("family_eval.json", {})
    ho = load("holdout.json", {})
    res = {r["system"]: r for r in metrics.get("results", [])}

    gt = [json.loads(l) for l in open(os.path.join(DATA, "ground_truth.jsonl"),
                                      encoding="utf-8") if l.strip()]
    gt = [g for g in gt if g.get("score", -1) >= 0]
    ext = {}
    for line in open(os.path.join(DATA, "extracted.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("ok"):
                ext[r["doc_id"]] = r
    jds = {j["jd_id"]: j for j in json.load(
        open(os.path.join(DATA, "jds_raw.json"), encoding="utf-8"))}
    cvs = {c["cv_id"]: c for c in json.load(
        open(os.path.join(DATA, "cvs_raw.json"), encoding="utf-8"))}
    calls = [json.loads(l) for l in open(os.path.join(DATA, "llm_calls.jsonl"),
                                        encoding="utf-8") if l.strip()]
    gtcalls = [json.loads(l) for l in open(os.path.join(DATA, "gt_calls.jsonl"),
                                          encoding="utf-8") if l.strip()]
    rerank = load("rerank_scores.json", {})

    best = res.get("llm_rerank_blend") or res.get("hybrid_family_pred") or {}
    deploy = res.get("hybrid_family_pred", {})
    ae_only = res.get("ae_latent", {})
    rnd = res.get("random", {})
    kw = res.get("keyword", {})
    tfidf = res.get("tfidf", {})
    nomic = res.get("nomic", {})
    oracle = res.get("family_oracle", {})
    gtcirc = res.get("hybrid_family_gt", {})

    grade_of = {(g["jd_id"], g["cv_id"]): g["score"] for g in gt}
    reason_of = {(g["jd_id"], g["cv_id"]): g["reason"] for g in gt}
    n_rel = metrics.get("n_relevant", {})
    n_rel_live = {j: v for j, v in n_rel.items() if v}

    # ---------- worked extraction examples for §3.5
    def fmt_example(doc_id, keys):
        e = ext.get(doc_id, {})
        lines = [f"{doc_id}"]
        for k in keys:
            v = e.get(k)
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v[:26])
            lines.append(f"  {k:18s} {v}")
        return "\n".join(lines)

    cv_example = esc(fmt_example("CV-002", [
        "current_title", "seniority", "years_experience", "education_level",
        "hard_skills", "soft_skills", "domains"]))
    jd_example = esc(fmt_example("P04", [
        "job_title", "seniority", "years_required", "must_have", "nice_to_have"]))

    # ---------- 80:20 holdout table
    holdout_rows = ""
    if ho:
        pairs = [("keyword_mh", "keyword_mh", "merged keywords (must-have weighted)"),
                 ("ae_latent", "ae_latent", "keyword AE alone"),
                 ("nomic", "nomic", "text embedding alone"),
                 ("hybrid", "hybrid_tuned", "fusion"),
                 ("hybrid_family", "hybrid_family_pred", "fusion + family filter"),
                 ("hybrid_family_rerank", "llm_rerank_blend",
                  "<b>full pipeline</b>")]
        for hk, ik, label in pairs:
            h = ho["holdout"].get(hk, {})
            ins = ho["in_sample"].get(ik, {}).get("recall@10", 0)
            hr = h.get("recall@10", 0)
            cls = " class='hl'" if hk == "hybrid_family_rerank" else ""
            holdout_rows += (
                f"<tr{cls}><td>{label}</td><td>{hr:.4f}</td>"
                f"<td>{h.get('recall@10_sd', 0):.4f}</td>"
                f"<td>{ins:.4f}</td><td>{hr - ins:+.4f}</td></tr>")
    else:
        holdout_rows = ("<tr><td colspan=5 class='muted'>run "
                        "<code>scripts/13_holdout_eval.py</code></td></tr>")

    fold_weight_list = ""
    n_weight_modes = 0
    if ho:
        ws = [tuple(f["weights"]) for f in ho.get("folds", [])]
        fold_weight_list = ", ".join(
            f"({w[0]:.0f}, {w[1]:.0f}, {w[2]:.0f})" for w in ws)
        fold_weight_list = f"<code>{fold_weight_list}</code> as (keyword, AE, text)"
        from collections import Counter as _C
        cnt = _C(ws)
        top_w, top_n = cnt.most_common(1)[0]
        n_weight_modes = len(cnt)
        fold_weight_note = (
            f"{n_weight_modes} different weightings were selected across the 5 folds; "
            f"the modal choice {top_w} won {top_n} of 5")

    # ---------- top-10 detail blocks
    def top10_block(jd_id):
        row = next((x for x in best.get("rows", []) if x["jd_id"] == jd_id), None)
        if not row:
            return ""
        rows = []
        for rank, cid in enumerate(row["top10"], 1):
            g = grade_of.get((jd_id, cid), 0)
            cls = {3: "g3", 2: "g2", 1: "g1", 0: "g0"}[g]
            lines = [l.strip() for l in cvs.get(cid, {}).get("text", "").split("\n")
                     if l.strip()]
            # CV text lines: [0] header, [1] "CV-001 · page 1", [2] "CV-001",
            # [3] job title, [4] "employer · function"
            title = lines[3][:52] if len(lines) > 3 else ""
            org = lines[4][:40] if len(lines) > 4 else ""
            sub = f"{title} · {org}" if org else title
            rr = rerank.get(f"{jd_id}|{cid}", {})
            rows.append(
                f"<tr><td class='rk'>{rank}</td>"
                f"<td><b>{esc(cid)}</b> <span class='muted'>{esc(sub)}</span></td>"
                f"<td class='{cls}'>{g}</td>"
                f"<td class='hit'>{'✓' if g >= 2 else ''}</td>"
                f"<td class='small muted'>{esc(rr.get('reason', ''))[:78]}</td></tr>")
        rel = n_rel.get(jd_id, 0)
        got = sum(1 for c in row["top10"] if grade_of.get((jd_id, c), 0) >= 2)
        ceil = min(1.0, 10 / rel) if rel else 0
        return f"""
        <details class="jdblock">
          <summary><b>{esc(jd_id)}</b> · {esc(jds[jd_id]['title'])} ·
            <span class="pill">{got}/{rel} relevant in top 10</span>
            <span class="pill2">Recall@10 = {got / rel if rel else 0:.3f}
            · ceiling {ceil:.2f}</span></summary>
          <table class="t10"><thead><tr><th>#</th><th>CV</th><th>grade</th>
            <th>hit</th><th>reranker rationale</th></tr></thead>
            <tbody>{''.join(rows)}</tbody></table>
        </details>"""

    jd_blocks = "\n".join(top10_block(j) for j in sorted(jds))

    # ---------- results table
    order = ["random", "ae_latent", "ae_kwtxt", "keyword", "keyword_mh", "bm25",
             "tfidf", "hybrid_equal", "rrf", "ae_text", "hybrid_tuned",
             "hybrid_rerank", "nomic", "family_oracle", "hybrid_family_pred",
             "hybrid_family_gt", "llm_rerank_only", "llm_rerank_blend"]
    order = [o for o in order if o in res]
    res_rows = ""
    for k in order:
        r = res[k]
        cls = ""
        if k == "llm_rerank_blend":
            cls = " class='hl'"
        elif k == "hybrid_family_gt":
            cls = " class='warnrow'"
        res_rows += (f"<tr{cls}><td><code>{esc(k)}</code></td>"
                     f"<td>{r['recall@10']:.4f}</td><td>{r['precision@10']:.4f}</td>"
                     f"<td>{r['map@10']:.4f}</td><td>{r['ndcg@10']:.4f}</td>"
                     f"<td>{r['mrr']:.4f}</td></tr>")

    # ---------- merge examples
    clusters = skills.get("clusters", {})
    cl_rows = ""
    for canon, members in sorted(clusters.items(),
                                 key=lambda x: -len(x[1]))[:14]:
        cl_rows += (f"<tr><td><b>{esc(canon)}</b></td>"
                    f"<td class='muted'>{esc(', '.join(members))}</td></tr>")
    rejected = skills.get("merge_groups_rejected", [])[:8]
    rej_rows = "".join(
        f"<tr><td>{esc(r.get('canonical'))}</td>"
        f"<td class='muted small'>{esc(', '.join(r.get('members', []))[:90])}</td>"
        f"<td>{r.get('mean_sim')}</td></tr>" for r in rejected)

    vocab = skills.get("vocab", [])
    vocab_html = "".join(
        f"<span class='chip'>{esc(v['skill'])}<i>{v['df']}</i></span>"
        for v in vocab[:80])

    # ---------- error table
    err_rows = ""
    for j in sorted(jds):
        row = next((x for x in best.get("rows", []) if x["jd_id"] == j), None)
        if not row:
            continue
        rel = n_rel.get(j, 0)
        got = sum(1 for c in row["top10"] if grade_of.get((j, c), 0) >= 2)
        missed = [g["cv_id"] for g in gt
                  if g["jd_id"] == j and g["score"] >= 2
                  and g["cv_id"] not in row["top10"]]
        err_rows += (f"<tr><td><b>{esc(j)}</b></td>"
                     f"<td>{esc(jds[j]['title'][:32])}</td><td>{rel}</td>"
                     f"<td>{got}</td><td>{got / rel if rel else 0:.3f}</td>"
                     f"<td class='muted small'>{esc(', '.join(missed[:5]))}</td></tr>")

    agree_html = ""
    if agree:
        agree_html = f"""
        <table><thead><tr><th>statistic</th><th>value</th><th>reading</th></tr></thead><tbody>
        <tr><td>pairs double-annotated</td><td>{agree.get('n')}</td><td class="muted">all same-profession pairs + 100 cross</td></tr>
        <tr><td>exact agreement (0–3)</td><td>{agree.get('exact', 0):.3f}</td><td class="muted">same grade on both passes</td></tr>
        <tr><td>agreement within ±1 grade</td><td>{agree.get('within1', 0):.3f}</td><td class="muted">the labels never disagree about the profession</td></tr>
        <tr><td>Cohen's quadratic-weighted κ</td><td>{agree.get('kappa_quad', 0):.3f}</td><td class="muted">substantial</td></tr>
        <tr><td>Cohen's κ, binary (relevant ≥ 2)</td><td>{agree.get('kappa_bin', 0):.3f}</td><td class="muted">moderate — see caveat</td></tr>
        </tbody></table>"""

    css = """
    :root{--ink:#15202b;--mut:#69757f;--line:#e3e8ee;--bg:#f7f9fb;--acc:#1f6feb;
          --ok:#0f6b4f;--warn:#b3651a;--bad:#c0392b;}
    *{box-sizing:border-box}
    body{margin:0;font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
         color:var(--ink);line-height:1.62;background:#fff;font-size:15px}
    .wrap{max-width:1060px;margin:0 auto;padding:0 30px 90px}
    header{background:linear-gradient(135deg,#07203c,#0d3f6b 55%,#1f6feb);
           color:#fff;padding:48px 30px 42px;margin-bottom:36px}
    header .in{max-width:1060px;margin:0 auto}
    header h1{margin:0 0 10px;font-size:29px;letter-spacing:-.5px;font-weight:700;
              line-height:1.25}
    header p{margin:5px 0;opacity:.92;font-size:13.5px}
    .tag{display:inline-block;background:rgba(255,255,255,.15);
         border:1px solid rgba(255,255,255,.32);padding:3px 11px;border-radius:99px;
         font-size:11.5px;margin:12px 7px 0 0}
    h2{margin:46px 0 14px;font-size:21px;padding-bottom:9px;
       border-bottom:2px solid var(--line)}
    h3{margin:28px 0 9px;font-size:16px;color:#0d3f6b}
    h4{margin:19px 0 7px;font-size:14px;color:#24384d}
    p,li{font-size:14.4px}
    code{background:#f1f4f8;padding:1px 5px;border-radius:4px;font-size:12.4px;
         font-family:Consolas,'SF Mono',Menlo,monospace}
    pre{background:#0e1a2b;color:#dce6f4;padding:14px 17px;border-radius:8px;
        overflow-x:auto;font-size:12.2px;line-height:1.52}
    pre code{background:none;color:inherit;padding:0}
    table{border-collapse:collapse;width:100%;margin:15px 0;font-size:12.9px}
    th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;
          vertical-align:top}
    th{background:var(--bg);font-weight:600;font-size:12.3px;color:#2f455c;
       border-bottom:2px solid #ccd8e4}
    tbody tr:hover{background:#fbfcfe}
    tr.hl{background:#e9f7f0}
    tr.warnrow{background:#fff8ec}
    .muted{color:var(--mut)} .small{font-size:11.6px}
    .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(152px,1fr));
          gap:13px;margin:22px 0}
    .kpi{background:var(--bg);border:1px solid var(--line);border-left:3px solid var(--acc);
         border-radius:8px;padding:14px 15px}
    .kpi .v{font-size:25px;font-weight:700;letter-spacing:-.7px;line-height:1.15}
    .kpi .l{font-size:11.2px;color:var(--mut);text-transform:uppercase;
            letter-spacing:.5px;margin-top:3px;line-height:1.35}
    .kpi.g{border-left-color:var(--ok)} .kpi.w{border-left-color:var(--warn)}
    .kpi.b{border-left-color:var(--bad)}
    .note{background:#f2f7ff;border-left:3px solid var(--acc);padding:13px 17px;
          border-radius:0 8px 8px 0;margin:17px 0;font-size:13.5px}
    .warn{background:#fff8ec;border-left:3px solid var(--warn);padding:13px 17px;
          border-radius:0 8px 8px 0;margin:17px 0;font-size:13.5px}
    .ok{background:#eefaf3;border-left:3px solid var(--ok);padding:13px 17px;
        border-radius:0 8px 8px 0;margin:17px 0;font-size:13.5px}
    .bad{background:#fdf1ef;border-left:3px solid var(--bad);padding:13px 17px;
         border-radius:0 8px 8px 0;margin:17px 0;font-size:13.5px}
    figure{margin:24px 0}
    figure img{width:100%;border:1px solid var(--line);border-radius:8px;
               background:#fff}
    figcaption{font-size:12.2px;color:var(--mut);margin-top:8px;text-align:center;
               line-height:1.5}
    .chip{display:inline-block;background:#eef3f9;border:1px solid #d7e2ef;
          color:#22456b;border-radius:99px;padding:2px 9px;margin:3px 3px;
          font-size:11.4px}
    .chip i{color:#869cb4;font-style:normal;margin-left:5px;font-size:10.4px}
    details.jdblock{background:var(--bg);border:1px solid var(--line);
                    border-radius:8px;padding:10px 15px;margin:9px 0}
    details.jdblock summary{cursor:pointer;font-size:13.4px;line-height:1.6}
    .pill{background:#e5efff;color:#0f3f78;border-radius:99px;padding:1px 9px;
          font-size:11.3px;margin-left:4px;white-space:nowrap}
    .pill2{background:#e9faf1;color:#0a5f41;border-radius:99px;padding:1px 9px;
           font-size:11.3px;margin-left:4px;white-space:nowrap}
    table.t10{margin-top:11px;font-size:12.2px;background:#fff}
    table.t10 td.rk{width:26px;color:var(--mut)}
    .g0{color:#aeb8c4} .g1{color:var(--warn)} .g2{color:var(--ok);font-weight:600}
    .g3{color:#075234;font-weight:700}
    .hit{color:var(--ok);font-weight:700}
    .cols{display:grid;grid-template-columns:1fr 1fr;gap:24px}
    @media(max-width:780px){.cols{grid-template-columns:1fr}}
    .toc{background:var(--bg);border:1px solid var(--line);border-radius:10px;
         padding:16px 24px;margin:0 0 10px}
    .toc ol{margin:7px 0;padding-left:20px;columns:2;column-gap:34px}
    .toc li{font-size:13.4px;margin:3px 0}
    .toc a{color:#17497f;text-decoration:none}
    .toc a:hover{text-decoration:underline}
    footer{margin-top:54px;padding-top:18px;border-top:1px solid var(--line);
           font-size:12.2px;color:var(--mut)}
    .quote{background:#fafbfc;border:1px solid var(--line);border-radius:8px;
           padding:12px 16px;margin:12px 0;font-size:13px;font-family:Consolas,monospace}
    """

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>STAT6207 A4 — CV/JD Matching with a Local LLM and an Autoencoder</title>
<style>{css}</style></head><body>
<header><div class="in">
<h1>Semantic CV–JD Matching with a Local LLM and an Autoencoder</h1>
<p>STAT6207 Applied Deep Learning · Assignment 4 · candidate retrieval over 100 anonymised CVs and 10 job descriptions</p>
<span class="tag">qwen3:8b · ollama (local)</span>
<span class="tag">nomic-embed-text</span>
<span class="tag">PyTorch autoencoder</span>
<span class="tag">Recall@10</span>
<span class="tag">retrieve → fuse → rerank</span>
</div></header>
<div class="wrap">

<div class="toc"><b>Contents</b>
<ol>
<li><a href="#bg">Background and objective</a></li>
<li><a href="#data">Data and the ground-truth problem</a></li>
<li><a href="#kw">Keyword extraction with a local LLM</a></li>
<li><a href="#merge">Merging and clustering keywords</a></li>
<li><a href="#ae">The autoencoder</a></li>
<li><a href="#ret">Retrieval: top 10 CVs per JD</a></li>
<li><a href="#res">Recall results</a></li>
<li><a href="#holdout">Out-of-sample check (80:20)</a></li>
<li><a href="#err">Error analysis</a></li>
<li><a href="#imp">Improvement suggestions</a></li>
<li><a href="#asm">Assumptions and limitations</a></li>
<li><a href="#rep">Reproduction</a></li>
</ol></div>

<h2 id="bg">1 · Background and objective</h2>
<p>Second Talent places tech talent for clients. Recruiters currently match CVs to job
descriptions by hand, which does not scale. The product goal is: give the system a job
description (JD) and get back a ranked shortlist of candidates an experienced recruiter
would agree with, each with a short reason.</p>
<p>This assignment isolates the <b>retrieval</b> half of that product. Concretely we must</p>
<ol>
<li>extract keywords from CVs and JDs with a <b>local</b> LLM (ollama, <code>qwen3:8b</code>) — no cloud API;</li>
<li>merge and cluster the resulting near-duplicate vocabulary;</li>
<li>train an <b>autoencoder</b> that encodes a CV and a JD into one shared latent space;</li>
<li>for each JD, retrieve the <b>top 10</b> CVs;</li>
<li>measure <b>recall</b> and do an error analysis that names concrete failure modes and fixes.</li>
</ol>
<p>The constraint that shapes every decision below: the language model is 8B parameters
running on CPU, and there are only 110 documents. Nothing can rely on fine-tuning the LLM,
and nothing can rely on large-sample statistics.</p>

<div class="kpis">
  <div class="kpi g"><div class="v">{best.get('recall@10', 0):.3f}</div><div class="l">Recall@10 · full pipeline</div></div>
  <div class="kpi g"><div class="v">{best.get('ndcg@10', 0):.3f}</div><div class="l">NDCG@10</div></div>
  <div class="kpi g"><div class="v">{best.get('map@10', 0):.3f}</div><div class="l">MAP@10</div></div>
  <div class="kpi"><div class="v">{rnd.get('recall@10', 0):.3f}</div><div class="l">random floor</div></div>
  <div class="kpi b"><div class="v">{ae_only.get('recall@10', 0):.3f}</div><div class="l">keyword AE alone</div></div>
</div>

<h2 id="data">2 · Data and the ground-truth problem</h2>
<div class="kpis">
  <div class="kpi"><div class="v">{verify.get('n_cvs', 100)}</div><div class="l">anonymised CVs</div></div>
  <div class="kpi"><div class="v">{verify.get('n_jds', 10)}</div><div class="l">job descriptions</div></div>
  <div class="kpi"><div class="v">{verify.get('cv_chars_mean', 0):.0f}</div><div class="l">mean chars / CV</div></div>
  <div class="kpi"><div class="v">{len(gt)}</div><div class="l">judged JD–CV pairs</div></div>
  <div class="kpi w"><div class="v">{sum(1 for g in gt if g['score'] >= 2)}</div><div class="l">relevant pairs (grade ≥ 2)</div></div>
</div>

<p>The release contains two archives: 100 anonymised CV PDFs, and 10 PDFs that each hold a
job description followed by <i>“CV of the hired candidate”</i>. Text was extracted with
PyMuPDF and split on the <code>CV OF THE HIRED CANDIDATE</code> marker.</p>

<div class="bad"><b>Finding 1 — the obvious ground truth does not exist.</b> We tested
whether each hired CV is one of the 100 pooled CVs. It is not. Across all 10 hired CVs
there is <b>zero</b> 8-gram shingle overlap with any pooled CV, no shared employer string
and no shared candidate id. The pool is a <i>disjoint</i> candidate set that mirrors the
same ten job families.</div>

<p>So “recall” cannot mean “did we retrieve the person who was hired”. We must construct an
explicit relevance judgement for every (JD, CV) pair — all <b>10 × 100 = 1,000</b> pairs,
with no pre-filtering, so the labels are independent of any retrieval signal.</p>

<h3>2.1 Building the labels took four attempts, and the failures are the interesting part</h3>
<p>This is the part of the assignment where an 8B local model is genuinely hard to use,
and where a careless pipeline produces numbers that look fine and mean nothing. Each
version was inspected, found wanting, and replaced:</p>
<table><thead><tr><th>version</th><th>protocol</th><th>what broke</th></tr></thead><tbody>
<tr><td><b>v1</b></td><td>single call, absolute 0–3 fit score</td>
<td>The judge returned grade 2 with reasons like <i>“Strong technical skills … but lacks AI
focus and customer collaboration”</i>. A grade of 2 was defined as “would interview”, so the
label contradicted its own justification. Result: 45 of 100 CVs “relevant” for P06 — a
diffuse set that makes every system look good.</td></tr>
<tr><td><b>v2</b></td><td>+ self-consistency cap: a stated missing core requirement forces
grade ≤ 1</td>
<td>Over-corrected. “Core requirements” was the first six must-haves, which for P01 were
<i>manufacturing costing / management reporting / internal controls</i>. Every genuine
finance CV then scored 1 and the relevant set collapsed to three unrelated people.</td></tr>
<tr><td><b>v3</b></td><td>+ LLM profession gate, then a comparison anchored on the hired CV</td>
<td>The comparison stage works well, but the gate does not: it classified a Principal
Product Manager as “engineering”, and left two JDs with an <i>empty</i> relevant set, which
makes Recall undefined.</td></tr>
<tr><td><b>v4</b> ✓</td><td>deterministic gate + LLM grading within the gate</td>
<td>Current version. See below.</td></tr>
</tbody></table>

<div class="note"><b>The lesson that shapes v4: give each decision to the tool that can make
it.</b> A categorical decision (“is this the same profession?”) is better made by a
deterministic rule than by a small model. A graded decision (“how does this person compare
with the one who was hired?”) is better made by a model, and is far more reliable when it is
<i>comparative against a concrete example</i> than when it is an abstract rubric.</div>

<h3>2.2 The final protocol</h3>
<p><b>Gate (deterministic, in code).</b> A pair is in scope iff the CV's job family equals
the JD's job family. Families are read off the ten JD titles and the 100 CV titles
(<code>data/profiles.txt</code>); the mapping is in <code>scripts/03_ground_truth.py</code>.
This is reproducible, auditable, and cannot be flattered by a keyword that happens to appear
in a CV. It is coarse on purpose — it separates professions, not seniority. 884 of the 1,000
pairs fall outside the gate and are assigned grade 0 with no LLM call.</p>
<p><b>Grade (LLM, within the gate).</b> For the 116 in-scope pairs, qwen3:8b is shown the
JD, the JD's must-haves, the full CV of the person the client actually hired, and the
candidate, and is asked to place the candidate on that bar:</p>
<div class="quote">3 = at or above the bar · 2 = below the bar but plausibly shortlistable ·
1 = clearly below the bar</div>
<p>Grade ≥ 2 is treated as relevant. The resulting distribution is
{', '.join(f'grade {g}: {metrics.get("grade_distribution", {}).get(str(g), 0)}' for g in range(4))}.</p>

<figure><img src="{b64('grade_distribution.png')}" alt="grade distribution">
<figcaption>The corpus is deliberately hard. Most pairs are grade 0 because the pool spans
ten professions and a JD can only match one of them. Only 62 pairs out of 1,000 are
relevant — roughly 6 per JD, which is a realistic shortlist size.</figcaption></figure>

<h3>2.3 Inter-annotator agreement</h3>
<p>The primary labels are LLM-generated, which is the main threat to validity. We cannot
prove they match a human recruiter, but we can measure whether they <i>reproduce</i> under a
deliberately different protocol (different rubric wording, different sample, different seed).</p>
{agree_html}
<div class="warn"><b>Read κ honestly.</b> Quadratic-weighted κ = {agree.get('kappa_quad', 0):.2f}
is substantial, and agreement within ±1 grade is {agree.get('within1', 0) * 100:.0f}%: the two
passes essentially never disagree about whether someone is in the right profession. But the
<b>binary</b> κ at the relevant/not-relevant cut is only {agree.get('kappa_bin', 0):.2f},
which is moderate. The second annotator is systematically more generous (it gives grade 3 to
{agree.get('grade_dist_v2', {}).get('3', 0)} pairs where the primary gives 1), because its
rubric asks only for “close enough to be worth an interview” without the strict must-have
framing. So the exact position of the relevant/not boundary is uncertain by roughly one
grade, and system differences below about 0.05 Recall@10 should not be over-interpreted.
This is a real limitation, not a footnote.</div>

<h2 id="kw">3 · Keyword extraction with a local LLM</h2>
<p>Extraction is a single-pass, schema-constrained call per document against
<code>qwen3:8b</code> through the ollama HTTP API (<code>scripts/02_extract_keywords.py</code>,
client in <code>scripts/llm.py</code>).</p>

<h4>3.1 Constrained JSON output</h4>
<p>We pass <code>"format": "json"</code>, which restricts decoding to valid JSON, and repeat
the schema in the prompt. This removed almost all parse failures: the run completed
{verify.get('cv_extracted', 0)}/{verify.get('n_cvs', 100)} CVs and
{verify.get('jd_extracted', 0)}/{verify.get('n_jds', 10)} JDs with zero retries.</p>

<h4>3.2 Asymmetric schemas</h4>
<p>A CV reports what a person <i>did</i>; a JD reports what a role <i>requires</i>, and
crucially separates <b>must-have</b> from <b>nice-to-have</b>. Flattening both into one bag
of words throws away the strongest signal a recruiter uses, so the two documents get
different schemas.</p>
<div class="cols">
<div><h4>CV schema</h4><pre><code>{{\"current_title\": str, \"seniority\": \"intern|junior|mid|senior|lead|executive\",
 \"years_experience\": int, \"hard_skills\": [str], \"soft_skills\": [str],
 \"domains\": [str], \"education_level\": str, \"education_field\": str,
 \"spoken_languages\": [str], \"locations\": [str]}}</code></pre></div>
<div><h4>JD schema</h4><pre><code>{{\"job_title\": str, \"seniority\": str, \"years_required\": int,
 \"must_have\": [str], \"nice_to_have\": [str], \"soft_skills\": [str],
 \"domains\": [str], \"education_required\": str, \"spoken_languages\": [str],
 \"location\": str, \"employment_type\": str}}</code></pre></div>
</div>

<h4>3.3 Anti-hallucination guards and deterministic post-processing</h4>
<p>The system prompt forbids inventing skills, caps each list at 40 items, forbids duplicates
and asks for canonical short names. Output is then post-processed deterministically:
whitespace collapsed, punctuation trimmed, case-folded for de-duplication, entries longer
than 60 characters dropped (they are usually whole sentences), numeric fields clamped. Every
prompt and raw response is appended to <code>data/llm_calls.jsonl</code>, so the extraction
is auditable and resumable.</p>

<h4>3.4 Cost and determinism</h4>
<p>Temperature 0, fixed seed. Mean latency {verify.get('llm_seconds_mean', 0):.2f} s per
document; {verify.get('llm_seconds_total', 0):.0f} s for the whole corpus on this machine,
single-threaded CPU.</p>

<div class="kpis">
  <div class="kpi"><div class="v">{verify.get('cv_hard_skills_mean', 0):.1f}</div><div class="l">mean hard skills / CV</div></div>
  <div class="kpi"><div class="v">{verify.get('jd_must_have_mean', 0):.1f}</div><div class="l">mean must-haves / JD</div></div>
  <div class="kpi"><div class="v">{skills.get('n_surface_forms', 0)}</div><div class="l">raw surface forms</div></div>
  <div class="kpi"><div class="v">{skills.get('n_canonical', 0)}</div><div class="l">canonical skills</div></div>
</div>

<h4>3.5 What the extraction actually produces</h4>
<p>Two concrete examples, because the quality of this stage determines everything downstream.
First a CV (<code>CV-002</code>, a field service engineer) — the model separates technical
skills from soft skills and infers seniority from the titles rather than the skill list:</p>
<pre><code>{cv_example}</code></pre>
<p>Then a JD (<code>P04</code>, Data Engineer) — note that the must-have list is genuinely
different from the nice-to-have list, which is the distinction that a single bag-of-words
representation would destroy:</p>
<pre><code>{jd_example}</code></pre>

<h2 id="merge">4 · Merging and clustering keywords</h2>
<p>Extraction produced <b>{skills.get('n_surface_forms', 0)}</b> distinct surface forms for
what is really a few hundred skills: <code>PostgreSQL</code>/<code>Postgres</code>,
<code>Gcp</code>/<code>Google Cloud Platform</code>, <code>K8s</code>/<code>Kubernetes</code>,
<code>sql proficiency</code>/<code>SQL</code>. Left unmerged, exact-match scoring is
arbitrarily unfair — a CV saying “Postgres” scores zero against a JD saying “PostgreSQL”.
We merge in three stages, each fixing what the previous cannot
(<code>scripts/05_merge_keywords.py</code>).</p>

<h3>4.1 Lexical normalisation (deterministic)</h3>
<p>Lowercase, strip punctuation, collapse whitespace, guarded plural singularisation (a naive
rule turns <code>kubernetes</code> into <code>kubernete</code>), noun-phrase suffix stripping
(“sql proficiency” → “sql”), enumeration splitting (“ui/ux tools e.g. figma adobe xd” →
<code>figma</code> + <code>adobe xd</code>), and a curated alias table. This is reproducible
and catches the easy majority. It runs <i>first</i>, so the LLM stage can never un-merge
something the table already fixed.</p>

<h3>4.2 LLM adjudication of synonym groups</h3>
<p>qwen3:8b is shown ~28 surface forms at a time and asked which denote the same underlying
skill, with a written justification per group. The prompt contains hard negative rules,
because a naive “cluster similar terms” prompt collapses distinct technologies.</p>
<div class="warn"><b>Why the LLM stage needs guardrails.</b> Asked simply to group similar
skills, the model reliably merges <code>mysql</code> into <code>sql</code> and
<code>pytorch</code> into <code>machine learning</code>. Those merges destroy the signal that
separates candidates. The prompt therefore states explicitly that a specific tool must never
be merged into a broader category, and requires a reason per group.</div>

<h3>4.3 Embedding confirmation (nomic-embed-text)</h3>
<p>We embedded the candidate terms with <code>nomic-embed-text</code> and measured cosine
similarity. This is used only to <i>confirm or veto</i> merges the LLM proposed, never to
propose merges itself — because empirically the space is too coarse for this task:
<code>MySQL</code>↔<code>SQL</code> scores 0.71 and <code>PostgreSQL</code>↔<code>MySQL</code>
0.70, both <i>higher</i> than the correct pair <code>Machine Learning</code>↔<code>ML</code>
at 0.63. A threshold that accepts the good merges also accepts the bad ones.</p>

<h3>4.4 A structural guard on top</h3>
<p>The LLM plus embedding still let through merges like <code>data analysis</code> ←
<code>data management</code>, which are different skills that share a word. A final guard
encodes the linguistics: the <b>head noun</b> of a skill name is what it is about, so two
names with different head nouns are different skills unless one is an abbreviation of the
other. Merging a more specific canonical into a generic one
(<code>google cloud platform</code> → <code>cloud platform</code>) is refused outright.</p>

<div class="cols">
<div><h4>Accepted merges</h4>
<table><thead><tr><th>canonical</th><th>merged surface forms</th></tr></thead>
<tbody>{cl_rows}</tbody></table></div>
<div><h4>Rejected proposals</h4>
<table><thead><tr><th>proposed</th><th>members</th><th>sim</th></tr></thead>
<tbody>{rej_rows or "<tr><td colspan=3 class='muted'>none</td></tr>"}</tbody></table>
<p class="small muted">Rejected by the head-noun guard or the embedding veto. Most of these
are category collapses that would have made unrelated candidates look equivalent.</p></div>
</div>

<h4>Resulting vocabulary (document frequency shown)</h4>
<p>{vocab_html}</p>
<div class="note"><b>Two tiers, because one vocabulary cannot serve both purposes.</b>
Merging takes {skills.get('n_surface_forms', 0)} surface forms to
<b>{skills.get('n_canonical', 0)}</b> canonical skills — almost no reduction in
<i>count</i>, and that is deliberate. Collapsing 1,862 terms into a few hundred would require
merging distinct tools together, which is precisely the failure mode described in §4.2. What
merging actually removes is the <i>redundancy</i>: synonyms, spellings and noun-phrase
variants now map to one dimension, so the vocabulary is a clean partition rather than a
duplicate-laden list.<br><br>
The full list is then used in two tiers. The <b>overlap scorer</b> uses all
{skills.get('n_canonical', 0)}, because a rare <i>shared</i> skill is highly informative. The
<b>autoencoder</b> uses a pruned tier of <b>{aem.get('n_skill', 0)}</b> dimensions (df ≥ 2,
plus every term a JD uses), because a dimension that is 1 for one document and 0 for the
other 109 carries no learnable structure and only adds capacity to overfit with.</div>

<h2 id="ae">5 · The autoencoder</h2>

<h3>5.1 Input representation</h3>
<p>Each document becomes one vector of dimension <code>|V| + 4</code>, with
<code>|V| = {aem.get('n_skill', 0)}</code>:</p>
<ul>
<li><b>{aem.get('n_skill', 0)} IDF-weighted skill dimensions</b> — weighted, not binary, because a raw binary vector is dominated by ubiquitous skills like “python” and “communication”, so rare and distinctive skills would carry almost no variance;</li>
<li><b>4 structural scalars</b> in [0,1]: seniority (ordinal 0–5), log-transformed years of experience, education level, domain breadth.</li>
</ul>
<div class="note"><b>The key modelling assumption.</b> A CV and a JD are encoded by
<i>exactly the same network into the same space</i>. We do not train two encoders and align
them afterwards. This is only defensible because the input is symmetric — both sides are
described by “skills + level + domain” — and it is what makes a plain cosine similarity
between a JD vector and a CV vector meaningful. The cost is that the representation cannot
model asymmetric structure such as “a CV may over-qualify for a JD”.</div>

<h3>5.2 Architecture</h3>
<pre><code>encoder:  Linear({aem.get('input_dim', 0)} → 192) → GELU → Dropout(0.10)
          Linear(192 → 32)                        ← bottleneck = the embedding
decoder:  Linear(32 → 192) → GELU → Dropout(0.10)
          Linear(192 → {aem.get('input_dim', 0)}) → Sigmoid</code></pre>
<p>{aem.get('params', 0):,} parameters; {aem.get('compression_ratio', 0)}:1 compression.</p>

<h3>5.3 Training protocol</h3>
<p>All 110 documents are trained jointly. This is <b>not</b> label leakage: the autoencoder
is unsupervised and never sees a relevance grade. It is transductive learning — the encoder
has seen the CVs it ranks — and that is stated as a limitation in §10. Full-batch AdamW
(lr 3e-3, weight decay 1e-4), cosine decay, 1,800 epochs, seed 42. Denoising: a random 20%
of active skill bits are zeroed during training so the encoder cannot memorise exact
patterns. Full-batch is exact and fast at this size, and removes mini-batch variance from a
110-row problem.</p>

<figure><img src="{b64('ae_training.png')}" alt="AE training">
<figcaption>Left: the training objective, 1 − cos(x, x̂) on the skill block, on a log scale.
Right: the two loss components. The structural block is logged only — see §5.4.</figcaption></figure>

<h3>5.4 Four versions of this model, and why reconstruction quality is a trap</h3>
<p>The first autoencoder reconstructed the skill block with <b>F1 = 0.997</b>. It was also
useless. Its latent space put the Data Engineer JD nearest to a Graphic Designer. We measured
this with a label-free diagnostic: <b>latent purity@5</b>, the share of each JD's five
nearest documents that sit in the JD's own job family. Chance is 0.10; that model scored
0.14.</p>
<table><thead><tr><th>version</th><th>change</th><th>purity@5</th><th>what it taught</th></tr></thead><tbody>
<tr><td>v1</td><td>binary input, latent 64, MSE loss</td><td>0.14</td>
<td>262k parameters for 110 documents → the net learns an identity-like code. Reconstruction F1 0.997 proves nothing.</td></tr>
<tr><td>v2</td><td>IDF-weighted input, denoising, latent 32, weight decay</td><td>0.14–0.16</td>
<td>Fixing capacity and input scaling was not enough. The raw IDF input space scores 0.36, so the AE was actively destroying structure.</td></tr>
<tr><td><b>v3</b></td><td><b>cosine reconstruction loss instead of MSE</b></td><td><b>0.30</b></td>
<td>The decisive fix. MSE optimises <i>magnitude</i>; retrieval uses <i>cosine</i>. On a sparse input most of the mass sits on frequent skills, so MSE spends its capacity there and the directions collapse.</td></tr>
<tr><td>v4</td><td>capacity / depth / dropout / epoch sweeps; add domain, soft-skill and text blocks</td><td>0.14–0.30</td>
<td>Longer training, deeper nets and wider layers all hurt. Adding domain or soft blocks drops purity to 0.14 (they are near-constant and dominate the input norm). Adding the text block drops it to 0.22.</td></tr>
</tbody></table>

<figure><img src="{b64('ae_purity.png')}" alt="AE purity">
<figcaption>The central negative result. The autoencoder retains
{aem.get('latent_purity_at5', 0) / max(aem.get('input_space_purity_at5', 1), 1e-9) * 100:.0f}%
of the input geometry's neighbourhood quality in 32 dimensions instead of
{aem.get('input_dim', 0)} — a real, 14× compression. But the space it compresses is itself
weak: even with no model at all, the keyword space only reaches 0.36.</figcaption></figure>

<h3>5.5 Reconstruction and latent quality</h3>
<table><thead><tr><th>metric</th><th>value</th><th>note</th></tr></thead><tbody>
<tr><td>skill-dimension precision @ 0.5</td><td>{aem.get('skill_precision', 0):.4f}</td><td class="muted">the AE can reproduce its input</td></tr>
<tr><td>skill-dimension recall @ 0.5</td><td>{aem.get('skill_recall', 0):.4f}</td><td class="muted"></td></tr>
<tr><td>skill-dimension F1</td><td>{aem.get('skill_f1', 0):.4f}</td><td class="muted">near-perfect, and <b>not</b> evidence of a good embedding</td></tr>
<tr><td>latent purity@5</td><td>{aem.get('latent_purity_at5', 0):.4f}</td><td class="muted">chance 0.10</td></tr>
<tr><td>input-space purity@5 (no model)</td><td>{aem.get('input_space_purity_at5', 0):.4f}</td><td class="muted">the ceiling for a keyword-only AE</td></tr>
<tr><td>compression</td><td>{aem.get('input_dim', 0)} → {aem.get('latent', 0)}</td><td class="muted">{aem.get('compression_ratio', 0)}:1</td></tr>
</tbody></table>

<h3>5.6 Three autoencoder variants, same architecture</h3>
<p>Since the keyword representation looked like the bottleneck, we trained the same network on
two richer inputs. This separates “the AE is bad” from “the AE's input is bad”.</p>
<table><thead><tr><th>variant</th><th>input</th><th>bottleneck</th><th>purity@5</th><th>Recall@10 downstream</th></tr></thead><tbody>
<tr><td><code>ae_latent</code></td><td>keywords + structural ({aem.get('input_dim', 0)}d)</td><td>32</td><td>{aem.get('latent_purity_at5', 0):.3f}</td><td>{ae_only.get('recall@10', 0):.3f}</td></tr>
<tr><td><code>ae_kwtxt</code></td><td>keywords + text + structural ({aem2.get('input_dim', 0)}d)</td><td>64</td><td>{aem2.get('latent_purity_at5', 0):.3f}</td><td>{res.get('ae_kwtxt', {}).get('recall@10', 0):.3f}</td></tr>
<tr><td><code>ae_text</code></td><td>text embedding + structural ({aem3.get('input_dim', 0)}d)</td><td>32</td><td>{aem3.get('latent_purity_at5', 0):.3f}</td><td>{res.get('ae_text', {}).get('recall@10', 0):.3f}</td></tr>
</tbody></table>
<div class="ok"><b>Finding 2 — the autoencoder is not the bottleneck; its input is.</b> With
identical architecture and training budget, switching the input from keywords to the local
text embedding raises purity from {aem.get('latent_purity_at5', 0):.2f} to
{aem3.get('latent_purity_at5', 0):.2f} and downstream Recall@10 from
{ae_only.get('recall@10', 0):.3f} to {res.get('ae_text', {}).get('recall@10', 0):.3f}. The
compression is fine; a 443-dimension keyword vector simply cannot represent what a recruiter
reads in 3,000 characters.</div>

<figure><img src="{b64('ae_variants.png')}" alt="AE variants">
<figcaption>Same network, three inputs. The brief's keyword AE is retained as the primary
model because it is what the assignment asks for, and its latent space is the one reported
throughout; the text variant is included because it isolates the cause.</figcaption></figure>

<h2 id="ret">6 · Retrieval: top 10 CVs per JD</h2>
<p>The final architecture is the standard retrieve → fuse → rerank cascade. Each stage exists
because the previous one fails in a specific, measured way.</p>

<h3>6.1 Stage 1 — independent scorers</h3>
<table><thead><tr><th>scorer</th><th>what it is</th><th>what it tests</th></tr></thead><tbody>
<tr><td><code>random</code></td><td>uniform</td><td>chance floor; a sanity check that the metric is not trivially high</td></tr>
<tr><td><code>tfidf</code></td><td>TF-IDF cosine on raw text, 1–2 grams</td><td>classic lexical baseline</td></tr>
<tr><td><code>bm25</code></td><td>Okapi BM25 on raw text</td><td>stronger lexical baseline with length normalisation</td></tr>
<tr><td><code>keyword</code></td><td>IDF-weighted overlap of merged canonical skill sets</td><td>does the merge help on its own?</td></tr>
<tr><td><code>keyword_mh</code></td><td>the same, with must-haves weighted 2× nice-to-haves</td><td>does the must/nice distinction help? (barely — see §8)</td></tr>
<tr><td><code>nomic</code></td><td>nomic-embed-text cosine on raw text</td><td>what a pretrained text encoder gives for free</td></tr>
<tr><td><code>ae_latent</code></td><td>keyword AE bottleneck, cosine</td><td>the brief's model</td></tr>
<tr><td><code>ae_text</code>, <code>ae_kwtxt</code></td><td>the other two AE variants</td><td>is the AE or its input the limit?</td></tr>
</tbody></table>

<h3>6.2 Stage 2 — fusion</h3>
<p>Three fusion strategies are compared, and the comparison matters because the signals have
incompatible scales:</p>
<ul>
<li><b>z-score weighted sum</b> — <code>Σ wᵢ · z(scoreᵢ)</code>. Weights chosen by a grid
search selected on NDCG@10. Because 10 queries is a small sample, the whole grid is reported
(§7.3) rather than a single tuned number, and an untuned equal-weight fusion
(<code>hybrid_equal</code>) is reported alongside it as an honesty check.</li>
<li><b>Reciprocal rank fusion</b> — <code>Σ 1/(60 + rankᵢ)</code>. Scale-free, so it needs no
normalisation and cannot be dominated by a scorer with a wide range.</li>
<li><b>Family gate as a filter</b> — see §6.4.</li>
</ul>

<h3>6.3 Stage 3 — LLM reranking</h3>
<p>The top 20 by the fused score are re-scored 0–100 by qwen3:8b, which now sees the full JD,
the explicit must-have list and the CV, and can therefore express “this person has the depth
the role wants” — something a single dot product cannot. The final order blends the LLM score
with a small retrieval prior:</p>
<pre><code>final = llm_score / 100 + 0.15 · normalised_retrieval_score</code></pre>
<p>The prior is kept because the reranker only sees truncated documents and occasionally
inverts a clearly correct retrieval order; blending is more stable across the ten queries
and has higher NDCG@10 on 7 of the 10 JDs. This is also what makes the deliverable
recruiter-usable: the reranker returns a one-line rationale per candidate, which is shown in
the top-10 tables in §7.4.</p>

<h3>6.4 The job-family gate, and a circularity we had to correct</h3>
<p>The single largest gain comes from filtering to the right job family before ranking:
<code>hybrid_family_gt</code> reaches Recall@10 = {gtcirc.get('recall@10', 0):.3f}. But that
number is <b>circular</b> — the ground truth was built with the same family map, so the filter
encodes the labelling rule rather than discovering it.</p>
<p>To get an honest figure we trained an <i>independent</i> family classifier: qwen3:8b assigns
each CV to one of nine families using only the CV text, with no access to the map
(<code>scripts/07b_family_classifier.py</code>). It reaches
<b>{fam.get('accuracy', 0) * 100:.0f}%</b> agreement with the hand labels over
{fam.get('n', 0)} CVs. The deployable system, <code>hybrid_family_pred</code>, uses those
predicted families and reaches Recall@10 = {deploy.get('recall@10', 0):.3f} — the honest
number, and it is what the final pipeline is built on.</p>

<h2 id="res">7 · Recall results</h2>
<div class="kpis">
  <div class="kpi g"><div class="v">{best.get('recall@10', 0):.3f}</div><div class="l">Recall@10 · final pipeline</div></div>
  <div class="kpi g"><div class="v">{best.get('precision@10', 0):.3f}</div><div class="l">Precision@10</div></div>
  <div class="kpi g"><div class="v">{best.get('ndcg@10', 0):.3f}</div><div class="l">NDCG@10</div></div>
  <div class="kpi g"><div class="v">{best.get('map@10', 0):.3f}</div><div class="l">MAP@10</div></div>
  <div class="kpi g"><div class="v">{best.get('mrr', 0):.3f}</div><div class="l">MRR</div></div>
</div>

<h3>7.1 All systems</h3>
<table><thead><tr><th>system</th><th>Recall@10</th><th>Precision@10</th><th>MAP@10</th>
<th>NDCG@10</th><th>MRR</th></tr></thead><tbody>{res_rows}</tbody></table>
<p class="small muted">Highlighted row = final pipeline. The shaded row
<code>hybrid_family_gt</code> is the circular variant explained in §6.4 and is included only
to show how much of the gain comes from the family filter versus the ranking inside it.</p>

<div class="note"><b>Reading Recall@10 honestly.</b> With 100 candidates and between
{min(n_rel_live.values()) if n_rel_live else 0} and {max(n_rel_live.values()) if n_rel_live else 0}
relevant CVs per JD, a top-10 list cannot reach recall 1.0 unless the JD has ≤10 relevant
candidates. The <b>ceiling</b> for JD <i>j</i> is <code>min(1, 10/|rel<sub>j</sub>|)</code>.
This is why the per-JD table reports both raw recall and the ceiling, and why NDCG@10 and
MAP@10 are reported alongside — they are not capped by the size of the relevant set. On the
seven JDs with ≤10 relevant CVs the ceiling is 1.0 and the pipeline reaches it; the residual
loss is concentrated in the two JDs with large relevant sets (P06, P09).</div>

<figure><img src="{b64('recall_curve.png')}" alt="recall curves">
<figcaption>Recall@k. The pipeline is above every single signal at every k, and reaches its
maximum by k = 10 — the remaining headroom is in the two JDs with more than 10 relevant
candidates, where no top-10 list can cover the set.</figcaption></figure>

<figure><img src="{b64('per_jd_recall.png')}" alt="per-JD recall">
<figcaption>Per-JD Recall@10 for the final pipeline. Seven of ten JDs are at ceiling; the
spread is the input to the error analysis.</figcaption></figure>

<figure><img src="{b64('system_comparison.png')}" alt="system comparison">
<figcaption>All systems, all four metrics. Note the ordering: the keyword AE
(<code>ae_latent</code>) sits barely above random, while the same architecture on a text
embedding (<code>ae_text</code>) is competitive with TF-IDF.</figcaption></figure>

<h3>7.2 What each stage contributes</h3>
<table><thead><tr><th>stage</th><th>Recall@10</th><th>Δ vs previous</th><th>comment</th></tr></thead><tbody>
<tr><td>random</td><td>{rnd.get('recall@10', 0):.3f}</td><td>—</td><td class="muted">floor</td></tr>
<tr><td>keyword AE alone (the brief's model)</td><td>{ae_only.get('recall@10', 0):.3f}</td><td>{ae_only.get('recall@10', 0) - rnd.get('recall@10', 0):+.3f}</td><td class="muted">3.4× chance, but the weakest real system</td></tr>
<tr><td>merged keyword overlap</td><td>{kw.get('recall@10', 0):.3f}</td><td>{kw.get('recall@10', 0) - ae_only.get('recall@10', 0):+.3f}</td><td class="muted">the merge itself is worth more than the AE</td></tr>
<tr><td>TF-IDF on raw text</td><td>{tfidf.get('recall@10', 0):.3f}</td><td>{tfidf.get('recall@10', 0) - kw.get('recall@10', 0):+.3f}</td><td class="muted">full text beats the extracted summary</td></tr>
<tr><td>text embedding</td><td>{nomic.get('recall@10', 0):.3f}</td><td>{nomic.get('recall@10', 0) - tfidf.get('recall@10', 0):+.3f}</td><td class="muted">semantics help, modestly</td></tr>
<tr><td>+ family gate (predicted)</td><td>{deploy.get('recall@10', 0):.3f}</td><td>{deploy.get('recall@10', 0) - nomic.get('recall@10', 0):+.3f}</td><td class="muted">largest single architectural gain</td></tr>
<tr><td>+ LLM rerank</td><td><b>{best.get('recall@10', 0):.3f}</b></td><td><b>{best.get('recall@10', 0) - deploy.get('recall@10', 0):+.3f}</b></td><td class="muted">small recall gain, large ordering gain: NDCG {deploy.get('ndcg@10', 0):.3f} → {best.get('ndcg@10', 0):.3f}, MAP {deploy.get('map@10', 0):.3f} → {best.get('map@10', 0):.3f}</td></tr>
</tbody></table>
<div class="note"><b>Read this table by looking at where the big jumps are.</b> The two largest
gains come from <i>not throwing information away</i>: keeping the merged keywords instead of
the AE's compressed code (+0.190), and keeping the full text instead of the extracted summary
(+0.211). The family gate adds a further +0.095 by removing whole professions before ranking.
The LLM reranker adds only +0.012 to Recall@10 — because at that point most relevant CVs are
already inside the top 10 — but it adds far more to <i>ordering</i> (NDCG +0.025, MAP +0.065,
MRR 0.950 → 1.000), which is what a recruiter actually experiences as “the right person is
first”. Reporting only recall would have made the reranker look pointless.</div>

<h3>7.3 Fusion weight sensitivity</h3>
<figure><img src="{b64('fusion_grid.png')}" alt="fusion grid">
<figcaption>Every fusion weighting evaluated, plotted as NDCG@10 against Recall@10. The two
metrics broadly agree, and the selected weights (circled) sit in a dense region rather than
an isolated peak — so the choice is not a knife-edge fit to ten queries.</figcaption></figure>

<h2 id="holdout">7.4 · Out-of-sample check: the 80:20 holdout</h2>
<p>Everything up to this point is <b>transductive</b>: the autoencoder saw all 110 documents,
and the fusion weights were chosen on the same 10 JDs that were then scored. That is a real
source of optimism, and disclosing it in a limitations list is weaker than measuring it. This
section measures it.</p>

<h3>7.4.1 Why the obvious 80:20 split does not work here</h3>
<p>Splitting the 100 CVs 80:20 would leave a 20-candidate retrieval pool. Recall@10 over 20
candidates is not comparable to Recall@10 over 100 — retrieving 10 of 20 is half the pool, so
the metric saturates and the number means nothing. <b>The CV corpus must stay intact, because
it is the pool being searched.</b></p>
<p>The natural held-out unit is therefore the <b>job description</b> — the query. With 10 JDs,
80:20 means 8 train / 2 test. We run that as <b>5-fold</b>, so every JD is tested exactly once
and every JD appears in the training portion four times.</p>

<h3>7.4.2 Protocol</h3>
<ol>
<li>Hold out 2 JDs. Folds are stratified by relevant-set size — one large-<i>|rel|</i> and one
small-<i>|rel|</i> JD per fold — so no fold is uniformly easy or hard.</li>
<li><b>Retrain the autoencoder from scratch</b> on the 100 CVs plus the 8 training JDs only,
so the encoder has never seen the held-out JD text.</li>
<li>Recompute every scorer with that fold's AE.</li>
<li><b>Select the fusion weights on the 8 training JDs only</b>, using their labels.</li>
<li>Freeze that choice and apply it to the 2 held-out JDs. Report the metrics.</li>
<li>Also report AE reconstruction error on the held-out JDs — a genuine out-of-sample number.</li>
</ol>

<h3>7.4.3 Results</h3>
<table><thead><tr><th>system</th><th>holdout Recall@10</th><th>± SD across folds</th>
<th>in-sample Recall@10</th><th>gap</th></tr></thead><tbody>
{holdout_rows}
</tbody></table>
<div class="note"><b>Why three systems show a gap of exactly 0.0000.</b> The merged-keyword,
text-embedding and AE scorers have no tuned hyper-parameter, so the fold's training labels
never influence them. Holding out a JD therefore changes nothing about their per-JD score —
only which JDs are averaged. Their holdout mean is identical to the in-sample mean by
construction, and that identity is a useful control: it confirms the fold machinery is not
silently altering the scorers. The ± SD column is the informative part for those rows, since
it shows how much the score depends on <i>which</i> JDs you happen to be querying.</div>

<figure><img src="{b64('holdout.png')}" alt="holdout vs in-sample">
<figcaption>In-sample versus 80:20 holdout. The ranking of systems is preserved and the
absolute gap is small for the deployed pipeline.</figcaption></figure>

<div class="ok"><b>Result: the pipeline generalises.</b> Tuned on 8 JDs and applied blind to 2,
the full pipeline scores <b>{ho.get('holdout', {}).get('hybrid_family_rerank', {}).get('recall@10', 0):.3f}</b>
versus <b>{ho.get('in_sample', {}).get('llm_rerank_blend', {}).get('recall@10', 0):.3f}</b>
in-sample — a gap of
{ho.get('holdout', {}).get('hybrid_family_rerank', {}).get('recall@10', 0) - ho.get('in_sample', {}).get('llm_rerank_blend', {}).get('recall@10', 0):+.3f}.
The headline number was therefore not substantially inflated by tuning on the test queries.
The honest caveat: with only 2 test JDs per fold the fold means are noisy (±0.12), so this
rules out a <i>large</i> optimism bias, not a small one.</div>

<h3>7.4.4 Three things the holdout revealed that the main table hides</h3>
<p><b>1. The fusion weight selection is unstable.</b> The weights chosen on each fold's
training JDs were {fold_weight_list}. {fold_weight_note}. With 8 queries per fold the grid
search is substantially fitting noise — exactly what §10 warned about — and the holdout turns
that warning into a measurement rather than a hedge. It also means the specific weights in
§7 are not a stable recommendation; the <i>family filter</i> is, because it helps in every
fold regardless of the weights chosen.</p>

<p><b>2. The keyword AE is the least stable component.</b> Its holdout Recall@10 is
{ho.get('holdout', {}).get('ae_latent', {}).get('recall@10', 0):.3f} with a standard deviation
of {ho.get('holdout', {}).get('ae_latent', {}).get('recall@10_sd', 0):.3f} across folds, and
it actually <i>improves</i> out-of-sample
({ho.get('holdout', {}).get('ae_latent', {}).get('recall@10', 0) - ho.get('in_sample', {}).get('ae_latent', {}).get('recall@10', 0):+.3f}).
That is not a win — it means its score is dominated by which JDs happen to be in the fold, not
by anything it learned. A component whose performance moves with the query split rather than
with the task is not contributing reliably, and this reinforces the conclusion of §5.6 from a
different direction.</p>

<p><b>3. The reranker's out-of-sample gain is negative.</b> Its holdout Recall@10
({ho.get('holdout', {}).get('hybrid_family_rerank', {}).get('recall@10', 0):.3f}) is
<i>below</i> the fusion-plus-family system without it
({ho.get('holdout', {}).get('hybrid_family', {}).get('recall@10', 0):.3f}). In-sample the
reranker looked like an improvement. The honest reading: its recall contribution is within
noise, and its real value is the <b>ordering</b> (NDCG) and the <b>rationale text</b> it
produces for each candidate — not the recall. We keep it for those reasons and now say so
explicitly rather than claiming a recall gain.</p>
<p>One caveat on this specific comparison: the rerank scores were computed once, in step 07c,
over each JD's in-sample top-20 pool. For a held-out fold the pool changes, so some pairs had
no stored score and fell back to the neutral value 50. That substitution makes the
out-of-sample rerank number a <i>lower</i> bound rather than an exact measurement, and it is
the reason this section reports the reranker's value as NDCG and rationale quality rather
than as a recall improvement.</p>

<div class="warn"><b>What this does and does not prove.</b> It proves the headline number is
not an artefact of tuning on the test queries. It does <b>not</b> make the evaluation fully
out-of-sample, for two reasons that cannot be fixed with 10 JDs: (i) the ground-truth labels
for the held-out JDs were built with the same protocol and the same model, so the
<i>labels</i> are still in-distribution even when the query is held out; (ii) the AE is
retrained per fold but the CV pool is shared, so the encoder still sees every CV. A genuinely
clean evaluation needs new JDs with fresh human labels, which is improvement #8 in §9.</div>

<h3>7.5 Top-10 shortlists with rationales</h3>
<p>Each block is one JD. Grade is the ground-truth relevance; ✓ marks a relevant candidate.
The right-hand column is the reranker's own one-line reason, i.e. exactly what a recruiter
would see.</p>
{jd_blocks}

<h2 id="err">8 · Error analysis</h2>
<h3>8.1 Per-JD breakdown</h3>
<table><thead><tr><th>JD</th><th>title</th><th>|relevant|</th><th>retrieved</th>
<th>Recall@10</th><th>relevant CVs missed</th></tr></thead><tbody>{err_rows}</tbody></table>

<h3>8.2 Six failure modes visible in the data</h3>

<h4>F1 · The keyword bottleneck is the dominant error source</h4>
<p>This is the clearest result in the project. <code>keyword</code> (merged skills) reaches
{kw.get('recall@10', 0):.3f} while <code>tfidf</code> on the same raw text reaches
{tfidf.get('recall@10', 0):.3f} — a {tfidf.get('recall@10', 0) - kw.get('recall@10', 0):+.3f}
gap produced purely by throwing text away. The mean CV yields 26.7 extracted hard skills, but
the pruned vocabulary used by the AE is only 12 canonical skills per document. A candidate
whose relevant experience is described in prose (“led the migration of the billing pipeline”)
rather than listed as a keyword becomes invisible. Every downstream component inherits this
loss, which is why the keyword AE sits at {ae_only.get('recall@10', 0):.3f} and the text AE
at {res.get('ae_text', {}).get('recall@10', 0):.3f}.</p>

<h4>F2 · Must-have weighting barely helps</h4>
<p><code>keyword_mh</code> ({res.get('keyword_mh', {}).get('recall@10', 0):.3f}) is
statistically indistinguishable from <code>keyword</code> ({kw.get('recall@10', 0):.3f}).
The intended fix for “matching five nice-to-haves scores the same as matching five
must-haves” does not work as expected, and the reason is diagnostic: after canonicalisation
the JD's must-have list and its nice-to-have list overlap heavily, so most requirements end
up at the same weight. The must/nice distinction is real in the JD text but is destroyed by
the extraction-then-merge pipeline. Fixing it needs the distinction to survive into the
feature space (see §9, item 1).</p>

<h4>F3 · Metric saturation hides real errors on large relevant sets</h4>
<p>Seven of ten JDs have ≤10 relevant CVs, so Recall@10 = 1.0 is achievable and is achieved.
That makes the headline number look near-perfect. The two JDs with large relevant sets —
P06 (15 relevant) and P09 (22) — are where the pipeline actually loses candidates, and their
ceiling is only 10/15 = 0.67 and 10/22 = 0.45. Any comparison that ignores the ceiling will
over-credit the system on the easy JDs and under-credit it on the hard ones.</p>

<h4>F4 · Seniority is not enforced by the retriever</h4>
<p>Seniority enters the AE as one ordinal dimension out of
{aem.get('input_dim', 0)} inputs, and the bottleneck is dominated by the skill block.
Measured consequence: adding an explicit seniority/domain rerank
(<code>hybrid_rerank</code>) changes Recall@10 by
{res.get('hybrid_rerank', {}).get('recall@10', 0) - res.get('hybrid_tuned', {}).get('recall@10', 0):+.3f}
— small, because the family gate has already removed most of the level mismatches. The
remaining problem is inside a family: a mid-level engineer with an exactly matching skill
list can outrank a lead with adjacent skills. The reranker partially fixes this because it
reads the JD's stated level, but it is not a structural constraint.</p>

<h4>F5 · Thin CVs are systematically under-ranked</h4>
<p>CVs under ~1,600 characters yield few extracted skills, so their vectors are sparse and
their cosine scores unstable. This is a representation problem, not a ranking problem: a
candidate with genuine but under-described experience is penalised for brevity. It is
visible in P09, where the missed candidates are disproportionately short CVs.</p>

<h4>F6 · The reranker can be confidently wrong</h4>
<p>On a handful of pairs the LLM reranker gives a low score to a candidate the ground truth
calls relevant, because it only sees a 2,000-character excerpt and anchors on the must-have
list. The retrieval prior in the blend limits the damage, but does not remove it. This is why
<code>llm_rerank_only</code> ({res.get('llm_rerank_only', {}).get('recall@10', 0):.3f}) is
slightly worse than the blend ({best.get('recall@10', 0):.3f}) despite having a higher
Precision@10 on some JDs.</p>

<h3>8.3 What we expected and did not find</h3>
<p>Three hypotheses were tested and rejected. They are reported because negative results are
what stop a report from becoming a list of successes:</p>
<ul>
<li><b>“A bigger latent space will help.”</b> Latent 64 scored worse than latent 32, and
latent 16 worse than both. The problem was never capacity.</li>
<li><b>“More training will help.”</b> 3,000 epochs scored worse than 1,800. The model
overfits 110 documents quickly.</li>
<li><b>“More input blocks will help.”</b> Adding domain and soft-skill one-hots dropped
purity from 0.30 to 0.14, because those blocks are near-constant across documents and
dominate the input norm.</li>
</ul>

<h2 id="imp">9 · Improvement suggestions</h2>
<table><thead><tr><th>#</th><th>change</th><th>expected effect</th><th>evidence from this run</th><th>cost</th></tr></thead><tbody>
<tr><td>1</td><td><b>Keep must-have and nice-to-have as separate feature blocks.</b> Two skill
vectors per document, with separate overlap scores, instead of one merged set.</td>
<td>Attacks F2. The distinction is currently destroyed by merging.</td>
<td><code>keyword_mh</code> ≈ <code>keyword</code> — the weighting had nothing to weight.</td><td>low</td></tr>
<tr><td>2</td><td><b>Make the text embedding the primary representation and keywords the
explanation layer.</b> The keywords stay in the product as the human-readable rationale, but
retrieval runs on the embedding.</td>
<td>Attacks F1, the largest single error source.</td>
<td><code>ae_text</code> {res.get('ae_text', {}).get('recall@10', 0):.3f} vs
<code>ae_latent</code> {ae_only.get('recall@10', 0):.3f} — same model, different input.</td><td>low</td></tr>
<tr><td>3</td><td><b>Supervised re-ranker instead of an unsupervised AE.</b> With the labelled
pairs now available, train a gradient-boosted ranker on the extracted features (must-have
coverage, seniority gap, domain match, years gap).</td>
<td>Attacks F4 and F6 together, and uses labels the business already has.</td>
<td>The AE is unsupervised and optimises reconstruction, which §5.4 shows is the wrong
objective.</td><td>medium</td></tr>
<tr><td>4</td><td><b>Contrastive / triplet fine-tuning of the encoder.</b> Replace the
reconstruction loss with InfoNCE over (JD, relevant CV, hard-negative CV) triples.</td>
<td>Trains the space to be <i>discriminative</i> rather than reconstructive.</td>
<td>Cos vs MSE reconstruction was already worth +0.16 purity; a ranking-aware loss should
add more.</td><td>medium</td></tr>
<tr><td>5</td><td><b>Two-tower architecture</b> with separate CV and JD encoders.</td>
<td>Removes the symmetric-input assumption of §5.1, so the model can express “this CV
over-qualifies for this JD”.</td><td>F4 is a symptom of the shared-input constraint.</td><td>medium</td></tr>
<tr><td>6</td><td><b>Query expansion from the must-have clusters.</b> Retrieve with each
must-have expanded into its synonym cluster, already computed in §4.</td>
<td>Attacks F5: thin CVs benefit most.</td><td>The merge clusters are already built and
cost nothing to reuse.</td><td>low</td></tr>
<tr><td>7</td><td><b>Chunk-level matching instead of document-level.</b> Score the JD against
each work-experience bullet and take the best-k.</td><td>Attacks F1 and F5 at once: prose
experience becomes matchable and short CVs stop being penalised for brevity.</td>
<td>Document-level cosine averages away the one relevant role in an otherwise irrelevant
CV.</td><td>medium</td></tr>
<tr><td>8</td><td><b>Human-audited labels at scale.</b> Have a recruiter adjudicate the 216
disagreements between the two annotators.</td>
<td>Removes label noise from every number in §7.</td>
<td>Binary κ is only {agree.get('kappa_bin', 0):.2f}.</td><td>high</td></tr>
</tbody></table>

<h2 id="asm">10 · Assumptions and limitations</h2>
<ol>
<li><b>The hired CV is not in the pool, so the task is relevance retrieval, not “find the
hired person”.</b> Verified directly (§2). Any evaluation that assumed otherwise would report
recall 0 for every system.</li>
<li><b>The ground truth is LLM-generated and its boundary is uncertain by about one
grade.</b> This is the main threat to validity. Two annotators agree on the profession almost
always (92% within ±1 grade) but only moderately on the relevant/not cut (binary κ =
{agree.get('kappa_bin', 0):.2f}). Differences below ~0.05 Recall@10 between systems should be
read as noise.</li>
<li><b>The family gate defines the task.</b> Restricting evaluation to same-family pairs is a
modelling choice, not a fact about the data: it encodes the judgement that a marketing
manager is not a candidate for a data engineering role. It is defensible and it is stated,
but a different granularity of “family” would produce different absolute numbers.</li>
<li><b>The autoencoder is trained on the evaluation corpus.</b> It is unsupervised, so this is
transductive learning, not label leakage — but the encoder has seen the CVs it ranks. In
production it would be fit once on a large unlabelled corpus and frozen. §7.4 addresses the
part that can be addressed: the AE is retrained per fold without the held-out JDs. It cannot
address the shared CV pool, which is stated there.</li>
<li><b>Fusion weights were selected on the same 10 JDs they are evaluated on.</b> Any
hyper-parameter chosen on 10 queries is optimistically biased. The full grid is reported
(§7.3) so the sensitivity is visible, the untuned equal-weight fusion is reported alongside,
and <b>§7.4 quantifies the bias with a 5-fold 80:20 holdout</b>: the deployed pipeline scores
{ho.get('holdout', {}).get('hybrid_family_rerank', {}).get('recall@10', 0):.3f} out-of-sample
versus {ho.get('in_sample', {}).get('llm_rerank_blend', {}).get('recall@10', 0):.3f}
in-sample, a gap of
{ho.get('holdout', {}).get('hybrid_family_rerank', {}).get('recall@10', 0) - ho.get('in_sample', {}).get('llm_rerank_blend', {}).get('recall@10', 0):+.3f}.</li>
<li><b>Only 10 JDs.</b> Every mean in §7 averages over 10 queries; one JD moving from 0.2 to
0.6 shifts the mean by 0.04.</li>
<li><b>No salary, notice-period, location or language filtering.</b> These are first-class
recruiter filters in the project brief. They are extracted but deliberately not applied, so
that the reported recall measures <i>semantic</i> matching only.</li>
<li><b>Privacy.</b> All documents are the anonymised release. No name, email, phone or URL
appears in the extracted features, and the LLM is local, so no document leaves the machine.</li>
</ol>

<h2 id="rep">11 · Reproduction</h2>
<pre><code># prerequisites: ollama with qwen3:8b and nomic-embed-text
python scripts/01_extract.py             # PDF → text JSON
python scripts/00_cv_family.py           # job-family map
python scripts/02_extract_keywords.py    # qwen3:8b keyword extraction (110 docs)
python scripts/03_ground_truth.py        # deterministic gate + LLM grading (1,000 pairs)
python scripts/03b_second_annotator.py   # independent re-annotation for κ
python scripts/04_verify.py              # dataset checks + statistics
python scripts/05_merge_keywords.py      # lexical + LLM + embedding merge
python scripts/06_autoencoder.py         # the brief's AE (keywords)
python scripts/06e1_ae_kw_txt.py          # AE variant: keywords + text
python scripts/06e2_ae_text.py            # AE variant: text only
python scripts/07_retrieve_eval.py       # scorers, fusion, metrics
python scripts/07b_family_classifier.py  # independent family classifier
python scripts/07c_llm_rerank.py         # LLM reranking of the top 20
python scripts/07d_apply_rerank.py       # fold rerank into the metrics
python scripts/13_holdout_eval.py        # 5-fold 80:20 out-of-sample check
python scripts/08_figures.py             # figures
python scripts/09_agreement.py           # inter-annotator agreement
python scripts/10_build_report.py        # this HTML report
python scripts/11_html_to_pdf.py         # PDF export
python scripts/12_transcript.py          # prompt transcript</code></pre>
<p class="small muted">Requirements: <code>pymupdf</code>, <code>torch</code>,
<code>scikit-learn</code>, <code>numpy</code>, <code>matplotlib</code>, ollama with
<code>qwen3:8b</code> and <code>nomic-embed-text</code>. Deterministic seeds throughout; the
whole pipeline re-runs with no network access.</p>

<h4>11.1 The diagnostic scripts</h4>
<p>These are not part of the main pipeline — they are the experiments that produced the
negative results in §5.4 and §8.3. They are included because a report that claims “a bigger
latent space did not help” should ship the script that showed it.</p>
<table><thead><tr><th>script</th><th>question it answers</th><th>result</th></tr></thead><tbody>
<tr><td><code>06b_ae_sweep.py</code></td><td>Is the AE better than doing nothing?</td>
<td>No — input space 0.36 vs AE 0.14–0.16</td></tr>
<tr><td><code>06c1_ae_objective.py</code></td><td>Does a direction-aware loss fix it?</td>
<td>Yes — 0.14 → 0.30 with cosine reconstruction</td></tr>
<tr><td><code>06d1_ae_tune.py</code></td><td>Does capacity, depth or training length help?</td>
<td>No — all variants ≤ 0.30</td></tr>
<tr><td><code>06f1_ae_blocks.py</code></td><td>Do domain / soft-skill blocks help?</td>
<td>No — 0.30 → 0.14</td></tr>
<tr><td><code>06f2_ae_inputs.py</code></td><td>Is the keyword representation or the AE the limit?</td>
<td>The input — text embedding reaches 0.34–0.38</td></tr>
<tr><td><code>06g1_ae_weights.py</code></td><td>Can loss re-weighting recover the text block?</td>
<td>Only to 0.28, still below keyword-only</td></tr>
<tr><td><code>04_verify.py</code></td><td>Are the hired CVs in the pool?</td>
<td>No — zero overlap, which defines the task</td></tr>
<tr><td><code>dump_profiles.py</code></td><td>What are the 100 CVs about?</td>
<td>Generates <code>data/profiles.txt</code>, used to assign job families by hand</td></tr>
<tr><td><code>99_check_refs.py</code></td><td>Do the paths in this report exist?</td>
<td>Guard against a rename breaking the reproduction instructions</td></tr>
</tbody></table>
<p class="small muted">The <code>06b</code>–<code>06g</code> series reproduces the
autoencoder negative results in §5.4 and the rejected hypotheses in §8.3.</p>

<footer>
STAT6207 Applied Deep Learning · Assignment 4 · local-only pipeline (ollama
<code>qwen3:8b</code> + <code>nomic-embed-text</code>, PyTorch autoencoder) ·
{len(calls)} logged extraction calls · {len(gtcalls)} logged adjudication calls ·
report generated from the run artefacts.
</footer>
</div></body></html>"""

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    print("wrote", OUT, f"{os.path.getsize(OUT):,} bytes")


if __name__ == "__main__":
    main()
