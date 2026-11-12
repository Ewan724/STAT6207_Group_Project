"""Step 5: merge / cluster similar keywords into canonical skill vocabularies.

Problem
    LLM extraction produced 2,254 distinct surface forms for a few hundred real
    skills: PostgreSQL/Postgres, Gcp/Google Cloud Platform, K8s/Kubernetes,
    "API Testing"/"API testing".  Exact matching on raw strings therefore
    penalises documents for spelling, not for substance.

Three stages, each fixing what the previous cannot
--------------------------------------------------
1. LEXICAL (deterministic)
   lowercase, punctuation strip, whitespace collapse, guarded plural
   singularisation, plus a curated alias table (postgres->postgresql,
   k8s->kubernetes, ...).  Reproducible, catches the easy majority, and is
   applied first so the LLM stage cannot *un*-merge something already fixed.

2. LLM (qwen3:8b)
   Batches of surface forms are shown to the model, which groups true synonyms
   and must justify every group.  Catches abbreviations and spellings no rule
   table anticipates.

3. EMBEDDING (nomic-embed-text)
   Cosine similarity confirms or vetoes groups proposed in (2); it never
   proposes merges itself.  The space is too coarse to drive merging alone:
   MySQL<->SQL scores 0.71 and PostgreSQL<->MySQL 0.70, both HIGHER than the
   correct pair Machine Learning<->ML at 0.63, so a threshold that accepts the
   good merges also accepts the bad ones.  Used as a sanity check only.

Separate vocabularies per field
    Hard skills, soft skills and domains are merged independently and kept as
    distinct feature blocks.  Pooling them would let "communication" compete
    with "postgresql" as a match signal, which is exactly the noise a recruiter
    would object to.

Output: data/skills.json
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

MIN_DF_HARD = 2
MIN_DF_SOFT = 2
MIN_DF_DOM = 2
SIM_CONFIRM = 0.93   # embedding confirms outright
SIM_WEAK = 0.72      # confirms only when the LLM also justified the merge
SIM_VETO = 0.55      # below this the embedding overrides the LLM

ALIAS = {
    "postgres": "postgresql", "psql": "postgresql", "postgre": "postgresql",
    "k8s": "kubernetes", "kube": "kubernetes", "kubernets": "kubernetes",
    "gcp": "google cloud platform", "google cloud": "google cloud platform",
    "aws": "amazon web services", "amazon web services": "amazon web services",
    "azure": "microsoft azure", "microsoft azure": "microsoft azure",
    "js": "javascript", "ts": "typescript", "nodejs": "node.js",
    "node": "node.js", "reactjs": "react", "react js": "react",
    "nextjs": "next.js", "next js": "next.js",
    "vuejs": "vue.js", "vue js": "vue.js",
    "ml": "machine learning", "dl": "deep learning",
    "nlp": "natural language processing",
    "llm": "large language model", "llms": "large language model",
    "genai": "generative ai", "gen ai": "generative ai",
    "ai": "artificial intelligence",
    "ci cd": "ci/cd", "cicd": "ci/cd",
    "rest": "rest api", "restful": "rest api", "restful api": "rest api",
    "rest apis": "rest api", "restful apis": "rest api",
    "qa": "quality assurance", "sdet": "quality assurance",
    "ui ux": "ui/ux design", "ui/ux": "ui/ux design",
    "ux": "ux design", "ui": "ui design",
    "hr": "human resources",
    "bd": "business development", "kam": "key account management",
    "sql server": "microsoft sql server", "mssql": "microsoft sql server",
    "powerbi": "power bi",
    "spark": "apache spark", "kafka": "apache kafka",
    "airflow": "apache airflow", "hadoop": "apache hadoop",
    "ms office": "microsoft office", "msoffice": "microsoft office",
    "excel": "microsoft excel", "ms excel": "microsoft excel",
    "advanced excel": "microsoft excel",
    "photoshop": "adobe photoshop", "illustrator": "adobe illustrator",
    "premiere pro": "adobe premiere pro",
    "after effects": "adobe after effects",
    "power point": "microsoft powerpoint", "powerpoint": "microsoft powerpoint",
    "a/b test": "a/b testing", "ab testing": "a/b testing",
    "fp a": "financial planning & analysis",
    "etl elt": "etl",
    "cnn": "convolutional neural network",
    "rl": "reinforcement learning",
    "sop": "standard operating procedure",
    "3pl": "third-party logistics",
    "wms": "warehouse management system",
    "oms": "order management system",
    "erp": "enterprise resource planning",
    "crm": "customer relationship management",
    "seo": "search engine optimisation",
    "sem": "search engine marketing",
    "gtm": "go-to-market strategy",
    "kpi": "key performance indicator",
    "e commerce": "e-commerce",
}

STOP = {"", "n/a", "na", "none", "various", "other", "others", "etc", "and",
        "misc", "multiple", "general", "skills", "experience", "knowledge",
        "skill", "tool", "technology"}

# Terms the extractor correctly finds in a JD but which are not *matchable skill
# dimensions*: degrees, personality traits and language requirements are handled
# by dedicated structural features, and leaving them in the hard-skill block
# would let "self-motivation" score the same as "postgresql".
DROP_PATTERNS = [
    re.compile(p) for p in [
        r"\b(degree|bachelor|master|phd|diploma|graduate|graduation|university|"
        r"certification|certificate|equivalent)\b",
        r"\b(mindset|motivation|self-motivat|attitude|ethic|passion|driven|"
        r"proactive|high-agency|detail-oriented|empathy|willingness|"
        r"fast learner|self-management|adaptab|flexib|resilien|curios)\b",
        r"\b(english|mandarin|cantonese|malay|indonesian|thai|vietnamese|"
        r"japanese|korean|language proficiency|spoken|written)\b",
        r"\b(year|month|salary|benefit|leave|insurance|visa|notice period)\b",
        r"\b(team player|teamwork spirit|hard work)\b",
    ]
]

# words ending in -s that are not plurals
KEEP_S = {"kubernetes", "serverless", "business", "analysis", "analytics",
          "logistics", "devops", "aws", "redis", "postgres", "microservices",
          "statistics", "physics", "economics", "news", "always", "series",
          "operations", "communications", "sales", "dns", "windows", "ios",
          "android", "nlp", "css", "sass", "scss", "jenkins", "sap",
          "ansible", "grafana", "terraform", "pandas", "numpy", "spark",
          "kafka", "jira", "confluence", "figma", "photoshop", "illustrator",
          "express", "typescript", "javascript", "gcp", "canvas", "iot",
          "business intelligence", "data analysis", "salesforce", "hubspot"}

NONWORD = re.compile(r"[^a-z0-9+#./& -]+")
VENDOR = re.compile(r"^(apache|adobe|microsoft|google|amazon|ibm|oracle|sap)\s+")

# The extractor writes noun phrases ("sql proficiency", "azure databricks
# experience").  Those suffixes carry no skill identity, so they are stripped
# deterministically before anything else looks at the term.  Order matters:
# longest suffix first.
SUFFIXES = [
    " hands on experience", " hands-on experience", " hands on", " hands-on",
    " working experience", " work experience", " professional experience",
    " proficiency", " proficient", " knowledge", " expertise", " competency",
    " competencies", " experience", " skills", " skill", " capability",
    " capabilities", " understanding", " familiarity", " exposure",
    " background", " ability", " abilities", " techniques", " technique",
    " principles", " concepts", " fundamentals", " practice", " practices",
    " tools", " tool", " programming",
]
PREFIXES = [
    "strong ", "advanced ", "expert ", "solid ", "good ", "excellent ",
    "proven ", "deep ", "hands on ", "hands-on ", "proficient in ",
    "experience in ", "experience with ", "knowledge of ", "understanding of ",
    "familiarity with ", "ability to ", "such as ", "including ",
    "e.g. ", "e.g., ", "basic ", "intermediate ",
]


def strip_affixes(s):
    s = s.strip()
    changed = True
    while changed:
        changed = False
        for p in PREFIXES:
            if s.startswith(p) and len(s) > len(p) + 2:
                s = s[len(p):].strip()
                changed = True
        for suf in SUFFIXES:
            if s.endswith(suf) and len(s) > len(suf) + 2:
                s = s[: -len(suf)].strip(" -/,")
                changed = True
                break
    return s


ENUM = re.compile(r"\s*(?:\be\.g\.,?|\bsuch as\b|\bincluding\b|\bi\.e\.,?|"
                  r"\blike\b|/|,)\s*", re.I)
ENUM_MARK = re.compile(r"(e\.g\.|such as|including|i\.e\.)", re.I)


def split_enum(s):
    """'ui/ux design tools e.g. figma adobe xd' -> concrete skills.

    The extractor sometimes copies an enumeration straight out of the JD.  Left
    whole, the phrase matches nothing on the CV side; split, "figma" and
    "adobe xd" match exactly what candidates write.  We only split when an
    explicit enumeration marker is present, so ordinary multi-word skills such
    as "machine learning" are untouched.
    """
    s = str(s)
    if not ENUM_MARK.search(s):
        return [s]
    parts = [p.strip(" .,;:-") for p in ENUM.split(s)]
    parts = [p for p in parts if len(p) > 2]
    return parts or [s]



def norm(s):
    s = str(s).lower().strip().replace("&", " and ")
    s = NONWORD.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" -./")
    s = strip_affixes(s)
    if (s.endswith("s") and len(s) > 3 and s not in KEEP_S
            and not s.endswith(("ss", "us", "is", "js", "as", "ps", "es"))):
        s = s[:-1]
    return s


def canonical_lexical(s):
    n = norm(s)
    if n in STOP or len(n) < 2:
        return None
    if n in ALIAS:
        return ALIAS[n]
    n2 = VENDOR.sub("", n)
    if n2 in ALIAS:
        return ALIAS[n2]
    return n


def canonical_hard(s):
    """Hard-skill canonicalisation: lexical form, then drop non-skill phrases."""
    n = canonical_lexical(s)
    if not n:
        return None
    for pat in DROP_PATTERNS:
        if pat.search(n):
            return None
    # a phrase with no alphabetic content or a single letter is not a skill
    if len(re.sub(r"[^a-z]", "", n)) < 2:
        return None
    return n


MERGE_SYSTEM = (
    "You build a controlled vocabulary of professional skills for a recruitment "
    "search engine. You group surface forms that denote the SAME underlying "
    "skill. Return strict JSON only."
)

MERGE_PROMPT = """Below is a numbered list of skill terms taken from CVs and job
descriptions. Group only the terms that denote the same underlying skill.

Rules:
- Merge true synonyms, spelling variants and abbreviation/brand pairs only,
  e.g. "postgres"/"postgresql", "k8s"/"kubernetes", "gcp"/"google cloud".
- NEVER merge a specific tool into a broader category. "mysql" and "sql" are
  different skills. "pytorch" and "machine learning" are different skills.
  "figma" and "design" are different skills.
- NEVER merge two different tools that happen to be used together.
- Every group needs at least 2 members, and you must give a one-line reason.
- Terms that belong to no group: simply omit them.

Terms:
{terms}

Return ONLY:
{{"groups":[{{"canonical":"...","members":["...","..."],"why":"short reason"}}]}}
"""


def llm_merge_groups(terms, batch=28):
    groups = []
    n_batch = (len(terms) + batch - 1) // batch
    for k in range(0, len(terms), batch):
        chunk = terms[k:k + batch]
        got = None
        raw_last = ""
        for attempt in range(2):
            sub = chunk if attempt == 0 else chunk[: max(8, len(chunk) // 2)]
            listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(sub))
            prompt = MERGE_PROMPT.format(terms=listing)
            try:
                raw_last = llm.chat(prompt, system=MERGE_SYSTEM, num_predict=1600,
                                    num_ctx=8192, seed=42 + attempt)
                got = llm.parse_json(raw_last)
                llm.append_jsonl(os.path.join(DATA, "merge_calls.jsonl"), {
                    "doc_id": f"batch{k // batch}", "kind": "merge",
                    "system": MERGE_SYSTEM, "prompt": prompt,
                    "raw": raw_last, "attempt": attempt})
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 1:
                    print(f"  batch {k // batch} failed: {e!r}", flush=True)
        if got:
            for g in got.get("groups", []):
                members = [str(m).strip() for m in (g.get("members") or [])
                           if str(m).strip()]
                canon = str(g.get("canonical", "")).strip()
                if len(members) < 2 or not canon:
                    continue
                groups.append({"canonical": canon, "members": members,
                               "why": str(g.get("why", ""))[:120]})
        print(f"  batch {k // batch + 1}/{n_batch} -> {len(groups)} groups",
              flush=True)
    return groups


def cos(a, b):
    d = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return sum(x * y for x, y in zip(a, b)) / (d + 1e-9)


# --------------------------------------------------------------- merge guard
# The LLM proposes groups; this guard decides whether a proposal is safe.
# It exists because the model will happily merge "data analysis" with
# "data management" (same words, different skills) when the embedding also
# thinks they are close.  The guard encodes the linguistic structure of a skill
# name: the HEAD NOUN is what the skill is about, and two skills with different
# head nouns are different skills unless one is an abbreviation of the other.

def head(s):
    """Last content word - the noun the phrase is really about."""
    toks = [w for w in s.split() if w not in ("of", "and", "for", "in", "the", "a")]
    return toks[-1] if toks else s


def is_acronym_pair(a, b):
    """'gcp' / 'google cloud platform', 'ml' / 'machine learning'."""
    sa, sb = a.split(), b.split()
    if len(sa) == 1 and len(sa[0]) <= 5 and len(sb) > 1:
        return sa[0][0] == sb[0][0] and (
            len(sa[0]) == len(sb) or sa[0] in "".join(w[0] for w in sb))
    if len(sb) == 1 and len(sb[0]) <= 5 and len(sa) > 1:
        return sb[0][0] == sa[0][0] and (
            len(sb[0]) == len(sa) or sb[0] in "".join(w[0] for w in sa))
    return False


def is_substring_pair(a, b):
    """'apache airflow' / 'airflow', 'postgresql' / 'postgresql database'.

    Direction matters.  The CANONICAL must be the more specific name and the
    member the shorter/vendor-stripped variant.  The reverse - merging
    "google cloud platform" into a generic "cloud platform" - is a category
    collapse and must be refused, because it destroys the identity of the
    product and lets unrelated cloud experience score as a match.
    """
    ta, tb = set(a.split()), set(b.split())   # a = canonical, b = member
    if ta == tb:
        return True
    if not (tb < ta):          # member must be a strict subset of canonical
        return False
    extra = ta - tb
    generic = {"database", "db", "api", "framework", "library", "language",
               "platform", "tool", "server", "cloud", "service", "system",
               "engine", "suite", "stack", "management", "development"}
    vendor = {"apache", "adobe", "microsoft", "google", "amazon", "ibm",
              "oracle", "sap", "gitlab", "github", "aws", "azure", "gcp",
              "elastic", "meta", "salesforce", "atlassian", "jetbrains",
              "apache", "premiere", "after", "effects"}
    return extra <= (generic | vendor)


def merge_ok(canon, member, mean_sim, why):
    """Decide whether canon <- member is a safe merge."""
    if canon == member:
        return True
    if is_acronym_pair(canon, member):
        return True
    if is_substring_pair(canon, member):
        return True
    # same head noun + one is a prefix/superset phrase, e.g. "brand positioning"
    # / "product positioning" -> still different, so require a real synonym
    # signal: identical head noun AND high embedding similarity AND justification
    if head(canon) == head(member):
        return mean_sim >= 0.90 and len(why) >= 8
    # different head nouns: only an extremely confident embedding may merge
    return mean_sim >= 0.97 and len(why) >= 8


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def merge_block(rows, field_list, min_df, tag, batch=28, canon_fn=None,
                keep_rare=False):
    canon_fn = canon_fn or canonical_lexical
    df = Counter()
    for r in rows:
        for f in field_list:
            for t in (r.get(f) or []):
                for piece in split_enum(t):
                    c = canon_fn(piece)
                    if c:
                        df[c] += 1
    print(f"[{tag}] {len(df)} canonical forms, "
          f"{sum(1 for v in df.values() if v >= min_df)} with df >= {min_df}",
          flush=True)

    target = [t for t, v in df.most_common() if v >= min_df]
    groups = llm_merge_groups(target, batch=batch) if len(target) > 1 else []
    print(f"[{tag}] {len(groups)} groups proposed", flush=True)

    mem_all = set()
    for g in groups:
        mem_all.add(canonical_lexical(g["canonical"]) or norm(g["canonical"]))
        for m in g["members"]:
            mc = canonical_lexical(m) or norm(m)
            if mc:
                mem_all.add(mc)
    mem_all = sorted(m for m in mem_all if m)
    vecs = dict(zip(mem_all, llm.embed(mem_all))) if mem_all else {}

    uf = UF()
    for t in df:
        uf.find(t)
    accepted, rejected = [], []
    for g in groups:
        canon = canonical_lexical(g["canonical"]) or norm(g["canonical"])
        mem = []
        for m in g["members"]:
            c = canonical_lexical(m) or norm(m)
            if c and c != canon and c not in mem:
                mem.append(c)
        if not canon or not mem:
            continue
        cv = vecs.get(canon)
        if cv is None:
            continue
        sims = [cos(vecs.get(m, cv), cv) for m in mem]
        mean_sim = sum(sims) / len(sims)
        safe = [m for m in mem
                if merge_ok(canon, m, cos(vecs.get(m, cv), cv), g["why"])]
        rec = {**g, "canonical": canon, "members": mem,
               "mean_sim": round(mean_sim, 3),
               "accepted_members": safe}
        if safe:
            for m in safe:
                uf.union(canon, m)
            accepted.append(rec)
        else:
            rejected.append({**rec, "why_rejected":
                             "guard/embedding veto (different head noun "
                             "or low similarity)"})
    print(f"[{tag}] accepted {len(accepted)}, vetoed {len(rejected)}", flush=True)

    clusters = defaultdict(set)
    for t in df:
        clusters[uf.find(t)].add(t)
    label = {}
    for root, mem in clusters.items():
        # prefer the form the corpus actually uses most; then the more compact
        # spelling.  Frequency first matters because a rare long form should not
        # become the canonical name for a widely used short one.
        label[root] = sorted(mem, key=lambda x: (-df.get(x, 0), len(x.split()),
                                                 len(x)))[0]
    alias = {m: label[uf.find(m)] for root, mem in clusters.items() for m in mem}

    doc_sets = {}
    for r in rows:
        sk = set()
        for f in field_list:
            for t in (r.get(f) or []):
                for piece in split_enum(t):
                    c = canon_fn(piece)
                    if c and c in alias:
                        sk.add(alias[c])
        doc_sets[r["doc_id"]] = sk

    vdf = Counter()
    for s in doc_sets.values():
        vdf.update(s)
    # Pruning rule: keep a skill if it is frequent enough OR if it appears in a
    # JD.  JD requirements are the query side, so dropping a term that only the
    # JD uses (e.g. "azure synapse") would silently remove a requirement from
    # the query.  For hard skills we keep rare terms too: a skill mentioned by a
    # single candidate is precisely what distinguishes them, and pruning it
    # removes signal rather than noise.
    jd_ids = {r["doc_id"] for r in rows if r["kind"] == "jd"}
    in_jd = {s for d, ss in doc_sets.items() if d in jd_ids for s in ss}
    keep = ({s for s, v in vdf.items() if v >= min_df} | in_jd)
    if keep_rare:
        keep |= set(vdf)
    doc_sets = {d: sorted(s & keep) for d, s in doc_sets.items()}
    vdf = Counter()
    for s in doc_sets.values():
        vdf.update(s)

    # Two tiers, used by different consumers:
    #   core  - df >= 2 (plus every JD term).  Dense enough to be a sane neural
    #           input; used to build the autoencoder's skill block.
    #   full  - every canonical skill.  Used by the IDF-weighted overlap scorer,
    #           where a rare shared skill is informative rather than noise.
    core = {s for s, v in vdf.items() if v >= min_df} | in_jd
    doc_core = {d: sorted(set(ss) & core) for d, ss in doc_sets.items()}

    real = {}
    for c in keep:
        mem = sorted(clusters[uf.find(c)])
        if len(mem) > 1:
            real[c] = mem
    return {
        "n_surface": len(df), "n_canonical": len(keep),
        "vocab": [{"skill": k, "df": v} for k, v in vdf.most_common()],
        "alias_map": {k: v for k, v in alias.items() if v in keep},
        "clusters": real,
        "accepted": accepted, "rejected": rejected[:30],
        "doc_sets": doc_sets,
        "doc_core": doc_core,
    }


def main():
    rows = [json.loads(l) for l in open(os.path.join(DATA, "extracted.jsonl"),
                                        encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("ok")]
    cvs = {c["cv_id"]: c for c in json.load(
        open(os.path.join(DATA, "cvs_raw.json"), encoding="utf-8"))}
    jds = {j["jd_id"]: j for j in json.load(
        open(os.path.join(DATA, "jds_raw.json"), encoding="utf-8"))}

    hard = merge_block(rows, ["hard_skills", "must_have", "nice_to_have"],
                       MIN_DF_HARD, "hard", canon_fn=canonical_hard,
                       keep_rare=True)
    soft = merge_block(rows, ["soft_skills"], MIN_DF_SOFT, "soft")
    dom = merge_block(rows, ["domains"], MIN_DF_DOM, "domain")

    out = {
        "n_surface_forms": hard["n_surface"],
        "n_canonical": hard["n_canonical"],
        "min_df": MIN_DF_HARD,
        "vocab": hard["vocab"],
        "alias_map": hard["alias_map"],
        "clusters": hard["clusters"],
        "merge_groups_accepted": hard["accepted"],
        "merge_groups_rejected": hard["rejected"],
        "doc_skills": hard["doc_sets"],
        "doc_skills_core": hard["doc_core"],
        "vocab_soft": soft["vocab"], "doc_soft": soft["doc_sets"],
        "clusters_soft": soft["clusters"],
        "vocab_dom": dom["vocab"], "doc_dom": dom["doc_sets"],
        "clusters_dom": dom["clusters"],
        "doc_meta": {**{k: {"kind": "cv"} for k in cvs},
                     **{k: {"kind": "jd", "title": v["title"]}
                        for k, v in jds.items()}},
    }
    json.dump(out, open(os.path.join(DATA, "skills.json"), "w", encoding="utf-8"),
              indent=1)

    print(f"\nhard: {hard['n_surface']} -> {hard['n_canonical']} canonical")
    print(f"surface forms hard={hard['n_surface']} soft={soft['n_surface']} "
          f"dom={dom['n_surface']}")
    print(f"soft: {soft['n_surface']} -> {soft['n_canonical']}")
    print(f"domain: {dom['n_surface']} -> {dom['n_canonical']}")
    print("top 30 hard:", [v["skill"] for v in hard["vocab"][:30]])
    print("\nlargest hard clusters:")
    for k, m in sorted(hard["clusters"].items(), key=lambda x: -len(x[1]))[:14]:
        print(f"   {k:26s} <- {m}")


if __name__ == "__main__":
    main()
