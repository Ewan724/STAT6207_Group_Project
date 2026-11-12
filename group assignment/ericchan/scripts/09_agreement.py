"""Step 9: inter-annotator agreement between the two annotation protocols."""
import json
import os
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")


def kappa(a, b, labels):
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def kappa_quad(a, b, k=4):
    """Cohen's kappa with quadratic weights, computed from the confusion matrix."""
    n = len(a)
    if n == 0:
        return 0.0
    O = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[x][y] += 1
    r = [sum(O[i]) for i in range(k)]
    c = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            E = r[i] * c[j] / n
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * O[i][j]
            den += w * E
    return 1 - num / den if den else 0.0


def main():
    gt = {}
    for line in open(os.path.join(DATA, "ground_truth.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("score", -1) >= 0:
                gt[(r["jd_id"], r["cv_id"])] = r["score"]
    v2 = {}
    p = os.path.join(DATA, "ground_truth_v2.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("score", -1) >= 0:
                    v2[(r["jd_id"], r["cv_id"])] = r["score"]

    keys = [k for k in v2 if k in gt]
    a = [gt[k] for k in keys]
    b = [v2[k] for k in keys]
    out = {
        "n": len(keys),
        "exact": round(sum(1 for x, y in zip(a, b) if x == y) / max(len(keys), 1), 4),
        "within1": round(sum(1 for x, y in zip(a, b) if abs(x - y) <= 1)
                         / max(len(keys), 1), 4),
        "kappa_quad": round(kappa_quad(a, b), 4),
        "kappa_bin": round(kappa([int(x >= 2) for x in a],
                                 [int(x >= 2) for x in b], [0, 1]), 4),
        "grade_dist_v1": dict(Counter(a)),
        "grade_dist_v2": dict(Counter(b)),
    }
    json.dump(out, open(os.path.join(DATA, "agreement.json"), "w", encoding="utf-8"),
              indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
