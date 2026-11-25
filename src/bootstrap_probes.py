"""Bootstrap 95% CIs for the intervention rates, and paired A-FT vs P-FT tests.

Reads the stored probe outputs, resamples items (1000 draws), and reports:
  - each model/arm's rate with a 95% CI
  - the paired A-FT minus P-FT difference with a 95% CI
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path("uploaded/results")
STEP5 = re.compile("step\\s*5\\s*[:\\uff1a]\\s*(-?\\d+(?:\\.\\d+)?)", re.I)
EQ = re.compile("=\\s*(-?\\d+(?:\\.\\d+)?)")
B = 1000
rng = np.random.default_rng(0)


def parse_result(t):
    if not t:
        return None
    m = STEP5.search(t)
    if m:
        return float(m.group(1))
    eq = EQ.findall(t)
    if eq:
        return float(eq[-1])
    m = re.search("-?\\d+(?:\\.\\d+)?", t)
    return float(m.group(0)) if m else None


# item_id -> per (model, arm, seed) -> tuple of indicators
data = defaultdict(dict)
for p in sorted(R.glob("probes_*.jsonl")):
    tag = p.stem.replace("probes_", "")
    m = re.match(r"(.+?)-(answer|process|scrambled)-s(\d+)", tag)
    if not m:
        continue
    model, arm, seed = m.groups()
    items = defaultdict(dict)
    for line in p.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            items[r["id"]][r["condition"]] = r
    for iid, c in items.items():
        if set(c) != {"keep", "flip", "neutral"}:
            continue
        vals = []
        for cond in c.values():
            v = parse_result(cond["output"])
            vals.append(1.0 if (v is not None and abs(v - cond["expected_result"]) <= 0.02) else 0.0)
        data[iid][(model, arm, seed)] = (
            float(c["keep"]["pred_answer"] == c["keep"]["gold_answer"]),
            float(c["flip"]["pred_answer"] == c["flip"]["expected_answer"]),
            float(c["neutral"]["pred_answer"] != c["neutral"]["expected_answer"]),
            sum(vals) / 3.0,
        )

models = sorted({k[0] for d in data.values() for k in d})
arms = ["answer", "process"]
seeds = sorted({k[2] for d in data.values() for k in d})
items = sorted(data)


def arm_matrix(model, arm, idx):
    out = []
    for iid in idx:
        vs = [data[iid].get((model, arm, s)) for s in seeds]
        vs = [v for v in vs if v is not None]
        if not vs:
            out.append(None)
            continue
        out.append(np.mean([v[arm_i] for v in vs]) for v in vs) if False else out.append(
            np.mean(np.array([list(v) for v in vs]), axis=0))
    return out


print(f"items={len(items)} seeds={len(seeds)}")
names = ["keep", "flip", "false_flip", "step5"]
for model in models:
    for arm in arms:
        mat = arm_matrix(model, arm, items)
        if any(x is None for x in mat):
            good = [i for i, x in enumerate(mat) if x is not None]
            sub = np.array([mat[i] for i in good])
        else:
            good, sub = list(range(len(items))), np.array(mat)
        n = len(sub)
        point = sub.mean(axis=0)
        boots = np.empty((B, 4))
        for b in range(B):
            boots[b] = sub[rng.integers(0, n, n)].mean(axis=0)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        cells = "  ".join(f"{names[j]}={point[j]:.3f}[{lo[j]:.3f},{hi[j]:.3f}]" for j in range(4))
        print(f"{model:5s} {arm:8s} n={n:3d}  {cells}")

print("\npaired differences (A-FT minus P-FT), 95% CI")
for model in models:
    a = arm_matrix(model, "answer", items)
    p = arm_matrix(model, "process", items)
    good = [i for i in range(len(items)) if a[i] is not None and p[i] is not None]
    A = np.array([a[i] for i in good])
    P = np.array([p[i] for i in good])
    d = A - P
    n = len(d)
    point = d.mean(axis=0)
    boots = np.empty((B, 4))
    for b in range(B):
        idx = rng.integers(0, n, n)
        boots[b] = d[idx].mean(axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    cells = "  ".join(
        f"{names[j]}={point[j]:+.3f}[{lo[j]:+.3f},{hi[j]:+.3f}]"
        + ("" if lo[j] * hi[j] > 0 else "*")
        for j in range(4))
    print(f"{model:5s} n={n:3d}  {cells}")
print("\n* = CI includes 0 (difference not significant)")
