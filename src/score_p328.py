"""Model-anchored scoring of the 328-item probes, side by side with the 157-item numbers."""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path("uploaded3/results") if Path("uploaded3/results").exists() else Path("uploaded3")
STEP5 = re.compile("step\\s*5\\s*[:\\uff1a]\\s*(-?\\d+(?:\\.\\d+)?)", re.I)
EQ = re.compile("=\\s*(-?\\d+(?:\\.\\d+)?)")


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


data = defaultdict(dict)
for p in sorted(R.glob("probes_*-p328.jsonl")):
    tag = p.stem.replace("probes_", "").replace("-p328", "")
    m = re.match(r"(.+?)-(answer|process)-s(\d+)", tag)
    model, arm, seed = m.groups()
    items = defaultdict(dict)
    for line in p.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            items[r["id"]][r["condition"]] = r
    for iid, c in items.items():
        if set(c) != {"keep", "flip", "neutral"}:
            continue
        k = c["keep"]["pred_answer"]
        f = c["flip"]["pred_answer"]
        h = c["neutral"]["pred_answer"]
        flip_ok = None if k is None or f is None else float(f != k)
        keep_ok = float(k == c["keep"]["gold_answer"]) if k else 0.0
        ff = None if k is None or h is None else float(h != k)
        res = [parse_result(cond["output"]) for cond in c.values()]
        s5 = np.mean([1.0 if (v is not None and abs(v - cond["expected_result"]) <= 0.02) else 0.0
                      for v, cond in zip(res, c.values())])
        data[iid][(model, arm, seed)] = (keep_ok, flip_ok, ff, s5)

items = sorted(data)
seeds = sorted({k[2] for d in data.values() for k in d})
rng = np.random.default_rng(0)


def mat(model, arm):
    out = []
    for iid in items:
        vs = [data[iid].get((model, arm, s)) for s in seeds]
        vs = [v for v in vs if v and v[1] is not None and v[2] is not None]
        out.append(np.mean(np.array([list(v) for v in vs], dtype=float), axis=0) if vs else None)
    return out


def ci(vals):
    a = np.array(vals, dtype=float)
    b = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(1000)])
    return a.mean(), *np.percentile(b, [2.5, 97.5])


print("=== 328 items, model-anchored (mean over seeds, 95% CI) ===")
for model in ["0.5b", "1.5b", "3b"]:
    for arm in ["answer", "process"]:
        rows = mat(model, arm)
        rows = [r for r in rows if r is not None]
        if not rows:
            print(f"{model:5s} {arm:8s} no parseable answers")
            continue
        keep, flip, ff, s5 = zip(*rows)
        cells = []
        for name, arr in [("keep", keep), ("flip", flip), ("false_flip", ff), ("step5", s5)]:
            m, lo, hi = ci(arr)
            cells.append(f"{name}={m:.3f}[{lo:.3f},{hi:.3f}]")
        print(f"{model:5s} {arm:8s} n={len(rows):3d}  " + "  ".join(cells))

print("\n=== paired A-FT minus P-FT (328 items) ===")
for model in ["0.5b", "1.5b", "3b"]:
    A, P = mat(model, "answer"), mat(model, "process")
    idx = [i for i in range(len(items)) if A[i] is not None and P[i] is not None]
    d = np.array([A[i] for i in idx]) - np.array([P[i] for i in idx])
    cells = []
    for j, name in enumerate(["keep", "flip", "false_flip", "step5"]):
        m, lo, hi = ci(d[:, j])
        sig = "" if lo * hi > 0 else "*"
        cells.append(f"{name}={m:+.3f}[{lo:+.3f},{hi:+.3f}]{sig}")
    print(f"{model:5s} n={len(idx):3d}  " + "  ".join(cells))
print("\n* = CI includes 0")

