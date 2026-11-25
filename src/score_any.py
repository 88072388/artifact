"""Model-anchored scoring of the 328-item probes for an arbitrary results dir.

Mirrors src/score_p328.py exactly (same definitions, same seed-averaging), but
takes the directory as an argument so 3B and 7B can be scored side by side:

    flip      = P(answer under flip differs from the model's own keep answer)
    keep      = P(keep answer equals gold)
    spurious  = P(answer under neutral differs from keep)
    step5     = mean over the three conditions of |parsed - expected| <= 0.02

    python src/score_any.py uploaded3/results results7b/results
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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


def collect(root):
    data = defaultdict(dict)
    for p in sorted(Path(root).glob("probes_*-p328.jsonl")):
        tag = p.stem.replace("probes_", "").replace("-p328", "")
        m = re.match(r"(.+?)-(answer|process)-s(\d+)", tag)
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
            k = c["keep"]["pred_answer"]
            f = c["flip"]["pred_answer"]
            h = c["neutral"]["pred_answer"]
            flip_ok = None if k is None or f is None else float(f != k)
            keep_ok = float(k == c["keep"]["gold_answer"]) if k else 0.0
            ff = None if k is None or h is None else float(h != k)
            res = [parse_result(cond["output"]) for cond in c.values()]
            s5 = np.mean([1.0 if (v is not None and abs(v - cond["expected_result"]) <= 0.02)
                          else 0.0 for v, cond in zip(res, c.values())])
            data[iid][(model, arm, seed)] = (keep_ok, flip_ok, ff, s5)
    return data


def scores(dirs, ci=False):
    """Per (model, arm): n and the four model-anchored metrics."""
    data = defaultdict(dict)
    for d in dirs:
        for iid, dct in collect(d).items():
            data[iid].update(dct)
    items = sorted(data)
    seeds = sorted({k[2] for d in data.values() for k in d})
    rng = np.random.default_rng(0)

    def ci_of(vals):
        a = np.array(vals, dtype=float)
        b = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(1000)])
        return a.mean(), *np.percentile(b, [2.5, 97.5])

    # sizes sort numerically ("0.5b"), foreign families ("llama") sort after them
    def _size_key(s):
        return (0, float(s.rstrip("b"))) if s[:1].isdigit() else (1, s)

    models = sorted({k[0] for d in data.values() for k in d}, key=_size_key)
    out = {}
    for model in models:
        for arm in ("answer", "process"):
            rows = []
            for iid in items:
                vs = [data[iid].get((model, arm, s)) for s in seeds]
                vs = [v for v in vs if v and v[1] is not None and v[2] is not None]
                if vs:
                    rows.append(np.mean(np.array([list(v) for v in vs], dtype=float), axis=0))
            if not rows:
                continue
            keep, flip, ff, s5 = zip(*rows)
            if ci:
                out[(model, arm)] = (len(rows),) + tuple(ci_of(a) for a in (keep, flip, ff, s5))
            else:
                out[(model, arm)] = (len(rows), np.mean(keep), np.mean(flip),
                                     np.mean(ff), np.mean(s5))
    return out


def main():
    dirs = sys.argv[1:]
    for (model, arm), v in scores(dirs, ci=True).items():
        cells = [f"keep={v[1][0]:.3f}[{v[1][1]:.3f},{v[1][2]:.3f}]",
                 f"flip={v[2][0]:.3f}[{v[2][1]:.3f},{v[2][2]:.3f}]",
                 f"false_flip={v[3][0]:.3f}[{v[3][1]:.3f},{v[3][2]:.3f}]",
                 f"step5={v[4][0]:.3f}[{v[4][1]:.3f},{v[4][2]:.3f}]"]
        print(f"{model:5s} {arm:8s} n={v[0]:3d}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
