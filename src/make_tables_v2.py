"""Collect v2 results into the paper's tables (mean +- std over seeds).

    python src/make_tables_v2.py      ->  results/tables_v2.md
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

COLS = [("accuracy", "Acc"), ("step_fidelity", "Fid"),
        ("false_correct_rate", "FalseCorr"), ("format_compliance", "Fmt")]
ALL = r"(answer|process|scrambled)"


def f(v):
    if v is None:
        return "n/a"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def mstd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{statistics.mean(xs):.3f} +/- {statistics.stdev(xs):.3f}"


def main():
    res = Path("results")
    rows = defaultdict(lambda: defaultdict(list))
    for p in res.glob("*.summary.json"):
        s = json.load(p.open(encoding="utf-8"))
        stem = p.name[: -len(".summary.json")]
        m = re.match(rf"(.+?)-{ALL}-s(\d+)-(.+)", stem)
        if not m:
            continue
        size, arm, seed, which = m.groups()
        s["_seed"] = seed
        rows[(size, arm)][which].append(s)

    probes = {}
    for p in res.glob("probes_*.json"):
        s = json.load(p.open(encoding="utf-8"))
        m = re.match(rf"(.+?)-{ALL}-s(\d+)", s["tag"])
        if not m:
            continue
        size, arm, seed = m.groups()
        probes.setdefault((size, arm), []).append(s)

    out = ["# v2 results", ""]

    out += ["## Table 1 - In-distribution (mean +/- std over seeds)", "",
            "| Model | Arm | Acc | Fid | FalseCorr | Fmt |", "|---|---|---|---|---|---|"]
    for (size, arm), sets in sorted(rows.items()):
        ss = sets.get("indist", [])
        out.append(f"| {size} | {arm} | " +
                   " | ".join(mstd([x.get(k) for x in ss]) for k, _ in COLS) + " |")

    out += ["", "## Table 2 - Distribution shift and shortcuts", "",
            "| Model | Arm | Set | Acc | Fid |", "|---|---|---|---|---|"]
    for (size, arm), sets in sorted(rows.items()):
        for which, label in [("unseen", "unseen structure"),
                             ("numremoved", "number removed"),
                             ("counterbench", "CounterBench (OOD)")]:
            ss = sets.get(which, [])
            if not ss:
                continue
            if which == "counterbench":
                out.append(f"| {size} | {arm} | {label} | "
                           f"{mstd([x.get('accuracy') for x in ss])} | n/a |")
                continue
            out.append(f"| {size} | {arm} | {label} | " +
                       " | ".join(mstd([x.get(k) for x in ss])
                                  for k in ("accuracy", "step_fidelity")) + " |")

    out += ["", "## Table 3 - Intervention probes", "",
            "| Model | Arm | n | keep | flip up | false-flip down | step5 follow | always-yes |",
            "|---|---|---|---|---|---|---|---|"]
    for (size, arm), ps in sorted(probes.items()):
        n = ps[0]["n_items"]
        out.append(
            f"| {size} | {arm} | {n} | "
            f"{mstd([x['keep_consistency'] for x in ps])} | "
            f"{mstd([x['flip_rate'] for x in ps])} | "
            f"{mstd([x['false_flip_rate'] for x in ps])} | "
            f"{mstd([x['step5_follow_rate'] for x in ps])} | "
            f"{ps[0]['always_yes_baseline']['flip_rate']:.3f} |")

    out += ["", "## Table 4 - Per-step fidelity (in-distribution)", "",
            "| Model | Arm | step0 | step1 | step2 | step3 | step4 | step5 | end |",
            "|---|---|---|---|---|---|---|---|---|"]
    for (size, arm), sets in sorted(rows.items()):
        ss = sets.get("indist", [])
        if not ss:
            continue
        cells = [mstd([x.get("per_step", {}).get(k) for x in ss])
                 for k in ["step0", "step1", "step2", "step3", "step4", "step5", "end"]]
        out.append(f"| {size} | {arm} | " + " | ".join(cells) + " |")

    txt = "\n".join(out) + "\n"
    (res / "tables_v2.md").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"wrote {res / 'tables_v2.md'}")


if __name__ == "__main__":
    main()
