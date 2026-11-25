"""Build the revised (v2) experimental materials.

Pipeline
--------
1. keep the 8,532 CLadder items that carry a step annotation (drops `backadj`)
2. drop ambiguous labels: numeric ground truth within 0.01 of the decision
   threshold (det-counterfactual excluded: there gt is a probability, gt==0 is a
   valid label, not an ambiguous one)
3. repair step5: annotate the terminal value as the ground-truth result, because
   the annotated arithmetic disagrees with it in a large share of items
4. split by story: 8 held-out stories (2 nonsense) -> in-distribution test,
   the whole `diamond` graph -> unseen-structure test, a validation slice, rest
   -> train
5. write A-FT (answer only) / P-FT (six steps + answer) training files
6. build the intervention probes: three conditions per item

Outputs land in ``work_v2/``.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    PROCESS_INSTRUCTION,
    build_prompt,
    has_steps,
    load_backgrounds,
    write_jsonl,
)

STEPS = ["step0", "step1", "step2", "step3", "step4", "step5"]
EXPR_OK = re.compile(r"^[0-9+\-*/(). ]+$")
P2 = re.compile(r"P\(\s*Y\s*=\s*1\s*\|\s*X\s*=\s*([01])\s*\)\s*=\s*([0-9.]+)")

# ------------------------------------------------------------------ filters

def numeric_gt(r) -> float | None:
    gt = r["meta"]["groundtruth"]
    if isinstance(gt, bool) or not isinstance(gt, (int, float)):
        return None
    return float(gt)


def is_ambiguous(r) -> bool:
    gt = numeric_gt(r)
    if gt is None:
        return False
    if r["meta"]["query_type"] == "det-counterfactual":
        return False  # gt in {0,1} is a probability, not an effect size
    return abs(gt) < 0.01


def repair_step5(r) -> tuple[dict, bool]:
    """Return a repaired reasoning dict; flag whether the item needed repair."""
    r = dict(r)
    reason = dict(r["reasoning"])
    gt = numeric_gt(r)
    if gt is None:
        return r, False
    s5 = (reason.get("step5") or "").strip()
    fixed = False
    if "=" in s5:
        lhs, rhs = s5.rsplit("=", 1)
        lhs_c = lhs.strip().replace(" ", "")
        try:
            value = float(rhs)
        except ValueError:
            value = None
        if EXPR_OK.match(lhs_c) and value is not None:
            try:
                computed = eval(lhs_c, {"__builtins__": {}}, {})
            except Exception:
                computed = None
            if computed is not None and abs(computed - gt) > 0.01:
                reason["step5"] = f"{lhs.strip()} = {gt:.2f}"
                fixed = True
            elif abs(value - gt) > 0.01:
                reason["step5"] = f"{lhs.strip()} = {gt:.2f}"
                fixed = True
    r["reasoning"] = reason
    return r, fixed


# ------------------------------------------------------------------ split

def choose_holdout_stories(records, n_real: int, n_nonsense: int, target: int, seed: int):
    """Pick stories whose item count is close to `target`, with 2 nonsense ones."""
    rng = random.Random(seed)
    by_story = defaultdict(int)
    for r in records:
        if r["meta"]["graph_id"] != "diamond":
            by_story[r["meta"]["story_id"]] += 1
    real = [s for s in by_story if not s.startswith("nonsense")]
    nons = [s for s in by_story if s.startswith("nonsense")]

    best, best_gap = None, 10**9
    for _ in range(20000):
        pick = rng.sample(real, n_real) + rng.sample(nons, n_nonsense)
        total = sum(by_story[s] for s in pick)
        gap = abs(total - target)
        if gap < best_gap:
            best, best_gap = pick, gap
        if gap == 0:
            break
    return best, sum(by_story[s] for s in best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/cladder-v1")
    ap.add_argument("--out-dir", default="work_v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=int, default=528)
    ap.add_argument("--indist-target", type=int, default=1089)
    ap.add_argument("--n-probe", type=int, default=0, help="deprecated, ignored")
    ap.add_argument("--probe-n-test", type=int, default=126)
    ap.add_argument("--probe-n-holdout", type=int, default=235)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    recs = json.load((data_dir / "cladder-v1-q-balanced.json").open(encoding="utf-8"))
    backgrounds = load_backgrounds(data_dir / "cladder-v1-meta-models.json")

    with_steps = [r for r in recs if has_steps(r)]
    dropped_backadj = len(recs) - len(with_steps)

    kept, dropped_amb, repaired = [], 0, 0
    for r in with_steps:
        if is_ambiguous(r):
            dropped_amb += 1
            continue
        r2, fixed = repair_step5(r)
        repaired += int(fixed)
        kept.append(r2)

    print(f"kept {len(kept)}  (dropped backadj {dropped_backadj}, ambiguous {dropped_amb}, "
          f"step5 repaired {repaired})")

    # --- split
    diamond = [r for r in kept if r["meta"]["graph_id"] == "diamond"]
    rest = [r for r in kept if r["meta"]["graph_id"] != "diamond"]

    holdout, held_n = choose_holdout_stories(
        rest, n_real=6, n_nonsense=2, target=args.indist_target, seed=args.seed
    )
    indist = [r for r in rest if r["meta"]["story_id"] in set(holdout)]
    pool = [r for r in rest if r["meta"]["story_id"] not in set(holdout)]
    rng.shuffle(pool)
    val = pool[: args.val_size]
    train = pool[args.val_size :]

    print(f"train {len(train)} / val {len(val)} / indist {len(indist)} / diamond {len(diamond)}")
    print(f"held-out stories ({len(holdout)}): {sorted(holdout)}")

    # --- supervision files
    def rows(rs, mode):
        out = []
        for r in rs:
            bg = backgrounds[r["meta"]["model_id"]]
            out.append({
                "id": r["question_id"],
                "prompt": build_prompt(r, bg, mode),
                "target": build_target_v2(r, mode),
            })
        return out

    write_jsonl(out_dir / "sft_answer_train.jsonl", rows(train, "answer"))
    write_jsonl(out_dir / "sft_process_train.jsonl", rows(train, "process"))
    write_jsonl(out_dir / "sft_scrambled_train.jsonl", rows_scrambled(train, backgrounds, args.seed))
    write_jsonl(out_dir / "val.jsonl", rows(val, "process"))

    # --- eval sets (process prompt, gold steps carried for scoring)
    def eval_rows(rs, num_removed=False):
        out = []
        for r in rs:
            rec = dict(r)
            if num_removed:
                # the probe keeps the background and the question only: without the
                # observed probabilities the item is not answerable in principle
                rec["given_info"] = ""
            out.append({
                "id": rec["question_id"],
                "prompt": build_prompt(rec, backgrounds[rec["meta"]["model_id"]], "process"),
                "gold_answer": rec["answer"],
                "gold_steps": rec["reasoning"],
                "query_type": rec["meta"]["query_type"],
                "rung": rec["meta"]["rung"],
                "graph_id": rec["meta"]["graph_id"],
                "story_id": rec["meta"]["story_id"],
                "num_removed": num_removed,
            })
        return out

    write_jsonl(out_dir / "eval_indist.jsonl", eval_rows(indist))
    write_jsonl(out_dir / "eval_unseen_struct.jsonl", eval_rows(diamond))
    six = {"ate", "ett", "correlation", "marginal", "nde", "nie"}
    write_jsonl(
        out_dir / "eval_numremoved.jsonl",
        eval_rows([r for r in indist + diamond if r["meta"]["query_type"] in six], True),
    )

    # --- intervention probes: drawn from the TEST sets, never from training items
    probes = build_probes(indist, diamond, backgrounds,
                          args.probe_n_test, args.probe_n_holdout, args.seed)
    write_jsonl(out_dir / "probes.jsonl", probes)

    manifest = {
        "total_raw": len(recs),
        "dropped_backadj_no_reasoning": dropped_backadj,
        "with_reasoning": len(with_steps),
        "dropped_ambiguous": dropped_amb,
        "step5_repaired": repaired,
        "usable": len(kept),
        "train": len(train), "val": len(val),
        "indist": len(indist), "unseen_struct": len(diamond),
        "scrambled": len(train),
        "heldout_stories": sorted(holdout),
        "probes": len(probes),
        "probe_items": len({p["id"] for p in probes}),
        "probe_by_type": dict(Counter(p["query_type"] for p in probes
                                      if p["condition"] == "keep")),
        "probe_by_source": dict(Counter(p["source"] for p in probes
                                        if p["condition"] == "keep")),
        "seed": args.seed,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def build_target_v2(r, mode):
    if mode == "answer":
        return f"Answer: {r['answer']}"
    rn = r["reasoning"]
    lines = [f"Step {i}: {(rn.get(k) or '').strip()}" for i, k in enumerate(STEPS)]
    lines.append(f"Conclusion: {(rn.get('end') or '').strip()}")
    lines.append(f"Answer: {r['answer']}")
    return "\n".join(lines)


# ------------------------------------------------------------------ scrambled control

def rows_scrambled(records, backgrounds, seed):
    """P-FT format and length, but Step 0-5 come from another item of the same type.

    Keeps this item's Answer, so the reasoning is deliberately wrong while the
    surface form is identical to P-FT: isolates the effect of *correct* steps.
    """
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for r in records:
        by_type[r["meta"]["query_type"]].append(r)
    queue = {}
    for qt, rs in by_type.items():
        shuffled = rs[:]
        rng.shuffle(shuffled)
        queue[qt] = shuffled

    out = []
    for r in records:
        qt = r["meta"]["query_type"]
        donor = queue[qt].pop(0)
        queue[qt].append(donor)
        if donor["question_id"] == r["question_id"] and len(queue[qt]) > 1:
            donor = queue[qt].pop(0)
            queue[qt].append(donor)
        rn = donor["reasoning"]
        lines = [f"Step {i}: {(rn.get(k) or '').strip()}" for i, k in enumerate(STEPS)]
        lines.append(f"Conclusion: {(rn.get('end') or '').strip()}")
        lines.append(f"Answer: {r['answer']}")
        out.append({
            "id": r["question_id"],
            "prompt": build_prompt(r, backgrounds[r["meta"]["model_id"]], "process"),
            "target": "\n".join(lines),
            "donor_id": donor["question_id"],
        })
    return out


# ------------------------------------------------------------------ probes

def _eligible(records):
    """Items whose answer is the sign of P(Y=1|X=1) - P(Y=1|X=0)."""
    pool = []
    for r in records:
        if r["meta"]["query_type"] not in ("ate", "ett"):
            continue
        d = dict(P2.findall(r["reasoning"].get("step4", "")))
        if "0" not in d or "1" not in d:
            continue
        p0, p1 = float(d["0"]), float(d["1"])
        if (r["answer"] == "yes") != (p1 > p0):
            continue
        if abs(p1 - p0) < 0.05:
            continue
        pool.append((r, p0, p1))
    return pool


def build_probes(test_recs, heldout_recs, backgrounds, n_test, n_hold, seed):
    """Three conditions per item (keep / flip / neutral), drawn from the TEST sets.

    The item is fed as a normal user turn; the gold Steps 0-4 become an assistant
    prefix, and the model must continue from Step 5. The question text is never
    edited, so only the numbers the model is handed change.
    """
    rng = random.Random(seed)
    picked = []
    for recs, n, source in ((test_recs, n_test, "test"),
                            (heldout_recs, n_hold, "heldout")):
        pool = _eligible(recs)
        rng.shuffle(pool)
        print(f"  probe pool [{source}]: {len(pool)} eligible, taking {min(n, len(pool))}")
        picked += [(r, p0, p1, source) for r, p0, p1 in pool[:n]]

    probes = []
    for r, p0, p1, source in picked:
        bg = backgrounds[r["meta"]["model_id"]]
        gold = r["reasoning"]
        user = f"{bg}\n\n{r['given_info']}\n\n{r['question']}\n\n{PROCESS_INSTRUCTION}"

        def prefix(a, b):
            return (
                f"Step 0: {gold.get('step0', '')}\n"
                f"Step 1: {gold.get('step1', '')}\n"
                f"Step 2: {gold.get('step2', '')}\n"
                f"Step 3: {gold.get('step3', '')}\n"
                f"Step 4: P(Y=1 | X=0) = {a:.2f}\n"
                f"        P(Y=1 | X=1) = {b:.2f}\n"
                "Step 5:"
            )

        shift = 0.1 if max(p0, p1) <= 0.9 else -0.1
        conditions = {
            "keep": (p0, p1, r["answer"]),
            "flip": (p1, p0, "no" if r["answer"] == "yes" else "yes"),
            "neutral": (round(p0 + shift, 2), round(p1 + shift, 2), r["answer"]),
        }
        for name, (a, b, expected) in conditions.items():
            probes.append({
                "probe_id": f"{r['question_id']}-{name}",
                "id": r["question_id"],
                "condition": name,
                "source": source,
                "query_type": r["meta"]["query_type"],
                "user": user,
                "assistant_prefix": prefix(a, b),
                "expected_answer": expected,
                "gold_answer": r["answer"],
                "p0": a, "p1": b,
                "expected_result": round(b - a, 4),
            })
    return probes


if __name__ == "__main__":
    main()
