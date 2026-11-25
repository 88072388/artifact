"""Build the 328-item probe set (advisor's suggestion, gap >= 0.05, both polarities).

Unlike the original 157-item set, no requirement is placed on the gold answer matching the
sign of p1-p0. Scoring is therefore done model-anchored afterwards: the flip condition is
correct when the model's answer inverts relative to its own keep answer, which tests
sensitivity rather than correctness. Step-5 follow stays objective.
"""
import json
import re
from pathlib import Path

P2 = re.compile(r"P\(\s*Y\s*=\s*1\s*\|\s*X\s*=\s*([01])\s*\)\s*=\s*([0-9.]+)")
raw = {str(r["question_id"]): r for r in
       json.load(open("data/cladder-v1/cladder-v1-q-balanced.json", encoding="utf-8"))}


def load(n):
    return [json.loads(l) for l in open(Path("work_v2") / n, encoding="utf-8") if l.strip()]


test, held = load("eval_indist.jsonl"), load("eval_unseen_struct.jsonl")


def eligible(rows):
    out = []
    for row in rows:
        r = raw.get(str(row["id"]))
        if not r or r["meta"]["query_type"] not in ("ate", "ett"):
            continue
        d = dict(P2.findall(r["reasoning"].get("step4", "")))
        if "0" not in d or "1" not in d:
            continue
        p0, p1 = float(d["0"]), float(d["1"])
        if abs(p1 - p0) < 0.05:
            continue
        out.append((row, p0, p1))
    return out


pool_t, pool_h = eligible(test), eligible(held)
print(f"pool test={len(pool_t)} heldout={len(pool_h)} total={len(pool_t)+len(pool_h)}")

probes = []
for source, pool in (("test", pool_t), ("heldout", pool_h)):
    for row, p0, p1 in pool:
        gold = row["gold_steps"]
        shift = 0.1 if max(p0, p1) <= 0.9 else -0.1

        def prefix(a, b):
            return (f"Step 0: {gold.get('step0','')}\nStep 1: {gold.get('step1','')}\n"
                    f"Step 2: {gold.get('step2','')}\nStep 3: {gold.get('step3','')}\n"
                    f"Step 4: P(Y=1 | X=0) = {a:.2f}\n"
                    f"        P(Y=1 | X=1) = {b:.2f}\nStep 5:")

        for name, (a, b) in {
            "keep": (p0, p1),
            "flip": (p1, p0),
            "neutral": (round(p0 + shift, 2), round(p1 + shift, 2)),
        }.items():
            probes.append({
                "probe_id": f"{row['id']}-{name}", "id": row["id"], "condition": name,
                "source": source, "query_type": row["query_type"], "user": row["prompt"],
                "assistant_prefix": prefix(a, b),
                # gold-based expectations are kept for reference only; scoring is model-anchored
                "expected_answer": "", "gold_answer": row["gold_answer"],
                "p0": a, "p1": b, "expected_result": round(b - a, 4),
            })

out = Path("work_v2/probes_p328.jsonl")
out.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in probes) + "\n",
               encoding="utf-8")
print(f"wrote {len(probes)} rows from {len(probes)//3} items -> {out}")
