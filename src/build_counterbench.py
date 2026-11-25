"""Convert CounterBench (1,000 alpha + 200 backdoor) into the eval JSONL format.

Source: https://huggingface.co/datasets/CounterBench/CounterBench
  data_balanced_alpha_V1.json     1,000 items, 4 types x 250
  data_balanced_backdoor_V2.json    200 items
  meta_model_*.json               per model_id background paragraph

CounterBench has no step annotation, so only accuracy is reported on it.

    python src/build_counterbench.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ANSWER_ONLY_INSTRUCTION, PROCESS_INSTRUCTION, write_jsonl


def load(path: Path):
    return json.load(path.open(encoding="utf-8"))


def build_prompt(background: str, rec: dict, mode: str) -> str:
    instr = ANSWER_ONLY_INSTRUCTION if mode == "answer" else PROCESS_INSTRUCTION
    return f"{background}\n\n{rec['given_info']}\n\n{rec['question']}\n\n{instr}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/counterbench")
    ap.add_argument("--out", default="work_v2/eval_counterbench.jsonl")
    ap.add_argument("--mode", default="process", choices=["answer", "process"])
    args = ap.parse_args()

    d = Path(args.data_dir)
    rows = []

    for stem, meta_name, subset in [
        ("data_balanced_alpha_V1", "meta_model_alpha_V1", "alpha"),
        ("data_balanced_backdoor_V2", "meta_model_backdoor_V2", "backdoor"),
    ]:
        items = load(d / f"{stem}.json")
        metas = {m["model_id"]: m for m in load(d / f"{meta_name}.json")}
        for it in items:
            mid = it["meta"]["model_id"]
            bg = metas.get(mid, {}).get("background", "")
            rows.append(
                {
                    "id": f"cb-{subset}-{it['question_id']}",
                    "prompt": build_prompt(bg, it, args.mode),
                    "gold_answer": it["answer"],
                    "gold_steps": {},          # no step annotation in CounterBench
                    "query_type": it["meta"].get("query_type", "det-counterfactual"),
                    "rung": it["meta"].get("rung", 3),
                    "graph_id": it["meta"].get("graph_id", "unknown"),
                    "story_id": it["meta"].get("story_id", "unknown"),
                    "subset": f"{subset}:{it['type']}",
                    "num_removed": False,
                }
            )

    write_jsonl(args.out, rows)
    from collections import Counter

    print(f"wrote {args.out}: {len(rows)} items")
    print("by subset:", dict(Counter(r["subset"] for r in rows)))
    print("answers:", dict(Counter(r["gold_answer"] for r in rows)))


if __name__ == "__main__":
    main()
