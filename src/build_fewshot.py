"""Build the P0-B few-shot CoT baseline inputs.

The baseline is Qwen2.5-3B-Instruct with no fine-tuning, prompted with three
worked examples from the *training* split (never the test split), in the same
format the P-FT arm was trained on. Three files are produced, all of which are
consumed by the existing runners without modification:

  work_v2/fs_indist.jsonl        -> src/evaluate.py --mode process   (Acc, F(q), FC)
  work_v2/probes_fs_prefix.jsonl -> src/evaluate_probes.py           (FR, UF, S5)
  work_v2/probes_fs_input.jsonl  -> src/input_test.py run            (FR, UF)

The exemplar block is prepended to the question text; the gold Steps 0-4 prefix
of the probe runs still follows it, so the model continues from the same place.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "work_v2"

TRAIN = W / "sft_process_train.jsonl"
INDIST = W / "eval_indist.jsonl"
PROBES = W / "probes_p328.jsonl"
INPUT = W / "probes_input_p328.jsonl"

OUT_BLOCK = W / "fs_exemplars.txt"
OUT_INDIST = W / "fs_indist.jsonl"
OUT_PROBES = W / "probes_fs_prefix.jsonl"
OUT_INPUT = W / "probes_fs_input.jsonl"

N_SHOT = 3


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def pick_exemplars(rows: list[dict], k: int) -> list[dict]:
    """One exemplar per query type, so the prompt is not dominated by one rung."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        q = r.get("query_type") or ""
        if q in seen:
            continue
        seen.add(q)
        out.append(r)
        if len(out) == k:
            break
    if len(out) < k:  # tiny train split: top up regardless of type
        for r in rows:
            if r not in out:
                out.append(r)
            if len(out) == k:
                break
    return out


def build_block(ex: list[dict]) -> str:
    from common import PROCESS_INSTRUCTION

    parts = []
    for r in ex:
        # keep only the question; the instruction is constant and is already
        # present once at the end of the target prompt
        q = r["prompt"].split(PROCESS_INSTRUCTION)[0].strip()
        parts.append(f"Question: {q}\n\n{r['target'].strip()}")
    head = "\n\n".join(parts)
    return (
        head
        + "\n\n"
        + "The examples above show the required format. Now answer the following "
        + "question in the same format, with the same six steps.\n\n"
    )


def prepend(block: str, text: str) -> str:
    return block + text


def main() -> None:
    ex = pick_exemplars(read_jsonl(TRAIN), N_SHOT)
    block = build_block(ex)
    OUT_BLOCK.write_text(block, encoding="utf-8")
    print(f"exemplars: {[r['id'] for r in ex]}  block chars={len(block)}")

    rows = read_jsonl(INDIST)
    with OUT_INDIST.open("w", encoding="utf-8") as f:
        for r in rows:
            r = dict(r)
            r["prompt"] = prepend(block, r["prompt"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_INDIST.name}  rows={len(rows)}")

    rows = read_jsonl(PROBES)
    with OUT_PROBES.open("w", encoding="utf-8") as f:
        for r in rows:
            r = dict(r)
            r["user"] = prepend(block, r["user"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PROBES.name}  rows={len(rows)}")

    rows = read_jsonl(INPUT)
    with OUT_INPUT.open("w", encoding="utf-8") as f:
        for r in rows:
            r = dict(r)
            r["prompt_process"] = prepend(block, r["prompt_process"])
            r["prompt_answer"] = prepend(block, r["prompt_answer"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_INPUT.name}  rows={len(rows)}")


if __name__ == "__main__":
    main()
