"""Input-level perturbation test (P0-A).

Three subcommands:

    python src/input_test.py build              # -> work_v2/probes_input_p328.jsonl
    python src/input_test.py run --model ... --adapter ... --arm answer --tag ...
    python src/input_test.py score --tags ...   # -> results/input_<tag>.summary.json

The 328-item probe set is reused, but the two probabilities are swapped inside the
question text itself, so an answer-only model can take the test in its native format
(no assistant prefix). Conditions: keep / flip (p0<->p1) / neutral (+0.05 to both).
Gold flips by construction for `flip`, stays put for `neutral`.

    bash scripts/v2_07_input.sh                 # all 18 adapters
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import ANSWER_ONLY_INSTRUCTION, PROCESS_INSTRUCTION  # noqa: E402

PROBES = Path("work_v2/probes_p328.jsonl")
OUT = Path("work_v2/probes_input_p328.jsonl")
PCT = re.compile(r"is (\d+(?:\.\d+)?)%")


def fmt(v: float) -> str:
    x = v * 100
    return f"{int(round(x))}" if abs(x - round(x)) < 1e-6 else f"{x:.1f}"


def swap_numbers(prompt: str, p0: float, p1: float) -> str | None:
    """Swap the two percentages in the question; None if the parse is not clean."""
    hits = list(PCT.finditer(prompt))
    if len(hits) != 2:
        return None
    got = [float(h.group(1)) / 100 for h in hits]
    if {round(got[0], 4), round(got[1], 4)} != {round(p0, 4), round(p1, 4)}:
        return None
    a, b = fmt(p1), fmt(p0)  # reversed on purpose
    out = prompt[: hits[0].start(1)] + a + prompt[hits[0].end(1) :]
    shift = len(a) - (hits[0].end(1) - hits[0].start(1))
    s, e = hits[1].start(1) + shift, hits[1].end(1) + shift
    return out[:s] + b + out[e:]


def shift_numbers(prompt: str, delta: float = 0.05) -> str | None:
    hits = list(PCT.finditer(prompt))
    if len(hits) != 2:
        return None
    vals = [float(h.group(1)) / 100 + delta for h in hits]
    if max(vals) > 0.99:
        return None
    out = prompt
    for h, v in sorted(zip(hits, vals), key=lambda kv: kv[0].start(1), reverse=True):
        out = out[: h.start(1)] + fmt(v) + out[h.end(1) :]
    return out


def to_answer_mode(prompt: str) -> str:
    base = prompt.split(PROCESS_INSTRUCTION)[0].rstrip()
    return base + "\n\n" + ANSWER_ONLY_INSTRUCTION


def build(_args=None) -> None:
    rows = [json.loads(l) for l in PROBES.open(encoding="utf-8")]
    keep = [r for r in rows if r["condition"] == "keep"]
    out, skipped = [], 0
    for r in keep:
        p0, p1 = float(r["p0"]), float(r["p1"])
        variants = {
            "keep": r["user"],
            "flip": swap_numbers(r["user"], p0, p1),
            "neutral": shift_numbers(r["user"], 0.05),
        }
        if any(v is None for v in variants.values()):
            skipped += 1
            continue
        for cond, prompt in variants.items():
            gold = r["gold_answer"] if cond != "flip" else ("no" if r["gold_answer"] == "yes" else "yes")
            out.append(
                {
                    "probe_id": f"{r['id']}-{cond}-input",
                    "id": r["id"],
                    "condition": cond,
                    "source": r["source"],
                    "query_type": r["query_type"],
                    "gold_answer": gold,
                    "prompt_process": prompt,
                    "prompt_answer": to_answer_mode(prompt),
                }
            )
    with OUT.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}  rows={len(out)}  items={len(out)//3}  skipped={skipped}")


def run(args) -> None:
    from transformers import AutoTokenizer

    from evaluate import chat_prompt, generate_hf, generate_vllm

    items = [json.loads(l) for l in Path(args.file).open(encoding="utf-8")]
    if getattr(args, "limit", 0):
        items = items[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:  # Llama-style tokenizers have no pad token
        tok.pad_token = tok.eos_token
    key = "prompt_answer" if args.arm == "answer" else "prompt_process"
    prompts = [chat_prompt(tok, it[key]) for it in items]

    args.max_tokens = 8 if args.arm == "answer" else 448
    # the few-shot baseline prepends ~1.2k tokens of exemplars, so keep enough
    # room left of max_tokens that truncation never eats the question itself
    args.max_len = max(args.max_len, 3072)
    args.gpu_mem = 0.85
    args.max_lora_rank = 32
    print(f"{len(prompts)} prompts, backend={args.backend}", flush=True)
    texts = generate_vllm(args, prompts) if args.backend == "vllm" else generate_hf(args, prompts)

    pat = re.compile(r"answer\s*[:\-]?\s*(yes|no)", re.I)
    rows = []
    for it, text in zip(items, texts):
        m = pat.findall(text)
        pred = m[-1].lower() if m else None
        if pred is None:  # last resort: any yes/no in the tail
            tail = re.findall(r"\b(yes|no)\b", text[-200:], re.I)
            pred = tail[-1].lower() if tail else None
        rows.append(
            {
                "id": it["id"],
                "condition": it["condition"],
                "query_type": it["query_type"],
                "gold_answer": it["gold_answer"],
                "pred_answer": pred,
                "output": text,
            }
        )

    out = Path(args.out_dir) / f"input_{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    parsed = sum(1 for r in rows if r["pred_answer"])
    print(f"wrote {out}  parsed {parsed}/{len(rows)}")


def score(args) -> None:
    import numpy as np

    for tag in args.tags:
        rows = [json.loads(l) for l in Path(f"results/input_{tag}.jsonl").open(encoding="utf-8")]
        by_id: dict[str, dict] = {}
        for r in rows:
            by_id.setdefault(r["id"], {})[r["condition"]] = r
        complete = {k: v for k, v in by_id.items() if len(v) == 3}
        a = {c: np.array([complete[k][c]["pred_answer"] or "?" for k in complete]) for c in ("keep", "flip", "neutral")}
        g = {c: np.array([complete[k][c]["gold_answer"] for k in complete]) for c in ("keep", "flip", "neutral")}

        fr = (a["flip"] != a["keep"]).mean()
        uf = (a["neutral"] != a["keep"]).mean()
        acc = {c: (a[c] == g[c]).mean() for c in a}
        n = len(complete)

        rng = np.random.default_rng(0)
        idx = rng.integers(0, n, size=(1000, n))
        boot = [(a["flip"][i] != a["keep"][i]).mean() for i in idx]
        lo, hi = np.percentile(boot, [2.5, 97.5])

        summary = {
            "tag": tag,
            "n": n,
            "flip_rate": round(float(fr), 4),
            "flip_ci": [round(float(lo), 4), round(float(hi), 4)],
            "spurious_flip": round(float(uf), 4),
            "acc_keep": round(float(acc["keep"]), 4),
            "acc_flip": round(float(acc["flip"]), 4),
            "acc_neutral": round(float(acc["neutral"]), 4),
            "parse_rate": round(float(np.mean([(a["keep"] != "?").mean()])), 4),
        }
        Path(f"results/input_{tag}.summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build")

    p = sub.add_parser("run")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--arm", choices=["answer", "process"], required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--file", default=str(OUT))
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-lora-rank", type=int, default=32)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--limit", type=int, default=0, help="smoke test: only the first N rows")

    s = sub.add_parser("score")
    s.add_argument("--tags", nargs="+", required=True)

    args = ap.parse_args()
    {"build": build, "run": run, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
