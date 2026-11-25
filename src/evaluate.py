"""Generation + scoring: accuracy, Step Fidelity, format compliance, false-correct rate.

    # base model, answer only
    python evaluate.py --model Qwen/Qwen2.5-3B-Instruct --file work/eval_indomain.jsonl --mode answer

    # LoRA adapter, process mode
    python evaluate.py --model Qwen/Qwen2.5-3B-Instruct --adapter runs/qwen3b-process/adapter \
        --file work/eval_indomain.jsonl --mode process --tag qwen3b-process-indomain

Greedy decoding only: temperature 0, so numbers are comparable across runs.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from common import parse_steps, read_jsonl
from fidelity import score_steps

SYSTEM = "You are a careful causal reasoner."
FIDELITY_THRESHOLD = 0.5
HIGH_FIDELITY = 0.8


def chat_prompt(tok, prompt: str) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_vllm(args, prompts):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_len,
        gpu_memory_utilization=args.gpu_mem,
        enable_lora=bool(args.adapter),
        max_lora_rank=args.max_lora_rank,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, stop=["\n\n\n"]
    )
    lora_req = (
        LoRARequest("adapter", 1, args.adapter) if args.adapter else None
    )
    outs = llm.generate(prompts, sampling, lora_request=lora_req)
    return [o.outputs[0].text for o in outs]


def generate_hf(args, prompts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    texts = []
    for i in range(0, len(prompts), args.bs):
        batch = prompts[i : i + args.bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len - args.max_tokens).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        texts.extend(tok.batch_decode(gen, skip_special_tokens=True))
        if i % (args.bs * 20) == 0:
            print(f"  hf-batch {i}/{len(prompts)}", flush=True)
    return texts


def summarize(rows: list) -> dict:
    n = len(rows)
    answered = [r for r in rows if r["pred_answer"]]
    correct = [r for r in answered if r["pred_answer"] == r["gold_answer"]]
    fids = [r["fidelity"] for r in rows]
    trustworthy = [r for r in correct if r["fidelity"] >= HIGH_FIDELITY]

    def agg(key):
        out = defaultdict(list)
        for r in rows:
            out[r[key]].append(r)
        return {
            str(k): {
                "n": len(v),
                "acc": round(
                    sum(1 for x in v if x["pred_answer"] == x["gold_answer"]) / len(v), 4
                ),
                "fidelity": round(statistics.mean([x["fidelity"] for x in v]), 4),
            }
            for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))
        }

    return {
        "n": n,
        "answer_coverage": round(len(answered) / n, 4) if n else 0.0,
        "accuracy": round(len(correct) / len(answered), 4) if answered else 0.0,
        "accuracy_all": round(len(correct) / n, 4) if n else 0.0,
        "step_fidelity": round(statistics.mean(fids), 4) if fids else 0.0,
        "format_compliance": round(
            statistics.mean([r["format_compliance"] for r in rows]), 4
        )
        if rows
        else 0.0,
        "false_correct_rate": round(
            sum(1 for r in correct if r["fidelity"] < FIDELITY_THRESHOLD) / len(correct), 4
        )
        if correct
        else 0.0,
        "faithful_correct_rate": round(len(trustworthy) / len(correct), 4) if correct else 0.0,
        "per_step": {
            k: round(
                statistics.mean([r[f"s_{k}"] for r in rows if r[f"s_{k}"] is not None]), 4
            )
            for k in ["step0", "step1", "step2", "step3", "step4", "step5", "end"]
            if any(r[f"s_{k}"] is not None for r in rows)
        },
        "by_query_type": agg("query_type"),
        "by_rung": agg("rung"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--file", required=True)
    ap.add_argument("--mode", choices=["answer", "process"], default="process")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    if args.max_tokens == 0:
        args.max_tokens = 8 if args.mode == "answer" else 448

    items = read_jsonl(args.file)
    if args.limit:
        items = items[: args.limit]
    # the jsonl prompt already contains the instruction; wrap it in the chat template
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.mode == "answer":
        from common import ANSWER_ONLY_INSTRUCTION, PROCESS_INSTRUCTION

        def as_answer_mode(p: str) -> str:
            base = p.split(PROCESS_INSTRUCTION)[0].rstrip()
            return base + "\n\n" + ANSWER_ONLY_INSTRUCTION

        prompts_text = [as_answer_mode(it["prompt"]) for it in items]
    else:
        prompts_text = [it["prompt"] for it in items]

    prompts = [chat_prompt(tok, p) for p in prompts_text]

    print(f"generating {len(prompts)} items with {args.backend} ...", flush=True)
    texts = (
        generate_vllm(args, prompts) if args.backend == "vllm" else generate_hf(args, prompts)
    )

    rows = []
    for it, text in zip(items, texts):
        pred = parse_steps(text)
        scored = score_steps(it["gold_steps"], pred)
        rows.append(
            {
                "id": it["id"],
                "query_type": it["query_type"],
                "rung": it["rung"],
                "gold_answer": it["gold_answer"],
                "pred_answer": pred["answer"],
                "correct": pred["answer"] == it["gold_answer"],
                "output": text,
                **scored,
            }
        )

    tag = args.tag or f"{Path(args.file).stem}-{args.mode}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    summary["tag"] = tag
    summary["file"] = args.file
    summary["mode"] = args.mode
    summary["model"] = args.model
    summary["adapter"] = args.adapter
    (out_dir / f"{tag}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("by_")}, indent=2))
    print(f"\nwrote {out_dir / f'{tag}.jsonl'} and .summary.json")


if __name__ == "__main__":
    main()
