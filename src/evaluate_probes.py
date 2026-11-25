"""Intervention probe runner: keep / flip / neutral + Step-5 follow rate.

The item is a normal user turn; the gold Steps 0-4 become an assistant prefix and
the model continues from Step 5. Only the two numbers in Step 4 change.

    python evaluate_probes.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter runs/q15-pft-s1/adapter --tag q15-pft-s1
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import extract_answer, read_jsonl

SYSTEM = "You are a careful causal reasoner."
STEP5_RE = re.compile("step\\s*5\\s*[:\\uff1a]\\s*(-?\\d+(?:\\.\\d+)?)", re.IGNORECASE)
EQ_RE = re.compile("=\\s*(-?\\d+(?:\\.\\d+)?)")


def parse_result(text):
    """The prefix already ends with 'Step 5:', so the continuation starts with the
    arithmetic, e.g. ' 0.35 - 0.52 = -0.17'. Take the value after the last '='."""
    if not text:
        return None
    m = STEP5_RE.search(text)
    if m:
        return float(m.group(1))
    eq = EQ_RE.findall(text)
    if eq:
        return float(eq[-1])
    m = re.search("-?\\d+(?:\\.\\d+)?", text)
    return float(m.group(0)) if m else None


def build_prompts(tok, probes):
    out = []
    for p in probes:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": p["user"]}]
        head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out.append(head + p["assistant_prefix"])
    return out


def gen(args, prompts):
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_len,
                  gpu_memory_utilization=args.gpu_mem, enable_lora=bool(args.adapter),
                  max_lora_rank=args.max_lora_rank, trust_remote_code=True)
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
        lora = LoRARequest("a", 1, args.adapter) if args.adapter else None
        return [o.outputs[0].text for o in llm.generate(prompts, sp, lora_request=lora)]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    texts = []
    for i in range(0, len(prompts), args.bs):
        enc = tok(prompts[i:i + args.bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_len - args.max_tokens).to(model.device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=False,
                                 temperature=None, top_p=None,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        texts.extend(tok.batch_decode(ids[:, enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True))
        print(f"  {i}/{len(prompts)}", flush=True)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--probes", default="work_v2/probes.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    probes = read_jsonl(args.probes)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = build_prompts(tok, probes)

    print(f"generating {len(prompts)} probe continuations ...", flush=True)
    texts = gen(args, prompts)

    rows = []
    for p, t in zip(probes, texts):
        v = parse_result(t)
        rows.append({**{k: p[k] for k in ("probe_id", "id", "condition", "source",
                                          "query_type", "gold_answer", "expected_answer",
                                          "p0", "p1", "expected_result")},
                     "pred_answer": extract_answer(t),
                     "pred_result": v,
                     "output": t})

    by_id = defaultdict(dict)
    for r in rows:
        by_id[r["id"]][r["condition"]] = r

    n = keep = flip = false_flip = follow = costepped = 0
    for conds in by_id.values():
        if set(conds) != {"keep", "flip", "neutral"}:
            continue
        n += 1
        keep += int(conds["keep"]["pred_answer"] == conds["keep"]["gold_answer"])
        flip += int(conds["flip"]["pred_answer"] == conds["flip"]["expected_answer"])
        false_flip += int(conds["neutral"]["pred_answer"] != conds["neutral"]["expected_answer"])
        for c in conds.values():
            if c["pred_result"] is not None:
                costepped += 1
                if abs(c["pred_result"] - c["expected_result"]) <= 0.02:
                    follow += 1

    yes_flip = sum(1 for c in by_id.values() if "flip" in c
                   and c["flip"]["expected_answer"] == "yes")
    out = {
        "tag": args.tag, "model": args.model, "adapter": args.adapter,
        "n_items": n,
        "keep_consistency": round(keep / n, 4) if n else 0.0,
        "flip_rate": round(flip / n, 4) if n else 0.0,
        "false_flip_rate": round(false_flip / n, 4) if n else 0.0,
        "step5_follow_rate": round(follow / costepped, 4) if costepped else 0.0,
        "result_parse_rate": round(costepped / max(1, len(rows)), 4),
        "answer_parse_rate": round(sum(1 for r in rows if r["pred_answer"]) / len(rows), 4),
        "always_yes_baseline": {"flip_rate": round(yes_flip / n, 4) if n else 0.0},
        "by_source": {},
    }
    for src in sorted({r["source"] for r in rows}):
        sub = [c for c in by_id.values() if "keep" in c and c["keep"]["source"] == src]
        if not sub:
            continue
        k = sum(1 for c in sub if c["keep"]["pred_answer"] == c["keep"]["gold_answer"])
        f = sum(1 for c in sub if c["flip"]["pred_answer"] == c["flip"]["expected_answer"])
        ff = sum(1 for c in sub if c["neutral"]["pred_answer"] != c["neutral"]["expected_answer"])
        out["by_source"][src] = {"n": len(sub), "keep": round(k / len(sub), 4),
                                 "flip": round(f / len(sub), 4),
                                 "false_flip": round(ff / len(sub), 4)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"probes_{args.tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / f"probes_{args.tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

