"""LoRA / QLoRA fine-tuning with loss masked to the completion only.

    # answer-only supervision
    python train_lora.py --train work/sft_answer_train.jsonl  --out runs/qwen3b-answer
    # process supervision (six steps + conclusion + answer)
    python train_lora.py --train work/sft_process_train.jsonl --out runs/qwen3b-process

Everything except the training file is kept identical between the two arms, so the
comparison isolates the supervision signal.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

SYSTEM = "You are a careful causal reasoner."


def build_example(tok, prompt: str, target: str, max_len: int):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    prompt_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    target_ids = tok(target, add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id

    max_target = max_len - 8
    target_ids = target_ids[:max_target]
    budget = max_len - len(target_ids) - 1
    if len(prompt_ids) > budget:
        prompt_ids = prompt_ids[-budget:]
    input_ids = prompt_ids + target_ids + [eos]
    labels = [-100] * len(prompt_ids) + target_ids + [eos]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N examples")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0})
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=args.train, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    ds = ds.map(
        lambda x: build_example(tok, x["prompt"], x["target"], args.max_len),
        remove_columns=ds.column_names,
        desc="tokenizing",
    )

    collator = DataCollatorForSeq2Seq(
        tok, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100
    )
    targs = TrainingArguments(
        output_dir=str(Path(args.out) / "ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")

    config = {
        "base_model": args.model,
        "train_file": str(args.train),
        "train_examples": len(ds),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.bs,
        "grad_accum": args.grad_accum,
        "effective_batch": args.bs * args.grad_accum,
        "max_len": args.max_len,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": "all-linear-attn-mlp",
        },
        "load_in_4bit": args.load_in_4bit,
        "seed": args.seed,
        "torch": torch.__version__,
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
