#!/usr/bin/env bash
# P0-B: few-shot CoT baseline (no fine-tuning, 3 exemplars).
#
#   SIZE=3b bash scripts/v2_08_fewshot.sh     # default
#   SIZE=7b bash scripts/v2_08_fewshot.sh     # cross-size control
#
# Prerequisite: python src/build_fewshot.py
# The exemplar block is ~1.2k tokens, so every run gets --max-len 3200 and a
# smaller batch; otherwise right-hand truncation would eat the question itself.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND=${BACKEND:-hf}
SIZE=${SIZE:-3b}
declare -A HF=(
  [0.5b]="Qwen/Qwen2.5-0.5B-Instruct"
  [1.5b]="Qwen/Qwen2.5-1.5B-Instruct"
  [3b]="Qwen/Qwen2.5-3B-Instruct"
  [7b]="Qwen/Qwen2.5-7B-Instruct"
)
MODEL="${HF[$SIZE]}"
TAG="fs${SIZE}"
echo "[few-shot] size=${SIZE} model=${MODEL} tag=${TAG}-*"

echo "== few-shot in-distribution (Acc / F(q) / FC / format)"
python src/evaluate.py --model "$MODEL" --file work_v2/fs_indist.jsonl \
    --mode process --tag "${TAG}-indist" --backend "$BACKEND" \
    --max-len 3200 --bs ${BS:-4}

echo "== few-shot prefix intervention (FR / UF / S5)"
python src/evaluate_probes.py --model "$MODEL" --probes work_v2/probes_fs_prefix.jsonl \
    --tag "${TAG}-probes" --backend "$BACKEND" --max-tokens 96 --max-len 3200 --bs ${BS:-4}

echo "== few-shot input perturbation (FR / UF)"
python src/input_test.py run --model "$MODEL" --arm process \
    --file work_v2/probes_fs_input.jsonl --tag "${TAG}-input" --backend "$BACKEND" \
    --max-len 3200 --bs ${BS:-4}

echo
echo "=== summary ==="
python src/input_test.py score --tags "${TAG}-input"
python - <<PY
import glob, json
for p in sorted(glob.glob("results/${TAG}-indist*.json")) + sorted(glob.glob("results/probes_${TAG}-*.json")):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(p, "unreadable:", e)
        continue
    keys = ["n", "accuracy", "accuracy_all", "step_fidelity", "false_correct_rate",
            "format_compliance", "answer_coverage", "flip_rate", "false_flip_rate",
            "keep_consistency", "step5_follow_rate"]
    print(p, {k: d[k] for k in keys if k in d})
PY
