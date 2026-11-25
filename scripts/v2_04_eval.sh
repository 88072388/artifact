#!/usr/bin/env bash
# STEP 5. Evaluate every finished adapter: main sets + shortcut probes + intervention probes.
# Resumable: a summary file that already exists is skipped.
#
#   bash scripts/v2_04_eval.sh                 # uses vLLM (fast)
#   BACKEND=hf bash scripts/v2_04_eval.sh      # fallback if vLLM cannot be installed
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-"0.5b 1.5b 3b"}
ARMS=${ARMS:-"answer process scrambled"}
SEEDS=${SEEDS:-"1 2 3"}
BACKEND=${BACKEND:-vllm}

declare -A HF=(
  [0.5b]="Qwen/Qwen2.5-0.5B-Instruct"
  [1.5b]="Qwen/Qwen2.5-1.5B-Instruct"
  [3b]="Qwen/Qwen2.5-3B-Instruct"
)

for size in $MODELS; do
  for arm in $ARMS; do
    for seed in $SEEDS; do
      tag="${size}-${arm}-s${seed}"
      ad="runs/${tag}/adapter"
      [ -f "${ad}/adapter_config.json" ] || { echo "MISSING ${ad}, skip"; continue; }
      mode="process"; [ "$arm" = "answer" ] && mode="answer"

      if [ ! -f "results/${tag}-indist.summary.json" ]; then
        echo "== ${tag} indist"
        python src/evaluate.py --model "${HF[$size]}" --adapter "$ad" \
            --file work_v2/eval_indist.jsonl --mode "$mode" --backend "$BACKEND" \
            --tag "${tag}-indist"
      fi

      if [ ! -f "results/${tag}-unseen.summary.json" ]; then
        echo "== ${tag} unseen structure"
        python src/evaluate.py --model "${HF[$size]}" --adapter "$ad" \
            --file work_v2/eval_unseen_struct.jsonl --mode "$mode" --backend "$BACKEND" \
            --tag "${tag}-unseen"
      fi

      if [ ! -f "results/${tag}-numremoved.summary.json" ]; then
        echo "== ${tag} number removed"
        python src/evaluate.py --model "${HF[$size]}" --adapter "$ad" \
            --file work_v2/eval_numremoved.jsonl --mode "$mode" --backend "$BACKEND" \
            --tag "${tag}-numremoved"
      fi

      if [ ! -f "results/${tag}-counterbench.summary.json" ]; then
        echo "== ${tag} CounterBench (OOD)"
        python src/evaluate.py --model "${HF[$size]}" --adapter "$ad" \
            --file work_v2/eval_counterbench.jsonl --mode "$mode" --backend "$BACKEND" \
            --tag "${tag}-counterbench"
      fi

      if [ ! -f "results/probes_${tag}.json" ]; then
        echo "== ${tag} intervention probes"
        python src/evaluate_probes.py --model "${HF[$size]}" --adapter "$ad" \
            --backend "$BACKEND" --tag "$tag"
      fi
    done
  done
done

echo
echo "=== building tables ==="
python src/make_tables_v2.py
echo
echo "Send results/tables_v2.md and the whole results/ folder back."
