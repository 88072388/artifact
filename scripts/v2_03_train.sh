#!/usr/bin/env bash
# STEP 4. Train the 21 runs of the execution plan:
#   smoke 2  +  q05 6  +  q15 9 (incl. 3 scrambled)  +  q3 6
# Resumable: a run whose run_config.json exists is skipped.
#
#   bash scripts/v2_03_train.sh              # everything (~15-20 GPU hours)
#   MODELS=3b bash scripts/v2_03_train.sh    # only 3B
#   SEEDS=1  bash scripts/v2_03_train.sh     # one seed
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-"0.5b 1.5b 3b"}
ARMS=${ARMS:-"answer process"}
SEEDS=${SEEDS:-"1 2 3"}
EPOCHS=${EPOCHS:-3}

declare -A HF=(
  [0.5b]="Qwen/Qwen2.5-0.5B-Instruct"
  [1.5b]="Qwen/Qwen2.5-1.5B-Instruct"
  [3b]="Qwen/Qwen2.5-3B-Instruct"
)
declare -A DATA=(
  [answer]="work_v2/sft_answer_train.jsonl"
  [process]="work_v2/sft_process_train.jsonl"
  [scrambled]="work_v2/sft_scrambled_train.jsonl"
)

run() {  # run <size> <arm> <seed>
  local size=$1 arm=$2 seed=$3
  local out="runs/${size}-${arm}-s${seed}"
  if [ -f "${out}/run_config.json" ]; then
    echo "SKIP ${out}"
    return
  fi
  echo "=================================================="
  echo "TRAIN ${out}  model=${HF[$size]}  data=${DATA[$arm]}"
  echo "start $(date)"
  echo "=================================================="
  python src/train_lora.py \
      --model "${HF[$size]}" --train "${DATA[$arm]}" --out "$out" \
      --epochs "$EPOCHS" --bs 4 --grad-accum 4 --lr 1e-4 \
      --seed "$seed" --load-in-4bit
  echo "end $(date)"
}

for size in $MODELS; do
  for arm in $ARMS; do
    for seed in $SEEDS; do
      run "$size" "$arm" "$seed"
    done
  done
  # scrambled control only for 1.5B (as in the plan)
  if [ "$size" = "1.5b" ]; then
    for seed in $SEEDS; do
      run "$size" scrambled "$seed"
    done
  fi
done

echo
echo "=== TRAINING DONE ==="
ls -d runs/*/adapter 2>/dev/null | wc -l
echo "adapters (expect 18 for 3 models x 2 arms x 3 seeds, +3 scrambled for 1.5B = 21)"
