#!/usr/bin/env bash
# P0-A: input-level perturbation test (no assistant prefix, native output format).
# Runs the 18 adapters on work_v2/probes_input_p328.jsonl (327 items x 3 conditions).
#
#   bash scripts/v2_07_input.sh                 # ~1-2 h on a 4090 with BACKEND=hf
#   BACKEND=vllm bash scripts/v2_07_input.sh    # faster if vLLM works on the box
set -euo pipefail
cd "$(dirname "$0")/.."

declare -A HF=(
  [0.5b]="Qwen/Qwen2.5-0.5B-Instruct"
  [1.5b]="Qwen/Qwen2.5-1.5B-Instruct"
  [3b]="Qwen/Qwen2.5-3B-Instruct"
  [7b]="Qwen/Qwen2.5-7B-Instruct"
)
BACKEND=${BACKEND:-hf}
TAGS=()

MODELS=${MODELS:-"0.5b 1.5b 3b"}

for size in $MODELS; do
  for arm in answer process; do
    for seed in 1 2 3; do
      tag="${size}-${arm}-s${seed}"
      out="results/input_${tag}.jsonl"
      if [ -f "$out" ]; then echo "SKIP ${tag}"; TAGS+=("$tag"); continue; fi
      ad="runs/${tag}/adapter"
      [ -f "${ad}/adapter_config.json" ] || { echo "MISSING ${ad}"; continue; }
      echo "== input test ${tag}"
      python src/input_test.py run --model "${HF[$size]}" --adapter "$ad" \
          --arm "$arm" --tag "$tag" --backend "$BACKEND"
      TAGS+=("$tag")
    done
  done
done

echo
echo "=== scoring ==="
python src/input_test.py score --tags "${TAGS[@]}"
