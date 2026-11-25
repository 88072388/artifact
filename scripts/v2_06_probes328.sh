#!/usr/bin/env bash
# Re-run the intervention test on the 328-item probe set (advisor's suggestion).
# Writes probes_<tag>-p328.json / .jsonl; the 157-item results are left untouched.
#
#   bash scripts/v2_06_probes328.sh
set -euo pipefail
cd "$(dirname "$0")/.."

declare -A HF=(
  [0.5b]="Qwen/Qwen2.5-0.5B-Instruct"
  [1.5b]="Qwen/Qwen2.5-1.5B-Instruct"
  [3b]="Qwen/Qwen2.5-3B-Instruct"
)
BACKEND=${BACKEND:-vllm}

for size in 0.5b 1.5b 3b; do
  for arm in answer process; do
    for seed in 1 2 3; do
      tag="${size}-${arm}-s${seed}"
      ad="runs/${tag}/adapter"
      [ -f "${ad}/adapter_config.json" ] || { echo "MISSING ${ad}"; continue; }
      if [ -f "results/probes_${tag}-p328.json" ]; then echo "SKIP ${tag}-p328"; continue; fi
      echo "== probes ${tag} (328 items)"
      python src/evaluate_probes.py --model "${HF[$size]}" --adapter "$ad" \
          --probes work_v2/probes_p328.jsonl --backend "$BACKEND" --tag "${tag}-p328"
    done
  done
done

echo
echo "=== DONE ==="
ls results/probes_*-p328.json 2>/dev/null | wc -l
echo "files (expect 18: 3 models x 2 arms x 3 seeds)"
