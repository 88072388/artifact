#!/usr/bin/env bash
# Zero-shot baseline: the base models with no fine-tuning, answer-only prompt.
# Resumable: a summary file that already exists is skipped.
#
#   bash scripts/v2_05_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

declare -A HF=(
  [0.5b]="Qwen/Qwen2.5-0.5B-Instruct"
  [1.5b]="Qwen/Qwen2.5-1.5B-Instruct"
  [3b]="Qwen/Qwen2.5-3B-Instruct"
)
SETS="indist unseen numremoved counterbench"
BACKEND=${BACKEND:-vllm}

for size in 0.5b 1.5b 3b; do
  for s in $SETS; do
    case $s in
      indist) f="work_v2/eval_indist.jsonl" ;;
      unseen) f="work_v2/eval_unseen_struct.jsonl" ;;
      numremoved) f="work_v2/eval_numremoved.jsonl" ;;
      counterbench) f="work_v2/eval_counterbench.jsonl" ;;
    esac
    tag="base-${size}-${s}"
    if [ -f "results/${tag}.summary.json" ]; then
      echo "SKIP ${tag}"
      continue
    fi
    echo "== ${tag}"
    python src/evaluate.py --model "${HF[$size]}" --file "$f" \
        --mode answer --backend "$BACKEND" --tag "$tag"
  done
done

echo
echo "=== BASELINE DONE ==="
ls results/base-*.summary.json 2>/dev/null | wc -l
echo "summaries (expect 12)"
