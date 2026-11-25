#!/usr/bin/env bash
# STEP 2. Build all data for the v2 plan. CPU only, ~1 minute, no GPU needed.
set -euo pipefail
cd "$(dirname "$0")/.."

python src/build_v2.py --data-dir data/cladder-v1 --out-dir work_v2
python src/build_counterbench.py --data-dir data/counterbench --out work_v2/eval_counterbench.jsonl

echo
echo "=== CHECK THESE NUMBERS ==="
python - <<'PY'
import json, pathlib
m = json.load(pathlib.Path("work_v2/manifest.json").open(encoding="utf-8"))
want = {"usable": 8303, "train": 5781, "val": 528,
        "indist": 1089, "unseen_struct": 905, "probes": 1083}
ok = True
for k, v in want.items():
    got = m.get(k)
    flag = "OK " if got == v else "!! "
    ok &= got == v
    print(f"  {flag}{k:14s} expected {v}, got {got}")

files = {"eval_numremoved.jsonl": 1499, "eval_counterbench.jsonl": 1200,
         "eval_indist.jsonl": 1089, "eval_unseen_struct.jsonl": 905,
         "probes.jsonl": 1083}
for f, v in files.items():
    p = pathlib.Path("work_v2") / f
    got = sum(1 for line in p.open(encoding="utf-8") if line.strip()) if p.exists() else -1
    flag = "OK " if got == v else "!! "
    ok &= got == v
    print(f"  {flag}{f:26s} expected {v}, got {got}")

print()
print("ALL GOOD — continue to v2_02_smoke.sh" if ok else
      "NUMBERS DIFFER — send this output back before training")
PY
