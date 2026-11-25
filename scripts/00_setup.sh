#!/usr/bin/env bash
# One-time environment setup on the cloud box. Run from the project root.
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf}
mkdir -p "$HF_HOME"

echo "== python =="
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda,
           'gpu', torch.cuda.get_device_name(0), 'vram_gb',
           round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))"

echo "== installing deps =="
pip install -q -r requirements.txt

echo "== sanity: can we see the dataset? =="
python - <<'PY'
import json, pathlib
p = pathlib.Path("data/cladder-v1/cladder-v1-q-balanced.json")
if not p.exists():
    raise SystemExit(f"missing {p} — upload the data/ folder first")
d = json.load(p.open(encoding="utf-8"))
ok = sum(1 for r in d if isinstance(r.get("reasoning"), dict))
print(f"records={len(d)} with_reasoning={ok}")
PY

echo "== done. next: bash scripts/01_prepare.sh =="
