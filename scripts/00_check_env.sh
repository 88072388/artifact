#!/usr/bin/env bash
# Environment self-check. Run this after any pip install, before training.
#   bash scripts/00_check_env.sh
set -u

python - <<'PY'
import importlib, sys

print("python", sys.version.split()[0])

mods = ["torch", "transformers", "peft", "datasets", "bitsandbytes",
        "sklearn", "numpy", "vllm"]
failed = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:13s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  FAIL {m:13s} {type(e).__name__}: {e}")
        failed.append(m)

ok_cuda = False
try:
    import torch
    ok_cuda = torch.cuda.is_available()
    print("  cuda available:", ok_cuda)
    if ok_cuda:
        p = torch.cuda.get_device_properties(0)
        print(f"  gpu: {torch.cuda.get_device_name(0)}  vram {p.total_memory/1024**3:.1f} GB")
except Exception as e:
    print("  FAIL torch cuda check:", e)

print()
if "vllm" in failed:
    try:
        import torch
        tv = torch.__version__.split("+")[0]
    except Exception:
        tv = "?"
    print("vLLM cannot be imported. Do NOT debug it — pick one of these:")
    print("  A) reinstall a vLLM built for your torch (%s):" % tv)
    print("       pip uninstall -y vllm")
    print("       pip install vllm            # latest, usually matches newest torch")
    print("  B) give up on vLLM and evaluate with the HF backend:")
    print("       BACKEND=hf bash scripts/v2_04_eval.sh   (slow, but works)")
elif not ok_cuda:
    print("CUDA is not available — you are probably on a CPU-only instance.")
    print("Switch the instance to a GPU machine before training.")
else:
    print("Environment looks fine. If numpy was downgraded, ignore the")
    print("scipy/pandas warnings: this project never imports scipy or pandas.")
PY
