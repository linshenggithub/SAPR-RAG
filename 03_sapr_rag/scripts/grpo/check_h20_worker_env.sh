#!/usr/bin/env bash
# 只读检查 H20/CUDA worker 是否满足 SAPR-RAG OPSD/GRPO 运行前提。
# 通过 mlx worker login <id> -- bash <this> 执行，避免复杂引号被 SSH 包装打乱。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

echo "== basic =="
hostname
pwd
date

echo
echo "== env =="
env | grep -E '^(ARNOLD_DEVICE_TYPE|ARNOLD_WORKER_GPU|ARNOLD_APPLIED_GPU_NUM|CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|MERLIN|MLX)' | sort | sed -n '1,120p'

echo
echo "== nvidia-smi =="
command -v nvidia-smi || true
nvidia-smi 2>&1 | sed -n '1,60p'

echo
echo "== python packages and cuda tensor =="
python - <<'PY'
import importlib
import sys

print("python", sys.version.replace("\n", " "))
print("executable", sys.executable)

for name in ["torch", "transformers", "swift", "vllm", "deepspeed", "datasets", "trl"]:
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "unknown"), getattr(module, "__file__", ""))
    except Exception as exc:
        print(name, "IMPORT_ERROR", repr(exc))

try:
    import torch
    print("torch.cuda.is_available", torch.cuda.is_available())
    print("torch.cuda.device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        x = torch.randn(1, device="cuda:0")
        print("cuda tensor ok", x.device, x.dtype, float(x.cpu()[0]))
except Exception as exc:
    print("torch cuda check error", repr(exc))
PY

echo
echo "== setup_env_opsd (cuda) =="
cd "$PROJ_ROOT" || exit 0
DEVICE_BACKEND=cuda bash 03_sapr_rag/scripts/grpo/setup_env_opsd.sh 2>&1 | sed -n '1,200p'

echo
echo "== project scripts dry-run (cuda) =="
DEVICE_BACKEND=cuda DRY_RUN=true \
    bash 03_sapr_rag/scripts/grpo/run_rollout_opsd.sh 2>&1 | sed -n '1,120p'

DEVICE_BACKEND=cuda ENABLE_OPSD=false DRY_RUN=true \
    DATASET="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_train_pilot.jsonl" \
    bash 03_sapr_rag/scripts/grpo/run_grpo_opsd.sh 2>&1 | sed -n '1,160p'
