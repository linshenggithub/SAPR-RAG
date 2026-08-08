#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

echo "== basic =="
hostname
pwd
date

echo
echo "== env =="
env | grep -E '^(ASCEND|NPU|CANN|LD_LIBRARY_PATH|PATH|PYTHONPATH|ARNOLD|MERLIN|MLX|PORT_LIST)' | sort | sed -n '1,160p'

echo
echo "== ascend paths =="
for p in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/driver \
    /usr/local/sbin/npu-smi \
    /usr/local/Ascend/ascend-toolkit/latest/bin/npu-smi
do
    if [ -e "$p" ]; then
        echo "FOUND $p"
    else
        echo "MISS  $p"
    fi
done

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "sourced /usr/local/Ascend/ascend-toolkit/set_env.sh"
else
    echo "WARN: Ascend set_env.sh not found"
fi

echo
echo "== npu-smi =="
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}"
command -v npu-smi || true
npu-smi info 2>&1 | sed -n '1,260p'

echo
echo "== python packages and npu tensor =="
python - <<'PY'
import importlib
import os
import sys

print("python", sys.version.replace("\n", " "))
print("executable", sys.executable)
for key in ["ASCEND_RT_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PATH"]:
    print(key, os.environ.get(key, ""))

for name in ["torch", "torch_npu", "transformers", "swift", "vllm", "vllm_ascend", "deepspeed", "datasets", "trl"]:
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "unknown"), getattr(module, "__file__", ""))
    except Exception as exc:
        print(name, "IMPORT_ERROR", repr(exc))

try:
    import torch
    print("torch.cuda.is_available", torch.cuda.is_available())
    print("has torch.npu", hasattr(torch, "npu"))
    if hasattr(torch, "npu"):
        print("torch.npu.device_count", torch.npu.device_count())
        x = torch.randn(1, device="npu:0")
        print("npu tensor ok", x.device, x.dtype, float(x.cpu()[0]))
except Exception as exc:
    print("torch npu check error", repr(exc))
PY

echo
echo "== project scripts dry-run =="
cd "$PROJ_ROOT" || exit 0
bash -n 03_sapr_rag/scripts/grpo/setup_env_opsd.sh \
    03_sapr_rag/scripts/grpo/run_grpo_opsd.sh \
    03_sapr_rag/scripts/grpo/run_rollout_opsd.sh \
    03_sapr_rag/scripts/grpo/run_retrieval_daemon_flexible.sh \
    03_sapr_rag/scripts/grpo/run_opsd_matched_pilot.sh

DEVICE_BACKEND=npu DRY_RUN=true \
    bash 03_sapr_rag/scripts/grpo/run_rollout_opsd.sh 2>&1 | sed -n '1,120p'

DEVICE_BACKEND=npu NPROC_PER_NODE=6 ENABLE_OPSD=false DRY_RUN=true \
    DATASET="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_train_pilot.jsonl" \
    bash 03_sapr_rag/scripts/grpo/run_grpo_opsd.sh 2>&1 | sed -n '1,160p'
