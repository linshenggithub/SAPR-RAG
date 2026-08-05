#!/usr/bin/env bash
# SAPR-RAG dynamic OPSD rollout server 入口。
# 这是增量脚本，不改动 baseline run_rollout.sh。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SWIFT_ROOT="${SWIFT_ROOT:-$(cd "$PROJ_ROOT/../ms-swift" 2>/dev/null && pwd || true)}"

[ -n "$SWIFT_ROOT" ] && [ -d "$SWIFT_ROOT" ] || {
    echo "[run_rollout_opsd] ERROR: SWIFT_ROOT not found. Set SWIFT_ROOT to the ms-swift checkout." >&2
    exit 2
}

BASE_MODEL="${BASE_MODEL:-$PROJ_ROOT/03_sapr_rag/models/Qwen2.5-7B-Instruct}"
SFT_LORA="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft/checkpoint-1650"
SFT_DPO_LORA="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft_dpo/checkpoint-395"
INIT_ADAPTER="${INIT_ADAPTER:-sft_dpo}"
case "$INIT_ADAPTER" in
    sft) RESOLVED_INIT_ADAPTER="$SFT_LORA" ;;
    sft_dpo) RESOLVED_INIT_ADAPTER="$SFT_DPO_LORA" ;;
    *) RESOLVED_INIT_ADAPTER="$INIT_ADAPTER" ;;
esac
ADAPTER_PATH="${ADAPTER_PATH:-$RESOLVED_INIT_ADAPTER}"
PLUGIN="$SCRIPT_DIR/plugin.py"

DEVICE_BACKEND="${DEVICE_BACKEND:-cuda}"
if [ "$DEVICE_BACKEND" = "npu" ] && [ -n "${ASCEND_VISIBLE_DEVICES:-}" ]; then
    DEFAULT_ROLLOUT_DEVICES="$(python - "$ASCEND_VISIBLE_DEVICES" <<'PY'
import sys
devices = [x for x in sys.argv[1].split(",") if x]
print(devices[6] if len(devices) >= 7 else (devices[-1] if devices else "0"))
PY
)"
else
    DEFAULT_ROLLOUT_DEVICES="${GPU:-0}"
fi
ROLLOUT_DEVICES="${ROLLOUT_DEVICES:-$DEFAULT_ROLLOUT_DEVICES}"
PORT="${PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
# 可注入：与常驻检索共卡时调低，给检索留余量（默认保持原 0.85）。
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
VLLM_ENABLE_LORA="${VLLM_ENABLE_LORA:-true}"
DRY_RUN="${DRY_RUN:-false}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

case "$DEVICE_BACKEND" in
    cuda)
        VISIBLE_DEVICES_ENV="CUDA_VISIBLE_DEVICES"
        DEVICE_LABEL="GPU"
        ;;
    npu)
        VISIBLE_DEVICES_ENV="ASCEND_RT_VISIBLE_DEVICES"
        DEVICE_LABEL="NPU"
        ;;
    *)
        echo "[run_rollout_opsd] ERROR: DEVICE_BACKEND must be cuda or npu, got: $DEVICE_BACKEND" >&2
        exit 2
        ;;
esac
for path in "$BASE_MODEL" "$ADAPTER_PATH"; do
    [ -d "$path" ] || { echo "[run_rollout_opsd] ERROR: path not found: $path" >&2; exit 2; }
done

cd "$SWIFT_ROOT"

echo "[run_rollout_opsd] backend=$DEVICE_BACKEND ${VISIBLE_DEVICES_ENV}=$ROLLOUT_DEVICES port=$PORT log=$LOG_DIR/rollout_opsd.log"
echo "[run_rollout_opsd] init_adapter=$INIT_ADAPTER resolved_adapter=$ADAPTER_PATH"
echo "[run_rollout_opsd] layout=rollout:${DEVICE_LABEL}${ROLLOUT_DEVICES}"
echo "[run_rollout_opsd] vllm_enable_lora=$VLLM_ENABLE_LORA"

CMD=(
    swift rollout
    --model "$BASE_MODEL"
    --adapters "$ADAPTER_PATH"
    --vllm_enable_lora "$VLLM_ENABLE_LORA"
    --vllm_use_async_engine true
    --multi_turn_scheduler sapr_rag_scheduler
    --external_plugins "$PLUGIN"
    --max_turns 6
    --vllm_max_model_len "$VLLM_MAX_MODEL_LEN"
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEM_UTIL"
    --host 127.0.0.1
    --port "$PORT"
)

if [ "$DRY_RUN" = "true" ]; then
    printf '[run_rollout_opsd] DRY_RUN command: %s=%q' "$VISIBLE_DEVICES_ENV" "$ROLLOUT_DEVICES"
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

env "$VISIBLE_DEVICES_ENV=$ROLLOUT_DEVICES" \
"${CMD[@]}" \
    2>&1 | tee "$LOG_DIR/rollout_opsd.log"
