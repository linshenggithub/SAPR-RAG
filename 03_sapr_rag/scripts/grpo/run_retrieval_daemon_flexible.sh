#!/usr/bin/env bash
# SAPR-RAG retrieval daemon flexible 入口，支持 cuda / npu / cpu。
# 这是增量脚本，不改动 baseline run_retrieval_daemon.sh。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

DEVICE_BACKEND="${DEVICE_BACKEND:-cuda}"
if [ "$DEVICE_BACKEND" = "npu" ] && [ -n "${ASCEND_VISIBLE_DEVICES:-}" ]; then
    DEFAULT_RETRIEVAL_DEVICES="$(python - "$ASCEND_VISIBLE_DEVICES" <<'PY'
import sys
devices = [x for x in sys.argv[1].split(",") if x]
print(devices[7] if len(devices) >= 8 else (devices[-1] if devices else "0"))
PY
)"
else
    DEFAULT_RETRIEVAL_DEVICES="${GPU:-0}"
fi
RETRIEVAL_DEVICES="${RETRIEVAL_DEVICES:-$DEFAULT_RETRIEVAL_DEVICES}"
PORT="${PORT:-8100}"
HOST="${HOST:-127.0.0.1}"
TEXT_TRUNCATE="${TEXT_TRUNCATE:-500}"
FAISS_DEVICE="${FAISS_DEVICE:-cpu}"
FAISS_GPU_ID="${FAISS_GPU_ID:-0}"
FAISS_GPU_FP16="${FAISS_GPU_FP16:-false}"
DRY_RUN="${DRY_RUN:-false}"

case "$DEVICE_BACKEND" in
    cuda)
        VISIBLE_DEVICES_ENV="CUDA_VISIBLE_DEVICES"
        RETRIEVAL_DEVICE="cuda:0"
        ;;
    npu)
        VISIBLE_DEVICES_ENV="ASCEND_RT_VISIBLE_DEVICES"
        RETRIEVAL_DEVICE="npu:0"
        ;;
    cpu)
        VISIBLE_DEVICES_ENV=""
        RETRIEVAL_DEVICE="cpu"
        ;;
    *)
        echo "[run_retrieval_daemon_flexible] ERROR: DEVICE_BACKEND must be cuda, npu, or cpu, got: $DEVICE_BACKEND" >&2
        exit 2
        ;;
esac

if [ -n "$VISIBLE_DEVICES_ENV" ]; then
    echo "[run_retrieval_daemon_flexible] backend=$DEVICE_BACKEND ${VISIBLE_DEVICES_ENV}=$RETRIEVAL_DEVICES device=$RETRIEVAL_DEVICE port=$PORT faiss_device=$FAISS_DEVICE"
else
    echo "[run_retrieval_daemon_flexible] backend=$DEVICE_BACKEND device=$RETRIEVAL_DEVICE port=$PORT faiss_device=$FAISS_DEVICE"
fi

CMD=(
    python "$SCRIPT_DIR/retrieval_daemon.py"
    --host "$HOST"
    --port "$PORT"
    --device "$RETRIEVAL_DEVICE"
    --text_truncate "$TEXT_TRUNCATE"
    --faiss_device "$FAISS_DEVICE"
    --faiss_gpu_id "$FAISS_GPU_ID"
)
[ "$FAISS_GPU_FP16" = "true" ] && CMD+=(--faiss_gpu_fp16)

if [ "$DRY_RUN" = "true" ]; then
    if [ -n "$VISIBLE_DEVICES_ENV" ]; then
        printf '[run_retrieval_daemon_flexible] DRY_RUN command: %s=%q' "$VISIBLE_DEVICES_ENV" "$RETRIEVAL_DEVICES"
    else
        printf '[run_retrieval_daemon_flexible] DRY_RUN command:'
    fi
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

if [ -n "$VISIBLE_DEVICES_ENV" ]; then
    env "$VISIBLE_DEVICES_ENV=$RETRIEVAL_DEVICES" "${CMD[@]}" 2>&1 | tee "$LOG_DIR/retrieval_daemon_flexible.log"
else
    "${CMD[@]}" 2>&1 | tee "$LOG_DIR/retrieval_daemon_flexible.log"
fi
