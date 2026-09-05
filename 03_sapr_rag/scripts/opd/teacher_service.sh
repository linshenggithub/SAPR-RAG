#!/usr/bin/env bash
# Manage the frozen external teacher used for sampled-token OPD scoring.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SWIFT_ROOT="${SWIFT_ROOT:-$(cd "$PROJ_ROOT/../ms-swift" && pwd)}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

TEACHER_MODEL="${TEACHER_MODEL:-$PROJ_ROOT/03_sapr_rag/models/Qwen2.5-14B-Instruct}"
TEACHER_ADAPTER="${TEACHER_ADAPTER:-}"
TEACHER_GPU="${TEACHER_GPU:-1}"
TEACHER_PORT="${TEACHER_PORT:-8040}"
TEACHER_HOST="${TEACHER_HOST:-127.0.0.1}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-8192}"
TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL:-0.82}"
TEACHER_MAX_LOGPROBS="${TEACHER_MAX_LOGPROBS:-1}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1200}"
RUN_NAME="${RUN_NAME:-sapr_opd}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/$RUN_NAME}"
PID_FILE="$LOG_DIR/teacher.pid"
LOG_FILE="$LOG_DIR/teacher.log"
ACTION="${1:-status}"

mkdir -p "$LOG_DIR"

health() {
    curl -fsS --max-time 5 "http://${TEACHER_HOST}:${TEACHER_PORT}/health/" 2>/dev/null
}

is_alive() {
    health >/dev/null
}

start() {
    [[ -d "$TEACHER_MODEL" ]] || {
        echo "[teacher] ERROR: model not found: $TEACHER_MODEL" >&2
        return 2
    }
    if is_alive; then
        echo "[teacher] already ready on ${TEACHER_HOST}:${TEACHER_PORT}"
        return 0
    fi

    local adapter_args=()
    [[ -n "$TEACHER_ADAPTER" ]] && adapter_args=(--adapters "$TEACHER_ADAPTER")
    echo "[teacher] starting model=$TEACHER_MODEL gpu=$TEACHER_GPU port=$TEACHER_PORT"
    (
        cd "$SWIFT_ROOT"
        exec setsid env CUDA_VISIBLE_DEVICES="$TEACHER_GPU" \
            swift deploy \
            --model "$TEACHER_MODEL" \
            "${adapter_args[@]}" \
            --infer_backend vllm \
            --torch_dtype bfloat16 \
            --host "$TEACHER_HOST" \
            --port "$TEACHER_PORT" \
            --max_logprobs "$TEACHER_MAX_LOGPROBS" \
            --max_length "$TEACHER_MAX_MODEL_LEN" \
            --vllm_max_model_len "$TEACHER_MAX_MODEL_LEN" \
            --vllm_gpu_memory_utilization "$TEACHER_GPU_MEM_UTIL"
    ) >"$LOG_FILE" 2>&1 &
    echo "$!" >"$PID_FILE"
    echo "[teacher] pid=$(cat "$PID_FILE") log=$LOG_FILE"
}

wait_ready() {
    local started
    started="$(date +%s)"
    until is_alive; do
        if [[ -f "$PID_FILE" ]] && ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "[teacher] ERROR: process exited before health became ready" >&2
            tail -n 120 "$LOG_FILE" >&2 || true
            return 3
        fi
        if (( $(date +%s) - started >= WAIT_TIMEOUT )); then
            echo "[teacher] ERROR: health timeout after ${WAIT_TIMEOUT}s" >&2
            tail -n 120 "$LOG_FILE" >&2 || true
            return 4
        fi
        sleep 10
    done
    echo "[teacher] READY $(health)"
}

status() {
    if is_alive; then
        echo "[teacher] READY $(health)"
    elif [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "[teacher] STARTING pid=$(cat "$PID_FILE")"
    else
        echo "[teacher] STOPPED"
        return 1
    fi
}

stop() {
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        local pid
        pid="$(cat "$PID_FILE")"
        local pgid
        pgid="$(ps -o pgid= -p "$pid" | tr -d ' ' || true)"
        [[ -n "$pgid" ]] && kill -TERM "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        echo "[teacher] stopped pid=$pid"
    else
        echo "[teacher] no managed process"
    fi
    rm -f "$PID_FILE"
}

case "$ACTION" in
    start) start ;;
    wait) wait_ready ;;
    status) status ;;
    stop) stop ;;
    restart) stop; sleep 2; start ;;
    *) echo "usage: $0 {start|wait|status|stop|restart}" >&2; exit 2 ;;
esac
