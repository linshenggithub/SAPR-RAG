#!/usr/bin/env bash
# Orchestrate retrieval, frozen 14B teacher, 7B rollout, and OPD training on one 8-GPU worker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
GRPO_DIR="$PROJ_ROOT/03_sapr_rag/scripts/grpo"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

RUN_NAME="${RUN_NAME:-sapr_opd_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/$RUN_NAME}"
TEACHER_MODEL="${TEACHER_MODEL:-$PROJ_ROOT/03_sapr_rag/models/Qwen2.5-14B-Instruct}"
TEACHER_GPU="${TEACHER_GPU:-1}"
TEACHER_PORT="${TEACHER_PORT:-8040}"
ROLLOUT_GPU="${ROLLOUT_GPU:-7}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8030}"
RETRIEVAL_GPU="${RETRIEVAL_GPU:-0}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8100}"
KEEP_SERVICES="${KEEP_SERVICES:-false}"
DRY_RUN="${DRY_RUN:-false}"
ROLLOUT_PID=""

mkdir -p "$LOG_DIR"

cleanup() {
    if [[ "$KEEP_SERVICES" == "true" ]]; then
        return
    fi
    if [[ -n "$ROLLOUT_PID" ]] && kill -0 "$ROLLOUT_PID" 2>/dev/null; then
        local pgid
        pgid="$(ps -o pgid= -p "$ROLLOUT_PID" | tr -d ' ' || true)"
        [[ -n "$pgid" ]] && kill -TERM "-$pgid" 2>/dev/null || kill -TERM "$ROLLOUT_PID" 2>/dev/null || true
    fi
    RUN_NAME="$RUN_NAME" LOG_DIR="$LOG_DIR" TEACHER_PORT="$TEACHER_PORT" \
        bash "$SCRIPT_DIR/teacher_service.sh" stop >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[orchestrator] DRY_RUN run=$RUN_NAME"
    env RUN_NAME="$RUN_NAME" LOG_DIR="$LOG_DIR" TEACHER_MODEL="$TEACHER_MODEL" \
        TEACHER_GPU="$TEACHER_GPU" TEACHER_PORT="$TEACHER_PORT" \
        bash "$SCRIPT_DIR/teacher_service.sh" status || true
    env RUN_NAME="$RUN_NAME" OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/opd/$RUN_NAME}" \
        TEACHER_URL="http://127.0.0.1:${TEACHER_PORT}" \
        ROLLOUT_PORT="$ROLLOUT_PORT" DRY_RUN=true \
        bash "$SCRIPT_DIR/run_sapr_opd.sh"
    exit 0
fi

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
[[ "$gpu_count" -ge 8 ]] || {
    echo "[orchestrator] ERROR: requires 8 visible GPUs, found $gpu_count" >&2
    exit 2
}

for path in \
    "$TEACHER_MODEL/config.json" \
    "$PROJ_ROOT/03_sapr_rag/models/Qwen2.5-7B-Instruct/config.json" \
    "$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft/checkpoint-1650/adapter_config.json"; do
    [[ -f "$path" ]] || { echo "[orchestrator] ERROR: missing $path" >&2; exit 2; }
done

python - "$PROJ_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
student = root / "03_sapr_rag/models/Qwen2.5-7B-Instruct"
teacher = root / "03_sapr_rag/models/Qwen2.5-14B-Instruct"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

student_cfg = json.loads((student / "config.json").read_text())
teacher_cfg = json.loads((teacher / "config.json").read_text())
if student_cfg["vocab_size"] != teacher_cfg["vocab_size"]:
    raise SystemExit("student/teacher vocab_size mismatch")
if sha256(student / "tokenizer.json") != sha256(teacher / "tokenizer.json"):
    raise SystemExit("student/teacher tokenizer.json mismatch")
print("[orchestrator] tokenizer alignment OK")
PY

env RETRIEVAL_GPU="$RETRIEVAL_GPU" RETRIEVAL_PORT="$RETRIEVAL_PORT" \
    bash "$GRPO_DIR/retrieval_service.sh" start
env RETRIEVAL_GPU="$RETRIEVAL_GPU" RETRIEVAL_PORT="$RETRIEVAL_PORT" WAIT_TIMEOUT=1800 \
    bash "$GRPO_DIR/retrieval_service.sh" wait

env RUN_NAME="$RUN_NAME" LOG_DIR="$LOG_DIR" TEACHER_MODEL="$TEACHER_MODEL" \
    TEACHER_GPU="$TEACHER_GPU" TEACHER_PORT="$TEACHER_PORT" \
    bash "$SCRIPT_DIR/teacher_service.sh" start
env RUN_NAME="$RUN_NAME" LOG_DIR="$LOG_DIR" TEACHER_MODEL="$TEACHER_MODEL" \
    TEACHER_GPU="$TEACHER_GPU" TEACHER_PORT="$TEACHER_PORT" WAIT_TIMEOUT=1200 \
    bash "$SCRIPT_DIR/teacher_service.sh" wait

setsid env \
    DEVICE_BACKEND=cuda \
    ROLLOUT_DEVICES="$ROLLOUT_GPU" \
    PORT="$ROLLOUT_PORT" \
    SAPR_RETRIEVAL_URL="http://127.0.0.1:${RETRIEVAL_PORT}" \
    SAPR_TOP_K=3 \
    SAPR_ENABLE_EVIDENCE_AGENT=true \
    MULTI_TURN_SCHEDULER=sapr_rag_scheduler \
    INIT_ADAPTER=sft \
    VLLM_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-8192}" \
    VLLM_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.85}" \
    bash "$GRPO_DIR/run_rollout_opsd.sh" \
    >"$LOG_DIR/rollout.log" 2>&1 < /dev/null &
ROLLOUT_PID="$!"
echo "$ROLLOUT_PID" >"$LOG_DIR/rollout.pid"

started="$(date +%s)"
until curl -fsS --max-time 5 "http://127.0.0.1:${ROLLOUT_PORT}/health/" \
    >"$LOG_DIR/rollout_health.json" 2>/dev/null; do
    if ! kill -0 "$ROLLOUT_PID" 2>/dev/null; then
        echo "[orchestrator] ERROR: rollout process exited" >&2
        tail -n 120 "$LOG_DIR/rollout.log" >&2 || true
        exit 3
    fi
    if (( $(date +%s) - started >= 1200 )); then
        echo "[orchestrator] ERROR: rollout health timeout" >&2
        tail -n 120 "$LOG_DIR/rollout.log" >&2 || true
        exit 4
    fi
    sleep 10
done

{
    echo "run_name=$RUN_NAME"
    echo "worker_id=${WORKER_ID:-unknown}"
    echo "teacher_model=$TEACHER_MODEL"
    echo "teacher_url=http://127.0.0.1:${TEACHER_PORT}"
    echo "retrieval_url=http://127.0.0.1:${RETRIEVAL_PORT}"
    echo "rollout_url=http://127.0.0.1:${ROLLOUT_PORT}"
    echo "started_at=$(date -Is)"
} >"$LOG_DIR/status.txt"

env \
    RUN_NAME="$RUN_NAME" \
    LOG_DIR="$LOG_DIR" \
    TEACHER_URL="http://127.0.0.1:${TEACHER_PORT}" \
    ROLLOUT_PORT="$ROLLOUT_PORT" \
    "$@" \
    bash "$SCRIPT_DIR/run_sapr_opd.sh" \
    2>&1 | tee "$LOG_DIR/train.log"

echo "completed_at=$(date -Is)" >>"$LOG_DIR/status.txt"
echo "completed=1" >>"$LOG_DIR/status.txt"
