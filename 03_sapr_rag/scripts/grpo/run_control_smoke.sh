#!/usr/bin/env bash
# SAPR-RAG matched CONTROL smoke（H20/CUDA, 20 steps, plain GRPO）。
# 三进程编排：retrieval daemon(CPU) -> rollout(GPU6) -> train(GPU0-5)。
# 只跑 plain control 臂，不跑 OPSD；用于验证链路、token 对齐、loss_mask、无 OOM。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOG_DIR="$SCRIPT_DIR/logs/smoke_control_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 关键：清掉 worker 继承的 PORT（11247），避免污染 rollout/retrieval 默认端口
unset PORT || true

RETRIEVAL_PORT=8100
ROLLOUT_PORT=8000
export SAPR_RETRIEVAL_URL="http://127.0.0.1:${RETRIEVAL_PORT}"

echo "[smoke] LOG_DIR=$LOG_DIR"
echo "[smoke] retrieval_port=$RETRIEVAL_PORT rollout_port=$ROLLOUT_PORT"

cleanup() {
    echo "[smoke] cleanup: stopping background daemons"
    [ -n "${RETR_PID:-}" ] && kill "$RETR_PID" 2>/dev/null || true
    [ -n "${ROLL_PID:-}" ] && kill "$ROLL_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---------- 1) retrieval daemon (CPU) ----------
echo "[smoke] starting retrieval daemon (CPU) ..."
DEVICE_BACKEND=cpu PORT=$RETRIEVAL_PORT \
    nohup bash "$SCRIPT_DIR/run_retrieval_daemon_flexible.sh" \
    > "$LOG_DIR/retrieval.log" 2>&1 &
RETR_PID=$!
echo "[smoke] retrieval pid=$RETR_PID; waiting for /health (index is 64G, allow up to 20min)"

ready=0
for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${RETRIEVAL_PORT}/health" >/dev/null 2>&1; then
        ready=1; break
    fi
    if ! kill -0 "$RETR_PID" 2>/dev/null; then
        echo "[smoke] ERROR: retrieval daemon exited early; tail:"; tail -40 "$LOG_DIR/retrieval.log"; exit 3
    fi
    sleep 5
done
[ "$ready" = 1 ] || { echo "[smoke] ERROR: retrieval not ready in time"; tail -40 "$LOG_DIR/retrieval.log"; exit 3; }
echo "[smoke] retrieval /health OK: $(curl -s http://127.0.0.1:${RETRIEVAL_PORT}/health)"

# ---------- 2) rollout server (GPU6) ----------
echo "[smoke] starting rollout server (GPU6) ..."
DEVICE_BACKEND=cuda ROLLOUT_DEVICES=6 PORT=$ROLLOUT_PORT INIT_ADAPTER=sft_dpo \
    nohup bash "$SCRIPT_DIR/run_rollout_opsd.sh" \
    > "$LOG_DIR/rollout.log" 2>&1 &
ROLL_PID=$!
echo "[smoke] rollout pid=$ROLL_PID; waiting for port $ROLLOUT_PORT (model load, allow up to 15min)"

ready=0
for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${ROLLOUT_PORT}/health" >/dev/null 2>&1 \
       || curl -s "http://127.0.0.1:${ROLLOUT_PORT}/" >/dev/null 2>&1; then
        ready=1; break
    fi
    if ! kill -0 "$ROLL_PID" 2>/dev/null; then
        echo "[smoke] ERROR: rollout exited early; tail:"; tail -60 "$LOG_DIR/rollout.log"; exit 4
    fi
    sleep 5
done
[ "$ready" = 1 ] || { echo "[smoke] ERROR: rollout not ready in time"; tail -60 "$LOG_DIR/rollout.log"; exit 4; }
echo "[smoke] rollout server responding on $ROLLOUT_PORT"
# 记录 rollout 实际监听端口（若 8000 被占 swift 会改口）
ACTUAL_PORT=$(grep -oE 'Uvicorn running on http://[0-9.]+:[0-9]+' "$LOG_DIR/rollout.log" | grep -oE '[0-9]+$' | tail -1)
ACTUAL_PORT=${ACTUAL_PORT:-$ROLLOUT_PORT}
echo "[smoke] rollout actual port=$ACTUAL_PORT"

# ---------- 3) train (GPU0-5), plain control, 20 steps ----------
echo "[smoke] starting plain control training (GPU0-5, MAX_STEPS=20) ..."
DEVICE_BACKEND=cuda \
TRAIN_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
INIT_ADAPTER=sft_dpo \
ENABLE_OPSD=false \
DATASET="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_train_pilot.jsonl" \
OUTPUT_DIR="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_pilot/plain_control_smoke" \
VLLM_PORT="$ACTUAL_PORT" \
MAX_STEPS=20 \
PER_DEVICE_BATCH_SIZE=1 \
MAX_LENGTH=2048 \
MAX_COMPLETION_LENGTH=4096 \
VLLM_MAX_MODEL_LEN=8192 \
DEEPSPEED=zero2 \
DRY_RUN=false \
    bash "$SCRIPT_DIR/run_grpo_opsd.sh" 2>&1 | tee "$LOG_DIR/train.log"

echo "[smoke] training finished (rc=${PIPESTATUS[0]}); logs in $LOG_DIR"
