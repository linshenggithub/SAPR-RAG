#!/usr/bin/env bash
# SAPR-RAG OPSD-arm smoke（H20/CUDA, 20 steps, dynamic OPSD）。
# 三进程编排：retrieval daemon(CPU) -> rollout(GPU6) -> train(GPU0-5)。
# 训练臂：ENABLE_OPSD=true + OPSD 数据集(带 teacher_prompt) + teacher_kl_coef。
# 与 control smoke 唯一差异是训练臂参数，retrieval/rollout 完全一致。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOG_DIR="$SCRIPT_DIR/logs/smoke_opsd_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

TEACHER_KL_COEF="${TEACHER_KL_COEF:-0.1}"

# 关键：清掉 worker 继承的 PORT，避免污染 rollout/retrieval 默认端口
unset PORT || true

RETRIEVAL_PORT=8100
ROLLOUT_PORT=8000
export SAPR_RETRIEVAL_URL="http://127.0.0.1:${RETRIEVAL_PORT}"

echo "[smoke-opsd] LOG_DIR=$LOG_DIR"
echo "[smoke-opsd] retrieval_port=$RETRIEVAL_PORT rollout_port=$ROLLOUT_PORT teacher_kl_coef=$TEACHER_KL_COEF"

cleanup() {
    echo "[smoke-opsd] cleanup: stopping background daemons"
    # 注意：只杀本 smoke 自己拉起的进程。常驻检索服务（REUSE_RETRIEVAL=1）不在此列，
    # 由 retrieval_service.sh 独立管理，smoke 结束不得带走它。
    [ -n "${RETR_PID:-}" ] && kill "$RETR_PID" 2>/dev/null || true
    [ -n "${ROLL_PID:-}" ] && kill "$ROLL_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---------- 1) retrieval daemon ----------
# 优先复用常驻服务（retrieval_service.sh 起的 GPU 长驻 daemon）：/health 存活即直接用，
# 免去每次冷启动重新 mmap 68GB 索引的十几分钟。仅当无常驻服务时才临时拉一个 CPU daemon。
if curl -sf "http://127.0.0.1:${RETRIEVAL_PORT}/health" >/dev/null 2>&1; then
    REUSE_RETRIEVAL=1
    echo "[smoke-opsd] reusing persistent retrieval service on port $RETRIEVAL_PORT: $(curl -s http://127.0.0.1:${RETRIEVAL_PORT}/health)"
else
    echo "[smoke-opsd] no persistent retrieval service; starting a temporary CPU daemon ..."
    DEVICE_BACKEND=cpu PORT=$RETRIEVAL_PORT \
        nohup bash "$SCRIPT_DIR/run_retrieval_daemon_flexible.sh" \
        > "$LOG_DIR/retrieval.log" 2>&1 &
    RETR_PID=$!
    echo "[smoke-opsd] retrieval pid=$RETR_PID; waiting for /health"
    ready=0
    for i in $(seq 1 240); do
        if curl -sf "http://127.0.0.1:${RETRIEVAL_PORT}/health" >/dev/null 2>&1; then ready=1; break; fi
        if ! kill -0 "$RETR_PID" 2>/dev/null; then
            echo "[smoke-opsd] ERROR: retrieval exited early"; tail -40 "$LOG_DIR/retrieval.log"; exit 3; fi
        sleep 5
    done
    [ "$ready" = 1 ] || { echo "[smoke-opsd] ERROR: retrieval not ready"; tail -40 "$LOG_DIR/retrieval.log"; exit 3; }
    echo "[smoke-opsd] retrieval /health OK: $(curl -s http://127.0.0.1:${RETRIEVAL_PORT}/health)"
fi

# ---------- 2) rollout server (GPU6) ----------
echo "[smoke-opsd] starting rollout server (GPU6) ..."
DEVICE_BACKEND=cuda ROLLOUT_DEVICES=6 PORT=$ROLLOUT_PORT INIT_ADAPTER=sft_dpo \
    nohup bash "$SCRIPT_DIR/run_rollout_opsd.sh" \
    > "$LOG_DIR/rollout.log" 2>&1 &
ROLL_PID=$!
echo "[smoke-opsd] rollout pid=$ROLL_PID; waiting for port $ROLLOUT_PORT"
ready=0
for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${ROLLOUT_PORT}/health" >/dev/null 2>&1 \
       || curl -s "http://127.0.0.1:${ROLLOUT_PORT}/" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$ROLL_PID" 2>/dev/null; then
        echo "[smoke-opsd] ERROR: rollout exited early"; tail -60 "$LOG_DIR/rollout.log"; exit 4; fi
    sleep 5
done
[ "$ready" = 1 ] || { echo "[smoke-opsd] ERROR: rollout not ready"; tail -60 "$LOG_DIR/rollout.log"; exit 4; }
ACTUAL_PORT=$(grep -oE 'Uvicorn running on http://[0-9.]+:[0-9]+' "$LOG_DIR/rollout.log" | grep -oE '[0-9]+$' | tail -1)
ACTUAL_PORT=${ACTUAL_PORT:-$ROLLOUT_PORT}
echo "[smoke-opsd] rollout actual port=$ACTUAL_PORT"

# ---------- 3) train (GPU0-5), OPSD arm, 20 steps ----------
echo "[smoke-opsd] starting OPSD training (GPU0-5, MAX_STEPS=20, ENABLE_OPSD=true) ..."
DEVICE_BACKEND=cuda \
TRAIN_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
INIT_ADAPTER=sft_dpo \
ENABLE_OPSD=true \
TEACHER_KL_COEF="$TEACHER_KL_COEF" \
DATASET="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_train_pilot_opsd.jsonl" \
OUTPUT_DIR="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_pilot/opsd_smoke_alpha0p1" \
VLLM_PORT="$ACTUAL_PORT" \
MAX_STEPS=20 \
PER_DEVICE_BATCH_SIZE=1 \
MAX_LENGTH=2048 \
MAX_COMPLETION_LENGTH=4096 \
VLLM_MAX_MODEL_LEN=8192 \
DEEPSPEED=zero2 \
DRY_RUN=false \
    bash "$SCRIPT_DIR/run_grpo_opsd.sh" 2>&1 | tee "$LOG_DIR/train.log"

echo "[smoke-opsd] training finished (rc=${PIPESTATUS[0]}); logs in $LOG_DIR"
