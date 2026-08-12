#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_NAME="opsd_multi_q001_a003_3src_s3000_actionfix_detached_20260812"
LOG_DIR="$SCRIPT_DIR/logs/$RUN_NAME"
PID_FILE="$LOG_DIR/outer.pid"

mkdir -p "$LOG_DIR"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already_running pid=$(cat "$PID_FILE")"
    exit 0
fi

nohup setsid env \
    TEACHER_ACTION_SCOPE=multi \
    TEACHER_QUERY_KL_COEF=0.01 \
    TEACHER_EVIDENCE_KL_COEF=0.0 \
    TEACHER_ANSWER_KL_COEF=0.03 \
    ENABLE_EVIDENCE_AGENT=true \
    MAX_STEPS=3000 \
    PER_DEVICE_BATCH_SIZE=2 \
    GRADIENT_ACCUMULATION_STEPS=4 \
    STEPS_PER_GENERATION=8 \
    NUM_GENERATIONS=8 \
    SAVE_STEPS=500 \
    SAVE_TOTAL_LIMIT=8 \
    DATASET="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_musique_train_multi_opsd.jsonl" \
    RUN_NAME="$RUN_NAME" \
    ROLLOUT_PORT=8030 \
    VLLM_GROUP_PORT=21239 \
    ROLLOUT_GPU=7 \
    TRAIN_DEVICES=2,3,4,5,6 \
    NPROC_PER_NODE=5 \
    bash "$SCRIPT_DIR/launch_action_scoped_opsd_worker.sh" \
    > "$LOG_DIR/outer.log" 2>&1 < /dev/null &

echo "$!" > "$PID_FILE"
echo "started pid=$! log=$LOG_DIR/outer.log"
