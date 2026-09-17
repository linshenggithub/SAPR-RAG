#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SCOPE="${TEACHER_ACTION_SCOPE:-multi}"
INIT_ADAPTER="${INIT_ADAPTER:-sft_dpo}"
MAX_STEPS="${MAX_STEPS:-1}"
ENABLE_EVIDENCE_AGENT="${ENABLE_EVIDENCE_AGENT:-true}"
MULTI_TURN_SCHEDULER="${MULTI_TURN_SCHEDULER:-sapr_rag_scheduler}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8030}"
GROUP_PORT="${VLLM_GROUP_PORT:-51309}"
ROLLOUT_GPU="${ROLLOUT_GPU:-7}"
# DP rollout（可选）：设置 ROLLOUT_GPU2 + ROLLOUT_PORT2 即启动第二个 rollout 副本，
# 训练侧把两个 (port, group_port) 都传给 VLLMClient 做数据并行采样 + 逐副本权重同步。
# 不设置 ROLLOUT_GPU2 时行为与原来完全一致（单副本）。
ROLLOUT_GPU2="${ROLLOUT_GPU2:-}"
ROLLOUT_PORT2="${ROLLOUT_PORT2:-8031}"
GROUP_PORT2="${VLLM_GROUP_PORT2:-51311}"
TRAIN_DEVICES="${TRAIN_DEVICES:-2,3,4,5,6}"
NPROC_PER_NODE="${NPROC_PER_NODE:-5}"
DATASET="${DATASET:-$PROJ_ROOT/data/grpo/hotpotqa_2wiki_action_opsd_smoke_100.jsonl}"
QUERY_COEF="${TEACHER_QUERY_KL_COEF:-0.01}"
EVIDENCE_COEF="${TEACHER_EVIDENCE_KL_COEF:-0.0}"
ANSWER_COEF="${TEACHER_ANSWER_KL_COEF:-0.03}"
ENABLE_TRUNCATION_REWARD="${ENABLE_TRUNCATION_REWARD:-false}"
TRUNCATION_REWARD_WEIGHT="${TRUNCATION_REWARD_WEIGHT:-0.5}"
RUN_NAME="${RUN_NAME:-opsd_scope_${SCOPE}_s${MAX_STEPS}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/$RUN_NAME}"
LOG_DIR="$SCRIPT_DIR/logs/$RUN_NAME"
STATUS="$LOG_DIR/status.txt"
ROLLOUT_PID=""
ROLLOUT_PID2=""

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

cleanup() {
    for _pid in "$ROLLOUT_PID" "$ROLLOUT_PID2"; do
        [ -n "$_pid" ] || continue
        kill -0 "$_pid" 2>/dev/null || continue
        local pgid
        pgid="$(ps -o pgid= -p "$_pid" | tr -d ' ' || true)"
        if [ -n "$pgid" ]; then
            kill -TERM "-$pgid" 2>/dev/null || true
        else
            kill -TERM "$_pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

{
    echo "run_name=$RUN_NAME"
    echo "init_adapter=$INIT_ADAPTER"
    echo "scope=$SCOPE"
    echo "query_coef=$QUERY_COEF"
    echo "evidence_coef=$EVIDENCE_COEF"
    echo "answer_coef=$ANSWER_COEF"
    echo "truncation_reward=$ENABLE_TRUNCATION_REWARD"
    echo "truncation_reward_weight=$TRUNCATION_REWARD_WEIGHT"
    echo "max_steps=$MAX_STEPS"
    echo "per_device_batch_size=${PER_DEVICE_BATCH_SIZE:-1}"
    echo "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-4}"
    echo "steps_per_generation=${STEPS_PER_GENERATION:-8}"
    echo "num_generations=${NUM_GENERATIONS:-8}"
    echo "save_steps=${SAVE_STEPS:-25}"
    echo "save_total_limit=${SAVE_TOTAL_LIMIT:-60}"
    echo "action_credit_mode=${ACTION_CREDIT_MODE:-off}"
    echo "action_credit_coef=${ACTION_CREDIT_COEF:-0.0}"
    echo "action_credit_scale=${ACTION_CREDIT_SCALE:-group_turn}"
    echo "action_credit_gate=${ACTION_CREDIT_GATE:-all}"
    echo "advantage_mode=${ADVANTAGE_MODE:-sequence}"
    echo "action_query_outcome_coef=${ACTION_QUERY_OUTCOME_COEF:-0.25}"
    echo "action_credit_clip=${ACTION_CREDIT_CLIP:-2.0}"
    echo "dynamic_sample=${DYNAMIC_SAMPLE:-false}"
    echo "max_resample_times=${MAX_RESAMPLE_TIMES:-3}"
    echo "overlong_filter=${OVERLONG_FILTER:-false}"
    echo "loss_type=${LOSS_TYPE:-grpo}"
    echo "evidence_agent=$ENABLE_EVIDENCE_AGENT"
    echo "scheduler=$MULTI_TURN_SCHEDULER"
    echo "rollout_port=$ROLLOUT_PORT"
    echo "group_port=$GROUP_PORT"
    echo "rollout_gpu=$ROLLOUT_GPU"
    echo "train_devices=$TRAIN_DEVICES"
    echo "dataset=$DATASET"
    echo "output_dir=$OUTPUT_DIR"
    echo "started_at=$(date -Is)"
} > "$STATUS"

curl -fsS --max-time 10 http://127.0.0.1:8100/health > "$LOG_DIR/retrieval_health.json"

nohup setsid env \
    DEVICE_BACKEND=cuda \
    ROLLOUT_DEVICES="$ROLLOUT_GPU" \
    PORT="$ROLLOUT_PORT" \
    SAPR_ENABLE_EVIDENCE_AGENT="$ENABLE_EVIDENCE_AGENT" \
    MULTI_TURN_SCHEDULER="$MULTI_TURN_SCHEDULER" \
    INIT_ADAPTER="$INIT_ADAPTER" \
    VLLM_MAX_MODEL_LEN=8192 \
    VLLM_GPU_MEM_UTIL=0.85 \
    bash "$SCRIPT_DIR/run_rollout_opsd.sh" \
    > "$LOG_DIR/rollout.log" 2>&1 < /dev/null &
ROLLOUT_PID="$!"
echo "$ROLLOUT_PID" > "$LOG_DIR/rollout.pid"

start_ts="$(date +%s)"
until curl -fsS --max-time 5 "http://127.0.0.1:${ROLLOUT_PORT}/health/" > "$LOG_DIR/rollout_health.json" 2>/dev/null; do
    if ! kill -0 "$ROLLOUT_PID" 2>/dev/null; then
        echo "rollout_died=1" >> "$STATUS"
        tail -n 120 "$LOG_DIR/rollout.log"
        exit 4
    fi
    if [ $(($(date +%s) - start_ts)) -ge 1200 ]; then
        echo "rollout_timeout=1" >> "$STATUS"
        tail -n 120 "$LOG_DIR/rollout.log"
        exit 5
    fi
    sleep 10
done
echo "rollout_ready_at=$(date -Is)" >> "$STATUS"

# 可选：启动第二个 rollout 副本（DP rollout）。
VLLM_PORT_LIST="$ROLLOUT_PORT"
VLLM_GROUP_PORT_LIST="$GROUP_PORT"
if [ -n "$ROLLOUT_GPU2" ]; then
    echo "rollout2_gpu=$ROLLOUT_GPU2" >> "$STATUS"
    echo "rollout2_port=$ROLLOUT_PORT2" >> "$STATUS"
    echo "rollout2_group_port=$GROUP_PORT2" >> "$STATUS"
    nohup setsid env \
        DEVICE_BACKEND=cuda \
        ROLLOUT_DEVICES="$ROLLOUT_GPU2" \
        PORT="$ROLLOUT_PORT2" \
        SAPR_ENABLE_EVIDENCE_AGENT="$ENABLE_EVIDENCE_AGENT" \
        MULTI_TURN_SCHEDULER="$MULTI_TURN_SCHEDULER" \
        INIT_ADAPTER="$INIT_ADAPTER" \
        VLLM_MAX_MODEL_LEN=8192 \
        VLLM_GPU_MEM_UTIL=0.85 \
        bash "$SCRIPT_DIR/run_rollout_opsd.sh" \
        > "$LOG_DIR/rollout2.log" 2>&1 < /dev/null &
    ROLLOUT_PID2="$!"
    echo "$ROLLOUT_PID2" > "$LOG_DIR/rollout2.pid"

    start_ts2="$(date +%s)"
    until curl -fsS --max-time 5 "http://127.0.0.1:${ROLLOUT_PORT2}/health/" > "$LOG_DIR/rollout2_health.json" 2>/dev/null; do
        if ! kill -0 "$ROLLOUT_PID2" 2>/dev/null; then
            echo "rollout2_died=1" >> "$STATUS"
            tail -n 120 "$LOG_DIR/rollout2.log"
            exit 4
        fi
        if [ $(($(date +%s) - start_ts2)) -ge 1200 ]; then
            echo "rollout2_timeout=1" >> "$STATUS"
            tail -n 120 "$LOG_DIR/rollout2.log"
            exit 5
        fi
        sleep 10
    done
    echo "rollout2_ready_at=$(date -Is)" >> "$STATUS"
    VLLM_PORT_LIST="$ROLLOUT_PORT $ROLLOUT_PORT2"
    VLLM_GROUP_PORT_LIST="$GROUP_PORT $GROUP_PORT2"
fi

env \
    DEVICE_BACKEND=cuda \
    TRAIN_DEVICES="$TRAIN_DEVICES" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    INIT_ADAPTER="$INIT_ADAPTER" \
    ENABLE_OPSD="${ENABLE_OPSD:-true}" \
    TEACHER_KL_COEF="${TEACHER_KL_COEF:-0.1}" \
    TEACHER_ACTION_SCOPE="$SCOPE" \
    TEACHER_QUERY_KL_COEF="$QUERY_COEF" \
    TEACHER_EVIDENCE_KL_COEF="$EVIDENCE_COEF" \
    TEACHER_ANSWER_KL_COEF="$ANSWER_COEF" \
    ENABLE_TRUNCATION_REWARD="$ENABLE_TRUNCATION_REWARD" \
    TRUNCATION_REWARD_WEIGHT="$TRUNCATION_REWARD_WEIGHT" \
    ENABLE_REWARD="${ENABLE_REWARD:-true}" \
    OPD_USE_GRPO_ADVANTAGE="${OPD_USE_GRPO_ADVANTAGE:-true}" \
    ACTION_CREDIT_MODE="${ACTION_CREDIT_MODE:-off}" \
    ACTION_CREDIT_COEF="${ACTION_CREDIT_COEF:-0.0}" \
    ACTION_CREDIT_SCALE="${ACTION_CREDIT_SCALE:-group_turn}" \
    ACTION_CREDIT_INFOS_KEY="${ACTION_CREDIT_INFOS_KEY:-query_evidence_gain}" \
    ACTION_CREDIT_GATE="${ACTION_CREDIT_GATE:-all}" \
    ADVANTAGE_MODE="${ADVANTAGE_MODE:-sequence}" \
    ACTION_QUERY_OUTCOME_COEF="${ACTION_QUERY_OUTCOME_COEF:-0.25}" \
    ACTION_CREDIT_CLIP="${ACTION_CREDIT_CLIP:-2.0}" \
    DYNAMIC_SAMPLE="${DYNAMIC_SAMPLE:-false}" \
    MAX_RESAMPLE_TIMES="${MAX_RESAMPLE_TIMES:-3}" \
    OVERLONG_FILTER="${OVERLONG_FILTER:-false}" \
    LOSS_TYPE="${LOSS_TYPE:-grpo}" \
    DATASET="$DATASET" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    VLLM_PORT="$VLLM_PORT_LIST" \
    VLLM_GROUP_PORT="$VLLM_GROUP_PORT_LIST" \
    MAX_STEPS="$MAX_STEPS" \
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}" \
    MAX_LENGTH="${MAX_LENGTH:-2048}" \
    MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}" \
    DEEPSPEED="${DEEPSPEED:-zero2}" \
    bash "$SCRIPT_DIR/run_grpo_opsd.sh" \
    2>&1 | tee "$LOG_DIR/train.log"

echo "completed_at=$(date -Is)" >> "$STATUS"
echo "completed=1" >> "$STATUS"
echo "[action-scoped-opsd] done output=$OUTPUT_DIR"
