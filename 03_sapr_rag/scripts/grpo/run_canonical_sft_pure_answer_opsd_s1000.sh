#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"
RUN_NAME="${RUN_NAME:-pure_answer_opsd_canonical_sft_q000_a003_3src_s1000_20260923}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8030}"
VLLM_GROUP_PORT="${VLLM_GROUP_PORT:-21241}"
ROLLOUT_GPU="${ROLLOUT_GPU:-7}"
TRAIN_DEVICES="${TRAIN_DEVICES:-2,3,4,5,6}"
NPROC_PER_NODE="${NPROC_PER_NODE:-5}"

# Pure Answer-only OPSD: no task reward and no outcome-level GRPO advantage.
exec env \
    INIT_ADAPTER="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft_canonical_fp16/checkpoint-4150" \
    ENABLE_OPSD=true \
    ENABLE_REWARD=false \
    OPD_USE_GRPO_ADVANTAGE=false \
    TEACHER_ACTION_SCOPE=multi \
    TEACHER_QUERY_KL_COEF=0.0 \
    TEACHER_EVIDENCE_KL_COEF=0.0 \
    TEACHER_ANSWER_KL_COEF=0.03 \
    ENABLE_TRUNCATION_REWARD=false \
    ACTION_CREDIT_MODE=off \
    ACTION_CREDIT_COEF=0.0 \
    ADVANTAGE_MODE=sequence \
    DYNAMIC_SAMPLE=false \
    OVERLONG_FILTER=false \
    LOSS_TYPE=grpo \
    ENABLE_EVIDENCE_AGENT=true \
    MAX_STEPS="$MAX_STEPS" \
    MAX_LENGTH=2048 \
    MAX_COMPLETION_LENGTH=4096 \
    PER_DEVICE_BATCH_SIZE=2 \
    GRADIENT_ACCUMULATION_STEPS=4 \
    STEPS_PER_GENERATION=8 \
    NUM_GENERATIONS=8 \
    SAVE_STEPS="$SAVE_STEPS" \
    SAVE_TOTAL_LIMIT="$SAVE_TOTAL_LIMIT" \
    DATASET="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_musique_train_multi_opsd.jsonl" \
    RUN_NAME="$RUN_NAME" \
    ROLLOUT_PORT="$ROLLOUT_PORT" \
    VLLM_GROUP_PORT="$VLLM_GROUP_PORT" \
    ROLLOUT_GPU="$ROLLOUT_GPU" \
    TRAIN_DEVICES="$TRAIN_DEVICES" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    bash "$SCRIPT_DIR/launch_action_scoped_opsd_worker.sh"
