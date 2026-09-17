#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# E18 pilot：Zero-Outcome Rescue GRPO。
#
# 相对 B（GRPO-only）仅在 sequence-level GRPO advantage 整组为 0 时启用
# query evidence credit；标准 GRPO 已能排序的组完全不受影响。相对 E17，
# gate 从 all 改为 zero_outcome，并把 coef 从 0.2 提到 0.5，使救援信号达到
# 可见尺度但仍低于常规标准化 advantage。
exec env \
    INIT_ADAPTER="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft_canonical_fp16/checkpoint-4150" \
    ENABLE_OPSD=false \
    ENABLE_REWARD=true \
    OPD_USE_GRPO_ADVANTAGE=true \
    ACTION_CREDIT_MODE=query_evidence \
    ACTION_CREDIT_COEF="${ACTION_CREDIT_COEF:-0.5}" \
    ACTION_CREDIT_SCALE=group_turn \
    ACTION_CREDIT_INFOS_KEY=query_evidence_gain \
    ACTION_CREDIT_GATE=zero_outcome \
    TEACHER_ACTION_SCOPE=multi \
    ENABLE_TRUNCATION_REWARD=false \
    ENABLE_EVIDENCE_AGENT=true \
    MAX_STEPS="${MAX_STEPS:-250}" \
    MAX_COMPLETION_LENGTH=4096 \
    PER_DEVICE_BATCH_SIZE=2 \
    GRADIENT_ACCUMULATION_STEPS=4 \
    STEPS_PER_GENERATION=8 \
    NUM_GENERATIONS=8 \
    SAVE_STEPS=125 \
    SAVE_TOTAL_LIMIT=4 \
    DATASET="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_musique_train_grpo_noteacher.jsonl" \
    RUN_NAME="zero_outcome_credit_pilot_coef${ACTION_CREDIT_COEF:-0.5}_s${MAX_STEPS:-250}_20260908" \
    ROLLOUT_PORT=8030 \
    VLLM_GROUP_PORT=21240 \
    ROLLOUT_GPU=7 \
    ROLLOUT_PORT2=8031 \
    VLLM_GROUP_PORT2=21242 \
    ROLLOUT_GPU2=1 \
    TRAIN_DEVICES=2,3,4,5,6 \
    NPROC_PER_NODE=5 \
    bash "$SCRIPT_DIR/launch_action_scoped_opsd_worker.sh"
