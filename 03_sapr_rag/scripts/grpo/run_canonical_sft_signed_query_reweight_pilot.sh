#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# E20 / G3 pilot: preserve the standard composite GRPO advantage everywhere,
# then reweight only Query tokens by bounded, outcome-signed evidence progress.
exec env \
    INIT_ADAPTER="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft_canonical_fp16/checkpoint-4150" \
    ENABLE_OPSD=false \
    ENABLE_REWARD=true \
    OPD_USE_GRPO_ADVANTAGE=true \
    ADVANTAGE_MODE=signed_query_reweight \
    ACTION_CREDIT_MODE=query_evidence \
    ACTION_CREDIT_COEF="${ACTION_CREDIT_COEF:-0.25}" \
    ACTION_CREDIT_CLIP="${ACTION_CREDIT_CLIP:-2.0}" \
    ACTION_CREDIT_SCALE=group_turn \
    ACTION_CREDIT_INFOS_KEY=query_evidence_gain \
    ACTION_CREDIT_GATE=all \
    TEACHER_ACTION_SCOPE=all \
    ENABLE_TRUNCATION_REWARD=false \
    ENABLE_EVIDENCE_AGENT=true \
    MAX_STEPS="${MAX_STEPS:-250}" \
    MAX_COMPLETION_LENGTH=4096 \
    PER_DEVICE_BATCH_SIZE=2 \
    GRADIENT_ACCUMULATION_STEPS=4 \
    STEPS_PER_GENERATION=8 \
    NUM_GENERATIONS=8 \
    SAVE_STEPS=125 \
    SAVE_TOTAL_LIMIT=2 \
    DATASET="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_musique_train_grpo_noteacher.jsonl" \
    RUN_NAME="signed_query_reweight_l${ACTION_CREDIT_COEF:-0.25}_c${ACTION_CREDIT_CLIP:-2.0}_s${MAX_STEPS:-250}_20260908" \
    ROLLOUT_PORT=8030 \
    VLLM_GROUP_PORT=21240 \
    ROLLOUT_GPU=7 \
    ROLLOUT_PORT2=8031 \
    VLLM_GROUP_PORT2=21242 \
    ROLLOUT_GPU2=1 \
    TRAIN_DEVICES=2,3,4,5,6 \
    NPROC_PER_NODE=5 \
    bash "$SCRIPT_DIR/launch_action_scoped_opsd_worker.sh"
