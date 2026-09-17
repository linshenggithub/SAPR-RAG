#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# E19 formal run: Hybrid Action-Causal GRPO (G2).
# Answer turns receive independently normalized answer F1. Query turns receive
# 0.5 * answer F1 advantage + 0.5 * per-turn marginal evidence advantage.
exec env \
    INIT_ADAPTER="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft_canonical_fp16/checkpoint-4150" \
    ENABLE_OPSD=false \
    ENABLE_REWARD=true \
    OPD_USE_GRPO_ADVANTAGE=true \
    ADVANTAGE_MODE=action_causal \
    ACTION_QUERY_OUTCOME_COEF="${ACTION_QUERY_OUTCOME_COEF:-0.5}" \
    ACTION_CREDIT_MODE=query_evidence \
    ACTION_CREDIT_COEF="${ACTION_CREDIT_COEF:-0.5}" \
    ACTION_CREDIT_SCALE=group_turn \
    ACTION_CREDIT_INFOS_KEY=query_evidence_gain \
    ACTION_CREDIT_GATE=all \
    TEACHER_ACTION_SCOPE=all \
    ENABLE_TRUNCATION_REWARD=false \
    ENABLE_EVIDENCE_AGENT=true \
    MAX_STEPS="${MAX_STEPS:-1000}" \
    MAX_COMPLETION_LENGTH=4096 \
    PER_DEVICE_BATCH_SIZE=2 \
    GRADIENT_ACCUMULATION_STEPS=4 \
    STEPS_PER_GENERATION=8 \
    NUM_GENERATIONS=8 \
    SAVE_STEPS="${SAVE_STEPS:-250}" \
    SAVE_TOTAL_LIMIT=4 \
    DATASET="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_musique_train_grpo_noteacher.jsonl" \
    RUN_NAME="action_causal_g2_a${ACTION_QUERY_OUTCOME_COEF:-0.5}_l${ACTION_CREDIT_COEF:-0.5}_s${MAX_STEPS:-1000}_20260908" \
    ROLLOUT_PORT=8030 \
    VLLM_GROUP_PORT=21240 \
    ROLLOUT_GPU=7 \
    ROLLOUT_PORT2=8031 \
    VLLM_GROUP_PORT2=21242 \
    ROLLOUT_GPU2=1 \
    TRAIN_DEVICES=2,3,4,5,6 \
    NPROC_PER_NODE=5 \
    bash "$SCRIPT_DIR/launch_action_scoped_opsd_worker.sh"
