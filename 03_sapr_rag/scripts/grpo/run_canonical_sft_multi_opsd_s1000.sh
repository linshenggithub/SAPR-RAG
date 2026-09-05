#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# E14 canonical-answer SFT -> matched Query/Answer action-scoped OPSD.
# This reproduces the effective E12 objective while changing only the SFT start.
exec env \
    INIT_ADAPTER="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft_canonical_fp16/checkpoint-4150" \
    TEACHER_ACTION_SCOPE=multi \
    TEACHER_QUERY_KL_COEF=0.01 \
    TEACHER_EVIDENCE_KL_COEF=0.0 \
    TEACHER_ANSWER_KL_COEF=0.03 \
    ENABLE_TRUNCATION_REWARD=false \
    ENABLE_EVIDENCE_AGENT=true \
    MAX_STEPS=1000 \
    MAX_COMPLETION_LENGTH=4096 \
    PER_DEVICE_BATCH_SIZE=2 \
    GRADIENT_ACCUMULATION_STEPS=4 \
    STEPS_PER_GENERATION=8 \
    NUM_GENERATIONS=8 \
    SAVE_STEPS=250 \
    SAVE_TOTAL_LIMIT=8 \
    DATASET="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_musique_train_multi_opsd.jsonl" \
    RUN_NAME=opsd_canonical_sft_q001_a003_3src_s1000_20260905 \
    ROLLOUT_PORT=8030 \
    VLLM_GROUP_PORT=21240 \
    ROLLOUT_GPU=7 \
    TRAIN_DEVICES=2,3,4,5,6 \
    NPROC_PER_NODE=5 \
    bash "$SCRIPT_DIR/launch_action_scoped_opsd_worker.sh"
