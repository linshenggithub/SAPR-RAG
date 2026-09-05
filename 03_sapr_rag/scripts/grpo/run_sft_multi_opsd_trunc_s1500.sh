#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Stability rerun from the original SFT adapter. Relative to the previous
# SFT->OPSD run, the only objective change is a -0.5 reward for trajectories
# truncated at the generation limit. Extra checkpoints improve early stopping.
exec env \
    INIT_ADAPTER=sft \
    TEACHER_ACTION_SCOPE=multi \
    TEACHER_QUERY_KL_COEF=0.01 \
    TEACHER_EVIDENCE_KL_COEF=0.0 \
    TEACHER_ANSWER_KL_COEF=0.03 \
    ENABLE_TRUNCATION_REWARD=true \
    TRUNCATION_REWARD_WEIGHT=0.5 \
    ENABLE_EVIDENCE_AGENT=true \
    MAX_STEPS=1500 \
    MAX_COMPLETION_LENGTH=4096 \
    PER_DEVICE_BATCH_SIZE=2 \
    GRADIENT_ACCUMULATION_STEPS=4 \
    STEPS_PER_GENERATION=8 \
    NUM_GENERATIONS=8 \
    SAVE_STEPS=250 \
    SAVE_TOTAL_LIMIT=8 \
    DATASET="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_musique_train_multi_opsd.jsonl" \
    RUN_NAME=opsd_sft_q001_a003_trunc05_3src_s1500_20260903 \
    ROLLOUT_PORT=8030 \
    VLLM_GROUP_PORT=21240 \
    ROLLOUT_GPU=7 \
    TRAIN_DEVICES=2,3,4,5,6 \
    NPROC_PER_NODE=5 \
    bash "$SCRIPT_DIR/launch_action_scoped_opsd_worker.sh"
