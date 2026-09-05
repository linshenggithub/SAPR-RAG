#!/usr/bin/env bash
# Two-step external-teacher OPD smoke on a balanced 24-row dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SOURCE="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_musique_train_multi_opsd.jsonl"
DATASET="${DATASET:-$PROJ_ROOT/data/grpo/opd_smoke_8x3.jsonl}"
RUN_NAME="${RUN_NAME:-opd_14b_failed_em_smoke_$(date +%Y%m%d_%H%M%S)}"
SMOKE_NUM_GENERATIONS="${NUM_GENERATIONS:-5}"

python "$SCRIPT_DIR/build_opd_dataset.py" \
    --input "$SOURCE" \
    --output "$DATASET" \
    --per-source-limit 8 \
    --seed 42

env \
    WORKER_ID="${WORKER_ID:-4220660}" \
    RUN_NAME="$RUN_NAME" \
    DATASET="$DATASET" \
    OPD_MODE=pure \
    TEACHER_SEQUENCE_GATE=failed_em \
    TEACHER_KL_COEF=0.01 \
    MAX_STEPS=2 \
    SAVE_STEPS=1 \
    SAVE_TOTAL_LIMIT=2 \
    PER_DEVICE_BATCH_SIZE=1 \
    GRADIENT_ACCUMULATION_STEPS=1 \
    STEPS_PER_GENERATION=1 \
    NUM_GENERATIONS="$SMOKE_NUM_GENERATIONS" \
    MAX_COMPLETION_LENGTH=2048 \
    "$@" \
    bash "$SCRIPT_DIR/launch_sapr_opd_worker.sh"
