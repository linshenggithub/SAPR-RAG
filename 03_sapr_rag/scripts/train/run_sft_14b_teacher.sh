#!/usr/bin/env bash
# Short protocol SFT for the external 14B OPD teacher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/sft_14b_teacher_lora.yaml}"
TRAIN_DEVICES="${TRAIN_DEVICES:-1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
RUN_NAME="${RUN_NAME:-sft_14b_teacher_s300_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$PROJ_ROOT/03_sapr_rag/scripts/train/logs/$RUN_NAME}"

mkdir -p "$LOG_DIR"
export WANDB_MODE=offline
export WANDB_PROJECT=sapr-rag
export WANDB_LOG_MODEL=false
export WANDB_DIR="$PROJ_ROOT/03_sapr_rag/saves/wandb"
export TOKENIZERS_PARALLELISM=false
# The shared ms-swift environment pins datasets 4.8.4. LLaMA-Factory's
# declared upper bound is conservative; keep the shared environment intact.
export DISABLE_VERSION_CHECK=1

echo "run_name=$RUN_NAME" >"$LOG_DIR/status.txt"
echo "train_devices=$TRAIN_DEVICES" >>"$LOG_DIR/status.txt"
echo "config=$CONFIG" >>"$LOG_DIR/status.txt"
echo "started_at=$(date -Is)" >>"$LOG_DIR/status.txt"

env CUDA_VISIBLE_DEVICES="$TRAIN_DEVICES" \
    FORCE_TORCHRUN=1 \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    llamafactory-cli train "$CONFIG" \
    2>&1 | tee "$LOG_DIR/train.log"

echo "completed_at=$(date -Is)" >>"$LOG_DIR/status.txt"
echo "completed=1" >>"$LOG_DIR/status.txt"
