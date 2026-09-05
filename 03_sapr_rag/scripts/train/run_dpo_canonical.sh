#!/usr/bin/env bash
# 在 E14 canonical SFT (checkpoint-4150) 基础上做 DPO，使用 worker 空闲 GPU1-7。
# GPU0 上是 retrieval daemon（别的任务在用），DPO 不需要检索，故跳过 GPU0。
set -euo pipefail

REPO=/mlx_devbox/users/mayi.summer/playground/SAPR-RAG
CFG=$REPO/03_sapr_rag/scripts/train/dpo_canonical_lora.yaml
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR=$REPO/03_sapr_rag/scripts/train/logs/dpo_canonical_${TS}
mkdir -p "$LOGDIR"
cp "$CFG" "$LOGDIR/config.yaml"

GPUS=1,2,3,4,5,6,7
NPROC=7

export CUDA_VISIBLE_DEVICES=$GPUS
export DISABLE_VERSION_CHECK=1
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.183.06
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=$NPROC
export WANDB_DISABLED=true

cd "$REPO"
echo "[dpo-start] $(date -Iseconds) gpus=$GPUS nproc=$NPROC ckpt=sft_canonical_fp16/checkpoint-4150" | tee "$LOGDIR/run.log"

nohup llamafactory-cli train "$CFG" >> "$LOGDIR/train.log" 2>&1 &
echo $! > "$LOGDIR/pid"
echo "[dpo-pid] $(cat $LOGDIR/pid)" | tee -a "$LOGDIR/run.log"
echo "logdir=$LOGDIR"
