#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# wandb：默认离线模式，避免 ~/.netrc 不可写报错
# 想用 wandb 时改成 export WANDB_MODE=online + 跑前 wandb login
export WANDB_MODE=offline
export WANDB_PROJECT=sapr-rag
export WANDB_LOG_MODEL=false
export WANDB_DIR=/mlx_devbox/users/mayi.summer/playground/SAPR-RAG/03_sapr_rag/saves/wandb
mkdir -p "$WANDB_DIR"

export TOKENIZERS_PARALLELISM=false

FORCE_TORCHRUN=1 llamafactory-cli train sft_lora.yaml
