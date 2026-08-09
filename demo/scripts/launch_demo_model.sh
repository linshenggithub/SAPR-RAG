#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source config/env_3090.sh

if [[ -f demo/.env ]]; then
  set -a
  source demo/.env
  set +a
fi

CONDA_ROOT="$(dirname "$(dirname "$SAPR_CONDA_BIN")")"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ROOT}/envs/reasonrag/bin/python}"
BASE_MODEL_NAME="${SAPR_DEMO_BASE_MODEL_NAME:-sapr-base}"
MODEL_NAME="${SAPR_DEMO_MODEL_NAME:-sapr-sft}"
MODEL_GPU="${SAPR_DEMO_MODEL_GPU:-0}"
LORA_PATH="${SAPR_DEMO_LORA_PATH:-03_sapr_rag/models/sapr-rag-sft-lora/checkpoint-1650}"

if [[ "$LORA_PATH" != /* ]]; then
  LORA_PATH="$REPO_ROOT/$LORA_PATH"
fi

if [[ ! -d "$SAPR_QWEN_BASE_MODEL_PATH" ]]; then
  echo "Base model not found: $SAPR_QWEN_BASE_MODEL_PATH" >&2
  exit 2
fi
if [[ ! -d "$LORA_PATH" ]]; then
  echo "LoRA adapter not found: $LORA_PATH" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$MODEL_GPU"
exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8001 \
  --model "$SAPR_QWEN_BASE_MODEL_PATH" \
  --served-model-name "$BASE_MODEL_NAME" \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.82 \
  --max-num-seqs 2 \
  --enforce-eager \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules "$MODEL_NAME=$LORA_PATH"
