#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source config/env_3090.sh

CONDA_ROOT="$(dirname "$(dirname "$SAPR_CONDA_BIN")")"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ROOT}/envs/reasonrag/bin/python}"

for path in "$SAPR_BGE_MODEL_PATH" "$SAPR_BGE_INDEX_PATH" "$SAPR_WIKI_CORPUS_PATH"; do
  if [[ ! -e "$path" ]]; then
    echo "Required retrieval asset not found: $path" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES=""
exec "$PYTHON_BIN" 03_sapr_rag/scripts/grpo/retrieval_daemon.py \
  --host 127.0.0.1 \
  --port 8100 \
  --device cpu \
  --faiss_device cpu \
  --bge_path "$SAPR_BGE_MODEL_PATH" \
  --index_path "$SAPR_BGE_INDEX_PATH" \
  --corpus_path "$SAPR_WIKI_CORPUS_PATH"
