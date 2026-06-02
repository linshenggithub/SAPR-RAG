#!/bin/bash
# Launch evidence decision point export for 30 examples
set -e

# 路径配置：默认值与 config/paths.py 保持一致，可通过 SAPR_* 环境变量覆盖
REASONRAG_ROOT="${SAPR_REASONRAG_ROOT:-${REASONRAG_ROOT:-/home/mayi/ReasonRAG}}"
CONDA_BIN="${SAPR_CONDA_BIN:-${CONDA_BIN:-/home/mayi/miniconda3/bin/conda}}"

# REPO_ROOT 由本脚本所在路径派生（即 SAPR-RAG 仓库根）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES=0
cd "${REASONRAG_ROOT}"

eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate reasonrag

LOG_DIR="${REPO_ROOT}/04_experiments/logs/20260529_evidence_decision_top10"
mkdir -p "$LOG_DIR"

python "${SCRIPT_DIR}/export_evidence_decision_points.py" --num_examples 30 2>&1 | tee "$LOG_DIR/run.log"
