#!/bin/bash
# Launch script for Evidence-only debug_result experiment on rag-5090
# Usage: bash launch_evidence_debug.sh
set -e

export CUDA_VISIBLE_DEVICES=0
cd /home/mayi/ReasonRAG

eval "$(/home/mayi/miniconda3/bin/conda shell.bash hook)"
conda activate reasonrag

LOG_DIR="/home/mayi/RAG/agentic-rag-process-optimization/04_experiments/logs/20260529_evidence_debug_30samples_v2"
mkdir -p "$LOG_DIR"

python run_evidence_debug.py 2>&1 | tee "$LOG_DIR/run.log"
