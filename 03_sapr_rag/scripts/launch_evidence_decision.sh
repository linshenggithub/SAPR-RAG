#!/bin/bash
# Launch evidence decision point export for 30 examples
set -e

export CUDA_VISIBLE_DEVICES=0
cd /home/mayi/ReasonRAG

eval "$(/home/mayi/miniconda3/bin/conda shell.bash hook)"
conda activate reasonrag

LOG_DIR="/home/mayi/RAG/agentic-rag-process-optimization/04_experiments/logs/20260529_evidence_decision_top10"
mkdir -p "$LOG_DIR"

python export_evidence_decisions.py --num_examples 30 2>&1 | tee "$LOG_DIR/run.log"
