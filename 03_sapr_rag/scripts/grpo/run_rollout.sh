#!/usr/bin/env bash
# 启动 swift rollout async server（GRPO 的轨迹生成端，方案 A 多轮调度）。
# 在单独终端执行；待打印监听 :8000 就绪后再启动 run_grpo.sh。
#
# 用法：
#   GPU=6 PORT=8000 bash run_rollout.sh
set -euo pipefail

PROJ_ROOT=/mlx_devbox/users/mayi.summer/playground/SAPR-RAG
SWIFT_ROOT=/mlx_devbox/users/mayi.summer/playground/ms-swift
SCRIPT_DIR="$PROJ_ROOT/03_sapr_rag/scripts/grpo"

BASE_MODEL="$PROJ_ROOT/03_sapr_rag/models/Qwen2.5-7B-Instruct"
SFT_LORA="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft/checkpoint-1650"
PLUGIN="$SCRIPT_DIR/plugin.py"

GPU="${GPU:-6}"
PORT="${PORT:-8000}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$SWIFT_ROOT"

echo "[run_rollout] GPU=$GPU port=$PORT  log -> $LOG_DIR/rollout.log"

CUDA_VISIBLE_DEVICES="$GPU" \
swift rollout \
    --model "$BASE_MODEL" \
    --adapters "$SFT_LORA" \
    --vllm_use_async_engine true \
    --multi_turn_scheduler sapr_rag_scheduler \
    --external_plugins "$PLUGIN" \
    --max_turns 6 \
    --vllm_max_model_len 8192 \
    --vllm_gpu_memory_utilization 0.85 \
    --host 127.0.0.1 \
    --port "$PORT" \
    2>&1 | tee "$LOG_DIR/rollout.log"
