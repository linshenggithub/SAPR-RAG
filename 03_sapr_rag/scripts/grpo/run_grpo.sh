#!/usr/bin/env bash
# SAPR-RAG GRPO 训练（server 模式，8 卡）。三进程：检索 daemon / swift rollout / swift rlhf。
#
# 前置：
#   1) 数据已构建：python build_grpo_dataset.py --corpus .../wiki18_extended.jsonl
#   2) 三进程分别在各自终端启动（本脚本默认只启动训练进程 C；
#      daemon 与 rollout 见下方注释，建议各开一个终端，便于看日志）。
#
# 卡分配（总 8）：daemon=GPU7  rollout=GPU6  train=GPU0-5
set -euo pipefail

PROJ_ROOT=/mlx_devbox/users/mayi.summer/playground/SAPR-RAG
SWIFT_ROOT=/mlx_devbox/users/mayi.summer/playground/ms-swift
SCRIPT_DIR="$PROJ_ROOT/03_sapr_rag/scripts/grpo"

BASE_MODEL="$PROJ_ROOT/03_sapr_rag/models/Qwen2.5-7B-Instruct"
SFT_LORA="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft/checkpoint-1650"
DATASET="${DATASET:-$PROJ_ROOT/data/grpo/hotpotqa_train.jsonl}"
PLUGIN="$SCRIPT_DIR/plugin.py"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/grpo}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
# DEEPSPEED=zero2 (默认) | none (首跑 sanity 可设 none 跳过，需 deepspeed 未安装时用)
DEEPSPEED="${DEEPSPEED:-zero2}"
DS_ARG=()
[ "$DEEPSPEED" != "none" ] && DS_ARG=(--deepspeed "$DEEPSPEED")
RESUME_ARG=()
[ -n "$RESUME_FROM_CHECKPOINT" ] && RESUME_ARG=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ───────────────────────────────────────────────────────────────
# 进程 A：检索 daemon（在单独终端执行）
#   GPU=7 PORT=8100 bash run_retrieval_daemon.sh
#
# 进程 B：swift rollout async server（在单独终端执行，或直接用 run_rollout.sh）
#   CUDA_VISIBLE_DEVICES=6 \
#   swift rollout \
#       --model "$BASE_MODEL" \
#       --adapters "$SFT_LORA" \
#       --vllm_use_async_engine true \
#       --multi_turn_scheduler sapr_rag_scheduler \
#       --external_plugins "$PLUGIN" \
#       --max_turns 6 \
#       --vllm_max_model_len 8192 \
#       --vllm_gpu_memory_utilization 0.85 \
#       --host 127.0.0.1 \
#       --port 8000
#   注意：rollout 自身监听用 --host/--port；若 8000 被占，swift 会自动改端口，
#         训练端 --vllm_server_port 须以 rollout 实际打印的端口为准。
# ───────────────────────────────────────────────────────────────

# 进程 C：swift rlhf grpo 训练（本脚本主体）
cd "$SWIFT_ROOT"

echo "[run_grpo] dataset=$DATASET"
echo "[run_grpo] output_dir=$OUTPUT_DIR"
echo "[run_grpo] resume_from_checkpoint=${RESUME_FROM_CHECKPOINT:-<none>}"
echo "[run_grpo] vllm=${VLLM_HOST}:${VLLM_PORT} deepspeed=$DEEPSPEED"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model "$BASE_MODEL" \
    --adapters "$SFT_LORA" \
    --tuner_type lora \
    --external_plugins "$PLUGIN" \
    --reward_funcs sapr_f1 sapr_relevance sapr_format \
    --reward_weights 1.0 0.2 0.05 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host "$VLLM_HOST" \
    --vllm_server_port "$VLLM_PORT" \
    --vllm_server_pass_dataset true \
    --torch_dtype bfloat16 \
    --dataset "$DATASET" \
    --split_dataset_ratio 0 \
    --max_completion_length 8192 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --steps_per_generation 8 \
    --num_generations 8 \
    --learning_rate 1e-6 \
    --temperature 1.0 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --save_total_limit 60 \
    --save_steps 25 \
    --save_only_model true \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    "${DS_ARG[@]}" \
    "${RESUME_ARG[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --log_completions true \
    --num_iterations 1 \
    --report_to tensorboard \
    2>&1 | tee "$LOG_DIR/grpo_train.log"
