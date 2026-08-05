#!/usr/bin/env bash
# SAPR-RAG dynamic OPSD / matched GRPO 训练入口。
# 这是增量脚本，不改动 baseline run_grpo.sh。
#
# 用法：
#   DEVICE_BACKEND=npu ENABLE_OPSD=true DATASET=... bash run_grpo_opsd.sh
#   DEVICE_BACKEND=npu ENABLE_OPSD=false DATASET=... bash run_grpo_opsd.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SWIFT_ROOT="${SWIFT_ROOT:-$(cd "$PROJ_ROOT/../ms-swift" 2>/dev/null && pwd || true)}"

[ -n "$SWIFT_ROOT" ] && [ -d "$SWIFT_ROOT" ] || {
    echo "[run_grpo_opsd] ERROR: SWIFT_ROOT not found. Set SWIFT_ROOT to the ms-swift checkout." >&2
    exit 2
}

BASE_MODEL="${BASE_MODEL:-$PROJ_ROOT/03_sapr_rag/models/Qwen2.5-7B-Instruct}"
SFT_LORA="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft/checkpoint-1650"
SFT_DPO_LORA="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft_dpo/checkpoint-395"
INIT_ADAPTER="${INIT_ADAPTER:-sft_dpo}"
case "$INIT_ADAPTER" in
    sft) RESOLVED_INIT_ADAPTER="$SFT_LORA" ;;
    sft_dpo) RESOLVED_INIT_ADAPTER="$SFT_DPO_LORA" ;;
    *) RESOLVED_INIT_ADAPTER="$INIT_ADAPTER" ;;
esac
ADAPTER_PATH="${ADAPTER_PATH:-$RESOLVED_INIT_ADAPTER}"
DATASET="${DATASET:-$PROJ_ROOT/data/grpo/hotpotqa_2wiki_train_pilot_opsd.jsonl}"
PLUGIN="$SCRIPT_DIR/plugin.py"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
# weight-sync NCCL 通信组端口；默认 51299。
# 必须避开：29500（torchrun --master-port 默认，训练进程组已占）、8000（vllm server）、8100（retrieval）。
VLLM_GROUP_PORT="${VLLM_GROUP_PORT:-51299}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
ENABLE_OPSD="${ENABLE_OPSD:-true}"
TEACHER_KL_COEF="${TEACHER_KL_COEF:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-}"
DRY_RUN="${DRY_RUN:-false}"
DEVICE_BACKEND="${DEVICE_BACKEND:-cuda}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
DEEPSPEED="${DEEPSPEED:-zero2}"

if [ "$DEVICE_BACKEND" = "npu" ] && [ -n "${ASCEND_VISIBLE_DEVICES:-}" ]; then
    DEFAULT_TRAIN_DEVICES="$(python - "$ASCEND_VISIBLE_DEVICES" <<'PY'
import sys
devices = [x for x in sys.argv[1].split(",") if x]
print(",".join(devices[:6] if len(devices) >= 6 else devices))
PY
)"
else
    DEFAULT_TRAIN_DEVICES="0,1,2,3,4,5"
fi
TRAIN_DEVICES="${TRAIN_DEVICES:-$DEFAULT_TRAIN_DEVICES}"

case "$ENABLE_OPSD" in
    true|false) ;;
    *) echo "[run_grpo_opsd] ERROR: ENABLE_OPSD must be true or false, got: $ENABLE_OPSD" >&2; exit 2 ;;
esac
case "$DEVICE_BACKEND" in
    cuda)
        VISIBLE_DEVICES_ENV="CUDA_VISIBLE_DEVICES"
        DEVICE_LABEL="GPU"
        ;;
    npu)
        VISIBLE_DEVICES_ENV="ASCEND_RT_VISIBLE_DEVICES"
        DEVICE_LABEL="NPU"
        ;;
    *)
        echo "[run_grpo_opsd] ERROR: DEVICE_BACKEND must be cuda or npu, got: $DEVICE_BACKEND" >&2
        exit 2
        ;;
esac

for path in "$BASE_MODEL" "$ADAPTER_PATH"; do
    [ -d "$path" ] || { echo "[run_grpo_opsd] ERROR: path not found: $path" >&2; exit 2; }
done
[ -f "$DATASET" ] || { echo "[run_grpo_opsd] ERROR: dataset not found: $DATASET" >&2; exit 2; }

DATASET_HAS_TEACHER_PROMPT="$(
    python - "$DATASET" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    for line in f:
        if line.strip():
            print("true" if json.loads(line).get("teacher_prompt") else "false")
            break
    else:
        raise SystemExit("dataset is empty")
PY
)"
if [ "$ENABLE_OPSD" != "$DATASET_HAS_TEACHER_PROMPT" ]; then
    echo "[run_grpo_opsd] ERROR: ENABLE_OPSD=$ENABLE_OPSD but dataset teacher_prompt=$DATASET_HAS_TEACHER_PROMPT" >&2
    exit 2
fi

DS_ARG=()
[ "$DEEPSPEED" != "none" ] && DS_ARG=(--deepspeed "$DEEPSPEED")
RESUME_ARG=()
[ -n "$RESUME_FROM_CHECKPOINT" ] && RESUME_ARG=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
MAX_STEPS_ARG=()
[ -n "$MAX_STEPS" ] && MAX_STEPS_ARG=(--max_steps "$MAX_STEPS")
OPD_ARG=(--teacher_kl_coef 0)
[ "$ENABLE_OPSD" = "true" ] && OPD_ARG=(--teacher_kl_coef "$TEACHER_KL_COEF")

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$SWIFT_ROOT"

echo "[run_grpo_opsd] dataset=$DATASET"
echo "[run_grpo_opsd] init_adapter=$INIT_ADAPTER resolved_adapter=$ADAPTER_PATH"
echo "[run_grpo_opsd] output_dir=$OUTPUT_DIR"
echo "[run_grpo_opsd] backend=$DEVICE_BACKEND visible_env=$VISIBLE_DEVICES_ENV train_devices=$TRAIN_DEVICES nproc=$NPROC_PER_NODE"
echo "[run_grpo_opsd] layout=train:${DEVICE_LABEL}${TRAIN_DEVICES}"
echo "[run_grpo_opsd] vllm_server=${VLLM_HOST}:${VLLM_PORT} group_port=${VLLM_GROUP_PORT}"
echo "[run_grpo_opsd] opsd=$ENABLE_OPSD teacher_kl_coef=$([ "$ENABLE_OPSD" = "true" ] && echo "$TEACHER_KL_COEF" || echo 0)"

CMD=(
    swift rlhf
    --rlhf_type grpo
    --model "$BASE_MODEL"
    --adapters "$ADAPTER_PATH"
    --tuner_type lora
    --external_plugins "$PLUGIN"
    --reward_funcs sapr_f1 sapr_relevance sapr_format
    --reward_weights 1.0 0.2 0.05
    --use_vllm true
    --vllm_mode server
    --vllm_server_host "$VLLM_HOST"
    --vllm_server_port "$VLLM_PORT"
    --vllm_server_group_port "$VLLM_GROUP_PORT"
    --vllm_server_pass_dataset true
    --torch_dtype bfloat16
    --dataset "$DATASET"
    --split_dataset_ratio 0
    --max_length "$MAX_LENGTH"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --num_train_epochs 1
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
    --gradient_accumulation_steps 4
    --steps_per_generation 8
    --num_generations 8
    --learning_rate 1e-6
    --temperature 1.0
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --save_total_limit 60
    --save_steps 25
    --save_only_model true
    --logging_steps 1
    --warmup_ratio 0.05
    --dataloader_num_workers 4
    --dataset_num_proc 4
    "${OPD_ARG[@]}"
    "${DS_ARG[@]}"
    "${RESUME_ARG[@]}"
    "${MAX_STEPS_ARG[@]}"
    --output_dir "$OUTPUT_DIR"
    --log_completions true
    --num_iterations 1
    --report_to tensorboard
)

if [ "$DRY_RUN" = "true" ]; then
    printf '[run_grpo_opsd] DRY_RUN command: %s=%q NPROC_PER_NODE=%q' "$VISIBLE_DEVICES_ENV" "$TRAIN_DEVICES" "$NPROC_PER_NODE"
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

env "$VISIBLE_DEVICES_ENV=$TRAIN_DEVICES" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
"${CMD[@]}" \
    2>&1 | tee "$LOG_DIR/grpo_opsd_train.log"
