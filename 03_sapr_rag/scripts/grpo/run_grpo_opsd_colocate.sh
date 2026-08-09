#!/usr/bin/env bash
# SAPR-RAG dynamic OPSD training with colocated vLLM rollout.
#
# This entrypoint is intended for the full OPSD run after the server-mode
# pipeline has been validated. It assumes retrieval is provided by a persistent
# HTTP daemon and keeps all paths relative to the SAPR-RAG checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SWIFT_ROOT="${SWIFT_ROOT:-$(cd "$PROJ_ROOT/../ms-swift" 2>/dev/null && pwd || true)}"

[ -n "$SWIFT_ROOT" ] && [ -d "$SWIFT_ROOT" ] || {
    echo "[run_grpo_opsd_colocate] ERROR: SWIFT_ROOT not found. Set SWIFT_ROOT to the ms-swift checkout." >&2
    exit 2
}

BASE_MODEL="${BASE_MODEL:-$PROJ_ROOT/03_sapr_rag/models/Qwen2.5-7B-Instruct}"
SFT_LORA="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft/checkpoint-1650"
SFT_DPO_LORA="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft_dpo/checkpoint-395"
TUNER_TYPE="${TUNER_TYPE:-lora}"
MODEL_TYPE="${MODEL_TYPE:-}"
TEMPLATE_TYPE="${TEMPLATE_TYPE:-}"
INIT_ADAPTER="${INIT_ADAPTER:-sft_dpo}"
case "$INIT_ADAPTER" in
    none|"") RESOLVED_INIT_ADAPTER="" ;;
    sft) RESOLVED_INIT_ADAPTER="$SFT_LORA" ;;
    sft_dpo) RESOLVED_INIT_ADAPTER="$SFT_DPO_LORA" ;;
    *) RESOLVED_INIT_ADAPTER="$INIT_ADAPTER" ;;
esac
ADAPTER_PATH="${ADAPTER_PATH:-$RESOLVED_INIT_ADAPTER}"
REF_MODEL="${REF_MODEL:-$BASE_MODEL}"
REF_MODEL_TYPE="${REF_MODEL_TYPE:-$MODEL_TYPE}"
DATASET="${DATASET:-$PROJ_ROOT/data/grpo/hotpotqa_2wiki_train_opsd.jsonl}"
PLUGIN="${PLUGIN:-$SCRIPT_DIR/plugin.py}"
RETRIEVAL_URL="${SAPR_RETRIEVAL_URL:-${RETRIEVAL_URL:-http://127.0.0.1:8100}}"

# Default layout leaves device 0 available for a persistent retrieval daemon.
TRAIN_DEVICES="${TRAIN_DEVICES:-1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(python - "$TRAIN_DEVICES" <<'PY'
import sys
print(len([x for x in sys.argv[1].split(',') if x.strip()]))
PY
)}"
MASTER_PORT="${MASTER_PORT:-$(python - <<'PY'
import socket
with socket.socket() as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
PY
)}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-7}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-1}"
ENABLE_OPSD="${ENABLE_OPSD:-true}"
TEACHER_KL_COEF="${TEACHER_KL_COEF:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BETA="${BETA:-0.04}"
DEEPSPEED="${DEEPSPEED:-none}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
MAX_TURNS="${MAX_TURNS:-6}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.20}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
LORA_RANK="${LORA_RANK:-16}"
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-30}"
MAX_STEPS="${MAX_STEPS:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
REWARD_FUNCS="${REWARD_FUNCS:-sapr_f1 sapr_relevance sapr_format}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0 0.2 0.05}"
REPEAT_QUERY_CAP="${REPEAT_QUERY_CAP:-3}"
DRY_RUN="${DRY_RUN:-false}"

RUN_NAME="${RUN_NAME:-opsd_colocate_effect_pbs${PER_DEVICE_BATCH_SIZE}_g${NUM_GENERATIONS}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_colocate_full/$RUN_NAME}"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

case "$TUNER_TYPE" in
    lora)
        REQUIRED_MODEL_PATHS=("$BASE_MODEL")
        [ -z "$ADAPTER_PATH" ] || REQUIRED_MODEL_PATHS+=("$ADAPTER_PATH")
        ;;
    full)
        REQUIRED_MODEL_PATHS=("$BASE_MODEL" "$REF_MODEL")
        if [ "$DEEPSPEED" = "none" ]; then
            echo "[run_grpo_opsd_colocate] ERROR: full training requires DEEPSPEED=zero3 or zero3_offload." >&2
            exit 2
        fi
        ;;
    *) echo "[run_grpo_opsd_colocate] ERROR: TUNER_TYPE must be lora or full, got: $TUNER_TYPE" >&2; exit 2 ;;
esac
for path in "${REQUIRED_MODEL_PATHS[@]}"; do
    [ -d "$path" ] || { echo "[run_grpo_opsd_colocate] ERROR: path not found: $path" >&2; exit 2; }
done
[ -f "$DATASET" ] || { echo "[run_grpo_opsd_colocate] ERROR: dataset not found: $DATASET" >&2; exit 2; }
[ -f "$PLUGIN" ] || { echo "[run_grpo_opsd_colocate] ERROR: plugin not found: $PLUGIN" >&2; exit 2; }

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
case "$ENABLE_OPSD" in
    true|false) ;;
    *) echo "[run_grpo_opsd_colocate] ERROR: ENABLE_OPSD must be true or false, got: $ENABLE_OPSD" >&2; exit 2 ;;
esac
if [ "$ENABLE_OPSD" != "$DATASET_HAS_TEACHER_PROMPT" ]; then
    echo "[run_grpo_opsd_colocate] ERROR: ENABLE_OPSD=$ENABLE_OPSD but dataset teacher_prompt=$DATASET_HAS_TEACHER_PROMPT" >&2
    exit 2
fi

GLOBAL_COMPLETION_BATCH=$((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE))
if [ $((GLOBAL_COMPLETION_BATCH % NUM_GENERATIONS)) -ne 0 ]; then
    echo "[run_grpo_opsd_colocate] ERROR: nproc*per_device_batch_size=$GLOBAL_COMPLETION_BATCH must be divisible by num_generations=$NUM_GENERATIONS." >&2
    exit 2
fi
PROMPTS_PER_STEP=$((GLOBAL_COMPLETION_BATCH / NUM_GENERATIONS))

read -r -a REWARD_FUNC_ARGS <<< "$REWARD_FUNCS"
read -r -a REWARD_WEIGHT_ARGS <<< "$REWARD_WEIGHTS"
if [ "${#REWARD_FUNC_ARGS[@]}" -eq 0 ] || [ "${#REWARD_FUNC_ARGS[@]}" -ne "${#REWARD_WEIGHT_ARGS[@]}" ]; then
    echo "[run_grpo_opsd_colocate] ERROR: reward func/weight count mismatch: funcs='$REWARD_FUNCS' weights='$REWARD_WEIGHTS'" >&2
    exit 2
fi

MAX_STEPS_ARG=()
[ -n "$MAX_STEPS" ] && MAX_STEPS_ARG=(--max_steps "$MAX_STEPS")
RESUME_ARG=()
[ -n "$RESUME_FROM_CHECKPOINT" ] && RESUME_ARG=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$SWIFT_ROOT"

echo "[run_grpo_opsd_colocate] dataset=$DATASET"
echo "[run_grpo_opsd_colocate] tuner_type=$TUNER_TYPE base_model=$BASE_MODEL ref_model=$REF_MODEL"
[ "$TUNER_TYPE" = "lora" ] && echo "[run_grpo_opsd_colocate] init_adapter=$INIT_ADAPTER resolved_adapter=$ADAPTER_PATH"
echo "[run_grpo_opsd_colocate] output_dir=$OUTPUT_DIR"
echo "[run_grpo_opsd_colocate] train_devices=$TRAIN_DEVICES nproc=$NPROC_PER_NODE master_port=$MASTER_PORT"
echo "[run_grpo_opsd_colocate] retrieval_url=$RETRIEVAL_URL"
echo "[run_grpo_opsd_colocate] per_device_batch_size=$PER_DEVICE_BATCH_SIZE num_generations=$NUM_GENERATIONS prompts_per_step=$PROMPTS_PER_STEP grad_accum=$GRADIENT_ACCUMULATION_STEPS"
echo "[run_grpo_opsd_colocate] opsd=$ENABLE_OPSD teacher_kl_coef=$([ "$ENABLE_OPSD" = "true" ] && echo "$TEACHER_KL_COEF" || echo 0) beta=$BETA lr=$LEARNING_RATE deepspeed=$DEEPSPEED"
echo "[run_grpo_opsd_colocate] reward_funcs=$REWARD_FUNCS reward_weights=$REWARD_WEIGHTS repeat_query_cap=$REPEAT_QUERY_CAP"

OPD_ARG=(--teacher_kl_coef 0)
[ "$ENABLE_OPSD" = "true" ] && OPD_ARG=(--teacher_kl_coef "$TEACHER_KL_COEF")

MODEL_ARGS=(--model "$BASE_MODEL" --tuner_type "$TUNER_TYPE")
VLLM_TUNER_ARGS=(--vllm_enable_lora false)
DEEPSPEED_ARGS=()
[ -n "$MODEL_TYPE" ] && MODEL_ARGS+=(--model_type "$MODEL_TYPE")
[ -n "$TEMPLATE_TYPE" ] && MODEL_ARGS+=(--template "$TEMPLATE_TYPE")
if [ "$TUNER_TYPE" = "lora" ]; then
    [ -z "$ADAPTER_PATH" ] || MODEL_ARGS+=(--adapters "$ADAPTER_PATH")
    MODEL_ARGS+=(--lora_rank "$LORA_RANK")
    VLLM_TUNER_ARGS=(--vllm_enable_lora true --vllm_max_lora_rank "$LORA_RANK")
else
    MODEL_ARGS+=(--ref_model "$REF_MODEL")
    [ -n "$REF_MODEL_TYPE" ] && MODEL_ARGS+=(--ref_model_type "$REF_MODEL_TYPE")
fi
[ "$DEEPSPEED" != "none" ] && DEEPSPEED_ARGS=(--deepspeed "$DEEPSPEED")

CMD=(
    swift rlhf
    --rlhf_type grpo
    "${MODEL_ARGS[@]}"
    --external_plugins "$PLUGIN"
    --reward_funcs "${REWARD_FUNC_ARGS[@]}"
    --reward_weights "${REWARD_WEIGHT_ARGS[@]}"
    --beta "$BETA"
    --use_vllm true
    --vllm_mode colocate
    "${VLLM_TUNER_ARGS[@]}"
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --vllm_tensor_parallel_size 1
    --vllm_max_model_len "$VLLM_MAX_MODEL_LEN"
    --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS"
    --sleep_level 0
    --multi_turn_scheduler sapr_rag_scheduler
    --max_turns "$MAX_TURNS"
    --torch_dtype bfloat16
    --dataset "$DATASET"
    --split_dataset_ratio 0
    --max_length "$MAX_LENGTH"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --num_train_epochs 1
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --steps_per_generation "$STEPS_PER_GENERATION"
    --num_generations "$NUM_GENERATIONS"
    --learning_rate "$LEARNING_RATE"
    --temperature 1.0
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --save_steps "$SAVE_STEPS"
    --save_only_model true
    --logging_steps 1
    --warmup_ratio 0.05
    --dataloader_num_workers 4
    --dataset_num_proc 4
    "${DEEPSPEED_ARGS[@]}"
    "${OPD_ARG[@]}"
    "${RESUME_ARG[@]}"
    "${MAX_STEPS_ARG[@]}"
    --output_dir "$OUTPUT_DIR"
    --log_completions true
    --num_iterations 1
    --report_to tensorboard
)

if [ "$DRY_RUN" = "true" ]; then
    printf '[run_grpo_opsd_colocate] DRY_RUN command: SAPR_RETRIEVAL_URL=%q CUDA_VISIBLE_DEVICES=%q NPROC_PER_NODE=%q MASTER_PORT=%q' \
        "$RETRIEVAL_URL" "$TRAIN_DEVICES" "$NPROC_PER_NODE" "$MASTER_PORT"
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

SAPR_RETRIEVAL_URL="$RETRIEVAL_URL" \
SAPR_MAX_TURNS="$MAX_TURNS" \
SAPR_REPEAT_QUERY_CAP="$REPEAT_QUERY_CAP" \
CUDA_VISIBLE_DEVICES="$TRAIN_DEVICES" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
MASTER_PORT="$MASTER_PORT" \
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
