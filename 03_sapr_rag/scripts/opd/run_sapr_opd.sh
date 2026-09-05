#!/usr/bin/env bash
# Train the 7B student with a state-aligned external teacher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SWIFT_ROOT="${SWIFT_ROOT:-$(cd "$PROJ_ROOT/../ms-swift" && pwd)}"
GRPO_DIR="$PROJ_ROOT/03_sapr_rag/scripts/grpo"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

BASE_MODEL="${BASE_MODEL:-$PROJ_ROOT/03_sapr_rag/models/Qwen2.5-7B-Instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft/checkpoint-1650}"
DATASET="${DATASET:-$PROJ_ROOT/data/grpo/hotpotqa_2wiki_musique_train_opd.jsonl}"
RUN_NAME="${RUN_NAME:-sapr_opd_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/opd/$RUN_NAME}"
TEACHER_URL="${TEACHER_URL:-http://127.0.0.1:8040}"
ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8030}"
VLLM_GROUP_PORT="${VLLM_GROUP_PORT:-21240}"
TRAIN_DEVICES="${TRAIN_DEVICES:-2,3,4,5,6}"
NPROC_PER_NODE="${NPROC_PER_NODE:-5}"

OPD_MODE="${OPD_MODE:-pure}"
TEACHER_KL_COEF="${TEACHER_KL_COEF:-0.01}"
TEACHER_SEQUENCE_GATE="${TEACHER_SEQUENCE_GATE:-failed_em}"
TEACHER_SEQUENCE_GATE_THRESHOLD="${TEACHER_SEQUENCE_GATE_THRESHOLD:-1.0}"
MAX_STEPS="${MAX_STEPS:-1000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
PADDING_FREE="${PADDING_FREE:-false}"
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
REFERENCE_KL_COEF="${REFERENCE_KL_COEF:-0.04}"
DRY_RUN="${DRY_RUN:-false}"

case "$OPD_MODE" in
    pure) USE_GRPO_ADVANTAGE=false ;;
    hybrid) USE_GRPO_ADVANTAGE=true ;;
    *) echo "[run_sapr_opd] ERROR: OPD_MODE must be pure or hybrid" >&2; exit 2 ;;
esac

for path in "$SWIFT_ROOT" "$BASE_MODEL" "$SFT_ADAPTER"; do
    [[ -e "$path" ]] || { echo "[run_sapr_opd] ERROR: missing path: $path" >&2; exit 2; }
done
[[ -f "$DATASET" ]] || { echo "[run_sapr_opd] ERROR: missing dataset: $DATASET" >&2; exit 2; }

python - "$DATASET" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    row = json.loads(next(line for line in handle if line.strip()))
leaked = sorted(key for key in row if key.startswith("teacher_"))
if leaked:
    raise SystemExit(f"[run_sapr_opd] privileged teacher fields are forbidden: {leaked}")
PY

if [[ "$DRY_RUN" != "true" ]]; then
    curl -fsS --max-time 10 "$TEACHER_URL/health/" >/dev/null || {
        echo "[run_sapr_opd] ERROR: teacher is not ready: $TEACHER_URL" >&2
        exit 3
    }
    curl -fsS --max-time 10 "http://${ROLLOUT_HOST}:${ROLLOUT_PORT}/health/" >/dev/null || {
        echo "[run_sapr_opd] ERROR: rollout is not ready: ${ROLLOUT_HOST}:${ROLLOUT_PORT}" >&2
        exit 3
    }
fi

mkdir -p "$OUTPUT_DIR"
cd "$SWIFT_ROOT"

CMD=(
    swift rlhf
    --rlhf_type grpo
    --model "$BASE_MODEL"
    --adapters "$SFT_ADAPTER"
    --tuner_type lora
    --external_plugins "$GRPO_DIR/plugin.py"
    --reward_funcs sapr_em sapr_f1 sapr_relevance sapr_format
    --reward_weights 0.0 1.0 0.2 0.05
    --teacher_model_server "$TEACHER_URL"
    --teacher_kl_coef "$TEACHER_KL_COEF"
    --teacher_action_scope all
    --teacher_sequence_gate "$TEACHER_SEQUENCE_GATE"
    --teacher_sequence_gate_threshold "$TEACHER_SEQUENCE_GATE_THRESHOLD"
    --opd_use_grpo_advantage "$USE_GRPO_ADVANTAGE"
    --use_vllm true
    --vllm_mode server
    --vllm_server_host "$ROLLOUT_HOST"
    --vllm_server_port "$ROLLOUT_PORT"
    --vllm_server_group_port "$VLLM_GROUP_PORT"
    --vllm_server_pass_dataset true
    --torch_dtype bfloat16
    --dataset "$DATASET"
    --split_dataset_ratio 0
    --max_length "$MAX_LENGTH"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --num_train_epochs 1
    --max_steps "$MAX_STEPS"
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --steps_per_generation "$STEPS_PER_GENERATION"
    --num_generations "$NUM_GENERATIONS"
    --learning_rate "$LEARNING_RATE"
    --temperature 1.0
    --beta "$REFERENCE_KL_COEF"
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --padding_free "$PADDING_FREE"
    --deepspeed zero2
    --save_strategy "$SAVE_STRATEGY"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --save_steps "$SAVE_STEPS"
    --save_only_model true
    --logging_steps 1
    --warmup_ratio 0.05
    --dataloader_num_workers 4
    --dataset_num_proc 4
    --output_dir "$OUTPUT_DIR"
    --log_completions true
    --num_iterations 1
    --report_to tensorboard
)

echo "[run_sapr_opd] mode=$OPD_MODE dataset=$DATASET"
echo "[run_sapr_opd] teacher=$TEACHER_URL gate=$TEACHER_SEQUENCE_GATE beta=$TEACHER_KL_COEF"
echo "[run_sapr_opd] train_devices=$TRAIN_DEVICES rollout=${ROLLOUT_HOST}:${ROLLOUT_PORT}"
echo "[run_sapr_opd] batch=$PER_DEVICE_BATCH_SIZE gas=$GRADIENT_ACCUMULATION_STEPS spg=$STEPS_PER_GENERATION gc=$GRADIENT_CHECKPOINTING padding_free=$PADDING_FREE"
echo "[run_sapr_opd] output=$OUTPUT_DIR"

if [[ "$DRY_RUN" == "true" ]]; then
    printf '[run_sapr_opd] DRY_RUN command: CUDA_VISIBLE_DEVICES=%q NPROC_PER_NODE=%q' \
        "$TRAIN_DEVICES" "$NPROC_PER_NODE"
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

env CUDA_VISIBLE_DEVICES="$TRAIN_DEVICES" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    "${CMD[@]}"
