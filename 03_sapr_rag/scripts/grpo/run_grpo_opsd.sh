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
# DP rollout：VLLM_PORT / VLLM_GROUP_PORT / VLLM_HOST 支持空格分隔的多值（多个 rollout 副本）。
# 单值时行为与原来完全一致（数组只有一个元素）。ms-swift 的这三个参数均为 List 类型，
# 训练侧 VLLMClient 会把采样请求按副本数分片并行、并向每个副本各建一条 NCCL 通信组同步权重。
read -r -a VLLM_PORT_ARR <<< "$VLLM_PORT"
read -r -a VLLM_GROUP_PORT_ARR <<< "$VLLM_GROUP_PORT"
read -r -a VLLM_HOST_ARR <<< "$VLLM_HOST"
# host 只给一个但有多个 port 时，自动把 host 复制到与 port 等长。
if [ "${#VLLM_HOST_ARR[@]}" -eq 1 ] && [ "${#VLLM_PORT_ARR[@]}" -gt 1 ]; then
    _h="${VLLM_HOST_ARR[0]}"; VLLM_HOST_ARR=(); for _ in "${VLLM_PORT_ARR[@]}"; do VLLM_HOST_ARR+=("$_h"); done
fi
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
ENABLE_OPSD="${ENABLE_OPSD:-true}"
TEACHER_KL_COEF="${TEACHER_KL_COEF:-0.1}"
TEACHER_ACTION_SCOPE="${TEACHER_ACTION_SCOPE:-all}"
TEACHER_QUERY_KL_COEF="${TEACHER_QUERY_KL_COEF:-0.01}"
TEACHER_EVIDENCE_KL_COEF="${TEACHER_EVIDENCE_KL_COEF:-0.0}"
TEACHER_ANSWER_KL_COEF="${TEACHER_ANSWER_KL_COEF:-0.03}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-60}"
MAX_STEPS="${MAX_STEPS:-}"
ENABLE_TRUNCATION_REWARD="${ENABLE_TRUNCATION_REWARD:-false}"
TRUNCATION_REWARD_WEIGHT="${TRUNCATION_REWARD_WEIGHT:-0.5}"
# 纯 OPSD（无 RL reward）对照：ENABLE_REWARD=false 关闭全部任务 reward，
# OPD_USE_GRPO_ADVANTAGE=false 丢弃 GRPO 组内 advantage，只保留 teacher log-ratio 信号。
ENABLE_REWARD="${ENABLE_REWARD:-true}"
OPD_USE_GRPO_ADVANTAGE="${OPD_USE_GRPO_ADVANTAGE:-true}"
# Evidence-Attributed GRPO（动作级证据信用）：ACTION_CREDIT_MODE=query_evidence 打开后，
# 追加 sapr_query_evidence_gain（权重恒 0，仅产出逐轮向量），并向 swift 传 --action_credit_*。
ACTION_CREDIT_MODE="${ACTION_CREDIT_MODE:-off}"
ACTION_CREDIT_COEF="${ACTION_CREDIT_COEF:-0.0}"
ACTION_CREDIT_SCALE="${ACTION_CREDIT_SCALE:-group_turn}"
ACTION_CREDIT_INFOS_KEY="${ACTION_CREDIT_INFOS_KEY:-query_evidence_gain}"
ACTION_CREDIT_GATE="${ACTION_CREDIT_GATE:-all}"
ADVANTAGE_MODE="${ADVANTAGE_MODE:-sequence}"
ACTION_QUERY_OUTCOME_COEF="${ACTION_QUERY_OUTCOME_COEF:-0.25}"
ACTION_CREDIT_CLIP="${ACTION_CREDIT_CLIP:-2.0}"
DYNAMIC_SAMPLE="${DYNAMIC_SAMPLE:-false}"
MAX_RESAMPLE_TIMES="${MAX_RESAMPLE_TIMES:-3}"
OVERLONG_FILTER="${OVERLONG_FILTER:-false}"
LOSS_TYPE="${LOSS_TYPE:-grpo}"
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
case "$ENABLE_TRUNCATION_REWARD" in
    true|false) ;;
    *) echo "[run_grpo_opsd] ERROR: ENABLE_TRUNCATION_REWARD must be true or false, got: $ENABLE_TRUNCATION_REWARD" >&2; exit 2 ;;
esac
case "$ENABLE_REWARD" in
    true|false) ;;
    *) echo "[run_grpo_opsd] ERROR: ENABLE_REWARD must be true or false, got: $ENABLE_REWARD" >&2; exit 2 ;;
esac
case "$OPD_USE_GRPO_ADVANTAGE" in
    true|false) ;;
    *) echo "[run_grpo_opsd] ERROR: OPD_USE_GRPO_ADVANTAGE must be true or false, got: $OPD_USE_GRPO_ADVANTAGE" >&2; exit 2 ;;
esac
case "$DYNAMIC_SAMPLE" in
    true|false) ;;
    *) echo "[run_grpo_opsd] ERROR: DYNAMIC_SAMPLE must be true or false, got: $DYNAMIC_SAMPLE" >&2; exit 2 ;;
esac
case "$OVERLONG_FILTER" in
    true|false) ;;
    *) echo "[run_grpo_opsd] ERROR: OVERLONG_FILTER must be true or false, got: $OVERLONG_FILTER" >&2; exit 2 ;;
esac
if ! [[ "$MAX_RESAMPLE_TIMES" =~ ^[1-9][0-9]*$ ]]; then
    echo "[run_grpo_opsd] ERROR: MAX_RESAMPLE_TIMES must be a positive integer, got: $MAX_RESAMPLE_TIMES" >&2
    exit 2
fi
case "$LOSS_TYPE" in
    grpo|bnpo|dr_grpo|dapo|cispo|sapo|real|fipo) ;;
    *) echo "[run_grpo_opsd] ERROR: unsupported LOSS_TYPE=$LOSS_TYPE" >&2; exit 2 ;;
esac
case "$ACTION_CREDIT_MODE" in
    off|query_evidence) ;;
    *) echo "[run_grpo_opsd] ERROR: ACTION_CREDIT_MODE must be off or query_evidence, got: $ACTION_CREDIT_MODE" >&2; exit 2 ;;
esac
case "$ACTION_CREDIT_GATE" in
    all|zero_outcome) ;;
    *) echo "[run_grpo_opsd] ERROR: ACTION_CREDIT_GATE must be all or zero_outcome, got: $ACTION_CREDIT_GATE" >&2; exit 2 ;;
esac
case "$ADVANTAGE_MODE" in
    sequence|action_causal|signed_query_reweight|causal_return) ;;
    *) echo "[run_grpo_opsd] ERROR: unsupported ADVANTAGE_MODE=$ADVANTAGE_MODE" >&2; exit 2 ;;
esac
if [ "$ADVANTAGE_MODE" != "sequence" ] \
        && { [ "$ACTION_CREDIT_MODE" != "query_evidence" ] || [ "$ACTION_CREDIT_GATE" != "all" ]; }; then
    echo "[run_grpo_opsd] ERROR: $ADVANTAGE_MODE requires ACTION_CREDIT_MODE=query_evidence and ACTION_CREDIT_GATE=all" >&2
    exit 2
fi
if [ "$ACTION_CREDIT_MODE" != "off" ] && [ "$ENABLE_REWARD" = "false" ]; then
    echo "[run_grpo_opsd] ERROR: ACTION_CREDIT_MODE requires ENABLE_REWARD=true (additive on GRPO advantage)" >&2; exit 2
fi
if [ "$ENABLE_REWARD" = "false" ] && [ "$ENABLE_OPSD" != "true" ]; then
    echo "[run_grpo_opsd] ERROR: ENABLE_REWARD=false (pure OPSD) requires ENABLE_OPSD=true" >&2; exit 2
fi
case "$TEACHER_ACTION_SCOPE" in
    all|query|evidence|answer|multi) ;;
    *) echo "[run_grpo_opsd] ERROR: invalid TEACHER_ACTION_SCOPE=$TEACHER_ACTION_SCOPE" >&2; exit 2 ;;
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

DATASET_TEACHER_FIELDS="$(
    python - "$DATASET" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            fields = [
                key for key in (
                    "teacher_prompt",
                    "teacher_query_prompt",
                    "teacher_evidence_prompt",
                    "teacher_answer_prompt",
                )
                if row.get(key)
            ]
            print(",".join(fields) if fields else "none")
            break
    else:
        raise SystemExit("dataset is empty")
PY
)"
if [ "$ENABLE_OPSD" = "true" ] && [ "$DATASET_TEACHER_FIELDS" = "none" ]; then
    echo "[run_grpo_opsd] ERROR: ENABLE_OPSD=true but dataset has no teacher prompt fields" >&2
    exit 2
fi
if [ "$ENABLE_OPSD" = "false" ] && [ "$DATASET_TEACHER_FIELDS" != "none" ]; then
    echo "[run_grpo_opsd] ERROR: ENABLE_OPSD=false but dataset has $DATASET_TEACHER_FIELDS" >&2
    exit 2
fi
if [ "$ENABLE_OPSD" = "true" ] && [ "$TEACHER_ACTION_SCOPE" != "multi" ] \
        && [[ ",$DATASET_TEACHER_FIELDS," != *",teacher_prompt,"* ]]; then
    echo "[run_grpo_opsd] ERROR: single-scope mode requires legacy teacher_prompt; use TEACHER_ACTION_SCOPE=multi for scoped fields" >&2
    exit 2
fi

DS_ARG=()
[ "$DEEPSPEED" != "none" ] && DS_ARG=(--deepspeed "$DEEPSPEED")
RESUME_ARG=()
[ -n "$RESUME_FROM_CHECKPOINT" ] && RESUME_ARG=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
MAX_STEPS_ARG=()
[ -n "$MAX_STEPS" ] && MAX_STEPS_ARG=(--max_steps "$MAX_STEPS")
OPD_ARG=(--teacher_kl_coef 0 --teacher_action_scope all)
if [ "$ENABLE_OPSD" = "true" ] && [ "$TEACHER_ACTION_SCOPE" = "multi" ]; then
    OPD_ARG=(
        --teacher_kl_coef 0
        --teacher_action_scope multi
        --teacher_query_kl_coef "$TEACHER_QUERY_KL_COEF"
        --teacher_evidence_kl_coef "$TEACHER_EVIDENCE_KL_COEF"
        --teacher_answer_kl_coef "$TEACHER_ANSWER_KL_COEF"
    )
elif [ "$ENABLE_OPSD" = "true" ]; then
    OPD_ARG=(
        --teacher_kl_coef "$TEACHER_KL_COEF"
        --teacher_action_scope "$TEACHER_ACTION_SCOPE"
    )
fi

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$SWIFT_ROOT"

echo "[run_grpo_opsd] dataset=$DATASET"
echo "[run_grpo_opsd] init_adapter=$INIT_ADAPTER resolved_adapter=$ADAPTER_PATH"
echo "[run_grpo_opsd] output_dir=$OUTPUT_DIR"
echo "[run_grpo_opsd] backend=$DEVICE_BACKEND visible_env=$VISIBLE_DEVICES_ENV train_devices=$TRAIN_DEVICES nproc=$NPROC_PER_NODE"
echo "[run_grpo_opsd] layout=train:${DEVICE_LABEL}${TRAIN_DEVICES}"
echo "[run_grpo_opsd] vllm_server=${VLLM_HOST}:${VLLM_PORT} group_port=${VLLM_GROUP_PORT}"
echo "[run_grpo_opsd] opsd=$ENABLE_OPSD teacher_fields=$DATASET_TEACHER_FIELDS action_scope=$TEACHER_ACTION_SCOPE"
echo "[run_grpo_opsd] teacher_coefs=global:$TEACHER_KL_COEF query:$TEACHER_QUERY_KL_COEF evidence:$TEACHER_EVIDENCE_KL_COEF answer:$TEACHER_ANSWER_KL_COEF"
echo "[run_grpo_opsd] advantage_mode=$ADVANTAGE_MODE query_outcome_coef=$ACTION_QUERY_OUTCOME_COEF credit_clip=$ACTION_CREDIT_CLIP"
echo "[run_grpo_opsd] loss_type=$LOSS_TYPE dynamic_sample=$DYNAMIC_SAMPLE max_resample_times=$MAX_RESAMPLE_TIMES overlong_filter=$OVERLONG_FILTER"

REWARD_FUNCS=(sapr_f1 sapr_relevance sapr_format)
REWARD_WEIGHTS=(1.0 0.2 0.05)
if [ "$ENABLE_TRUNCATION_REWARD" = "true" ]; then
    REWARD_FUNCS+=(sapr_truncation)
    REWARD_WEIGHTS+=("$TRUNCATION_REWARD_WEIGHT")
fi
if [ "$ACTION_CREDIT_MODE" != "off" ]; then
    # 权重 0：该 reward 只把逐轮证据增益写入 rollout_infos，不改变标量 reward。
    REWARD_FUNCS+=(sapr_query_evidence_gain)
    REWARD_WEIGHTS+=(0.0)
fi
REWARD_ARG=(--reward_funcs "${REWARD_FUNCS[@]}" --reward_weights "${REWARD_WEIGHTS[@]}")
if [ "$ENABLE_REWARD" = "false" ]; then
    REWARD_FUNCS=()
    REWARD_WEIGHTS=()
    REWARD_ARG=()
fi
echo "[run_grpo_opsd] enable_reward=$ENABLE_REWARD opd_use_grpo_advantage=$OPD_USE_GRPO_ADVANTAGE"
echo "[run_grpo_opsd] reward_funcs=${REWARD_FUNCS[*]:-<none>} reward_weights=${REWARD_WEIGHTS[*]:-<none>}"

CMD=(
    swift rlhf
    --rlhf_type grpo
    --model "$BASE_MODEL"
    --adapters "$ADAPTER_PATH"
    --tuner_type lora
    --external_plugins "$PLUGIN"
    "${REWARD_ARG[@]}"
    --opd_use_grpo_advantage "$OPD_USE_GRPO_ADVANTAGE"
    --action_credit_mode "$ACTION_CREDIT_MODE"
    --action_credit_coef "$ACTION_CREDIT_COEF"
    --action_credit_scale "$ACTION_CREDIT_SCALE"
    --action_credit_infos_key "$ACTION_CREDIT_INFOS_KEY"
    --action_credit_gate "$ACTION_CREDIT_GATE"
    --advantage_mode "$ADVANTAGE_MODE"
    --action_query_outcome_coef "$ACTION_QUERY_OUTCOME_COEF"
    --action_credit_clip "$ACTION_CREDIT_CLIP"
    --use_vllm true
    --vllm_mode server
    --vllm_server_host "${VLLM_HOST_ARR[@]}"
    --vllm_server_port "${VLLM_PORT_ARR[@]}"
    --vllm_server_group_port "${VLLM_GROUP_PORT_ARR[@]}"
    --vllm_server_pass_dataset true
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
    --loss_type "$LOSS_TYPE"
    --dynamic_sample "$DYNAMIC_SAMPLE"
    --max_resample_times "$MAX_RESAMPLE_TIMES"
    --overlong_filter "$OVERLONG_FILTER"
    --learning_rate 1e-6
    --temperature 1.0
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --save_steps "$SAVE_STEPS"
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
