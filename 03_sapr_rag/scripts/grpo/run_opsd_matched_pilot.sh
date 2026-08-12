#!/usr/bin/env bash
# Matched plain GRPO vs dynamic OPSD pilot 编排脚本。
#
# 默认只 dry-run，打印 Ascend/CUDA 统一入口命令，不启动训练。
# 典型 Ascend 910B2 用法：
#   DEVICE_BACKEND=npu TARGET=all DRY_RUN=true bash run_opsd_matched_pilot.sh
#   DEVICE_BACKEND=npu TARGET=opsd TEACHER_KL_COEFS="0.05 0.10 0.20" DRY_RUN=false bash run_opsd_matched_pilot.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

DEVICE_BACKEND="${DEVICE_BACKEND:-npu}"
RETRIEVAL_BACKEND="${RETRIEVAL_BACKEND:-cpu}"
if [ "$DEVICE_BACKEND" = "npu" ] && [ -n "${ASCEND_VISIBLE_DEVICES:-}" ]; then
    DEFAULT_TRAIN_DEVICES="$(python - "$ASCEND_VISIBLE_DEVICES" <<'PY'
import sys
devices = [x for x in sys.argv[1].split(",") if x]
print(",".join(devices[:6] if len(devices) >= 6 else devices))
PY
)"
    DEFAULT_ROLLOUT_DEVICES="$(python - "$ASCEND_VISIBLE_DEVICES" <<'PY'
import sys
devices = [x for x in sys.argv[1].split(",") if x]
print(devices[6] if len(devices) >= 7 else (devices[-1] if devices else "0"))
PY
)"
    DEFAULT_RETRIEVAL_DEVICES="$(python - "$ASCEND_VISIBLE_DEVICES" <<'PY'
import sys
devices = [x for x in sys.argv[1].split(",") if x]
print(devices[7] if len(devices) >= 8 else (devices[-1] if devices else "0"))
PY
)"
else
    DEFAULT_TRAIN_DEVICES="0,1,2,3,4,5"
    DEFAULT_ROLLOUT_DEVICES="6"
    DEFAULT_RETRIEVAL_DEVICES="7"
fi
TRAIN_DEVICES="${TRAIN_DEVICES:-$DEFAULT_TRAIN_DEVICES}"
ROLLOUT_DEVICES="${ROLLOUT_DEVICES:-$DEFAULT_ROLLOUT_DEVICES}"
RETRIEVAL_DEVICES="${RETRIEVAL_DEVICES:-$DEFAULT_RETRIEVAL_DEVICES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
PORT="${ROLLOUT_PORT:-8000}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8100}"
INIT_ADAPTER="${INIT_ADAPTER:-sft_dpo}"
MAX_STEPS="${MAX_STEPS:-20}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
DEEPSPEED="${DEEPSPEED:-zero2}"
DRY_RUN="${DRY_RUN:-true}"
TARGET="${TARGET:-all}"  # data | retrieval | rollout | control | opsd | all
TEACHER_KL_COEFS="${TEACHER_KL_COEFS:-0.05 0.10 0.20}"
TEACHER_ACTION_SCOPE="${TEACHER_ACTION_SCOPE:-all}"

PLAIN_DATASET="${PLAIN_DATASET:-$PROJ_ROOT/data/grpo/hotpotqa_2wiki_train_pilot.jsonl}"
OPSD_DATASET="${OPSD_DATASET:-$PROJ_ROOT/data/grpo/hotpotqa_2wiki_train_pilot_opsd.jsonl}"
PILOT_MAX_TOTAL="${PILOT_MAX_TOTAL:-100}"
SEED="${SEED:-42}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_pilot}"
CONTROL_OUTPUT_DIR="${CONTROL_OUTPUT_DIR:-$BASE_OUTPUT_DIR/plain_control}"
OPSD_OUTPUT_ROOT="${OPSD_OUTPUT_ROOT:-$BASE_OUTPUT_DIR/opsd}"

run_or_print() {
    printf '[pilot] %q' "$1"
    shift
    printf ' %q' "$@"
    printf '\n'
    if [ "$DRY_RUN" != "true" ]; then
        "$@"
    fi
}

build_data() {
    run_or_print plain-data \
        python "$SCRIPT_DIR/build_grpo_dataset_mixed_opsd.py" \
            --output "$PLAIN_DATASET" \
            --max_total "$PILOT_MAX_TOTAL" \
            --seed "$SEED" \
            --teacher_prompt_mode none

    run_or_print opsd-data \
        python "$SCRIPT_DIR/build_grpo_dataset_mixed_opsd.py" \
            --output "$OPSD_DATASET" \
            --max_total "$PILOT_MAX_TOTAL" \
            --seed "$SEED" \
            --teacher_prompt_mode gold

    if [ "$DRY_RUN" = "true" ]; then
        echo "[pilot] DRY_RUN: skip matched data diff. Set DRY_RUN=false TARGET=data to build real files."
    else
        python - "$PLAIN_DATASET" "$OPSD_DATASET" <<'PY'
import json
import sys

plain_path, opsd_path = sys.argv[1:]
teacher_keys = {
    "teacher_prompt",
    "teacher_prompt_version",
    "teacher_prompt_source",
    "teacher_prompt_fallback",
    "teacher_prompt_truncated",
    "teacher_prompt_tokens",
    "teacher_prompt_chars",
    "teacher_prompt_tokenizer",
}

with open(plain_path) as fp, open(opsd_path) as fo:
    plain_rows = [json.loads(line) for line in fp if line.strip()]
    opsd_rows = [json.loads(line) for line in fo if line.strip()]

assert len(plain_rows) == len(opsd_rows), (len(plain_rows), len(opsd_rows))
for i, (plain, opsd) in enumerate(zip(plain_rows, opsd_rows)):
    stripped = {k: v for k, v in opsd.items() if k not in teacher_keys}
    assert plain == stripped, f"matched diff at row {i}"
    assert opsd.get("teacher_prompt"), f"missing teacher_prompt at row {i}"
print(f"[pilot] matched data diff OK: rows={len(plain_rows)} teacher_keys={sorted(teacher_keys)}")
PY
        python "$SCRIPT_DIR/sanity_check_opsd.py" --skip_daemon --dataset "$OPSD_DATASET" --max_rows 0
    fi
}

run_retrieval() {
    DEVICE_BACKEND="$RETRIEVAL_BACKEND" \
    RETRIEVAL_DEVICES="$RETRIEVAL_DEVICES" \
    PORT="$RETRIEVAL_PORT" \
    DRY_RUN="$DRY_RUN" \
    bash "$SCRIPT_DIR/run_retrieval_daemon_flexible.sh"
}

run_rollout() {
    DEVICE_BACKEND="$DEVICE_BACKEND" \
    ROLLOUT_DEVICES="$ROLLOUT_DEVICES" \
    PORT="$PORT" \
    INIT_ADAPTER="$INIT_ADAPTER" \
    VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
    DRY_RUN="$DRY_RUN" \
    bash "$SCRIPT_DIR/run_rollout_opsd.sh"
}

run_control() {
    DEVICE_BACKEND="$DEVICE_BACKEND" \
    TRAIN_DEVICES="$TRAIN_DEVICES" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    INIT_ADAPTER="$INIT_ADAPTER" \
    ENABLE_OPSD=false \
    DATASET="$PLAIN_DATASET" \
    OUTPUT_DIR="$CONTROL_OUTPUT_DIR" \
    VLLM_PORT="$PORT" \
    MAX_STEPS="$MAX_STEPS" \
    PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
    MAX_LENGTH="$MAX_LENGTH" \
    MAX_COMPLETION_LENGTH="$MAX_COMPLETION_LENGTH" \
    VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
    DEEPSPEED="$DEEPSPEED" \
    DRY_RUN="$DRY_RUN" \
    bash "$SCRIPT_DIR/run_grpo_opsd.sh"
}

run_opsd() {
    for coef in $TEACHER_KL_COEFS; do
        local tag
        tag="$(echo "$coef" | tr '.' 'p')"
        DEVICE_BACKEND="$DEVICE_BACKEND" \
        TRAIN_DEVICES="$TRAIN_DEVICES" \
        NPROC_PER_NODE="$NPROC_PER_NODE" \
        INIT_ADAPTER="$INIT_ADAPTER" \
        ENABLE_OPSD=true \
        TEACHER_KL_COEF="$coef" \
        TEACHER_ACTION_SCOPE="$TEACHER_ACTION_SCOPE" \
        DATASET="$OPSD_DATASET" \
        OUTPUT_DIR="$OPSD_OUTPUT_ROOT/scope_${TEACHER_ACTION_SCOPE}_alpha_${tag}" \
        VLLM_PORT="$PORT" \
        MAX_STEPS="$MAX_STEPS" \
        PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
        MAX_LENGTH="$MAX_LENGTH" \
        MAX_COMPLETION_LENGTH="$MAX_COMPLETION_LENGTH" \
        VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
        DEEPSPEED="$DEEPSPEED" \
        DRY_RUN="$DRY_RUN" \
        bash "$SCRIPT_DIR/run_grpo_opsd.sh"
    done
}

echo "[pilot] target=$TARGET backend=$DEVICE_BACKEND dry_run=$DRY_RUN init_adapter=$INIT_ADAPTER"
echo "[pilot] plain_dataset=$PLAIN_DATASET"
echo "[pilot] opsd_dataset=$OPSD_DATASET"
echo "[pilot] max_steps=$MAX_STEPS teacher_kl_coefs=$TEACHER_KL_COEFS teacher_action_scope=$TEACHER_ACTION_SCOPE"

case "$TARGET" in
    data) build_data ;;
    retrieval) run_retrieval ;;
    rollout) run_rollout ;;
    control) run_control ;;
    opsd) run_opsd ;;
    all)
        build_data
        run_retrieval
        run_rollout
        run_control
        run_opsd
        ;;
    *)
        echo "[pilot] ERROR: TARGET must be data, retrieval, rollout, control, opsd, or all; got $TARGET" >&2
        exit 2
        ;;
esac
