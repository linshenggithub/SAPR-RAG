#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SWIFT_ROOT="${SWIFT_ROOT:-$(cd "$PROJ_ROOT/../ms-swift" && pwd)}"

MERGED_MODEL="${MERGED_MODEL:-/tmp/sapr_grpo_full/models/qwen2_5_7b_sft_ckpt1650_merged}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/sapr_grpo_full/checkpoints}"
RUN_NAME="${RUN_NAME:-grpo_full_control_sft_merged_pbs2_g7_epoch1_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-16}"

LOG_DIR="$SCRIPT_DIR/logs"
LAUNCH_LOG="$LOG_DIR/${RUN_NAME}.launcher.log"
PID_FILE="$LOG_DIR/${RUN_NAME}.pid"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

for path in \
    "$MERGED_MODEL/config.json" \
    "$MERGED_MODEL/model.safetensors.index.json" \
    "$PROJ_ROOT/data/grpo/hotpotqa_2wiki_train.jsonl"; do
    [ -e "$path" ] || { echo "[launch-full-grpo] ERROR: required path not found: $path" >&2; exit 2; }
done

python - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8100/health", timeout=5) as resp:
    payload = json.loads(resp.read().decode("utf-8"))
print("[preflight] retrieval health:", json.dumps(payload, ensure_ascii=False))
if payload.get("status") != "ok":
    raise SystemExit(f"retrieval daemon unhealthy: {payload}")
PY

python - "$MERGED_MODEL" <<'PY'
import json
import pathlib
import sys

model_dir = pathlib.Path(sys.argv[1]).resolve()
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
weight_files = sorted(set(index["weight_map"].values()))
missing = [name for name in weight_files if not (model_dir / name).is_file()]
if missing:
    raise SystemExit(f"merged model has missing weight shards: {missing}")
print(f"[preflight] merged_model={model_dir} weight_shards={len(weight_files)}")
print(f"[preflight] policy_model={model_dir}")
print(f"[preflight] ref_model={model_dir}")
PY

mapfile -t TRAIN_GPU_USED_MIB < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F, '$1 + 0 >= 1 {gsub(/ /, "", $2); print $2}'
)
for used_mib in "${TRAIN_GPU_USED_MIB[@]}"; do
    if [ "$used_mib" -gt 1024 ]; then
        echo "[launch-full-grpo] ERROR: at least one training GPU (1-7) uses more than 1 GiB." >&2
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >&2
        exit 2
    fi
done

echo "[launch-full-grpo] run_name=$RUN_NAME"
echo "[launch-full-grpo] merged_model=$MERGED_MODEL"
echo "[launch-full-grpo] output_dir=$OUTPUT_DIR"
echo "[launch-full-grpo] max_steps=${MAX_STEPS:-full_epoch}"
echo "[launch-full-grpo] log=$LAUNCH_LOG"

nohup setsid bash -lc "
set -euo pipefail
cd '$PROJ_ROOT'
env \
  SAPR_RAG_ROOT='$PROJ_ROOT' \
  SWIFT_ROOT='$SWIFT_ROOT' \
  TUNER_TYPE=full \
  BASE_MODEL='$MERGED_MODEL' \
  MODEL_TYPE=qwen2 \
  TEMPLATE_TYPE=qwen2_5 \
  REF_MODEL='$MERGED_MODEL' \
  REF_MODEL_TYPE=qwen2 \
  DEEPSPEED=zero3 \
  ENABLE_OPSD=false \
  DATASET='$PROJ_ROOT/data/grpo/hotpotqa_2wiki_train.jsonl' \
  RUN_NAME='$RUN_NAME' \
  OUTPUT_DIR='$OUTPUT_DIR' \
  TRAIN_DEVICES=1,2,3,4,5,6,7 \
  NPROC_PER_NODE=7 \
  PER_DEVICE_BATCH_SIZE=2 \
  NUM_GENERATIONS=7 \
  GRADIENT_ACCUMULATION_STEPS=1 \
  STEPS_PER_GENERATION=1 \
  TEACHER_KL_COEF=0 \
  BETA=0.04 \
  LEARNING_RATE=1e-6 \
  MAX_LENGTH=2048 \
  MAX_COMPLETION_LENGTH=1024 \
  MAX_TURNS=6 \
  VLLM_GPU_MEMORY_UTILIZATION=0.20 \
  VLLM_MAX_NUM_SEQS=16 \
  VLLM_MAX_MODEL_LEN=8192 \
  SAVE_STEPS='$SAVE_STEPS' \
  SAVE_TOTAL_LIMIT='$SAVE_TOTAL_LIMIT' \
  MAX_STEPS='$MAX_STEPS' \
  bash '$SCRIPT_DIR/run_grpo_opsd_colocate.sh'
" >"$LAUNCH_LOG" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$PID_FILE"
sleep 8
if ! kill -0 "$pid" 2>/dev/null; then
    echo "[launch-full-grpo] process exited early; tail follows"
    tail -n 160 "$LAUNCH_LOG" || true
    exit 3
fi

echo "[launch-full-grpo] started pid=$pid pid_file=$PID_FILE"
tail -n 60 "$LAUNCH_LOG" || true
