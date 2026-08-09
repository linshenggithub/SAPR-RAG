#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SWIFT_ROOT="${SWIFT_ROOT:-$(cd "$PROJ_ROOT/../ms-swift" && pwd)}"

MERGED_MODEL="${MERGED_MODEL:-/tmp/sapr_grpo_reward_v2/models/qwen2_5_7b_sft_ckpt1650_merged}"
DATASET="${DATASET:-$PROJ_ROOT/data/grpo/hotpotqa_2wiki_train_reward_v2.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/sapr_grpo_reward_v2/checkpoints}"
RUN_NAME="${RUN_NAME:-grpo_reward_v2_lora_pbs2_g7_s500_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"

REWARD_FUNCS="${REWARD_FUNCS:-sapr_f1 sapr_relevance sapr_format sapr_turn_cost sapr_repeat_query sapr_max_turn}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0 0.15 0.05 0.02 0.15 0.50}"

LOG_DIR="$SCRIPT_DIR/logs"
LAUNCH_LOG="$LOG_DIR/${RUN_NAME}.launcher.log"
PID_FILE="$LOG_DIR/${RUN_NAME}.pid"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

for path in \
    "$MERGED_MODEL/config.json" \
    "$MERGED_MODEL/model.safetensors.index.json" \
    "$DATASET"; do
    [ -e "$path" ] || {
        echo "[launch-reward-v2] ERROR: required path not found: $path" >&2
        exit 2
    }
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

mapfile -t TRAIN_GPU_USED_MIB < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F, '$1 + 0 >= 1 {gsub(/ /, "", $2); print $2}'
)
for used_mib in "${TRAIN_GPU_USED_MIB[@]}"; do
    if [ "$used_mib" -gt 1024 ]; then
        echo "[launch-reward-v2] ERROR: at least one training GPU (1-7) uses more than 1 GiB." >&2
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >&2
        exit 2
    fi
done

echo "[launch-reward-v2] run_name=$RUN_NAME"
echo "[launch-reward-v2] merged_model=$MERGED_MODEL"
echo "[launch-reward-v2] dataset=$DATASET"
echo "[launch-reward-v2] output_dir=$OUTPUT_DIR"
echo "[launch-reward-v2] max_steps=$MAX_STEPS"
echo "[launch-reward-v2] reward_funcs=$REWARD_FUNCS"
echo "[launch-reward-v2] reward_weights=$REWARD_WEIGHTS"
echo "[launch-reward-v2] log=$LAUNCH_LOG"

nohup setsid bash -lc "
set -euo pipefail
cd '$PROJ_ROOT'
env \
  SAPR_RAG_ROOT='$PROJ_ROOT' \
  SWIFT_ROOT='$SWIFT_ROOT' \
  TUNER_TYPE=lora \
  BASE_MODEL='$MERGED_MODEL' \
  MODEL_TYPE=qwen2 \
  TEMPLATE_TYPE=qwen2_5 \
  INIT_ADAPTER=none \
  ENABLE_OPSD=false \
  DATASET='$DATASET' \
  RUN_NAME='$RUN_NAME' \
  OUTPUT_DIR='$OUTPUT_DIR' \
  TRAIN_DEVICES=1,2,3,4,5,6,7 \
  NPROC_PER_NODE=7 \
  PER_DEVICE_BATCH_SIZE=2 \
  NUM_GENERATIONS=7 \
  GRADIENT_ACCUMULATION_STEPS=1 \
  STEPS_PER_GENERATION=1 \
  BETA=0.04 \
  LEARNING_RATE=1e-6 \
  LORA_RANK=16 \
  MAX_LENGTH=2048 \
  MAX_COMPLETION_LENGTH=1024 \
  MAX_TURNS=6 \
  VLLM_GPU_MEMORY_UTILIZATION=0.20 \
  VLLM_MAX_NUM_SEQS=16 \
  VLLM_MAX_MODEL_LEN=8192 \
  SAVE_STEPS='$SAVE_STEPS' \
  SAVE_TOTAL_LIMIT='$SAVE_TOTAL_LIMIT' \
  MAX_STEPS='$MAX_STEPS' \
  REWARD_FUNCS='$REWARD_FUNCS' \
  REWARD_WEIGHTS='$REWARD_WEIGHTS' \
  REPEAT_QUERY_CAP=3 \
  bash '$SCRIPT_DIR/run_grpo_opsd_colocate.sh'
" >"$LAUNCH_LOG" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$PID_FILE"
sleep 8
if ! kill -0 "$pid" 2>/dev/null; then
    echo "[launch-reward-v2] process exited early; tail follows"
    tail -n 160 "$LAUNCH_LOG" || true
    exit 3
fi

echo "[launch-reward-v2] started pid=$pid pid_file=$PID_FILE"
tail -n 80 "$LAUNCH_LOG" || true
