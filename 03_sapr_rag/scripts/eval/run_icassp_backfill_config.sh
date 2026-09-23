#!/usr/bin/env bash
# Run one ICASSP inference-budget configuration on a GPU released early.
set -euo pipefail

if (( $# != 4 )); then
  echo "Usage: $0 CONFIG GPU PORT OUT_ROOT" >&2
  exit 2
fi

CONFIG="$1"
GPU="$2"
PORT="$3"
OUT_ROOT="$4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
EVAL_ENTRY="$SCRIPT_DIR/eval_action_opsd_3src.sh"
RUN_ROOT="${RUN_ROOT:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/opsd_canonical_sft_q001_a003_3src_s1000_20260905/v0-20260905-151754}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-1000}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8100}"
DATASETS_CSV="${DATASETS_CSV:-hotpotqa,2wikimultihopqa,musique}"
SUBSET_SEED="${SUBSET_SEED:-20260922}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"
RETRY_BACKOFF="${RETRY_BACKOFF:-5}"
RESUME="${RESUME:-false}"

if [[ "$CONFIG" =~ ^s([0-7])_k([1-9][0-9]*)$ ]]; then
  MAX_SEARCHES="${BASH_REMATCH[1]}"
  TOP_K="${BASH_REMATCH[2]}"
else
  echo "ERROR: invalid config: $CONFIG" >&2
  exit 2
fi

CONFIG_OUT="$OUT_ROOT/$CONFIG"
if [[ "$RESUME" != "true" ]] \
    && find "$CONFIG_OUT" -name results.jsonl -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: result already exists: $CONFIG_OUT" >&2
  exit 5
fi

mkdir -p "$CONFIG_OUT"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

STARTED_AT="$(date -Is)"
STARTED_EPOCH="$(date +%s)"
echo "[backfill] start config=$CONFIG gpu=$GPU port=$PORT"

set +e
env \
  SAPR_RAG_ROOT="$PROJ_ROOT" \
  MODE=full \
  RUN_ROOT="$RUN_ROOT" \
  CHECKPOINT_STEP="$CHECKPOINT_STEP" \
  CHECKPOINT_STEPS="$CHECKPOINT_STEP" \
  N_SUBSET=20 \
  SUBSET_SELECTION=hash \
  SUBSET_SEED="$SUBSET_SEED" \
  ROLLOUT_GPU="$GPU" \
  ROLLOUT_PORT="$PORT" \
  RETRIEVAL_URL="$RETRIEVAL_URL" \
  DATASETS_CSV="$DATASETS_CSV" \
  TOP_K="$TOP_K" \
  MAX_SEARCHES="$MAX_SEARCHES" \
  MAX_TURNS="$((MAX_SEARCHES + 1))" \
  FORCE_FINAL_ANSWER=true \
  RUN_BOOTSTRAP=false \
  REQUEST_RETRIES="$REQUEST_RETRIES" \
  RETRY_BACKOFF="$RETRY_BACKOFF" \
  RESUME="$RESUME" \
  OUT_ROOT="$CONFIG_OUT" \
  DRY_RUN=false \
  bash "$EVAL_ENTRY" \
  >"$CONFIG_OUT/orchestrator.log" 2>&1
STATUS="$?"
set -e

COMPLETED_AT="$(date -Is)"
COMPLETED_EPOCH="$(date +%s)"
{
  printf 'config\tgpu\tport\tphase\tstarted_at\tcompleted_at\twall_seconds\texit_code\n'
  printf '%s\t%s\t%s\tfull-backfill\t%s\t%s\t%s\t%s\n' \
    "$CONFIG" "$GPU" "$PORT" "$STARTED_AT" "$COMPLETED_AT" \
    "$((COMPLETED_EPOCH - STARTED_EPOCH))" "$STATUS"
} >"$CONFIG_OUT/runtime.tsv"

echo "[backfill] done config=$CONFIG exit_code=$STATUS"
exit "$STATUS"
