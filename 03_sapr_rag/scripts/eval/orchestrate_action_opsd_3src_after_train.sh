#!/usr/bin/env bash
# Wait for the formal action-scoped OPSD run, then perform checkpoint
# selection and full three-dataset evaluation without competing for its GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TRAIN_SESSION="${TRAIN_SESSION:-opsd_multi_s3000}"
RUN_ROOT="${RUN_ROOT:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/opsd_multi_q001_a003_3src_s3000_actionfix_tmux_20260812/v0-20260812-064707}"
FINAL_STEP="${FINAL_STEP:-3000}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-500,1000,1500,2000,2500,3000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
ROLLOUT_GPU="${ROLLOUT_GPU:-7}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8031}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8100}"
RUN_TAG="${RUN_TAG:-20260812}"
OUT_ROOT="${OUT_ROOT:-$PROJ_ROOT/data/eval_results/action_opsd_3src_s3000_${RUN_TAG}}"
DRY_RUN="${DRY_RUN:-false}"

EVAL_SCRIPT="$SCRIPT_DIR/eval_action_opsd_3src.sh"
FINAL_CKPT="$RUN_ROOT/checkpoint-$FINAL_STEP"
TRAIN_LOGGING="$RUN_ROOT/logging.jsonl"
STATUS="$OUT_ROOT/orchestrator_status.txt"

mkdir -p "$OUT_ROOT"

record_status() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS"
}

latest_step() {
  if [[ ! -s "$TRAIN_LOGGING" ]]; then
    printf '0\n'
    return
  fi
  tail -n 1 "$TRAIN_LOGGING" |
    jq -r '."global_step/max_steps" | split("/")[0]'
}

require_final_checkpoint() {
  local required
  for required in adapter_config.json adapter_model.safetensors trainer_state.json; do
    [[ -s "$FINAL_CKPT/$required" ]] || {
      record_status "state=failed reason=incomplete_final_checkpoint missing=$FINAL_CKPT/$required"
      exit 3
    }
  done
}

{
  echo "train_session=$TRAIN_SESSION"
  echo "run_root=$RUN_ROOT"
  echo "final_step=$FINAL_STEP"
  echo "checkpoint_steps=$CHECKPOINT_STEPS"
  echo "rollout_gpu=$ROLLOUT_GPU"
  echo "rollout_port=$ROLLOUT_PORT"
  echo "retrieval_url=$RETRIEVAL_URL"
  echo "out_root=$OUT_ROOT"
} >"$OUT_ROOT/orchestrator_config.txt"

if [[ "$DRY_RUN" == "true" ]]; then
  record_status "state=dry_run"
  cat "$OUT_ROOT/orchestrator_config.txt"
  exit 0
fi

record_status "state=waiting_for_training current_step=$(latest_step)"
while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  sleep "$POLL_SECONDS"
  record_status "state=training current_step=$(latest_step)"
done

record_status "state=training_session_finished current_step=$(latest_step)"
require_final_checkpoint

# Allow the training ranks and rollout server to release GPU memory and ports.
sleep 60
curl -fsS --max-time 10 "$RETRIEVAL_URL/health" \
  >"$OUT_ROOT/retrieval_health_before_eval.json"

record_status "state=checkpoint_sweep_started"
MODE=sweep \
RUN_ROOT="$RUN_ROOT" \
CHECKPOINT_STEPS="$CHECKPOINT_STEPS" \
OUT_ROOT="$OUT_ROOT" \
ROLLOUT_GPU="$ROLLOUT_GPU" \
ROLLOUT_PORT="$ROLLOUT_PORT" \
RETRIEVAL_URL="$RETRIEVAL_URL" \
RUN_TAG="$RUN_TAG" \
  bash "$EVAL_SCRIPT" 2>&1 | tee "$OUT_ROOT/sweep.log"

BEST_STEP="$(cat "$OUT_ROOT/best_checkpoint_step.txt")"
record_status "state=full_eval_started best_step=$BEST_STEP"
MODE=full \
RUN_ROOT="$RUN_ROOT" \
CHECKPOINT_STEP="$BEST_STEP" \
OUT_ROOT="$OUT_ROOT" \
ROLLOUT_GPU="$ROLLOUT_GPU" \
ROLLOUT_PORT="$ROLLOUT_PORT" \
RETRIEVAL_URL="$RETRIEVAL_URL" \
RUN_TAG="$RUN_TAG" \
  bash "$EVAL_SCRIPT" 2>&1 | tee "$OUT_ROOT/full.log"

record_status "state=complete best_step=$BEST_STEP"
