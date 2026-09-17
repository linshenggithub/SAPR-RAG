#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TRAIN_SESSION="${TRAIN_SESSION:-e19_action_causal_s1000}"
RUN_NAME="${RUN_NAME:-action_causal_g2_a0.5_l0.5_s1000_20260908}"
RUN_VERSION="${RUN_VERSION:-v0-20260908-065449}"
RUN_ROOT="${RUN_ROOT:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/$RUN_NAME/$RUN_VERSION}"
TRAIN_STATUS="${TRAIN_STATUS:-$PROJ_ROOT/03_sapr_rag/scripts/grpo/logs/$RUN_NAME/status.txt}"
TRAIN_LOG="${TRAIN_LOG:-$PROJ_ROOT/03_sapr_rag/scripts/grpo/logs/$RUN_NAME/train.log}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-250,500,750,1000}"
FINAL_STEP="${FINAL_STEP:-1000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
ROLLOUT_GPU="${ROLLOUT_GPU:-7}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8030}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8100}"
OUT_ROOT="${OUT_ROOT:-$PROJ_ROOT/data/eval_results/E19_action_causal_s1000_20260908}"

EVAL_SCRIPT="$SCRIPT_DIR/eval_action_opsd_3src.sh"
ORCH_STATUS="$OUT_ROOT/orchestrator_status.txt"
FINAL_CKPT="$RUN_ROOT/checkpoint-$FINAL_STEP"

mkdir -p "$OUT_ROOT"

record_status() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$ORCH_STATUS"
}

latest_step() {
    local logging="$RUN_ROOT/logging.jsonl"
    if [[ ! -s "$logging" ]]; then
        printf '0\n'
        return
    fi
    tail -n 1 "$logging" | jq -r '."global_step/max_steps" | split("/")[0]'
}

pane_dead() {
    tmux display-message -p -t "$TRAIN_SESSION" '#{pane_dead}' 2>/dev/null || printf '1\n'
}

record_status "state=waiting_for_training current_step=$(latest_step)"
while ! grep -qx 'completed=1' "$TRAIN_STATUS" 2>/dev/null; do
    if grep -Eq 'rollout(_|2_)?(died|timeout)=1' "$TRAIN_STATUS" 2>/dev/null; then
        record_status "state=failed reason=rollout_failure current_step=$(latest_step)"
        exit 3
    fi
    if [[ "$(pane_dead)" == "1" ]]; then
        record_status "state=failed reason=training_session_dead current_step=$(latest_step)"
        tail -n 120 "$TRAIN_LOG" >>"$ORCH_STATUS" 2>/dev/null || true
        exit 4
    fi
    sleep "$POLL_SECONDS"
    record_status "state=training current_step=$(latest_step)"
done

for required in adapter_config.json adapter_model.safetensors trainer_state.json; do
    [[ -s "$FINAL_CKPT/$required" ]] || {
        record_status "state=failed reason=incomplete_final_checkpoint missing=$FINAL_CKPT/$required"
        exit 5
    }
done

sleep 60
curl -fsS --max-time 10 "$RETRIEVAL_URL/health" \
    >"$OUT_ROOT/retrieval_health_before_eval.json"

record_status "state=checkpoint_sweep_started"
MODE=sweep \
RUN_ROOT="$RUN_ROOT" \
CHECKPOINT_STEPS="$CHECKPOINT_STEPS" \
N_SUBSET=1000 \
SUBSET_SELECTION=hash \
SUBSET_SEED=20260908 \
OUT_ROOT="$OUT_ROOT" \
ROLLOUT_GPU="$ROLLOUT_GPU" \
ROLLOUT_PORT="$ROLLOUT_PORT" \
RETRIEVAL_URL="$RETRIEVAL_URL" \
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
    bash "$EVAL_SCRIPT" 2>&1 | tee "$OUT_ROOT/full.log"

record_status "state=complete best_step=$BEST_STEP"
