#!/usr/bin/env bash
# Fill GPUs released by the first batch, then validate and summarize all runs.
set -uo pipefail

if (( $# != 2 )); then
  echo "Usage: $0 OUT_ROOT SUSPENDED_ORCHESTRATOR_PID" >&2
  exit 2
fi

OUT_ROOT="$1"
ORCHESTRATOR_PID="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$OUT_ROOT/dynamic_dispatcher.log"
STATUS_FILE="$OUT_ROOT/dynamic_dispatcher_status.tsv"
PRIMARY_CONFIGS=(s1_k3 s2_k3 s3_k3 s4_k3 s5_k3 s6_k3)
BACKFILLS=(s5_k1 s5_k5)
ALL_CONFIGS=(s0_k3 s1_k3 s2_k3 s3_k3 s4_k3 s5_k3 s6_k3 s7_k3 s5_k1 s5_k5)

exec >>"$LOG" 2>&1
echo "[dispatcher] started_at=$(date -Is)"

claim_released_slot() {
  local primary
  local runtime
  local exit_code
  local gpu
  local port

  while true; do
    for primary in "${PRIMARY_CONFIGS[@]}"; do
      runtime="$OUT_ROOT/$primary/runtime.tsv"
      [[ -f "$runtime" ]] || continue
      [[ ! -e "$OUT_ROOT/.slot_claimed_$primary" ]] || continue

      exit_code="$(awk -F '\t' 'NR == 2 {print $8}' "$runtime")"
      if [[ "$exit_code" != "0" ]]; then
        echo "[dispatcher] primary failed: $primary exit_code=$exit_code"
        return 1
      fi

      gpu="$(awk -F '\t' 'NR == 2 {print $2}' "$runtime")"
      port="$(awk -F '\t' 'NR == 2 {print $3}' "$runtime")"
      : >"$OUT_ROOT/.slot_claimed_$primary"
      printf '%s %s %s\n' "$primary" "$gpu" "$port"
      return 0
    done
    sleep 30
  done
}

backfill_pids=()
for config in "${BACKFILLS[@]}"; do
  if ! slot="$(claim_released_slot)"; then
    echo "[dispatcher] unable to claim slot for $config"
    exit 3
  fi
  read -r primary gpu port <<<"$slot"
  mkdir -p "$OUT_ROOT/$config"
  echo "[dispatcher] launch config=$config gpu=$gpu port=$port released_by=$primary"
  bash "$SCRIPT_DIR/run_icassp_backfill_config.sh" \
    "$config" "$gpu" "$port" "$OUT_ROOT" \
    >"$OUT_ROOT/$config/backfill_launcher.log" 2>&1 &
  backfill_pids+=("$!")
done

backfill_failures=0
for pid in "${backfill_pids[@]}"; do
  if ! wait "$pid"; then
    backfill_failures=$((backfill_failures + 1))
  fi
done

while true; do
  missing=0
  for config in "${ALL_CONFIGS[@]}"; do
    [[ -f "$OUT_ROOT/$config/runtime.tsv" ]] || missing=$((missing + 1))
  done
  (( missing == 0 )) && break
  echo "[dispatcher] waiting_for_runtime_files=$missing at $(date -Is)"
  sleep 60
done

run_failures="$backfill_failures"
for config in "${ALL_CONFIGS[@]}"; do
  exit_code="$(awk -F '\t' 'NR == 2 {print $8}' "$OUT_ROOT/$config/runtime.tsv")"
  if [[ "$exit_code" != "0" ]]; then
    echo "[dispatcher] failed config=$config exit_code=$exit_code"
    run_failures=$((run_failures + 1))
  fi
done

if kill -0 "$ORCHESTRATOR_PID" 2>/dev/null; then
  kill -TERM "$ORCHESTRATOR_PID" 2>/dev/null || true
  kill -CONT "$ORCHESTRATOR_PID" 2>/dev/null || true
  sleep 3
  kill -KILL "$ORCHESTRATOR_PID" 2>/dev/null || true
fi

validation_status=99
summary_status=99
if (( run_failures == 0 )); then
  python "$SCRIPT_DIR/validate_icassp_inference_budget.py" \
    --root "$OUT_ROOT" \
    --stage full \
    --checkpoint-step 1000 \
    --configs s5_k3,s0_k3,s1_k3,s2_k3,s3_k3,s4_k3,s6_k3,s7_k3,s5_k1,s5_k5 \
    >"$OUT_ROOT/dynamic_validation.log" 2>&1
  validation_status="$?"

  if (( validation_status == 0 )); then
    python "$SCRIPT_DIR/summarize_icassp_inference_budget.py" \
      --root "$OUT_ROOT" \
      --bootstrap-samples 10000 \
      >"$OUT_ROOT/dynamic_summary.log" 2>&1
    summary_status="$?"
  fi
fi

{
  printf 'completed_at\trun_failures\tvalidation_exit_code\tsummary_exit_code\n'
  printf '%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "$run_failures" "$validation_status" "$summary_status"
} >"$STATUS_FILE"

echo "[dispatcher] completed run_failures=$run_failures validation=$validation_status summary=$summary_status"
(( run_failures == 0 && validation_status == 0 && summary_status == 0 ))
