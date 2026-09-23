#!/usr/bin/env bash
# Run the ICASSP inference-budget ablation for the fixed C/E16 checkpoint-1000.
# Supports dry-run, fixed hash smoke, and full dev evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
EVAL_ENTRY="$SCRIPT_DIR/eval_action_opsd_3src.sh"

RUN_ROOT="${RUN_ROOT:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/opsd_canonical_sft_q001_a003_3src_s1000_20260905/v0-20260905-151754}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-1000}"
PHASE="${PHASE:-dry-run}"
CONFIGS_CSV="${CONFIGS_CSV:-s5_k3,s0_k3,s1_k3,s2_k3,s3_k3,s4_k3,s6_k3,s7_k3,s5_k1,s5_k5}"
GPUS_CSV="${GPUS_CSV:-1,2,3}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
PORT_BASE="${PORT_BASE:-8130}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8100}"
DATASETS_CSV="${DATASETS_CSV:-hotpotqa,2wikimultihopqa,musique}"
N_SUBSET="${N_SUBSET:-20}"
SUBSET_SEED="${SUBSET_SEED:-20260922}"
RUN_TAG="${RUN_TAG:-20260922}"
OUT_ROOT="${OUT_ROOT:-$PROJ_ROOT/data/eval_results/icassp_inference_budget_full_ckpt1000_${RUN_TAG}}"
ALLOW_EXISTING="${ALLOW_EXISTING:-false}"
RESUME="${RESUME:-false}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"
RETRY_BACKOFF="${RETRY_BACKOFF:-5}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-10}"
GPU_MONITOR_PID=""

case "$PHASE" in
  dry-run|smoke|full) ;;
  *) echo "ERROR: PHASE must be dry-run, smoke, or full, got: $PHASE" >&2; exit 2 ;;
esac
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: MAX_PARALLEL must be a positive integer" >&2
  exit 2
}
[[ "$ALLOW_EXISTING" == "true" || "$ALLOW_EXISTING" == "false" ]] || {
  echo "ERROR: ALLOW_EXISTING must be true or false" >&2
  exit 2
}
[[ "$RESUME" == "true" || "$RESUME" == "false" ]] || {
  echo "ERROR: RESUME must be true or false" >&2
  exit 2
}
[[ "$GPU_MONITOR_INTERVAL" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: GPU_MONITOR_INTERVAL must be a positive integer" >&2
  exit 2
}
[[ "$REQUEST_RETRIES" =~ ^[0-9]+$ ]] || {
  echo "ERROR: REQUEST_RETRIES must be a non-negative integer" >&2
  exit 2
}

IFS=',' read -r -a CONFIGS <<<"$CONFIGS_CSV"
IFS=',' read -r -a GPUS <<<"$GPUS_CSV"
(( MAX_PARALLEL <= ${#GPUS[@]} )) || {
  echo "ERROR: MAX_PARALLEL=$MAX_PARALLEL exceeds GPU slots=${#GPUS[@]}" >&2
  exit 2
}

parse_config() {
  local config="$1"
  if [[ "$config" =~ ^s([0-7])_k([1-9][0-9]*)$ ]]; then
    printf '%s %s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
  else
    echo "ERROR: invalid config name: $config (expected s0_k3 style)" >&2
    return 2
  fi
}

mkdir -p "$OUT_ROOT"

if [[ "$PHASE" != "dry-run" ]]; then
  export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
  export no_proxy="$NO_PROXY"
  health="$(curl -fsS --max-time 5 "$RETRIEVAL_URL/health")" || {
    echo "ERROR: retrieval service unavailable: $RETRIEVAL_URL/health" >&2
    exit 3
  }
  python - "$health" <<'PY'
import json
import sys

health = json.loads(sys.argv[1])
expected = {
    "status": "ok",
    "n_vectors": 22352695,
    "n_docs": 22352695,
    "text_truncate": 500,
    "faiss_device": "gpu",
    "faiss_gpu_id": 0,
    "faiss_gpu_fp16": False,
}
bad = {key: (health.get(key), value) for key, value in expected.items() if health.get(key) != value}
if bad:
    raise SystemExit(f"retrieval health mismatch: {bad}; full={health}")
print("[preflight] retrieval health verified")
PY
  printf '%s\n' "$health" >"$OUT_ROOT/retrieval_health.json"
fi

MANIFEST="$OUT_ROOT/orchestrator_${PHASE}.txt"
{
  echo "phase=$PHASE"
  echo "run_root=$RUN_ROOT"
  echo "checkpoint_step=$CHECKPOINT_STEP"
  echo "configs=$CONFIGS_CSV"
  echo "gpus=$GPUS_CSV"
  echo "max_parallel=$MAX_PARALLEL"
  echo "port_base=$PORT_BASE"
  echo "retrieval_url=$RETRIEVAL_URL"
  echo "datasets=$DATASETS_CSV"
  echo "n_subset=$N_SUBSET"
  echo "subset_seed=$SUBSET_SEED"
  echo "allow_existing=$ALLOW_EXISTING"
  echo "resume=$RESUME"
  echo "request_retries=$REQUEST_RETRIES"
  echo "retry_backoff=$RETRY_BACKOFF"
  echo "gpu_monitor_interval=$GPU_MONITOR_INTERVAL"
  echo "git_commit=$(git -C "$PROJ_ROOT" rev-parse HEAD)"
  echo "started_at=$(date -Is)"
} >"$MANIFEST"

sha256sum \
  "$EVAL_ENTRY" \
  "$SCRIPT_DIR/run_direct_rollout_eval.py" \
  "$SCRIPT_DIR/score.py" \
  "$PROJ_ROOT/03_sapr_rag/scripts/grpo/plugin.py" \
  "$RUN_ROOT/checkpoint-$CHECKPOINT_STEP/adapter_config.json" \
  "$RUN_ROOT/checkpoint-$CHECKPOINT_STEP/adapter_model.safetensors" \
  "$PROJ_ROOT/data/eval/hotpotqa/dev.jsonl" \
  "$PROJ_ROOT/data/eval/2wikimultihopqa/dev.jsonl" \
  "$PROJ_ROOT/data/eval/musique/dev.jsonl" \
  >"$OUT_ROOT/input_sha256.txt"

monitor_gpus() {
  while true; do
    local timestamp
    timestamp="$(date -Is)"
    nvidia-smi \
      --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits \
      | sed "s/^/${timestamp},/"
    sleep "$GPU_MONITOR_INTERVAL"
  done
}

cleanup_monitor() {
  if [[ -n "${GPU_MONITOR_PID:-}" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup_monitor EXIT

if [[ "$PHASE" != "dry-run" ]]; then
  printf 'timestamp,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_pct\n' \
    >"$OUT_ROOT/gpu_metrics.csv"
  monitor_gpus >>"$OUT_ROOT/gpu_metrics.csv" 2>>"$OUT_ROOT/gpu_monitor.log" &
  GPU_MONITOR_PID="$!"
fi

run_one() {
  local config="$1"
  local gpu="$2"
  local port="$3"
  local values
  local max_searches
  local top_k
  local config_out
  local mode
  local dry_run
  local started_at
  local started_epoch
  local completed_at
  local completed_epoch
  local status

  values="$(parse_config "$config")"
  read -r max_searches top_k <<<"$values"
  config_out="$OUT_ROOT/$config"

  if [[ "$ALLOW_EXISTING" != "true" && "$RESUME" != "true" ]] \
      && find "$config_out" -name results.jsonl -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: result already exists for $config: $config_out" >&2
    return 5
  fi
  mkdir -p "$config_out"
  started_at="$(date -Is)"
  started_epoch="$(date +%s)"

  case "$PHASE" in
    dry-run)
      mode=full
      dry_run=true
      ;;
    smoke)
      mode=sweep
      dry_run=false
      ;;
    full)
      mode=full
      dry_run=false
      ;;
  esac

  echo "[orchestrator] start config=$config gpu=$gpu port=$port phase=$PHASE"
  set +e
  env \
    SAPR_RAG_ROOT="$PROJ_ROOT" \
    MODE="$mode" \
    RUN_ROOT="$RUN_ROOT" \
    CHECKPOINT_STEP="$CHECKPOINT_STEP" \
    CHECKPOINT_STEPS="$CHECKPOINT_STEP" \
    N_SUBSET="$N_SUBSET" \
    SUBSET_SELECTION=hash \
    SUBSET_SEED="$SUBSET_SEED" \
    ROLLOUT_GPU="$gpu" \
    ROLLOUT_PORT="$port" \
    RETRIEVAL_URL="$RETRIEVAL_URL" \
    DATASETS_CSV="$DATASETS_CSV" \
    TOP_K="$top_k" \
    MAX_SEARCHES="$max_searches" \
    MAX_TURNS="$((max_searches + 1))" \
    FORCE_FINAL_ANSWER=true \
    RUN_BOOTSTRAP=false \
    REQUEST_RETRIES="$REQUEST_RETRIES" \
    RETRY_BACKOFF="$RETRY_BACKOFF" \
    RESUME="$RESUME" \
    OUT_ROOT="$config_out" \
    DRY_RUN="$dry_run" \
    bash "$EVAL_ENTRY" \
    >"$config_out/orchestrator.log" 2>&1
  status="$?"
  set -e
  completed_at="$(date -Is)"
  completed_epoch="$(date +%s)"
  {
    printf 'config\tgpu\tport\tphase\tstarted_at\tcompleted_at\twall_seconds\texit_code\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$config" "$gpu" "$port" "$PHASE" "$started_at" "$completed_at" \
      "$((completed_epoch - started_epoch))" "$status"
  } >"$config_out/runtime.tsv"
  if (( status != 0 )); then
    return "$status"
  fi
  echo "[orchestrator] done config=$config"
}

failures=0
for ((start=0; start<${#CONFIGS[@]}; start+=MAX_PARALLEL)); do
  pids=()
  names=()
  for ((slot=0; slot<MAX_PARALLEL && start+slot<${#CONFIGS[@]}; slot++)); do
    config="${CONFIGS[$((start + slot))]}"
    gpu="${GPUS[$slot]}"
    port="$((PORT_BASE + slot))"
    run_one "$config" "$gpu" "$port" &
    pids+=("$!")
    names+=("$config")
  done

  for ((i=0; i<${#pids[@]}; i++)); do
    if ! wait "${pids[$i]}"; then
      echo "ERROR: config failed: ${names[$i]}" >&2
      failures=$((failures + 1))
    fi
  done
  (( failures == 0 )) || break
done

echo "completed_at=$(date -Is)" >>"$MANIFEST"
echo "failures=$failures" >>"$MANIFEST"
(( failures == 0 )) || exit 6

if [[ "$PHASE" == "smoke" ]]; then
  python "$SCRIPT_DIR/validate_icassp_inference_budget.py" \
    --root "$OUT_ROOT" \
    --stage selection \
    --checkpoint-step "$CHECKPOINT_STEP" \
    --configs "$CONFIGS_CSV" \
    --expected-per-dataset "$N_SUBSET"
elif [[ "$PHASE" == "full" ]]; then
  python "$SCRIPT_DIR/validate_icassp_inference_budget.py" \
    --root "$OUT_ROOT" \
    --stage full \
    --checkpoint-step "$CHECKPOINT_STEP" \
    --configs "$CONFIGS_CSV"
fi

echo "[orchestrator] all requested configs completed: $OUT_ROOT"
