#!/usr/bin/env bash
# Evaluate the three-source action-scoped OPSD run with the training-time
# scheduler, Evidence Agent, and BGE+FAISS Top-3 retrieval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SWIFT_ROOT="${SWIFT_ROOT:-$(cd "$PROJ_ROOT/../ms-swift" && pwd)}"
GRPO_DIR="$PROJ_ROOT/03_sapr_rag/scripts/grpo"

MODE="${MODE:-sweep}"
RUN_ROOT="${RUN_ROOT:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/opsd_multi_q001_a003_3src_s3000_actionfix_tmux_20260812/v0-20260812-064707}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-500,1000,1500,2000,2500,3000}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"
N_SUBSET="${N_SUBSET:-200}"
ROLLOUT_GPU="${ROLLOUT_GPU:-7}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8031}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-512}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-20000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$PROJ_ROOT/data/eval_results/action_opsd_3src_${MODE}_${RUN_TAG}}"
DRY_RUN="${DRY_RUN:-false}"

DATASETS_CSV="${DATASETS_CSV:-hotpotqa,2wikimultihopqa,musique}"
IFS=',' read -r -a DATASETS <<<"$DATASETS_CSV"
declare -A BASELINES=(
  [hotpotqa]="$PROJ_ROOT/data/eval_results/hotpotqa/sft_dpo_20260610_145349/merged.jsonl"
  [2wikimultihopqa]="$PROJ_ROOT/data/eval_results/2wikimultihopqa/sft_dpo_20260610_155526/merged.jsonl"
  [musique]="$PROJ_ROOT/data/eval_results/musique/sft_dpo_20260610_142115/merged.jsonl"
)
declare -A REASONRAG_F1=(
  [hotpotqa]="0.489"
  [2wikimultihopqa]="0.372"
  [musique]="0.321"
)
ROLL_PID=""

cleanup_rollout() {
  if [[ -z "${ROLL_PID:-}" ]] || ! kill -0 "$ROLL_PID" 2>/dev/null; then
    ROLL_PID=""
    return
  fi

  local pgid
  pgid="$(ps -o pgid= -p "$ROLL_PID" | tr -d ' ' || true)"
  echo "[cleanup] stopping rollout pid=$ROLL_PID pgid=${pgid:-unknown}"
  if [[ -n "${pgid:-}" ]]; then
    kill -TERM "-$pgid" 2>/dev/null || true
    sleep 15
    kill -KILL "-$pgid" 2>/dev/null || true
  else
    kill -TERM "$ROLL_PID" 2>/dev/null || true
  fi
  ROLL_PID=""
}
trap cleanup_rollout EXIT

require_file() {
  [[ -f "$1" ]] || {
    echo "ERROR: missing file: $1" >&2
    exit 2
  }
}

checkpoint_path() {
  printf '%s/checkpoint-%s\n' "$RUN_ROOT" "$1"
}

start_rollout() {
  local step="$1"
  local ckpt
  local log_dir
  local started

  ckpt="$(checkpoint_path "$step")"
  require_file "$ckpt/adapter_config.json"
  require_file "$ckpt/adapter_model.safetensors"

  log_dir="$OUT_ROOT/rollout_logs/checkpoint-${step}"
  mkdir -p "$log_dir"
  echo "[rollout] checkpoint=$step gpu=$ROLLOUT_GPU port=$ROLLOUT_PORT"
  setsid env \
    SAPR_RAG_ROOT="$PROJ_ROOT" \
    SWIFT_ROOT="$SWIFT_ROOT" \
    DEVICE_BACKEND=cuda \
    ROLLOUT_DEVICES="$ROLLOUT_GPU" \
    PORT="$ROLLOUT_PORT" \
    SAPR_RETRIEVAL_URL="$RETRIEVAL_URL" \
    SAPR_TOP_K=3 \
    SAPR_ENABLE_EVIDENCE_AGENT=true \
    SAPR_EVIDENCE_MAX_TOKENS=128 \
    MULTI_TURN_SCHEDULER=sapr_rag_scheduler \
    INIT_ADAPTER=none \
    ADAPTER_PATH="$ckpt" \
    VLLM_MAX_MODEL_LEN=8192 \
    VLLM_GPU_MEM_UTIL=0.85 \
    bash "$GRPO_DIR/run_rollout_opsd.sh" \
    >"$log_dir/rollout.log" 2>&1 < /dev/null &
  ROLL_PID="$!"
  echo "$ROLL_PID" > "$log_dir/rollout.pid"

  started="$(date +%s)"
  while true; do
    if curl -fsS --max-time 5 "http://127.0.0.1:${ROLLOUT_PORT}/health/" \
        >"$log_dir/health.json"; then
      echo "[rollout] checkpoint=$step ready"
      return
    fi
    if ! kill -0 "$ROLL_PID" 2>/dev/null; then
      echo "ERROR: rollout died for checkpoint-$step" >&2
      tail -n 120 "$log_dir/rollout.log" >&2 || true
      exit 3
    fi
    if (( $(date +%s) - started >= 1500 )); then
      echo "ERROR: rollout health timeout for checkpoint-$step" >&2
      tail -n 120 "$log_dir/rollout.log" >&2 || true
      exit 4
    fi
    sleep 10
  done
}

eval_one() {
  local step="$1"
  local dataset="$2"
  local input="$3"
  local stage="$4"
  local out_dir="$OUT_ROOT/$stage/checkpoint-${step}/$dataset"
  local raw="$out_dir/results.jsonl"
  local metrics="$out_dir/metrics.json"
  local expected
  local rows
  local unique_ids

  mkdir -p "$out_dir/logs"
  expected="$(wc -l < "$input")"
  echo "[eval] stage=$stage checkpoint=$step dataset=$dataset rows=$expected"

  python "$SCRIPT_DIR/run_direct_rollout_eval.py" \
    --input_jsonl "$input" \
    --output_jsonl "$raw" \
    --rollout_url "http://127.0.0.1:${ROLLOUT_PORT}" \
    --batch_size "$BATCH_SIZE" \
    --max_turns 6 \
    --max_tokens "$MAX_TOKENS" \
    2>&1 | tee "$out_dir/logs/eval.log"

  rows="$(wc -l < "$raw")"
  unique_ids="$(jq -r '.id' "$raw" | sort -u | wc -l)"
  if [[ "$rows" -ne "$expected" || "$unique_ids" -ne "$expected" ]]; then
    echo "ERROR: result integrity failed for $dataset: rows=$rows unique=$unique_ids expected=$expected" >&2
    exit 5
  fi

  python "$SCRIPT_DIR/score.py" \
    --input "$raw" \
    --output "$metrics" \
    2>&1 | tee "$out_dir/logs/score.log"
}

prepare_subsets() {
  local dataset
  local input
  local subset_dir="$OUT_ROOT/selection_inputs"

  mkdir -p "$subset_dir"
  for dataset in "${DATASETS[@]}"; do
    input="$PROJ_ROOT/data/eval/$dataset/dev.jsonl"
    require_file "$input"
    head -n "$N_SUBSET" "$input" >"$subset_dir/${dataset}_first${N_SUBSET}.jsonl"
  done
}

write_sweep_summary() {
  local summary="$OUT_ROOT/selection_summary.tsv"
  local ranking="$OUT_ROOT/selection_ranking.tsv"
  local step
  local dataset
  local metrics

  printf 'step\tdataset\tn_total\tem\tf1\tcover_em\tmax_turns_rate\tempty_evidence_rate\n' >"$summary"
  IFS=',' read -r -a steps <<<"$CHECKPOINT_STEPS"
  for step in "${steps[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      metrics="$OUT_ROOT/selection/checkpoint-${step}/$dataset/metrics.json"
      require_file "$metrics"
      jq -r --arg step "$step" --arg dataset "$dataset" \
        '[$step,$dataset,.n_total,.em,.f1,.cover_em,.max_turns_rate,.empty_evidence_rate] | @tsv' \
        "$metrics" >>"$summary"
    done
  done

  awk -F '\t' '
    NR > 1 {
      n[$1] += 1
      em[$1] += $4
      f1[$1] += $5
      cover[$1] += $6
    }
    END {
      print "step\tmacro_f1\tmacro_em\tmacro_cover_em"
      for (step in n) {
        printf "%s\t%.8f\t%.8f\t%.8f\n", step, f1[step]/n[step], em[step]/n[step], cover[step]/n[step]
      }
    }
  ' "$summary" >"$ranking.unsorted"
  {
    head -n 1 "$ranking.unsorted"
    tail -n +2 "$ranking.unsorted" | sort -t $'\t' -k2,2nr -k3,3nr -k4,4nr
  } >"$ranking"
  rm -f "$ranking.unsorted"

  tail -n +2 "$ranking" | head -n 1 | cut -f1 >"$OUT_ROOT/best_checkpoint_step.txt"
  echo "[sweep] best checkpoint=$(cat "$OUT_ROOT/best_checkpoint_step.txt")"
  cat "$ranking"
}

run_sweep() {
  local step
  local dataset
  local input

  prepare_subsets
  IFS=',' read -r -a steps <<<"$CHECKPOINT_STEPS"
  for step in "${steps[@]}"; do
    start_rollout "$step"
    for dataset in "${DATASETS[@]}"; do
      input="$OUT_ROOT/selection_inputs/${dataset}_first${N_SUBSET}.jsonl"
      eval_one "$step" "$dataset" "$input" selection
    done
    cleanup_rollout
  done
  write_sweep_summary
}

run_full() {
  local step="$CHECKPOINT_STEP"
  local dataset
  local input
  local candidate
  local bootstrap_output
  local -a reasonrag_args

  if [[ -z "$step" ]]; then
    require_file "$OUT_ROOT/best_checkpoint_step.txt"
    step="$(cat "$OUT_ROOT/best_checkpoint_step.txt")"
  fi

  start_rollout "$step"
  for dataset in "${DATASETS[@]}"; do
    input="$PROJ_ROOT/data/eval/$dataset/dev.jsonl"
    require_file "$input"
    eval_one "$step" "$dataset" "$input" full

    candidate="$OUT_ROOT/full/checkpoint-${step}/$dataset/results.jsonl"
    bootstrap_output="$OUT_ROOT/full/checkpoint-${step}/$dataset/paired_bootstrap_vs_sft_dpo.json"
    require_file "${BASELINES[$dataset]}"
    reasonrag_args=(--reasonrag_f1 "${REASONRAG_F1[$dataset]}")
    if [[ "$dataset" == "hotpotqa" ]]; then
      reasonrag_args+=(--reasonrag_em 0.384)
    fi
    python "$SCRIPT_DIR/paired_bootstrap.py" \
      --candidate "$candidate" \
      --baseline "${BASELINES[$dataset]}" \
      --output "$bootstrap_output" \
      --samples "$BOOTSTRAP_SAMPLES" \
      --seed 20260812 \
      "${reasonrag_args[@]}" \
      2>&1 | tee "$OUT_ROOT/full/checkpoint-${step}/$dataset/logs/bootstrap.log"
  done
  cleanup_rollout
  echo "$step" >"$OUT_ROOT/full_checkpoint_step.txt"
}

mkdir -p "$OUT_ROOT"
CONFIG_FILE="$OUT_ROOT/config_${MODE}.txt"
{
  echo "mode=$MODE"
  echo "run_root=$RUN_ROOT"
  echo "checkpoint_steps=$CHECKPOINT_STEPS"
  echo "checkpoint_step=$CHECKPOINT_STEP"
  echo "n_subset=$N_SUBSET"
  echo "rollout_gpu=$ROLLOUT_GPU"
  echo "rollout_port=$ROLLOUT_PORT"
  echo "retrieval_url=$RETRIEVAL_URL"
  echo "bootstrap_samples=$BOOTSTRAP_SAMPLES"
  echo "datasets=$DATASETS_CSV"
  echo "evidence_agent=true"
  echo "top_k=3"
  echo "started_at=$(date -Is)"
} >"$CONFIG_FILE"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[dry-run] configuration validated; no rollout or evaluation started"
  cat "$CONFIG_FILE"
  exit 0
fi

case "$MODE" in
  sweep)
    run_sweep
    ;;
  full)
    [[ -n "$CHECKPOINT_STEP" || -f "$OUT_ROOT/best_checkpoint_step.txt" ]] || {
      echo "ERROR: MODE=full requires CHECKPOINT_STEP or an existing best_checkpoint_step.txt in OUT_ROOT" >&2
      exit 2
    }
    run_full
    ;;
  *)
    echo "ERROR: MODE must be sweep or full, got: $MODE" >&2
    exit 2
    ;;
esac

echo "completed_at=$(date -Is)" >>"$CONFIG_FILE"
echo "[done] output=$OUT_ROOT"
