#!/usr/bin/env bash
# Evaluate the frozen 14B teacher with the same retrieval/evidence pipeline as the student.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
GRPO_DIR="$PROJ_ROOT/03_sapr_rag/scripts/grpo"
EVAL_DIR="$PROJ_ROOT/03_sapr_rag/scripts/eval"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

TEACHER_MODEL="${TEACHER_MODEL:-$PROJ_ROOT/03_sapr_rag/models/Qwen2.5-14B-Instruct}"
TEACHER_ADAPTER="${TEACHER_ADAPTER:-}"
TEACHER_GPU="${TEACHER_GPU:-1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8041}"
RETRIEVAL_GPU="${RETRIEVAL_GPU:-0}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8100}"
N_SUBSET="${N_SUBSET:-200}"
RUN_NAME="${RUN_NAME:-teacher_14b_ceiling_200}"
OUT_ROOT="${OUT_ROOT:-$PROJ_ROOT/data/eval_results/$RUN_NAME}"
ROLLOUT_PID=""

declare -A BASELINES=(
    [hotpotqa]="$PROJ_ROOT/data/eval_results/hotpotqa/20260608_175824/merged.jsonl"
    [2wikimultihopqa]="$PROJ_ROOT/data/eval_results/2wikimultihopqa/sft_20260609_232951/merged.jsonl"
    [musique]="$PROJ_ROOT/data/eval_results/musique/sft_20260610_112233/merged.jsonl"
)
DATASETS=(hotpotqa 2wikimultihopqa musique)

cleanup() {
    if [[ -n "$ROLLOUT_PID" ]] && kill -0 "$ROLLOUT_PID" 2>/dev/null; then
        local pgid
        pgid="$(ps -o pgid= -p "$ROLLOUT_PID" | tr -d ' ' || true)"
        [[ -n "$pgid" ]] && kill -TERM "-$pgid" 2>/dev/null || kill -TERM "$ROLLOUT_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

mkdir -p "$OUT_ROOT/inputs" "$OUT_ROOT/logs"
[[ -d "$TEACHER_MODEL" ]] || { echo "ERROR: missing teacher model: $TEACHER_MODEL" >&2; exit 2; }
for dataset in "${DATASETS[@]}"; do
    input="$PROJ_ROOT/data/eval/$dataset/dev.jsonl"
    baseline="${BASELINES[$dataset]}"
    [[ -f "$input" ]] || { echo "ERROR: missing eval input: $input" >&2; exit 2; }
    [[ -f "$baseline" ]] || { echo "ERROR: missing SFT baseline: $baseline" >&2; exit 2; }
    head -n "$N_SUBSET" "$input" >"$OUT_ROOT/inputs/${dataset}.jsonl"
done

env RETRIEVAL_GPU="$RETRIEVAL_GPU" RETRIEVAL_PORT="$RETRIEVAL_PORT" \
    bash "$GRPO_DIR/retrieval_service.sh" start
env RETRIEVAL_GPU="$RETRIEVAL_GPU" RETRIEVAL_PORT="$RETRIEVAL_PORT" WAIT_TIMEOUT=1800 \
    bash "$GRPO_DIR/retrieval_service.sh" wait

setsid env \
    DEVICE_BACKEND=cuda \
    BASE_MODEL="$TEACHER_MODEL" \
    INIT_ADAPTER="${TEACHER_ADAPTER:-none}" \
    ROLLOUT_DEVICES="$TEACHER_GPU" \
    PORT="$ROLLOUT_PORT" \
    SAPR_RETRIEVAL_URL="http://127.0.0.1:${RETRIEVAL_PORT}" \
    SAPR_TOP_K=3 \
    SAPR_ENABLE_EVIDENCE_AGENT=true \
    MULTI_TURN_SCHEDULER=sapr_rag_scheduler \
    VLLM_MAX_MODEL_LEN=8192 \
    VLLM_GPU_MEM_UTIL=0.88 \
    bash "$GRPO_DIR/run_rollout_opsd.sh" \
    >"$OUT_ROOT/logs/teacher_rollout.log" 2>&1 < /dev/null &
ROLLOUT_PID="$!"
echo "$ROLLOUT_PID" >"$OUT_ROOT/logs/teacher_rollout.pid"

started="$(date +%s)"
until curl -fsS --max-time 5 "http://127.0.0.1:${ROLLOUT_PORT}/health/" \
    >"$OUT_ROOT/logs/teacher_health.json" 2>/dev/null; do
    if ! kill -0 "$ROLLOUT_PID" 2>/dev/null; then
        echo "ERROR: teacher rollout exited before becoming ready" >&2
        tail -n 120 "$OUT_ROOT/logs/teacher_rollout.log" >&2 || true
        exit 3
    fi
    if (( $(date +%s) - started >= 1200 )); then
        echo "ERROR: teacher rollout health timeout" >&2
        tail -n 120 "$OUT_ROOT/logs/teacher_rollout.log" >&2 || true
        exit 4
    fi
    sleep 10
done

teacher_args=()
baseline_args=()
for dataset in "${DATASETS[@]}"; do
    out_dir="$OUT_ROOT/$dataset"
    mkdir -p "$out_dir"
    python "$EVAL_DIR/run_direct_rollout_eval.py" \
        --input_jsonl "$OUT_ROOT/inputs/${dataset}.jsonl" \
        --output_jsonl "$out_dir/results.jsonl" \
        --rollout_url "http://127.0.0.1:${ROLLOUT_PORT}" \
        --batch_size 32 \
        --max_turns 6 \
        --max_tokens 512 \
        2>&1 | tee "$out_dir/eval.log"
    python "$EVAL_DIR/score.py" \
        --input "$out_dir/results.jsonl" \
        --output "$out_dir/metrics.json" \
        2>&1 | tee "$out_dir/score.log"
    teacher_args+=(--teacher "$dataset=$out_dir/results.jsonl")
    baseline_args+=(--baseline "$dataset=${BASELINES[$dataset]}")
done

python "$SCRIPT_DIR/check_teacher_ceiling.py" \
    "${teacher_args[@]}" \
    "${baseline_args[@]}" \
    --min-macro-f1-delta 0.05 \
    --output "$OUT_ROOT/ceiling_gate.json"
