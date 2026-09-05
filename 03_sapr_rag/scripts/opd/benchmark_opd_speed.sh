#!/usr/bin/env bash
# Compare throughput-preserving OPD configurations on one 8-GPU worker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
STAMP="${BENCHMARK_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PREFIX="${BENCHMARK_PREFIX:-opd_speed_${STAMP}}"
MAX_STEPS="${MAX_STEPS:-6}"
FIRST_RUN="${PREFIX}_baseline"
FIRST_LOG_DIR="$SCRIPT_DIR/logs/$FIRST_RUN"
RESULT_DIR="$PROJ_ROOT/data/eval_results/$PREFIX"
SUMMARY="$RESULT_DIR/summary.tsv"
GPU_METRICS="$RESULT_DIR/gpu_metrics.csv"
CURRENT_VARIANT="$RESULT_DIR/current_variant"
TELEMETRY_PID=""

mkdir -p "$RESULT_DIR"
printf 'variant\tpbs\tgas\tspg\tgradient_checkpointing\tmax_steps\n' >"$SUMMARY"
printf 'idle\n' >"$CURRENT_VARIANT"
printf 'timestamp,variant,index,utilization_gpu_pct,memory_used_mib,memory_total_mib\n' >"$GPU_METRICS"

cleanup_services() {
    if [[ -n "$TELEMETRY_PID" ]]; then
        kill "$TELEMETRY_PID" 2>/dev/null || true
    fi
    if [[ -f "$FIRST_LOG_DIR/rollout.pid" ]]; then
        rollout_pid="$(cat "$FIRST_LOG_DIR/rollout.pid")"
        if kill -0 "$rollout_pid" 2>/dev/null; then
            rollout_pgid="$(ps -o pgid= -p "$rollout_pid" | tr -d ' ' || true)"
            if [[ -n "$rollout_pgid" ]]; then
                kill -TERM "-$rollout_pgid" 2>/dev/null || true
            fi
        fi
    fi
    RUN_NAME="$FIRST_RUN" LOG_DIR="$FIRST_LOG_DIR" \
        bash "$SCRIPT_DIR/teacher_service.sh" stop >/dev/null 2>&1 || true
}
trap cleanup_services EXIT

(
    while true; do
        timestamp="$(date -Is)"
        variant="$(cat "$CURRENT_VARIANT")"
        nvidia-smi \
            --query-gpu=index,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits |
            sed "s/^/$timestamp,$variant,/"
        sleep 5
    done
) >>"$GPU_METRICS" 2>&1 &
TELEMETRY_PID="$!"

record_variant() {
    local variant="$1"
    local pbs="$2"
    local gas="$3"
    local spg="$4"
    local gradient_checkpointing="$5"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$variant" "$pbs" "$gas" "$spg" "$gradient_checkpointing" "$MAX_STEPS" >>"$SUMMARY"
}

run_direct_variant() {
    local variant="$1"
    local pbs="$2"
    local gas="$3"
    local spg="$4"
    local gradient_checkpointing="$5"
    local run_name="${PREFIX}_${variant}"
    local log_dir="$SCRIPT_DIR/logs/$run_name"

    mkdir -p "$log_dir"
    record_variant "$variant" "$pbs" "$gas" "$spg" "$gradient_checkpointing"
    printf '%s\n' "$variant" >"$CURRENT_VARIANT"
    env \
        RUN_NAME="$run_name" \
        LOG_DIR="$log_dir" \
        OUTPUT_DIR="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/opd/$run_name" \
        TEACHER_URL=http://127.0.0.1:8040 \
        ROLLOUT_PORT=8030 \
        OPD_MODE=pure \
        TEACHER_SEQUENCE_GATE=failed_em \
        TEACHER_KL_COEF=0.01 \
        MAX_STEPS="$MAX_STEPS" \
        PER_DEVICE_BATCH_SIZE="$pbs" \
        GRADIENT_ACCUMULATION_STEPS="$gas" \
        STEPS_PER_GENERATION="$spg" \
        NUM_GENERATIONS=5 \
        GRADIENT_CHECKPOINTING="$gradient_checkpointing" \
        PADDING_FREE=false \
        SAVE_STRATEGY=no \
        bash "$SCRIPT_DIR/run_sapr_opd.sh" \
        >"$log_dir/train.log" 2>&1
    printf 'idle\n' >"$CURRENT_VARIANT"
}

# Start shared retrieval/teacher/rollout services while running the baseline.
record_variant baseline 1 4 1 true
printf 'baseline\n' >"$CURRENT_VARIANT"
env \
    WORKER_ID="${WORKER_ID:-4220660}" \
    RUN_NAME="$FIRST_RUN" \
    MAX_STEPS="$MAX_STEPS" \
    PER_DEVICE_BATCH_SIZE=1 \
    GRADIENT_ACCUMULATION_STEPS=4 \
    STEPS_PER_GENERATION=1 \
    NUM_GENERATIONS=5 \
    GRADIENT_CHECKPOINTING=true \
    PADDING_FREE=false \
    SAVE_STRATEGY=no \
    KEEP_SERVICES=true \
    bash "$SCRIPT_DIR/run_opd_formal.sh"
printf 'idle\n' >"$CURRENT_VARIANT"

run_direct_variant spg2 1 4 2 true
run_direct_variant gc_off_spg1 1 4 1 false
run_direct_variant gc_off_spg2 1 4 2 false
run_direct_variant batch2_gc_off_spg2 2 2 2 false

python "$SCRIPT_DIR/summarize_opd_speed.py" \
    --result-dir "$RESULT_DIR" \
    --log-root "$SCRIPT_DIR/logs" \
    --prefix "$PREFIX"
echo "[benchmark] completed: $SUMMARY"
