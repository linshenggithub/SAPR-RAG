#!/usr/bin/env bash
# Formal 1,000-step SFT -> failed-only external-teacher OPD run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SOURCE="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_musique_train_multi_opsd.jsonl"
DATASET="${DATASET:-$PROJ_ROOT/data/grpo/hotpotqa_2wiki_musique_train_opd.jsonl}"
RUN_NAME="${RUN_NAME:-opd_sft_14b_failed_em_s1000_$(date +%Y%m%d_%H%M%S)}"
TEACHER_ADAPTER="${TEACHER_ADAPTER:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_14b/lora/sft_teacher/checkpoint-300}"
CEILING_REPORT="${CEILING_REPORT:-$PROJ_ROOT/data/eval_results/teacher_14b_sft_ceiling_50/ceiling_gate.json}"
SKIP_TEACHER_CEILING="${SKIP_TEACHER_CEILING:-false}"

if [[ "$SKIP_TEACHER_CEILING" != "true" ]]; then
    python - "$CEILING_REPORT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"teacher ceiling report is missing: {path}")
report = json.loads(path.read_text())
if not report.get("summary", {}).get("passed"):
    raise SystemExit(f"teacher ceiling gate did not pass: {path}")
print(f"[formal] teacher ceiling passed: {path}")
PY
fi

if [[ ! -f "$DATASET" ]]; then
    python "$SCRIPT_DIR/build_opd_dataset.py" \
        --input "$SOURCE" \
        --output "$DATASET" \
        --seed 42
fi

env \
    WORKER_ID="${WORKER_ID:-4220660}" \
    RUN_NAME="$RUN_NAME" \
    DATASET="$DATASET" \
    TEACHER_ADAPTER="$TEACHER_ADAPTER" \
    OPD_MODE=pure \
    TEACHER_SEQUENCE_GATE=failed_em \
    TEACHER_KL_COEF="${TEACHER_KL_COEF:-0.01}" \
    MAX_STEPS="${MAX_STEPS:-1000}" \
    SAVE_STEPS="${SAVE_STEPS:-250}" \
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-6}" \
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}" \
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}" \
    STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-2}" \
    NUM_GENERATIONS="${NUM_GENERATIONS:-5}" \
    MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}" \
    GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}" \
    PADDING_FREE="${PADDING_FREE:-false}" \
    "$@" \
    bash "$SCRIPT_DIR/launch_sapr_opd_worker.sh"
