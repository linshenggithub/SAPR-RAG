#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
: "${WORKER_RUN_ROOT:?Set WORKER_RUN_ROOT to the full-parameter GRPO run directory}"
INPUT="${INPUT:-${PROJ_ROOT}/data/eval/hotpotqa/dev.jsonl}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8100}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PROJ_ROOT}/data/eval_results/hotpotqa/full_grpo_sweep_${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EVAL_SCRIPT="${PROJ_ROOT}/03_sapr_rag/scripts/eval/agent_infer.py"
MERGE_SCRIPT="${PROJ_ROOT}/03_sapr_rag/scripts/eval/merge_shards.py"
SCORE_SCRIPT="${PROJ_ROOT}/03_sapr_rag/scripts/eval/score.py"

LABELS=(ckpt2500 ckpt3000 ckpt3660)
CHECKPOINTS=(
  "${WORKER_RUN_ROOT}/checkpoint-2500"
  "${WORKER_RUN_ROOT}/checkpoint-3000"
  "${WORKER_RUN_ROOT}/checkpoint-3660"
)
GPU_GROUPS=("1 2" "3 4" "5 6 7")

mkdir -p "${OUT_ROOT}"
STATUS_FILE="${OUT_ROOT}/status.txt"
exec > >(tee -a "${OUT_ROOT}/launcher.log") 2>&1

echo "run_tag=${RUN_TAG}" | tee "${STATUS_FILE}"
echo "input=${INPUT}" | tee -a "${STATUS_FILE}"
echo "retrieval_url=${RETRIEVAL_URL}" | tee -a "${STATUS_FILE}"
echo "started_at=$(date -Iseconds)" | tee -a "${STATUS_FILE}"

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: missing input ${INPUT}" | tee -a "${STATUS_FILE}"
  exit 1
fi

for ckpt in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "${ckpt}/model.safetensors.index.json" ]]; then
    echo "ERROR: incomplete checkpoint ${ckpt}" | tee -a "${STATUS_FILE}"
    exit 1
  fi
done

PIDS=()
PID_LABELS=()
PID_SHARDS=()
PID_GPUS=()

for idx in "${!LABELS[@]}"; do
  label="${LABELS[$idx]}"
  ckpt="${CHECKPOINTS[$idx]}"
  read -r -a gpus <<< "${GPU_GROUPS[$idx]}"
  num_shards="${#gpus[@]}"
  out_dir="${OUT_ROOT}/${label}"
  mkdir -p "${out_dir}/logs"

  {
    echo "label=${label}"
    echo "checkpoint=${ckpt}"
    echo "gpus=${GPU_GROUPS[$idx]}"
    echo "num_shards=${num_shards}"
  } > "${out_dir}/config.txt"

  for shard_id in "${!gpus[@]}"; do
    gpu="${gpus[$shard_id]}"
    out_file="${out_dir}/shard_${shard_id}.jsonl"
    log_file="${out_dir}/logs/shard_${shard_id}.log"
    cache_root="/tmp/sapr_eval_cache/${RUN_TAG}/${label}/gpu${gpu}"
    mkdir -p "${cache_root}/vllm" "${cache_root}/inductor"

    echo "[launch] ${label} shard=${shard_id}/${num_shards} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS=12 \
    OPENBLAS_NUM_THREADS=12 \
    MKL_NUM_THREADS=12 \
    VLLM_CACHE_ROOT="${cache_root}/vllm" \
    TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor" \
      "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
        --backend vllm \
        --base_model "${ckpt}" \
        --no_lora \
        --input_jsonl "${INPUT}" \
        --output_jsonl "${out_file}" \
        --shard_id "${shard_id}" \
        --num_shards "${num_shards}" \
        --cohort_size 16 \
        --retrieval_url "${RETRIEVAL_URL}" \
        --top_k 3 \
        --max_turns 6 \
        --max_tokens 512 \
        --evidence_max_tokens 128 \
        --temperature 0 \
        --top_p 1 \
        --gpu_memory_utilization 0.20 \
        --max_model_len 8192 \
        --vllm_dtype bfloat16 \
        --vllm_max_num_seqs 16 \
        --vllm_max_num_batched_tokens 16384 \
        --vllm_cuda_graph \
        --vllm_disable_custom_all_reduce \
        > "${log_file}" 2>&1 &

    PIDS+=("$!")
    PID_LABELS+=("${label}")
    PID_SHARDS+=("${shard_id}")
    PID_GPUS+=("${gpu}")
    sleep 3
  done
done

echo "launch_pids=${PIDS[*]}" | tee -a "${STATUS_FILE}"

FAIL=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  if wait "${pid}"; then
    echo "[done] ${PID_LABELS[$idx]} shard=${PID_SHARDS[$idx]} gpu=${PID_GPUS[$idx]}"
  else
    rc=$?
    echo "[failed] ${PID_LABELS[$idx]} shard=${PID_SHARDS[$idx]} gpu=${PID_GPUS[$idx]} rc=${rc}"
    FAIL=$((FAIL + 1))
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "result=failed failed_shards=${FAIL}" | tee -a "${STATUS_FILE}"
  exit 1
fi

EXPECTED="$(wc -l < "${INPUT}")"
for label in "${LABELS[@]}"; do
  out_dir="${OUT_ROOT}/${label}"
  echo "[merge] ${label}"
  "${PYTHON_BIN}" "${MERGE_SCRIPT}" \
    --shard_dir "${out_dir}" \
    --output "${out_dir}/merged.jsonl" \
    > "${out_dir}/merge.log" 2>&1

  "${PYTHON_BIN}" - "${out_dir}/merged.jsonl" "${EXPECTED}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
rows = [json.loads(line) for line in path.open() if line.strip()]
ids = [row.get("id") for row in rows]
exceptions = sum(str(row.get("error", "")).startswith("exception:") for row in rows)
if len(rows) != expected:
    raise SystemExit(f"row count mismatch: {len(rows)} != {expected}")
if len(set(ids)) != expected:
    raise SystemExit(f"unique id mismatch: {len(set(ids))} != {expected}")
if exceptions:
    raise SystemExit(f"cohort exceptions found: {exceptions}")
print(f"validated rows={len(rows)} unique_ids={len(set(ids))} exceptions={exceptions}")
PY

  "${PYTHON_BIN}" "${SCORE_SCRIPT}" \
    --input "${out_dir}/merged.jsonl" \
    --output "${out_dir}/metrics.json" \
    > "${out_dir}/score.log" 2>&1
  cat "${out_dir}/metrics.json"
done

echo "result=success" | tee -a "${STATUS_FILE}"
echo "completed_at=$(date -Iseconds)" | tee -a "${STATUS_FILE}"
echo "out_root=${OUT_ROOT}" | tee -a "${STATUS_FILE}"
