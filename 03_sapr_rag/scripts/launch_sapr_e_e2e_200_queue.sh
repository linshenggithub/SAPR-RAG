#!/usr/bin/env bash
set -euo pipefail

# 路径配置：默认值与 config/paths.py 保持一致，可通过 SAPR_* 环境变量覆盖
RUN_ID="${RUN_ID:-20260531_sapr_e_e2e_200_maxtok256}"
REASONRAG_ROOT="${SAPR_REASONRAG_ROOT:-${REASONRAG_ROOT:-/home/mayi/ReasonRAG}}"
CONDA_BIN="${SAPR_CONDA_BIN:-${CONDA_BIN:-/home/mayi/miniconda3/bin/conda}}"
FREE_MEM_THRESHOLD_MB="${FREE_MEM_THRESHOLD_MB:-1000}"
SLEEP_SEC="${SLEEP_SEC:-120}"

# REPO_ROOT 由本脚本所在路径派生（即 SAPR-RAG 仓库根）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_DIR="${REPO_ROOT}/04_experiments/logs/${RUN_ID}"
mkdir -p "${BASE_DIR}/baseline" "${BASE_DIR}/sapr_e"

pick_free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v th="${FREE_MEM_THRESHOLD_MB}" '
        {
          gsub(/ /, "", $1);
          gsub(/ /, "", $2);
          if ($2 < th) {
            print $1;
            exit;
          }
        }'
}

wait_for_gpu() {
  local gpu
  while true; do
    gpu="$(pick_free_gpu || true)"
    if [[ -n "${gpu}" ]]; then
      echo "${gpu}"
      return 0
    fi
    echo "[$(date -Is)] no free GPU below ${FREE_MEM_THRESHOLD_MB} MiB; sleeping ${SLEEP_SEC}s" >&2
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >&2
    sleep "${SLEEP_SEC}"
  done
}

run_mode() {
  local mode="$1"
  local gpu="$2"
  local log="${BASE_DIR}/${mode}/run.log"
  echo "[$(date -Is)] starting mode=${mode} on gpu=${gpu}; log=${log}"
  cd "${REASONRAG_ROOT}"
  eval "$("${CONDA_BIN}" shell.bash hook)"
  conda activate reasonrag
  # 透传给 Python 脚本，让 config/paths.py 也读取到这些值
  export SAPR_REASONRAG_ROOT="${REASONRAG_ROOT}"
  python "${SCRIPT_DIR}/run_sapr_e_e2e.py" \
    --num_examples 200 \
    --mode "${mode}" \
    --run_id "${RUN_ID}" \
    --max_tokens 256 \
    --gpu_id "${gpu}" \
    > "${log}" 2>&1
  echo "[$(date -Is)] finished mode=${mode} on gpu=${gpu}"
}

echo "[$(date -Is)] SAPR-E e2e 200 queue started"
echo "RUN_ID=${RUN_ID}"
echo "REASONRAG_ROOT=${REASONRAG_ROOT}"
echo "REPO_ROOT=${REPO_ROOT}"

gpu="$(wait_for_gpu)"
run_mode baseline "${gpu}"

gpu="$(wait_for_gpu)"
run_mode sapr_e "${gpu}"

echo "[$(date -Is)] SAPR-E e2e 200 queue completed"
