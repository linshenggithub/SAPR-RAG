#!/usr/bin/env bash
# 8 卡 H20 数据并行推理 SAPR-RAG dev 集
# 用法：
#   bash run_dp8.sh hotpotqa                 # SFT LoRA（默认）
#   NO_LORA=1 bash run_dp8.sh hotpotqa       # zero-shot 纯 backbone 对照
#   LORA_PATH=/path/to/sft_dpo/checkpoint-395 SETTING_NAME=sft_dpo bash run_dp8.sh hotpotqa  # 用自定义 LoRA
#   NUM_SHARDS=4 bash run_dp8.sh hotpotqa    # 改并发档（标定 FAISS 争抢）
#   RESUME_DIR=/path/to/zeroshot_xxx NO_LORA=1 bash run_dp8.sh hotpotqa  # 断点续跑（worker 被杀后）
set -euo pipefail

DATASET="${1:?usage: run_dp8.sh <dataset>}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GMU="${GMU:-0.85}"            # 每卡 vllm 显存利用率
STAGGER_SEC="${STAGGER_SEC:-5}"    # 错峰启动间隔（秒），默认 5s 缩短"半空窗"避免被平台回收
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"  # vllm context 上限，砍半提升 KV cache 利用率
COHORT_SIZE="${COHORT_SIZE:-64}"   # 批处理 cohort 大小；小 cohort 让 GPU/检索高频交替，避免长空窗被平台回收
NO_LORA="${NO_LORA:-0}"            # 1 = zero-shot 纯 backbone（不挂 LoRA）
LORA_PATH="${LORA_PATH:-}"         # 自定义 LoRA 路径（如 sft_dpo/checkpoint-395）；为空时用脚本默认 SFT
SETTING_NAME="${SETTING_NAME:-}"   # 自定义 setting 名（如 sft_dpo），影响输出目录命名

# zero-shot 与 sft 结果分目录存，避免混淆
if [[ "${NO_LORA}" == "1" ]]; then
  SETTING="zeroshot"
  LORA_FLAG="--no_lora"
elif [[ -n "${LORA_PATH}" ]]; then
  SETTING="${SETTING_NAME:-custom}"
  LORA_FLAG="--lora_path ${LORA_PATH}"
else
  SETTING="sft"
  LORA_FLAG=""
fi

# FAISS CPU 线程：避免 N 进程各自抢满所有核导致超额订阅（上次 8×174 线程抢 174 核把检索拖垮）。
# 每进程分到 总核数/进程数，留 ~10% 余量给 vllm/系统。
NCPU="$(nproc)"
OMP_PER_PROC="${OMP_PER_PROC:-$(( NCPU * 9 / 10 / NUM_SHARDS ))}"
[[ "${OMP_PER_PROC}" -lt 1 ]] && OMP_PER_PROC=1

PROJ_ROOT="/mlx_devbox/users/mayi.summer/playground/SAPR-RAG"
SCRIPT="${PROJ_ROOT}/03_sapr_rag/scripts/eval/agent_infer.py"
INPUT="${PROJ_ROOT}/data/eval/${DATASET}/dev.jsonl"
RESUME_DIR="${RESUME_DIR:-}"
if [[ -n "${RESUME_DIR}" ]]; then
  OUT_DIR="${RESUME_DIR}"
  RESUME_FLAG="--resume"
  echo "[run_dp8] RESUME mode -> reuse ${OUT_DIR}"
else
  OUT_DIR="${PROJ_ROOT}/data/eval_results/${DATASET}/${SETTING}_$(date +%Y%m%d_%H%M%S)"
  RESUME_FLAG=""
fi
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

if [[ ! -f "${INPUT}" ]]; then
  echo "[ERR] dev jsonl 不存在: ${INPUT}"
  echo "      先跑：python 03_sapr_rag/scripts/eval/fetch_flashrag_dev.py"
  exit 1
fi

echo "[run_dp8] dataset=${DATASET} setting=${SETTING}"
echo "[run_dp8] num_shards=${NUM_SHARDS} gmu=${GMU} stagger=${STAGGER_SEC}s max_model_len=${MAX_MODEL_LEN}"
echo "[run_dp8] cohort_size=${COHORT_SIZE} cpu_cores=${NCPU} omp_per_proc=${OMP_PER_PROC} (${NUM_SHARDS}x${OMP_PER_PROC}=$(( NUM_SHARDS * OMP_PER_PROC )) <= ${NCPU})"
echo "[run_dp8] input=${INPUT}"
echo "[run_dp8] out_dir=${OUT_DIR}"
n_total=$(wc -l < "${INPUT}")
echo "[run_dp8] total questions=${n_total}"

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  out_file="${OUT_DIR}/shard_${i}.jsonl"
  log_file="${LOG_DIR}/shard_${i}.log"
  [[ -n "${RESUME_FLAG}" ]] && log_file="${LOG_DIR}/shard_${i}.resume.log"
  echo "[run_dp8] launching shard ${i} on GPU ${i} -> ${out_file}"
  CUDA_VISIBLE_DEVICES=${i} \
  OMP_NUM_THREADS=${OMP_PER_PROC} \
  OPENBLAS_NUM_THREADS=${OMP_PER_PROC} \
  MKL_NUM_THREADS=${OMP_PER_PROC} \
    python "${SCRIPT}" \
      --backend vllm \
      --input_jsonl "${INPUT}" \
      --output_jsonl "${out_file}" \
      --shard_id ${i} \
      --num_shards ${NUM_SHARDS} \
      --cohort_size ${COHORT_SIZE} \
      --gpu_memory_utilization ${GMU} \
      --max_model_len ${MAX_MODEL_LEN} \
      ${LORA_FLAG} \
      ${RESUME_FLAG} \
      > "${log_file}" 2>&1 &
  PIDS+=($!)
  if [[ $i -lt $((NUM_SHARDS - 1)) ]]; then
    sleep ${STAGGER_SEC}
  fi
done

echo "[run_dp8] all shards launched, pids=${PIDS[*]}"
echo "[run_dp8] waiting ..."
FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=$((FAIL + 1))
    echo "[run_dp8] pid ${pid} FAILED"
  fi
done

if [[ ${FAIL} -gt 0 ]]; then
  echo "[run_dp8] ${FAIL} shard(s) failed; check ${LOG_DIR}/"
  exit 1
fi

echo "[run_dp8] all shards done; merging ..."
python "${PROJ_ROOT}/03_sapr_rag/scripts/eval/merge_shards.py" \
  --shard_dir "${OUT_DIR}" \
  --output "${OUT_DIR}/merged.jsonl"

echo "[run_dp8] scoring ..."
python "${PROJ_ROOT}/03_sapr_rag/scripts/eval/score.py" \
  --input "${OUT_DIR}/merged.jsonl" \
  --output "${OUT_DIR}/metrics.json"

echo "[run_dp8] done -> ${OUT_DIR}"
