#!/usr/bin/env bash
# SAPR-R v1 数据构造一键 launcher：step2 → step3 → step4 → step5 顺序执行。
#
# 设计:
#   - 每步失败立刻停止（set -e）；前一步产物未生成时显式报错，避免静默拼错
#   - 每步独立 log；都支持断点续跑，重跑同命令自动跳过已完成
#   - 路径全部走 config/env_*.sh + config/paths.py，不写死绝对路径
#   - 跑前必须 source 对应机器 env：source config/env_5090.sh
#
# 跑法（5090，全量 5k）:
#   source config/env_5090.sh
#   conda activate reasonrag
#   bash 03_sapr_rag/data/build_v1/launch_build_v1_data.sh
#
# 自定义环境变量:
#   RUN_NAME       输出子目录名，默认 v1_5k
#   N_SAMPLES      step2 抽样数，默认 5000
#   SEED           step2 随机种子，默认 42
#   TOP_K          step3 检索 top-K，默认 10
#   MAX_WORKERS_S2 step2 并发，默认 30
#   MAX_WORKERS_S4 step4 并发，默认 50
#   CHUNK_SIZE_S4  step4 每块大小，默认 2000
#   ALPHA_S5       step5 cls_label 权重，默认 0.7
#   NORM_MODE_S5   step5 retriever_score 归一化方式，默认 minmax
#   DEV_RATIO_S5   step5 dev 划分比例，默认 0.1
#   SKIP_STEP2     非空跳过 step2（用已有产物）
#   SKIP_STEP3     非空跳过 step3
#   SKIP_STEP4     非空跳过 step4
#   SKIP_STEP5     非空跳过 step5
#   LIMIT_DEBUG    若设置，所有 step 共用此 --limit-debug 值（smoke 用）
#
# 示例（小批 smoke）:
#   LIMIT_DEBUG=10 RUN_NAME=v1_smoke bash 03_sapr_rag/data/build_v1/launch_build_v1_data.sh

set -euo pipefail

# ── 仓内路径派生 ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ── 配置 ──
RUN_NAME="${RUN_NAME:-v1_5k}"
N_SAMPLES="${N_SAMPLES:-5000}"
SEED="${SEED:-42}"
TOP_K="${TOP_K:-10}"
MAX_WORKERS_S2="${MAX_WORKERS_S2:-30}"
MAX_WORKERS_S4="${MAX_WORKERS_S4:-50}"
CHUNK_SIZE_S4="${CHUNK_SIZE_S4:-2000}"

OUT_DIR="${REPO_ROOT}/03_sapr_rag/data/build_v1/out/${RUN_NAME}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# ── 必需环境变量自检 ──
: "${SAPR_HOTPOTQA_TRAIN_PATH:?source config/env_*.sh first (HOTPOTQA train jsonl missing)}"
: "${SAPR_BGE_INDEX_PATH:?source config/env_*.sh first (BGE FAISS index missing)}"
: "${SAPR_BGE_MODEL_PATH:?source config/env_*.sh first (BGE model dir missing)}"
: "${SAPR_WIKI_CORPUS_PATH:?source config/env_*.sh first (wiki corpus jsonl missing)}"

if [[ -z "${DEEPSEEK_API_KEY:-}" && -z "${DMXAPI_API_KEY:-}" ]]; then
  if [[ -f "${REPO_ROOT}/03_sapr_rag/.env" ]]; then
    echo "[info] DEEPSEEK_API_KEY / DMXAPI_API_KEY not in env; will rely on ${REPO_ROOT}/03_sapr_rag/.env"
  else
    echo "[error] DEEPSEEK_API_KEY / DMXAPI_API_KEY not set, and 03_sapr_rag/.env missing" >&2
    exit 1
  fi
fi

# ── 通用 LIMIT_DEBUG 透传 ──
DEBUG_FLAG=()
if [[ -n "${LIMIT_DEBUG:-}" ]]; then
  DEBUG_FLAG=(--limit-debug "${LIMIT_DEBUG}")
  echo "[info] LIMIT_DEBUG=${LIMIT_DEBUG} 模式：所有 step 仅跑前 ${LIMIT_DEBUG} 条样本/单元/task"
fi

PY="python"

echo "============================================================"
echo "SAPR-R v1 data build launcher"
echo "RUN_NAME       = ${RUN_NAME}"
echo "OUT_DIR        = ${OUT_DIR}"
echo "N_SAMPLES      = ${N_SAMPLES}"
echo "SEED           = ${SEED}"
echo "TOP_K          = ${TOP_K}"
echo "MAX_WORKERS_S2 = ${MAX_WORKERS_S2}"
echo "MAX_WORKERS_S4 = ${MAX_WORKERS_S4}"
echo "CHUNK_SIZE_S4  = ${CHUNK_SIZE_S4}"
echo "ALPHA_S5       = ${ALPHA_S5}"
echo "NORM_MODE_S5   = ${NORM_MODE_S5}"
echo "DEV_RATIO_S5   = ${DEV_RATIO_S5}"
echo "============================================================"

cd "${REPO_ROOT}"

# ── step2 ──
STEP2_OUT="${OUT_DIR}/reasoning_steps.jsonl"
if [[ -n "${SKIP_STEP2:-}" ]]; then
  echo "[skip] step2 (SKIP_STEP2 set)"
  if [[ ! -f "${STEP2_OUT}" ]]; then
    echo "[error] step2 skipped but ${STEP2_OUT} not found" >&2
    exit 1
  fi
else
  echo "[$(date -Is)] step2 BEGIN  → ${LOG_DIR}/step2.log"
  ${PY} 03_sapr_rag/data/build_v1/step2_generate_reasoning_steps.py \
    --n-samples "${N_SAMPLES}" \
    --seed "${SEED}" \
    --max-workers "${MAX_WORKERS_S2}" \
    --out-dir "${OUT_DIR}" \
    "${DEBUG_FLAG[@]}" \
    2>&1 | tee "${LOG_DIR}/step2.log"
  echo "[$(date -Is)] step2 DONE"
fi

if [[ ! -s "${STEP2_OUT}" ]]; then
  echo "[error] step2 output empty: ${STEP2_OUT}" >&2
  exit 1
fi

# ── step3 ──
STEP3_OUT="${OUT_DIR}/candidates.jsonl"
if [[ -n "${SKIP_STEP3:-}" ]]; then
  echo "[skip] step3 (SKIP_STEP3 set)"
  if [[ ! -f "${STEP3_OUT}" ]]; then
    echo "[error] step3 skipped but ${STEP3_OUT} not found" >&2
    exit 1
  fi
else
  echo "[$(date -Is)] step3 BEGIN  → ${LOG_DIR}/step3.log"
  ${PY} 03_sapr_rag/data/build_v1/step3_retrieve_candidates.py \
    --in-dir "${OUT_DIR}" \
    --out-dir "${OUT_DIR}" \
    --top-k "${TOP_K}" \
    "${DEBUG_FLAG[@]}" \
    2>&1 | tee "${LOG_DIR}/step3.log"
  echo "[$(date -Is)] step3 DONE"
fi

if [[ ! -s "${STEP3_OUT}" ]]; then
  echo "[error] step3 output empty: ${STEP3_OUT}" >&2
  exit 1
fi

# ── step4 ──
STEP4_OUT="${OUT_DIR}/cls_labels.jsonl"
if [[ -n "${SKIP_STEP4:-}" ]]; then
  echo "[skip] step4 (SKIP_STEP4 set)"
else
  echo "[$(date -Is)] step4 BEGIN  → ${LOG_DIR}/step4.log"
  ${PY} 03_sapr_rag/data/build_v1/step4_label_cls.py \
    --in-dir "${OUT_DIR}" \
    --out-dir "${OUT_DIR}" \
    --max-workers "${MAX_WORKERS_S4}" \
    --chunk-size "${CHUNK_SIZE_S4}" \
    "${DEBUG_FLAG[@]}" \
    2>&1 | tee "${LOG_DIR}/step4.log"
  echo "[$(date -Is)] step4 DONE"
fi

if [[ ! -s "${STEP4_OUT}" ]]; then
  echo "[error] step4 output empty: ${STEP4_OUT}" >&2
  exit 1
fi

# ── step5 ──
STEP5_TRAIN="${OUT_DIR}/train.jsonl"
STEP5_DEV="${OUT_DIR}/dev.jsonl"
if [[ -n "${SKIP_STEP5:-}" ]]; then
  echo "[skip] step5 (SKIP_STEP5 set)"
else
  echo "[$(date -Is)] step5 BEGIN  → ${LOG_DIR}/step5.log"
  ${PY} 03_sapr_rag/data/build_v1/step5_assemble_train_jsonl.py \
    --in-dir "${OUT_DIR}" \
    --out-dir "${OUT_DIR}" \
    --alpha "${ALPHA_S5}" \
    --norm-mode "${NORM_MODE_S5}" \
    --dev-ratio "${DEV_RATIO_S5}" \
    --seed "${SEED}" \
    "${DEBUG_FLAG[@]}" \
    2>&1 | tee "${LOG_DIR}/step5.log"
  echo "[$(date -Is)] step5 DONE"
fi

if [[ ! -s "${STEP5_TRAIN}" ]]; then
  echo "[error] step5 train output empty: ${STEP5_TRAIN}" >&2
  exit 1
fi

# ── 汇总 ──
echo "============================================================"
echo "SAPR-R v1 data build COMPLETED"
echo "OUT_DIR = ${OUT_DIR}"
echo "  reasoning_steps.jsonl : $(wc -l < "${STEP2_OUT}" | tr -d ' ') 行"
echo "  candidates.jsonl      : $(wc -l < "${STEP3_OUT}" | tr -d ' ') 行"
if [[ -f "${STEP4_OUT}" ]]; then
  echo "  cls_labels.jsonl      : $(wc -l < "${STEP4_OUT}" | tr -d ' ') 行"
fi
if [[ -f "${STEP5_TRAIN}" ]]; then
  echo "  train.jsonl           : $(wc -l < "${STEP5_TRAIN}" | tr -d ' ') 行"
fi
if [[ -f "${STEP5_DEV}" ]]; then
  echo "  dev.jsonl             : $(wc -l < "${STEP5_DEV}" | tr -d ' ') 行"
fi
echo "logs in ${LOG_DIR}/"
echo "next: 用 train.jsonl/dev.jsonl 训 reranker（写 dataset.py / model.py / loss.py / train.py）"
echo "============================================================"
