#!/usr/bin/env bash
# 阶段 2：GPU 任务 —— BGE 编码 + FAISS Flat 索引构建
#
# **必须在 GPU worker 上跑**（开发机一般无 GPU）。前置：launch_download.sh 已完成。
#
# 总耗时：
#   1× H20 100GB:       ~2-3 小时
#   4× H20 100GB:       ~30-60 分钟
#   1× RTX 3090 8GB:    ~24-48 小时
#
# 用法（单卡）：
#   source config/env_local.sh
#   bash 03_sapr_rag/data/build_v1/00_prep/launch_build_index.sh
#
# 用法（多卡，推荐 H20 4 卡）：
#   source config/env_local.sh
#   NPROC_PER_NODE=4 BATCH_SIZE=512 bash 03_sapr_rag/data/build_v1/00_prep/launch_build_index.sh
#
# 可调环境变量：
#   NPROC_PER_NODE=4    多卡 BGE 编码 GPU 数（默认 1）
#   BATCH_SIZE=256      BGE forward batch size（H20 100GB 建议 512）
#   MAX_SEQ_LEN=256     BGE max sequence length（与 ReasonRAG 一致）
#   DTYPE=fp16          fp16 / fp32（fp16 更省显存且与 ReasonRAG 对齐）
#   CHUNKS_DIR=/tmp/sapr_chunks       中间产物路径（chunk_*.npy ~32GB）；
#                                     默认 ${REPO_ROOT}/data/index/_chunks_extended（NFS 共享但占 NFS 32GB）；
#                                     worker 上推荐 export 到本地大盘 /tmp 避免压 NFS。
#   INDEX_OUT=/tmp/sapr_index/...     最终 FAISS 索引落盘路径（~64GB）；
#                                     默认 ${SAPR_BGE_INDEX_PATH}（NFS 仓内）；
#                                     若 NFS 容量紧张，先写本地 /tmp，跑完再 rsync 回仓内。
#   SKIP_INDEX_COPY=1   仅跑编码 + 建索引，不自动 cp 回 ${SAPR_BGE_INDEX_PATH}
#
# 重跑安全：chunk_*.npy 已存在的会跳过，崩溃后接续重跑即可。
#
# Worker 推荐用法（chunks + index 全写本地 /tmp，跑完手动 rsync 回 NFS）：
#   CHUNKS_DIR=/tmp/sapr_chunks \
#   INDEX_OUT=/tmp/sapr_index/bge_extended_Flat.index \
#   NPROC_PER_NODE=4 BATCH_SIZE=512 \
#   bash 03_sapr_rag/data/build_v1/00_prep/launch_build_index.sh
#   # 跑完
#   rsync -av --progress /tmp/sapr_index/ ${SAPR_REPO_ROOT}/data/index/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# ---- 必需 env 自检 ----
: "${SAPR_BGE_MODEL_PATH:?source config/env_*.sh first (BGE model missing)}"
: "${SAPR_WIKI_CORPUS_PATH:?source config/env_*.sh first (wiki corpus path missing)}"
: "${SAPR_BGE_INDEX_PATH:?source config/env_*.sh first (BGE index path missing)}"

# 前置产物校验
if [[ ! -s "${SAPR_WIKI_CORPUS_PATH}" ]]; then
    echo "[err] corpus not found at ${SAPR_WIKI_CORPUS_PATH}"
    echo "      run launch_download.sh first."
    exit 1
fi
if [[ ! -d "${SAPR_BGE_MODEL_PATH}" ]]; then
    echo "[err] BGE model dir not found at ${SAPR_BGE_MODEL_PATH}"
    echo "      run launch_download.sh first."
    exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
DTYPE="${DTYPE:-fp16}"

# chunks / index 落盘路径（worker 上建议 override 到 /tmp）
CHUNKS_DIR="${CHUNKS_DIR:-${REPO_ROOT}/data/index/_chunks_extended}"
INDEX_OUT="${INDEX_OUT:-${SAPR_BGE_INDEX_PATH}}"
SKIP_INDEX_COPY="${SKIP_INDEX_COPY:-0}"

mkdir -p "$(dirname "${INDEX_OUT}")" "${CHUNKS_DIR}"

LOG_DIR="${REPO_ROOT}/data/logs/prep"
mkdir -p "${LOG_DIR}"

echo "================================================================"
echo "[stage 2: GPU] BGE encode + FAISS IndexFlatIP build"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}  BATCH_SIZE=${BATCH_SIZE}"
echo "  MAX_SEQ_LEN=${MAX_SEQ_LEN}  DTYPE=${DTYPE}"
echo "  Corpus:     ${SAPR_WIKI_CORPUS_PATH} ($(wc -l < "${SAPR_WIKI_CORPUS_PATH}") lines)"
echo "  ChunksDir:  ${CHUNKS_DIR}"
echo "  IndexOut:   ${INDEX_OUT}"
if [[ "${INDEX_OUT}" != "${SAPR_BGE_INDEX_PATH}" ]]; then
    echo "  (final NFS target: ${SAPR_BGE_INDEX_PATH}; copy after build: $([[ "${SKIP_INDEX_COPY}" == "1" ]] && echo SKIP || echo YES))"
fi
echo "================================================================"

# ---- 检查 GPU 可见性 ----
if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "[gpu check]"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
else
    echo "[warn] nvidia-smi not found; this may not be a GPU worker"
fi

# ---- BGE encode + FAISS build ----
if [[ "${NPROC_PER_NODE}" == "1" ]]; then
    echo
    echo "[run] single-GPU mode"
    python "${SCRIPT_DIR}/build_extended_index.py" \
        --chunks-dir "${CHUNKS_DIR}" \
        --index-out "${INDEX_OUT}" \
        --batch-size "${BATCH_SIZE}" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --dtype "${DTYPE}" \
        2>&1 | tee "${LOG_DIR}/build_index.log"
else
    echo
    echo "[run] multi-GPU mode (${NPROC_PER_NODE} cards)"
    echo "      phase A: torchrun encode-only (each rank writes its own chunks)"
    torchrun --nproc_per_node="${NPROC_PER_NODE}" \
        "${SCRIPT_DIR}/build_extended_index.py" \
        --chunks-dir "${CHUNKS_DIR}" \
        --index-out "${INDEX_OUT}" \
        --batch-size "${BATCH_SIZE}" \
        --max-seq-len "${MAX_SEQ_LEN}" \
        --dtype "${DTYPE}" \
        --encode-only \
        2>&1 | tee "${LOG_DIR}/encode_chunks.log"
    echo
    echo "      phase B: rank 0 single process FAISS .add"
    python "${SCRIPT_DIR}/build_extended_index.py" \
        --chunks-dir "${CHUNKS_DIR}" \
        --index-out "${INDEX_OUT}" \
        --build-only \
        2>&1 | tee "${LOG_DIR}/build_index.log"
fi

if [[ ! -s "${INDEX_OUT}" ]]; then
    echo "[err] empty index at ${INDEX_OUT}"
    exit 1
fi

# ---- 若 INDEX_OUT 不在 NFS 目标上，自动 rsync 一份过去（保证开发机能立刻看到）----
if [[ "${INDEX_OUT}" != "${SAPR_BGE_INDEX_PATH}" && "${SKIP_INDEX_COPY}" != "1" ]]; then
    echo
    echo "[copy] rsync index to NFS target ..."
    echo "       src: ${INDEX_OUT}"
    echo "       dst: ${SAPR_BGE_INDEX_PATH}"
    mkdir -p "$(dirname "${SAPR_BGE_INDEX_PATH}")"
    rsync -av --progress "${INDEX_OUT}" "${SAPR_BGE_INDEX_PATH}"
    # 同步 meta.json（与 .index 同目录、同前缀）
    META_SRC="${INDEX_OUT%.index}.meta.json"
    META_DST="${SAPR_BGE_INDEX_PATH%.index}.meta.json"
    if [[ -f "${META_SRC}" ]]; then
        rsync -av "${META_SRC}" "${META_DST}"
    fi
fi

FINAL_INDEX="${SAPR_BGE_INDEX_PATH}"
[[ "${SKIP_INDEX_COPY}" == "1" && "${INDEX_OUT}" != "${SAPR_BGE_INDEX_PATH}" ]] && FINAL_INDEX="${INDEX_OUT}"

echo
echo "================================================================"
echo "[stage 2 done] FAISS index ready."
echo "  Index size: $(du -h "${FINAL_INDEX}" | cut -f1)"
echo "  Index path: ${FINAL_INDEX}"
echo "================================================================"
echo
echo "Next: bash 03_sapr_rag/data/build_v1/launch_build_v1_data.sh"
