#!/usr/bin/env bash
# 阶段 1：CPU 任务 —— 下载远端资产 + 拼 wiki18_extended.jsonl
#
# 这个 launcher **不需要 GPU**，可以直接在开发机上跑（不用 launch GPU worker）。
# 跑完后再用 launch_build_index.sh 在 GPU worker 上跑 BGE 编码 + FAISS。
#
# 总耗时：
#   - 下载远端资产 (huggingface):   1-2 小时（取决于网速）
#   - 拼 wiki18_extended.jsonl:     3-5 分钟（纯 IO）
#
# 用法：
#   source config/env_local.sh
#   bash 03_sapr_rag/data/build_v1/00_prep/launch_download.sh
#
# 可调环境变量：
#   SKIP_DOWNLOAD=1            跳过下载（已有资产时）
#   SKIP_CORPUS=1              跳过 corpus 拼接
#   LIMIT_DEBUG_CORPUS=10000   smoke：只拼前 N 条做小 corpus（0=全量）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# ---- 必需 env 自检 ----
: "${SAPR_HOTPOTQA_TRAIN_PATH:?source config/env_*.sh first (HOTPOTQA train missing)}"
: "${SAPR_WIKI_CORPUS_PATH:?source config/env_*.sh first (wiki corpus path missing)}"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_CORPUS="${SKIP_CORPUS:-0}"
LIMIT_DEBUG_CORPUS="${LIMIT_DEBUG_CORPUS:-0}"

LOG_DIR="${REPO_ROOT}/data/logs/prep"
mkdir -p "${LOG_DIR}"

echo "================================================================"
echo "[stage 1: CPU] download remote assets + concat extended corpus"
echo "  SKIP_DOWNLOAD=${SKIP_DOWNLOAD}  SKIP_CORPUS=${SKIP_CORPUS}"
echo "  LIMIT_DEBUG_CORPUS=${LIMIT_DEBUG_CORPUS}  (0 = full corpus)"
echo "================================================================"

# ---- 1. 下载资产 ----
if [[ "${SKIP_DOWNLOAD}" == "0" ]]; then
    echo
    echo "[1/2] Downloading remote assets ..."
    # 注意：stderr 不重定向，让 hf 的 tqdm 进度条看到 tty 正常刷新；
    # 只把 stdout 通过 tee 落盘到 download.log（进度条本就在 stderr，不入 log）。
    bash "${SCRIPT_DIR}/download_assets.sh" > >(tee "${LOG_DIR}/download.log")
else
    echo "[1/2] SKIP_DOWNLOAD=1, skip"
fi

# ---- 2. 拼 wiki18_extended.jsonl ----
if [[ "${SKIP_CORPUS}" == "0" ]]; then
    echo
    echo "[2/2] Building wiki18_extended.jsonl ..."
    LIMIT_FLAGS=""
    if [[ "${LIMIT_DEBUG_CORPUS}" != "0" ]]; then
        LIMIT_FLAGS="--limit-base ${LIMIT_DEBUG_CORPUS} --limit-extend ${LIMIT_DEBUG_CORPUS}"
        echo "      [smoke] using --limit-base ${LIMIT_DEBUG_CORPUS} --limit-extend ${LIMIT_DEBUG_CORPUS}"
    fi
    python "${SCRIPT_DIR}/build_extended_corpus.py" ${LIMIT_FLAGS} 2>&1 | tee "${LOG_DIR}/build_corpus.log"
    if [[ ! -s "${SAPR_WIKI_CORPUS_PATH}" ]]; then
        echo "[err] empty corpus at ${SAPR_WIKI_CORPUS_PATH}"
        exit 1
    fi
    echo "      corpus lines: $(wc -l < "${SAPR_WIKI_CORPUS_PATH}")"
else
    echo "[2/2] SKIP_CORPUS=1, skip"
fi

echo
echo "================================================================"
echo "[stage 1 done] CPU tasks complete."
echo "  HotpotQA:       ${SAPR_HOTPOTQA_TRAIN_PATH}"
echo "  Corpus:         ${SAPR_WIKI_CORPUS_PATH}"
echo "================================================================"
echo
echo "Next (on GPU worker):"
echo "  source config/env_local.sh"
echo "  NPROC_PER_NODE=4 BATCH_SIZE=512 bash 03_sapr_rag/data/build_v1/00_prep/launch_build_index.sh"
