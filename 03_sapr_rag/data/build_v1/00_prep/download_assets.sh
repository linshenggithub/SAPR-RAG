#!/usr/bin/env bash
# 一键下载 SAPR-R v1 离线数据构造 + extended index 重建所需的全部远端资产
#
# 落盘位置（仓内）：
#   - data/raw/hotpotqa/train.jsonl                   ← FlashRAG 格式，step2 直采输入
#   - data/raw/hotpotqa/dev.jsonl                     ← FlashRAG 格式，预留
#   - data/raw/wiki18_100w.jsonl                      ← FlashRAG wiki18 标准 corpus（21M 段）
#   - data/raw/RAG_extend_corpus/                     ← reasonrag/RAG_extend_corpus（1.34M 段）
#   - models/bge-base-en-v1.5/                        ← BGE encoder
#
# 总体积 ~7 GB；下载耗时 1-2 小时（取决于网速）。
#
# 用法：
#   source config/env_local.sh
#   bash 03_sapr_rag/data/build_v1/00_prep/download_assets.sh
#
# 重跑安全：huggingface-cli 自带断点续传，重复执行会跳过已下载的文件。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

DATA_RAW="${REPO_ROOT}/data/raw"
MODELS_DIR="${REPO_ROOT}/models"

mkdir -p "${DATA_RAW}/hotpotqa" "${DATA_RAW}/RAG_extend_corpus" "${MODELS_DIR}"

# ---- HF 镜像（开发机直连 huggingface.co 不可达；source env_local.sh 会预设此值）----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "[info] HF_ENDPOINT=${HF_ENDPOINT}"

# ---- 检查 hf CLI 是否可用（新版替代 huggingface-cli）----
# 旧版 huggingface-cli 在新版 huggingface_hub 中已废弃，运行只会打印 deprecation hint。
if ! command -v hf >/dev/null 2>&1; then
    echo "[err] 'hf' CLI not found. Install via:"
    echo "      pip install -U huggingface_hub"
    echo "      pip install -r 03_sapr_rag/requirements_v1_offline.txt"
    exit 1
fi

# ---- 1. FlashRAG 版 HotpotQA（step2 直采输入：question + golden_answers + supporting_facts）----
echo "[1/4] Downloading FlashRAG-formatted HotpotQA (train + dev) ..."
hf download \
    RUC-NLPIR/FlashRAG_datasets \
    "hotpotqa/train.jsonl" "hotpotqa/dev.jsonl" \
    --repo-type dataset \
    --local-dir "${DATA_RAW}"
ls -lh "${DATA_RAW}/hotpotqa/" || true

# ---- 2. FlashRAG wiki18_100w corpus（标准 21M 段 base corpus）----
# repo 里只放了 zip 压缩包（~5 GB），没有 raw jsonl，需要先下 zip 再解压。
echo "[2/4] Downloading FlashRAG wiki18_100w corpus (~5 GB zip, base corpus for extended) ..."
if [[ -f "${DATA_RAW}/wiki18_100w.jsonl" ]]; then
    echo "      jsonl already exists, skip download/unzip: ${DATA_RAW}/wiki18_100w.jsonl"
else
    hf download \
        RUC-NLPIR/FlashRAG_datasets \
        "retrieval-corpus/wiki18_100w.zip" \
        --repo-type dataset \
        --local-dir "${DATA_RAW}"
    # 解压：zip 内是 wiki18_100w.jsonl
    if ! command -v unzip >/dev/null 2>&1; then
        echo "[err] 'unzip' not found. Install: apt-get install -y unzip  (or: pip install python-unzip)"
        exit 1
    fi
    echo "      unzipping wiki18_100w.zip → wiki18_100w.jsonl ..."
    unzip -o "${DATA_RAW}/retrieval-corpus/wiki18_100w.zip" -d "${DATA_RAW}/retrieval-corpus/"
    # 移到统一位置；删除 zip 节省 5 GB
    if [[ -f "${DATA_RAW}/retrieval-corpus/wiki18_100w.jsonl" ]]; then
        mv "${DATA_RAW}/retrieval-corpus/wiki18_100w.jsonl" "${DATA_RAW}/wiki18_100w.jsonl"
        rm -f "${DATA_RAW}/retrieval-corpus/wiki18_100w.zip"
        rmdir "${DATA_RAW}/retrieval-corpus" 2>/dev/null || true
    else
        echo "[err] unzip succeeded but wiki18_100w.jsonl not found"
        exit 1
    fi
fi
ls -lh "${DATA_RAW}/wiki18_100w.jsonl" || true

# ---- 3. ReasonRAG 扩展段（1.34M 段，extended 的"扩展"部分）----
echo "[3/4] Downloading reasonrag/RAG_extend_corpus (~1.34M extra paragraphs) ..."
hf download \
    reasonrag/RAG_extend_corpus \
    --repo-type dataset \
    --local-dir "${DATA_RAW}/RAG_extend_corpus"
ls -lh "${DATA_RAW}/RAG_extend_corpus/" | head -20 || true

# ---- 4. BGE encoder 模型 ----
echo "[4/4] Downloading BGE encoder (BAAI/bge-base-en-v1.5) ..."
hf download \
    BAAI/bge-base-en-v1.5 \
    --local-dir "${MODELS_DIR}/bge-base-en-v1.5"
ls -lh "${MODELS_DIR}/bge-base-en-v1.5/" | head -10 || true

echo
echo "[done] All assets downloaded."
echo "  HotpotQA:      ${DATA_RAW}/hotpotqa/{train,dev}.jsonl"
echo "  wiki18_100w:   ${DATA_RAW}/wiki18_100w.jsonl"
echo "  extend_corpus: ${DATA_RAW}/RAG_extend_corpus/"
echo "  BGE model:     ${MODELS_DIR}/bge-base-en-v1.5/"
echo
echo "Next: python 03_sapr_rag/data/build_v1/00_prep/build_extended_corpus.py"
