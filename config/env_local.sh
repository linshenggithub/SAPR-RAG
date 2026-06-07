#!/usr/bin/env bash
# 本地 dev box 环境变量配置（mlx_devbox + GPU worker 模式）
#
# 用法：
#   source config/env_local.sh
#   # 阶段 1（CPU，开发机直接跑）：下载 + 拼 corpus，~1-2 小时
#   bash 03_sapr_rag/data/build_v1/00_prep/launch_download.sh
#   # 阶段 2（GPU worker）：BGE 编码 + FAISS 建索引，4×H20 ~30-60 分钟
#   NPROC_PER_NODE=4 BATCH_SIZE=512 bash 03_sapr_rag/data/build_v1/00_prep/launch_build_index.sh
#   # 阶段 3（CPU/网络，开发机直接跑）：step2-5 离线数据构造
#   bash 03_sapr_rag/data/build_v1/launch_build_v1_data.sh
#
# 资产存放：仓内 SAPR-RAG/data/ 和 SAPR-RAG/models/（已加 .gitignore）
# 仅覆盖 SAPR-R v1 离线数据构造管线（step2-step5）；
# REASONRAG_ROOT / FLASHRAG_ROOT / LORA_MODEL_PATH 这一阶段不用。
#
# corpus / index 路线：**extended**（与 ReasonRAG 数据生成 / SAPR-E v0.30 主线对齐）
#   = wiki18_100w.jsonl ⊕ reasonrag/RAG_extend_corpus → wiki18_extended.jsonl
#   index = BGE-base-en-v1.5 编码 + FAISS IndexFlatIP

# 仓库根（绝对路径派生）
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- HuggingFace Hub 镜像（开发机直连 huggingface.co 不可达，走国内镜像）----
# 影响 hf CLI / huggingface_hub.snapshot_download / from_pretrained 等所有走 hub 的调用
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---- HotpotQA train jsonl (SAPR-R v1 step2 输入) ----
# FlashRAG 格式：{id, question, golden_answers, metadata.supporting_facts}
export SAPR_HOTPOTQA_TRAIN_PATH="${_REPO_ROOT}/data/raw/hotpotqa/train.jsonl"

# ---- HotpotQA dev jsonl（v1 不直接用，留个占位避免别处 import 失败）----
export SAPR_HOTPOTQA_DEV_PATH="${_REPO_ROOT}/data/raw/hotpotqa/dev.jsonl"

# ---- BGE encoder 模型 ----
export SAPR_BGE_MODEL_PATH="${_REPO_ROOT}/models/bge-base-en-v1.5"

# ---- BGE FAISS index（extended 全量，由 build_extended_index.py 产出）----
export SAPR_BGE_INDEX_PATH="${_REPO_ROOT}/data/index/bge_extended_Flat.index"

# ---- 维基百科 corpus（extended，由 build_extended_corpus.py 产出）----
export SAPR_WIKI_CORPUS_PATH="${_REPO_ROOT}/data/corpus/wiki18_extended.jsonl"

# ---- 以下变量 v1 数据构造阶段不用，留空即可（下游脚本访问时会报清晰错误）----
# 训练 + 推理阶段会再补：
# export SAPR_REASONRAG_ROOT=""
# export SAPR_FLASHRAG_ROOT=""
# export SAPR_LORA_MODEL_PATH=""
# export SAPR_CONDA_BIN=""
# export SAPR_REASONRAG_OUTPUT_DIR=""

unset _REPO_ROOT
echo "[env] sourced config/env_local.sh (mlx_devbox, extended corpus/index aligned with ReasonRAG)"
