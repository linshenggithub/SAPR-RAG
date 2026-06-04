#!/usr/bin/env bash
# 5090 服务器（rag-5090，3 x RTX 5090）环境变量配置
#
# 用法：
#   source config/env_5090.sh
#   python gate0/run_mcts_typed_vs_scalar_pilot.py --mode sanity
#
# 状态：已按当前 5090 机器 expm11 上的真实路径确认。
# 已知线索来自 gate0/GATE0_STATUS.md §6.3 与本机路径审计：
#   /home/mayi/ReasonRAG_modified/indexes/bge_extended/bge_Flat.index   (BGE extended index)
#   /nas/mayi/RAG/corpus/wiki18_extended.jsonl                  (extended corpus)
#   /nas/mayi/RAG/corpus/wiki18_100w.jsonl                      (non-extended corpus)
#   /home/mayi/ReasonRAG_modified                                (current runnable ReasonRAG repo)
#
# 填值原则：
# 1. ssh rag-5090 后用 `ls` 确认每条路径都存在；
# 2. extended / non-extended 二选一，且 BGE_INDEX_PATH 与 WIKI_CORPUS_PATH 必须配套；
# 3. 不确定的留 "TODO_XXX"，让 config/paths.py 在用到时报错而不是默默错。

# ---- ReasonRAG MCTS reward_data*.json 目录 ----
# 用途：gate0/sample_branch_points.py / relabel_q_with_gpt4o.py 的输入
# Gate 0 已暂停；仅保留给历史脚本使用。
export SAPR_REASONRAG_OUTPUT_DIR="/home/mayi/SAPR-RAG/gate0/data/reasonrag_mcts"

# ---- ReasonRAG 仓库根 ----
export SAPR_REASONRAG_ROOT="/home/mayi/ReasonRAG_modified"

# ---- FlashRAG 仓库根 ----
# 当前 SAPR-E e2e 依赖 reasonrag 环境里的 flashrag 包；该变量主要给历史 Gate0 脚本使用。
export SAPR_FLASHRAG_ROOT="/home/mayi/ReasonRAG_modified"

# ---- BGE Flat 检索索引 ----
# 中期答辩主线对齐 modified baseline：extended index + extended corpus
export SAPR_BGE_INDEX_PATH="/home/mayi/ReasonRAG_modified/indexes/bge_extended/bge_Flat.index"

# ---- BGE encoder 模型 ----
export SAPR_BGE_MODEL_PATH="/nas/mayi/RAG/retrievers/bge-base-en-v1.5"

# ---- 维基百科 corpus ----
# 与 BGE_INDEX_PATH 配套：用 extended index 就配 extended corpus
export SAPR_WIKI_CORPUS_PATH="/nas/mayi/RAG/corpus/wiki18_extended.jsonl"

# ---- HotpotQA dev jsonl ----
export SAPR_HOTPOTQA_DEV_PATH="/home/mayi/ReasonRAG_modified/dataset/hotpotqa/dev.jsonl"

# ---- HotpotQA train jsonl (SAPR-R v1 离线数据构造输入) ----
export SAPR_HOTPOTQA_TRAIN_PATH="/home/mayi/ReasonRAG_modified/dataset/hotpotqa/train.jsonl"

# ---- 实验 generator ----
# 对齐用户 baseline：Qwen2.5-7B-Instruct-ReasonRAG LoRA 合并后的完整模型。
# /home/mayi/models/Qwen2.5-7B-Instruct-ReasonRAG-Lora 是 adapter 目录，不是 vLLM 直接使用的完整模型。
export SAPR_LORA_MODEL_PATH="/home/mayi/LLaMA-Factory/examples/merge_lora/output/qwen2.5-7B-lora-dpo-RAG-ProGuide"

# ---- conda 可执行 ----
export SAPR_CONDA_BIN="/home/mayi/miniconda3/bin/conda"

echo "[env] sourced config/env_5090.sh (expm11 / 5090, extended corpus/index)"
