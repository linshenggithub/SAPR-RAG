#!/usr/bin/env bash
# 5090 服务器（rag-5090，3 x RTX 5090）环境变量配置
#
# 用法：
#   source config/env_5090.sh
#   python gate0/run_mcts_typed_vs_scalar_pilot.py --mode sanity
#
# 状态：⚠️ 大部分值是 TODO 占位，第一次在 5090 上跑实验前必须 ssh 上去确认实际路径再填回来。
# 已知线索来自 gate0/GATE0_STATUS.md §6.3：
#   /home/mayi/ReasonRAG/indexes/bge_extended/bge_Flat.index   (BGE extended index)
#   /nas/mayi/RAG/corpus/wiki18_extended.jsonl                  (extended corpus)
#   /nas/mayi/RAG/corpus/wiki18_100w.jsonl                      (non-extended corpus)
#   /home/mayi/ReasonRAG                                         (ReasonRAG repo)
#
# 填值原则：
# 1. ssh rag-5090 后用 `ls` 确认每条路径都存在；
# 2. extended / non-extended 二选一，且 BGE_INDEX_PATH 与 WIKI_CORPUS_PATH 必须配套；
# 3. 不确定的留 "TODO_XXX"，让 config/paths.py 在用到时报错而不是默默错。

# ---- ReasonRAG MCTS reward_data*.json 目录 ----
# 用途：gate0/sample_branch_points.py / relabel_q_with_gpt4o.py 的输入
# 注意：5090 上 ReasonRAG 仓库在 /home/mayi/ReasonRAG，但 reward_data 是否同步过来 / 在哪个子目录待确认
export SAPR_REASONRAG_OUTPUT_DIR="TODO_5090_REASONRAG_OUTPUT_DIR"

# ---- ReasonRAG 仓库根 ----
export SAPR_REASONRAG_ROOT="/home/mayi/ReasonRAG"

# ---- FlashRAG 仓库根 ----
# 5090 上 FlashRAG 装在哪儿待确认（可能是 site-packages 全局装的，那这条就用不到）
export SAPR_FLASHRAG_ROOT="TODO_5090_FLASHRAG_ROOT"

# ---- BGE Flat 检索索引 ----
# Gate 0 推荐用 extended（对齐论文 data_generation.py + extended corpus）
export SAPR_BGE_INDEX_PATH="/home/mayi/ReasonRAG/indexes/bge_extended/bge_Flat.index"

# ---- BGE encoder 模型 ----
export SAPR_BGE_MODEL_PATH="TODO_5090_BGE_MODEL_PATH"

# ---- 维基百科 corpus ----
# 与 BGE_INDEX_PATH 配套：用 extended index 就配 extended corpus
export SAPR_WIKI_CORPUS_PATH="/nas/mayi/RAG/corpus/wiki18_extended.jsonl"

# ---- HotpotQA dev jsonl ----
export SAPR_HOTPOTQA_DEV_PATH="TODO_5090_HOTPOTQA_DEV_PATH"

# ---- qwen2.5-7B-LoRA-DPO 合并模型 ----
export SAPR_LORA_MODEL_PATH="TODO_5090_LORA_MODEL_PATH"

# ---- conda 可执行 ----
export SAPR_CONDA_BIN="TODO_5090_CONDA_BIN"

echo "[env] sourced config/env_5090.sh (rag-5090 / 3xRTX5090) — 注意：含 TODO 占位，跑实验前先填全"
