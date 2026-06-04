#!/usr/bin/env bash
# 3090 服务器（4 x RTX 3090）环境变量配置
#
# 用法：
#   source config/env_3090.sh
#   python gate0/run_mcts_typed_vs_scalar_pilot.py --mode sanity
#
# 路径来源：gate0/GATE0_STATUS.md §6.3 已审计的本机资源清单
# 修改原则：如果某条路径在你这台 3090 上不存在，先在文件里改完再 source
#
# 缺值标注：
#   值带 "TODO_XXX" 的表示未确认；用之前先填正确路径。

# ---- ReasonRAG MCTS reward_data*.json 目录 ----
# 用途：gate0/sample_branch_points.py / relabel_q_with_gpt4o.py 的输入
export SAPR_REASONRAG_OUTPUT_DIR="/home/mayi/RAG/ReasonRAG/output/hotpotqa"

# ---- ReasonRAG 仓库根 ----
# 用途：03_sapr_rag/scripts/run_*.py 把它加进 sys.path
export SAPR_REASONRAG_ROOT="/home/mayi/ReasonRAG"

# ---- FlashRAG 仓库根 ----
# 用途：gate0/run_mcts_typed_vs_scalar_pilot.py 把它加进 sys.path
export SAPR_FLASHRAG_ROOT="/home/mayi/RAG/FlashRAG"

# ---- BGE Flat 检索索引 ----
# 注意：3090 这份对齐 inference.py（非 extended）；要对齐 data_generation.py 请改成 extended
export SAPR_BGE_INDEX_PATH="/home/mayi/RAG/retriever/bgeindex/bge_Flat.index"

# ---- BGE encoder 模型 ----
export SAPR_BGE_MODEL_PATH="/home/mayi/RAG/retriever/bge-base-en-v1.5"

# ---- 维基百科 corpus（FlashRAG jsonl）----
# 注意：与 BGE_INDEX_PATH 必须配套（都是 extended 或都是非 extended）
export SAPR_WIKI_CORPUS_PATH="/home/mayi/RAG/corpus/wiki18_100w.jsonl"

# ---- HotpotQA dev jsonl ----
# 用途：gate0 GPT-4o pilot 输入 query 来源
export SAPR_HOTPOTQA_DEV_PATH="/home/mayi/RAG/ReasonRAG/dataset/hotpotqa/dev.jsonl"

# ---- HotpotQA train jsonl (SAPR-R v1 离线数据构造输入) ----
export SAPR_HOTPOTQA_TRAIN_PATH="/home/mayi/RAG/ReasonRAG/dataset/hotpotqa/train.jsonl"

# ---- qwen2.5-7B-LoRA-DPO 合并模型 ----
# 用途：03_sapr_rag/scripts/run_*.py 的 generator
export SAPR_LORA_MODEL_PATH="/home/mayi/LLaMA-Factory/examples/merge_lora/output/qwen2.5-7B-lora-dpo-RAG-ProGuide"

# ---- conda 可执行 ----
# 用途：launch_*.sh 包装脚本
export SAPR_CONDA_BIN="/home/mayi/miniconda3/bin/conda"

echo "[env] sourced config/env_3090.sh (3090 / 4xRTX3090)"
