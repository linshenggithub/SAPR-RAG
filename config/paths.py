"""仓库外部依赖路径单点配置。

设计原则：
- 仓库内路径用 ``Path(__file__).resolve().parents[N]`` 派生，**不在此文件配置**
- 仓库外路径（ReasonRAG 训练数据、Wiki 语料、BGE 索引等）写在这里，集中管理
- 每个路径都可以通过同名环境变量覆盖，方便跨机器使用

跨机器使用示例（无需改代码）：

.. code-block:: bash

    # 在 5090 上跑实验
    export SAPR_REASONRAG_OUTPUT_DIR=/home/mayi/ReasonRAG/output/hotpotqa
    export SAPR_WIKI_CORPUS_PATH=/nas/mayi/RAG/corpus/wiki18_extended.jsonl
    python gate0/sample_branch_points.py

    # 在本地 mlx_devbox 上跑（如果有数据软链）
    export SAPR_REASONRAG_OUTPUT_DIR=/path/to/local/reward_data_dir
    python gate0/sample_branch_points.py
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# 仓库根目录（用于其他模块构造仓内相对路径时引用）
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def _path_from_env(env_var: str, default: str) -> Path:
    """从环境变量读取路径，未设置则用默认值。"""
    return Path(os.environ.get(env_var, default))


# ---------------------------------------------------------------------------
# 仓库外路径（默认值对应 3090 服务器布局；其他机器请用环境变量覆盖）
# ---------------------------------------------------------------------------

# ReasonRAG MCTS reward_data*.json 目录
# 来源：ReasonRAG 仓库自己跑出的训练数据
REASONRAG_OUTPUT_DIR: Path = _path_from_env(
    "SAPR_REASONRAG_OUTPUT_DIR",
    "/home/mayi/RAG/ReasonRAG/output/hotpotqa",
)

# 维基百科 corpus（FlashRAG 格式 jsonl）
# 注意：extended 版本（含 PopQA/HotpotQA/2Wiki augment）和原版 wiki18_100w.jsonl 是不同的
# 默认指向 extended，对齐 ReasonRAG data_generation.py 的配置
WIKI_CORPUS_PATH: Path = _path_from_env(
    "SAPR_WIKI_CORPUS_PATH",
    "/nas/mayi/RAG/corpus/wiki18_extended.jsonl",
)

# BGE Flat 检索索引
# 注意：默认指向非 extended 版（与 inference.py 对齐）；
# 若要对齐 data_generation.py 的 extended 版，请用 SAPR_BGE_INDEX_PATH 覆盖
BGE_INDEX_PATH: Path = _path_from_env(
    "SAPR_BGE_INDEX_PATH",
    "/home/mayi/RAG/retriever/bgeindex/bge_Flat.index",
)

# ReasonRAG 仓库根目录（用于把它加入 sys.path 以 import flashrag 配置等）
REASONRAG_ROOT: Path = _path_from_env(
    "SAPR_REASONRAG_ROOT",
    "/home/mayi/ReasonRAG",
)

# BGE 检索 encoder 模型目录（bge-base-en-v1.5）
BGE_MODEL_PATH: Path = _path_from_env(
    "SAPR_BGE_MODEL_PATH",
    "/nas/mayi/RAG/retrievers/bge-base-en-v1.5",
)

# qwen2.5-7B-lora-dpo-RAG-ProGuide （ReasonRAG DPO 后的合并模型）
LORA_MODEL_PATH: Path = _path_from_env(
    "SAPR_LORA_MODEL_PATH",
    "/home/mayi/LLaMA-Factory/examples/merge_lora/output/qwen2.5-7B-lora-dpo-RAG-ProGuide",
)

# Conda 可执行文件（仅 launch_*.sh 需要；其他 Python 脚本不依赖）
CONDA_BIN: Path = _path_from_env(
    "SAPR_CONDA_BIN",
    "/home/mayi/miniconda3/bin/conda",
)

# HotpotQA dev jsonl（gate0 GPT-4o pilot 用作输入 query 来源）
HOTPOTQA_DEV_PATH: Path = _path_from_env(
    "SAPR_HOTPOTQA_DEV_PATH",
    "/home/mayi/RAG/ReasonRAG/dataset/hotpotqa/dev.jsonl",
)


__all__ = [
    "REPO_ROOT",
    "REASONRAG_OUTPUT_DIR",
    "WIKI_CORPUS_PATH",
    "BGE_INDEX_PATH",
    "REASONRAG_ROOT",
    "BGE_MODEL_PATH",
    "LORA_MODEL_PATH",
    "CONDA_BIN",
    "HOTPOTQA_DEV_PATH",
]
