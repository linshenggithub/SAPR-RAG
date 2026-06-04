"""仓库外部依赖路径单点配置。

设计原则：
- 仓内路径用 ``Path(__file__).resolve().parents[N]`` 派生，**不在此文件配置**
- 仓外路径（数据集、索引、模型、外部仓库等）写在这里，集中管理
- **不内置任何机器特定的默认值**：每台机器自己 source ``config/env_*.sh`` 设置环境变量
- 未设置的路径在被使用时会抛 ``RuntimeError``，避免误用其他机器的路径

跨机器使用示例（无需改代码）：

.. code-block:: bash

    # 在 5090 上跑实验
    source config/env_5090.sh
    python gate0/run_mcts_typed_vs_scalar_pilot.py

    # 在 3090 上跑实验
    source config/env_3090.sh
    python gate0/run_mcts_typed_vs_scalar_pilot.py

如果某条路径忘记 export，脚本会在用到它时报：

    RuntimeError: SAPR_BGE_INDEX_PATH is not set. ...

而不是默默走到错误的默认路径。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 仓库根目录（用于其他模块构造仓内相对路径时引用）
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[1]


class _RequiredPath:
    """延迟取值的环境变量路径。

    属性访问 / str() 都会触发取值；如果未设置环境变量则抛 RuntimeError，
    并提示该 export 哪个变量。这样只在实际用到的脚本里报错，不会拖累所有 import。
    """

    __slots__ = ("env_var", "purpose")

    def __init__(self, env_var: str, purpose: str) -> None:
        self.env_var = env_var
        self.purpose = purpose

    def _resolve(self) -> Path:
        value = os.environ.get(self.env_var)
        if not value:
            raise RuntimeError(
                f"{self.env_var} is not set. "
                f"Purpose: {self.purpose}. "
                f"Fix: source config/env_3090.sh (or env_5090.sh) "
                f"before running this script."
            )
        return Path(value)

    # 让 _RequiredPath 在大部分使用场景下表现得像 Path
    def __fspath__(self) -> str:
        return str(self._resolve())

    def __str__(self) -> str:
        return str(self._resolve())

    def __repr__(self) -> str:
        value = os.environ.get(self.env_var)
        return f"_RequiredPath({self.env_var}={value!r})"

    def __truediv__(self, other) -> Path:
        return self._resolve() / other

    @property
    def value(self) -> Path:
        return self._resolve()


def _optional(env_var: str) -> Optional[Path]:
    """读取可选环境变量，未设置返回 None。"""
    value = os.environ.get(env_var)
    return Path(value) if value else None


# ---------------------------------------------------------------------------
# 仓外路径（必须由 config/env_*.sh 设置；不内置机器特定默认值）
# ---------------------------------------------------------------------------

# ReasonRAG MCTS reward_data*.json 目录（gate0 输入）
REASONRAG_OUTPUT_DIR = _RequiredPath(
    "SAPR_REASONRAG_OUTPUT_DIR",
    "ReasonRAG MCTS reward_data*.json directory",
)

# 维基百科 corpus（FlashRAG 格式 jsonl）
WIKI_CORPUS_PATH = _RequiredPath(
    "SAPR_WIKI_CORPUS_PATH",
    "Wikipedia corpus jsonl (FlashRAG format)",
)

# BGE Flat 检索索引
BGE_INDEX_PATH = _RequiredPath(
    "SAPR_BGE_INDEX_PATH",
    "BGE Flat FAISS index file",
)

# ReasonRAG 仓库根（用于把它加入 sys.path）
REASONRAG_ROOT = _RequiredPath(
    "SAPR_REASONRAG_ROOT",
    "ReasonRAG repo root (for sys.path injection)",
)

# BGE encoder 模型目录（bge-base-en-v1.5）
BGE_MODEL_PATH = _RequiredPath(
    "SAPR_BGE_MODEL_PATH",
    "BGE encoder model directory (bge-base-en-v1.5)",
)

# FlashRAG 仓库根（pilot 脚本依赖 flashrag.config / flashrag.retriever）
FLASHRAG_ROOT = _RequiredPath(
    "SAPR_FLASHRAG_ROOT",
    "FlashRAG repo root (for sys.path injection)",
)

# qwen2.5-7B-lora-dpo-RAG-ProGuide 合并模型（v0 evidence-only 推理用）
LORA_MODEL_PATH = _RequiredPath(
    "SAPR_LORA_MODEL_PATH",
    "qwen2.5-7B LoRA-DPO merged model directory",
)

# Conda 可执行（仅 launch_*.sh 用）
CONDA_BIN = _RequiredPath(
    "SAPR_CONDA_BIN",
    "conda executable (used by launch_*.sh)",
)

# HotpotQA dev jsonl（gate0 GPT-4o pilot 用作输入 query 来源）
HOTPOTQA_DEV_PATH = _RequiredPath(
    "SAPR_HOTPOTQA_DEV_PATH",
    "HotpotQA dev jsonl",
)

# HotpotQA train jsonl（SAPR-R v1 离线训练数据构造的输入）
HOTPOTQA_TRAIN_PATH = _RequiredPath(
    "SAPR_HOTPOTQA_TRAIN_PATH",
    "HotpotQA train jsonl (FlashRAG format) for SAPR-R v1 data construction",
)


__all__ = [
    "REPO_ROOT",
    "REASONRAG_OUTPUT_DIR",
    "WIKI_CORPUS_PATH",
    "BGE_INDEX_PATH",
    "REASONRAG_ROOT",
    "BGE_MODEL_PATH",
    "FLASHRAG_ROOT",
    "LORA_MODEL_PATH",
    "CONDA_BIN",
    "HOTPOTQA_DEV_PATH",
    "HOTPOTQA_TRAIN_PATH",
]
