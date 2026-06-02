"""SAPR-RAG 全局配置包。

统一存放跨脚本使用的配置项（路径、环境变量等），避免在多个脚本里重复硬编码。
"""

from .paths import (
    BGE_INDEX_PATH,
    BGE_MODEL_PATH,
    CONDA_BIN,
    HOTPOTQA_DEV_PATH,
    LORA_MODEL_PATH,
    REASONRAG_OUTPUT_DIR,
    REASONRAG_ROOT,
    REPO_ROOT,
    WIKI_CORPUS_PATH,
)

__all__ = [
    "BGE_INDEX_PATH",
    "BGE_MODEL_PATH",
    "CONDA_BIN",
    "HOTPOTQA_DEV_PATH",
    "LORA_MODEL_PATH",
    "REASONRAG_OUTPUT_DIR",
    "REASONRAG_ROOT",
    "REPO_ROOT",
    "WIKI_CORPUS_PATH",
]
