#!/usr/bin/env python3
"""
Sanity check for SAPR-RAG Evidence-only debug_result experiment.

What it does:
  1. Constructs flashrag Config (save_dir OUTSIDE /home/mayi/ReasonRAG/output/)
  2. Loads HotpotQA dev dataset, slices 30 examples
  3. Initializes ReasonRAGPipeline (retriever + generator)
  4. Prints key config values and save_dir for verification
  5. Does NOT run actual generation — exits after initialization check.

Usage (on rag-5090):
  conda activate reasonrag
  cd /home/mayi/ReasonRAG
  CUDA_VISIBLE_DEVICES=0 python sanity_check_evidence.py
"""

import os
import sys
import json
import datetime

# 让脚本能直接 `python 03_sapr_rag/scripts/xxx.py` 运行：把仓库根加进 sys.path
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.paths import (  # noqa: E402
    REASONRAG_ROOT,
    WIKI_CORPUS_PATH,
    BGE_MODEL_PATH,
    LORA_MODEL_PATH,
)

# ── Add ReasonRAG to path ──────────────────────────────────────
sys.path.insert(0, str(REASONRAG_ROOT))

from flashrag.config import Config
from flashrag.utils import get_dataset
from pipeline.reasonrag_pipeline import ReasonRAGPipeline

# ── Safety: output dir must be OUTSIDE ReasonRAG/output/ ───────
SAFE_SAVE_DIR = "/home/mayi/sapr_rag_debug_output"
FORBIDDEN_DIR = str(REASONRAG_ROOT / "output")

assert not os.path.abspath(SAFE_SAVE_DIR).startswith(
    os.path.abspath(FORBIDDEN_DIR)
), f"save_dir {SAFE_SAVE_DIR} must NOT be inside {FORBIDDEN_DIR}"

os.makedirs(SAFE_SAVE_DIR, exist_ok=True)

# ── Build config ───────────────────────────────────────────────
SLICE_SIZE = 30

config_dict = {
    # Data
    "data_dir": str(REASONRAG_ROOT / "dataset/"),
    "dataset_name": "hotpotqa",
    "split": ["dev", "test"],

    # Retrieval
    "index_path": str(REASONRAG_ROOT / "indexes/bge_extended/bge_Flat.index"),
    "retrieval_method": "bge",
    "corpus_path": str(WIKI_CORPUS_PATH),
    "faiss_gpu": False,

    # Model paths
    "model2path": {
        "bge": str(BGE_MODEL_PATH),
        "qwen2.5-instruct-ReasonRAG-lora": str(LORA_MODEL_PATH),
    },
    "model2pooling": {
        "bge": "cls",
    },
    "method2index": {
        "bge": None,
    },

    # Generator
    "generator_model": "qwen2.5-instruct-ReasonRAG-lora",
    "generator_batch_size": 1,
    "tensor_parallel_size": 1,       # single GPU
    "framework": "vllm",
    "gpu_id": "0",
    "gpu_memory_utilization": 0.8,   # need enough for 7B model on 32GB GPU
    "generator_max_input_len": 8192,  # ReasonRAG default, was 1024 from basic_config

    # Experiment
    "retrieval_topk": 10,
    "metrics": ["em", "f1", "acc", "recall", "precision"],
    "save_intermediate_data": True,
    "save_note": f"sapr_evidence_debug_{SLICE_SIZE}samples",
    "save_dir": SAFE_SAVE_DIR,
    "seed": 2024,
    "disable_save": False,           # let Config create the timestamped dir

    # Sampling
    "test_sample_num": None,         # we slice manually
    "random_sample": False,
}

print("=" * 70)
print("SAPR-RAG Evidence-only Sanity Check")
print("=" * 70)
print(f"Time      : {datetime.datetime.now().isoformat()}")
print(f"Server    : rag-5090")
print(f"GPU       : 0 (single)")
print(f"Slice     : {SLICE_SIZE} examples from HotpotQA dev")
print(f"Save dir  : {SAFE_SAVE_DIR} (outside ReasonRAG/output/)")
print(f"Corpus    : {config_dict['corpus_path']}")
print(f"Index     : {config_dict['index_path']}")
print(f"Model     : {config_dict['generator_model']}")
print("=" * 70)

# ── Step 1: Construct Config ───────────────────────────────────
print("\n[Step 1/4] Constructing flashrag Config ...")
config = Config(config_dict=config_dict)
actual_save_dir = config["save_dir"]
print(f"  ✓ Config created")
print(f"  ✓ Resolved save_dir: {actual_save_dir}")
print(f"  ✓ dataset_path: {config['dataset_path']}")
print(f"  ✓ generator_model_path: {config['generator_model_path']}")
print(f"  ✓ retrieval_model_path: {config['retrieval_model_path']}")

# Verify save_dir is safe
assert not os.path.abspath(actual_save_dir).startswith(
    os.path.abspath(FORBIDDEN_DIR)
), f"SAFETY VIOLATION: save_dir {actual_save_dir} is inside {FORBIDDEN_DIR}!"
print(f"  ✓ SAFETY CHECK PASSED: save_dir is outside {FORBIDDEN_DIR}")

# ── Step 2: Load HotpotQA dev and slice ────────────────────────
print(f"\n[Step 2/4] Loading HotpotQA dev dataset ...")
all_split = get_dataset(config)
dev_data = all_split.get("dev", None)
if dev_data is None:
    print("  ✗ dev split not found! Available splits:", list(all_split.keys()))
    sys.exit(1)

total_count = len(dev_data)
print(f"  ✓ Loaded {total_count} examples from HotpotQA dev")

# Slice first SLICE_SIZE examples
# flashrag Dataset stores Items in self.data; slice directly
from flashrag.dataset.dataset import Dataset as FlashRAGDataset

sliced_items = dev_data.data[:SLICE_SIZE]
sliced_data = FlashRAGDataset(
    config=config,              # first arg is config dict
    dataset_path=dev_data.dataset_path,
    data=sliced_items,          # already Item objects
    sample_num=None,
    random_sample=False,
)
print(f"  ✓ Sliced to {len(sliced_data)} examples")

# Print a sample
sample = sliced_data[0]
print(f"\n  Sample [0]:")
print(f"    id      : {sample.data.get('id', 'N/A')}")
print(f"    question: {sample.data.get('question', 'N/A')[:100]}...")
print(f"    answer  : {sample.data.get('golden_answers', 'N/A')}")

# ── Step 3: Initialize Pipeline ────────────────────────────────
print(f"\n[Step 3/4] Initializing ReasonRAGPipeline ...")
print(f"  (This will load vLLM model — expect ~60-90s on first run)")
pipeline = ReasonRAGPipeline(
    config,
    prompt_template=None,
    answer_format="answer",
    max_iter=3,          # matched compute: max_steps=3 per EXPERIMENT_PLAN
    max_children=2,
    max_rollouts=64,
)
print(f"  ✓ Pipeline initialized")
print(f"  ✓ max_iter=3, max_children=2, retrieval_topk={config['retrieval_topk']}")

# ── Step 4: Summary ────────────────────────────────────────────
print("\n" + "=" * 70)
print("SANITY CHECK PASSED — ALL SYSTEMS GO")
print("=" * 70)
print(f"  save_dir      : {actual_save_dir}")
print(f"  dataset       : HotpotQA dev, {len(sliced_data)} examples")
print(f"  GPU           : {config['gpu_id']} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
print(f"  gpu_num       : {config['gpu_num']}")
print(f"  device        : {config['device']}")
print(f"  model         : {config['generator_model']}")
print(f"  retriever     : {config['retrieval_method']}")
print(f"  top_k         : {config['retrieval_topk']}")
print(f"  max_iter      : 3")
print(f"  corpus_path   : {config['corpus_path']}")
print(f"  index_path    : {config['index_path']}")
print()
print("Ready for actual generation run.")
print("Command to run generation:")
print(f"  CUDA_VISIBLE_DEVICES=0 python run_evidence_debug.py")
print("=" * 70)
