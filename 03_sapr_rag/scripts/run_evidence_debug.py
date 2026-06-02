#!/usr/bin/env python3
"""
Run Evidence-only debug_result go/no-go experiment.

Runs ReasonRAG-LoRA baseline on 30 HotpotQA dev examples.
Outputs: trajectories, metrics, and intermediate data.

Usage (on rag-5090):
  conda activate reasonrag
  cd /home/mayi/ReasonRAG
  CUDA_VISIBLE_DEVICES=0 python run_evidence_debug.py
"""

import os
import sys
import json
import time
import datetime

# 让脚本能直接 `python 03_sapr_rag/scripts/xxx.py` 运行：把仓库根加进 sys.path
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.paths import (  # noqa: E402
    REPO_ROOT,
    REASONRAG_ROOT,
    WIKI_CORPUS_PATH,
    BGE_MODEL_PATH,
    LORA_MODEL_PATH,
)

# ── Add ReasonRAG to path ──────────────────────────────────────
sys.path.insert(0, str(REASONRAG_ROOT))

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.dataset.dataset import Dataset as FlashRAGDataset
from pipeline.reasonrag_pipeline import ReasonRAGPipeline

# ── Run ID ─────────────────────────────────────────────────────
RUN_ID = "20260529_evidence_debug_30samples_v2"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Output directories ─────────────────────────────────────────
SAVE_DIR = os.path.join(str(REPO_ROOT), "04_experiments", "logs", RUN_ID)
METRICS_DIR = os.path.join(str(REPO_ROOT), "04_experiments", "metrics", RUN_ID)
FORBIDDEN_DIR = str(REASONRAG_ROOT / "output")

# Safety check
assert not os.path.abspath(SAVE_DIR).startswith(
    os.path.abspath(FORBIDDEN_DIR)
), f"SAFETY: save_dir must not be inside {FORBIDDEN_DIR}"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────
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
    "tensor_parallel_size": 1,
    "framework": "vllm",
    "gpu_id": "0",
    "gpu_memory_utilization": 0.8,
    "generator_max_input_len": 8192,

    # Experiment (matched to ReasonRAG baseline: max_iter=8, topk=3)
    "retrieval_topk": 3,
    "metrics": ["em", "f1", "acc", "recall", "precision"],
    "save_intermediate_data": True,
    "save_note": RUN_ID,
    "save_dir": SAVE_DIR,
    "seed": 2024,
    "disable_save": False,

    # Sampling
    "test_sample_num": None,
    "random_sample": False,
}

print("=" * 70)
print(f"Evidence-only debug_result — RUN {RUN_ID}")
print("=" * 70)
print(f"Time      : {datetime.datetime.now().isoformat()}")
print(f"Server    : rag-5090, GPU 0")
print(f"Dataset   : HotpotQA dev, {SLICE_SIZE} examples")
print(f"Model     : qwen2.5-instruct-ReasonRAG-lora (ReasonRAG-LoRA)")
print(f"Save dir  : {SAVE_DIR}")
print(f"Metrics   : {METRICS_DIR}")
print(f"top_k=3, max_iter=8, max_children=2")
print("=" * 70)

# ── Build Config ───────────────────────────────────────────────
print("\n[1/4] Building config ...")
config = Config(config_dict=config_dict)
actual_save_dir = config["save_dir"]
assert not os.path.abspath(actual_save_dir).startswith(os.path.abspath(FORBIDDEN_DIR))
print(f"  ✓ save_dir: {actual_save_dir}")

# ── Load & slice data ──────────────────────────────────────────
print(f"\n[2/4] Loading HotpotQA dev, slicing {SLICE_SIZE} ...")
all_split = get_dataset(config)
dev_data = all_split["dev"]
print(f"  ✓ Total: {len(dev_data)} examples")

sliced_items = dev_data.data[:SLICE_SIZE]
sliced_data = FlashRAGDataset(
    config=config,
    dataset_path=dev_data.dataset_path,
    data=sliced_items,
    sample_num=None,
    random_sample=False,
)
print(f"  ✓ Sliced: {len(sliced_data)} examples")

# ── Init pipeline ──────────────────────────────────────────────
print(f"\n[3/4] Initializing ReasonRAGPipeline ...")
t0 = time.time()
pipeline = ReasonRAGPipeline(
    config,
    prompt_template=None,
    answer_format="answer",
    max_iter=8,          # ReasonRAG default, 3 was too low
    max_children=2,
    max_rollouts=64,
)
t_init = time.time() - t0
print(f"  ✓ Pipeline ready ({t_init:.1f}s)")

# ── Run ────────────────────────────────────────────────────────
print(f"\n[4/4] Running inference on {len(sliced_data)} examples ...")
t0 = time.time()
output_dataset = pipeline.run(sliced_data, batch_size=1, do_eval=True)
t_run = time.time() - t0
print(f"  ✓ Inference done ({t_run:.1f}s, {t_run/60:.1f}min)")

# ── Save metrics ───────────────────────────────────────────────
print(f"\n[Results] Collecting metrics ...")

# Extract per-item results
results = []
for item in output_dataset.data:
    d = item.to_dict()
    results.append({
        "id": d.get("id", ""),
        "question": d.get("question", ""),
        "golden_answers": d.get("golden_answers", []),
        "pred": d.get("pred", ""),
        "raw_pred": d.get("output", {}).get("raw_pred", ""),
        "em": d.get("output", {}).get("em", None),
        "f1": d.get("output", {}).get("f1", None),
        "acc": d.get("output", {}).get("acc", None),
    })

# Compute aggregate metrics
em_scores = [r["em"] for r in results if r["em"] is not None]
f1_scores = [r["f1"] for r in results if r["f1"] is not None]
acc_scores = [r["acc"] for r in results if r["acc"] is not None]

aggregate = {
    "run_id": RUN_ID,
    "timestamp": TIMESTAMP,
    "dataset": "hotpotqa_dev",
    "num_examples": len(results),
    "method": "ReasonRAG-LoRA baseline",
    "model": "qwen2.5-instruct-ReasonRAG-lora",
    "retriever": "bge-base-en-v1.5",
    "top_k": 3,
    "max_iter": 8,
    "max_children": 2,
    "init_time_s": round(t_init, 1),
    "run_time_s": round(t_run, 1),
    "avg_time_per_example_s": round(t_run / max(len(results), 1), 2),
    "em": round(sum(em_scores) / max(len(em_scores), 1), 4) if em_scores else None,
    "f1": round(sum(f1_scores) / max(len(f1_scores), 1), 4) if f1_scores else None,
    "acc": round(sum(acc_scores) / max(len(acc_scores), 1), 4) if acc_scores else None,
    "label": "debug_result",
    "gpu": "rag-5090:0",
    "save_dir": actual_save_dir,
}

# Save per-item results
results_path = os.path.join(METRICS_DIR, "per_item_results.jsonl")
with open(results_path, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"  ✓ Per-item results: {results_path}")

# Save aggregate metrics
metrics_path = os.path.join(METRICS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(aggregate, f, indent=2, ensure_ascii=False)
print(f"  ✓ Aggregate metrics: {metrics_path}")

# Print summary
print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE — debug_result")
print("=" * 70)
print(f"  EM  : {aggregate['em']}")
print(f"  F1  : {aggregate['f1']}")
print(f"  ACC : {aggregate['acc']}")
print(f"  Time: {t_run:.1f}s ({t_run/60:.1f}min), avg {aggregate['avg_time_per_example_s']:.1f}s/example")
print(f"  N   : {len(results)}")
print(f"  Log : {actual_save_dir}")
print(f"  Out : {METRICS_DIR}")
print("=" * 70)
