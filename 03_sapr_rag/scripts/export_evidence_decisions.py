#!/usr/bin/env python3
"""
Export evidence decision points from ReasonRAG pipeline.

Runs ReasonRAG-LoRA on HotpotQA dev subset with retrieval_topk=10,
intercepts all retrieval calls, and exports structured decision points
showing candidate evidence at each reasoning step.

Usage (on rag-5090):
  conda activate reasonrag
  cd /home/mayi/ReasonRAG
  CUDA_VISIBLE_DEVICES=0 python export_evidence_decisions.py --num_examples 3
"""

import os
import sys
import json
import re
import time
import datetime
import argparse
from copy import deepcopy

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

sys.path.insert(0, str(REASONRAG_ROOT))

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.dataset.dataset import Dataset as FlashRAGDataset
from pipeline.reasonrag_pipeline import ReasonRAGPipeline

# ── CLI ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num_examples", type=int, default=3, help="Number of examples to process")
args = parser.parse_args()

SLICE_SIZE = args.num_examples
OUTPUT_DIR = os.path.join(str(REPO_ROOT), "04_experiments", "logs", "20260529_evidence_decision_top10")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evidence_decision_points.jsonl")
FORBIDDEN_DIR = str(REASONRAG_ROOT / "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────
config_dict = {
    "data_dir": str(REASONRAG_ROOT / "dataset/"),
    "dataset_name": "hotpotqa",
    "split": ["dev", "test"],
    "index_path": str(REASONRAG_ROOT / "indexes/bge_extended/bge_Flat.index"),
    "retrieval_method": "bge",
    "corpus_path": str(WIKI_CORPUS_PATH),
    "faiss_gpu": False,
    "model2path": {
        "bge": str(BGE_MODEL_PATH),
        "qwen2.5-instruct-ReasonRAG-lora": str(LORA_MODEL_PATH),
    },
    "model2pooling": {"bge": "cls"},
    "method2index": {"bge": None},
    "generator_model": "qwen2.5-instruct-ReasonRAG-lora",
    "generator_batch_size": 1,
    "tensor_parallel_size": 1,
    "framework": "vllm",
    "gpu_id": "0",
    "gpu_memory_utilization": 0.8,
    "generator_max_input_len": 8192,
    "retrieval_topk": 10,           # expose top-10 candidate evidence
    "metrics": ["em", "f1", "acc"],
    "save_intermediate_data": True,
    "save_note": f"evidence_decision_top10_{SLICE_SIZE}samples",
    "save_dir": OUTPUT_DIR,
    "seed": 2024,
    "disable_save": True,           # we handle saving ourselves
    "test_sample_num": None,
    "random_sample": False,
}

print("=" * 70)
print("Evidence Decision Point Exporter")
print("=" * 70)
print("Examples  : {}".format(SLICE_SIZE))
print("top_k     : 10")
print("max_iter  : 8")
print("Output    : {}".format(OUTPUT_FILE))
print("=" * 70)

# ── Build config + load data ───────────────────────────────────
print("\n[1/4] Building config ...")
config = Config(config_dict=config_dict)
print("  save_dir: {}".format(config["save_dir"]))

print("\n[2/4] Loading data, slicing {} ...".format(SLICE_SIZE))
all_split = get_dataset(config)
dev_data = all_split["dev"]
sliced_items = dev_data.data[:SLICE_SIZE]
sliced_data = FlashRAGDataset(
    config=config,
    dataset_path=dev_data.dataset_path,
    data=sliced_items,
    sample_num=None,
    random_sample=False,
)
print("  Sliced: {} examples".format(len(sliced_data)))

# ── Monkey-patch retriever to intercept all retrieval calls ─────
print("\n[3/4] Initializing pipeline with retrieval interceptor ...")

# Global storage: {(item_idx, step): {"query": str, "docs": [...], "scores": [...]}}
retrieval_log = {}
_call_counter = [0]  # mutable counter

# We'll patch run_batch on the pipeline instance after creation
# Strategy: wrap pipeline.run_batch to capture retrieval results at each step

t0 = time.time()
pipeline = ReasonRAGPipeline(
    config,
    prompt_template=None,
    answer_format="answer",
    max_iter=8,
    max_children=2,
    max_rollouts=64,
)
t_init = time.time() - t0
print("  Pipeline ready ({:.1f}s)".format(t_init))

# Patch the retriever's batch_search to also return scores
original_batch_search = pipeline.retriever.batch_search

def patched_batch_search(queries, **kwargs):
    # Always get scores
    result = pipeline.retriever._batch_search(queries, num=10, return_score=True)
    docs_list, scores_list = result
    # Store in log keyed by call order
    for i, (q, docs, scores) in enumerate(zip(queries, docs_list, scores_list)):
        _call_counter[0] += 1
        formatted_docs = []
        for d, s in zip(docs, scores):
            raw = d.get("contents", d.get("text", ""))
            # Corpus format: contents = "Title\nRest of text..."
            parts = raw.split("\n", 1)
            title = parts[0].strip().strip('"') if parts else ""
            text = parts[1].strip() if len(parts) > 1 else raw
            formatted_docs.append({
                "title": title,
                "text": text[:500],
                "score": float(s) if s is not None else None,
            })
        retrieval_log[_call_counter[0]] = {
            "query": q,
            "docs": formatted_docs,
        }
    return docs_list  # return original format (no scores)

pipeline.retriever.batch_search = patched_batch_search

# ── Run pipeline ───────────────────────────────────────────────
print("\n[4/4] Running inference (topk=10, capturing retrieval) ...")
t0 = time.time()

# We need to track which retrieval call belongs to which item at which step.
# The run_batch processes items iteratively. We'll track via dataset index.
# Since batch_size=1, each step processes one item at a time.

# Re-initialize call counter
_call_counter[0] = 0
retrieval_log.clear()

output_dataset = pipeline.run(sliced_data, batch_size=1, do_eval=True)
t_run = time.time() - t0
print("  Inference done ({:.1f}s)".format(t_run))
print("  Total retrieval calls: {}".format(len(retrieval_log)))

# ── Build decision points ──────────────────────────────────────
print("\n[Results] Building evidence decision points ...")

# Now we need to map retrieval calls back to items and steps.
# With batch_size=1, run_batch processes items sequentially.
# The retrieval happens once per step for the active items.
# Since batch_size=1, each retrieval call is for exactly 1 query.

decision_points = []
retrieval_idx = 0  # sequential counter

for item_idx, item in enumerate(output_dataset.data):
    out = item.to_dict()
    item_id = out.get("id", "")
    question = out.get("question", "")
    golden_answers = out.get("golden_answers", [])
    pred = out.get("output", {}).get("pred", "")
    metadata = out.get("metadata", {})
    supporting_facts = metadata.get("supporting_facts", {}) if isinstance(metadata, dict) else {}

    # Extract per-step data from intermediate nodes
    previous_thoughts = []
    for step in range(9):  # max 8 iterations + initial
        node_key = "intermediate_node_{}".format(step)
        node = out.get("output", {}).get(node_key)
        if node is None:
            break

        action = node.get("action_name", "unknown")
        response = node.get("response", "")
        raw_query = node.get("query", "")
        raw_answer = node.get("answer", "")

        # Extract query from response using regex (more reliable than node field)
        query_matches = re.findall(r'<query>(.*?)</query>', response)
        extracted_query = query_matches[-1].strip() if query_matches else None

        answer_matches = re.findall(r'<answer>(.*?)</answer>', response)
        extracted_answer = answer_matches[-1].strip() if answer_matches else None

        previous_thoughts.append(response)

        # For steps that involve retrieval (step >= 1), attach retrieval results
        step_retrieval = None
        if step >= 1:
            retrieval_idx += 1
            if retrieval_idx in retrieval_log:
                step_retrieval = retrieval_log[retrieval_idx]

        dp = {
            "item_id": item_id,
            "item_index": item_idx,
            "step": step,
            "action": action,
            "original_question": question,
            "golden_answers": golden_answers,
            "supporting_facts": supporting_facts,
            "history_thoughts": [t[:200] for t in previous_thoughts[:-1]],  # exclude current
            "current_response": response[:300],
            "extracted_query": extracted_query,
            "extracted_answer": extracted_answer,
            "retrieval_query": step_retrieval["query"] if step_retrieval else None,
            "retrieval_top10": step_retrieval["docs"] if step_retrieval else [],
            "is_final_step": (step == out.get("output", {}).get("iteration_count", 0)),
        }
        decision_points.append(dp)

    # Add summary point
    final_dp = {
        "item_id": item_id,
        "item_index": item_idx,
        "step": -1,
        "action": "summary",
        "original_question": question,
        "golden_answers": golden_answers,
        "supporting_facts": supporting_facts,
        "final_pred": pred,
        "total_steps": out.get("output", {}).get("iteration_count", 0),
        "flag": out.get("output", {}).get("flag"),
        "em": out.get("output", {}).get("metric_score", {}).get("em", 0),
        "f1": out.get("output", {}).get("metric_score", {}).get("f1", 0),
    }
    decision_points.append(final_dp)

# Write output
with open(OUTPUT_FILE, "w") as f:
    for dp in decision_points:
        f.write(json.dumps(dp, ensure_ascii=False) + "\n")

print("  Written {} decision points to {}".format(len(decision_points), OUTPUT_FILE))

# Print summary
retrieval_steps = [dp for dp in decision_points if dp.get("retrieval_top10")]
print("\n  Decision points with retrieval: {}".format(len(retrieval_steps)))
print("  Summary points: {}".format(len([dp for dp in decision_points if dp["step"] == -1])))
print("  Total points: {}".format(len(decision_points)))

# Print first retrieval step as schema preview
for dp in decision_points:
    if dp.get("retrieval_top10"):
        print("\n=== SCHEMA PREVIEW (first retrieval step) ===")
        print(json.dumps(dp, indent=2, ensure_ascii=False)[:1500])
        break

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
