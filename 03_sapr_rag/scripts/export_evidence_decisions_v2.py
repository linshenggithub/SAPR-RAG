#!/usr/bin/env python3
"""
Export evidence decision points from ReasonRAG pipeline (v2 — query-fixed).

Root cause of v1 bug: ReasonRAG batch run_batch uses extract_query() which
only matches "So the next query is ...". In batch mode the model never
generates this prefix — it reasons from its own knowledge. The pipeline
still calls retriever.batch_search(active_querys) with empty strings.

Fix: Override run_batch to capture active_querys, active_questions, and
active_previous_thoughts at the retrieval boundary. Extract inferred
subqueries from the concatenated thoughts using multi-pattern matching.
Do NOT change model behavior.

Usage (on rag-5090):
  conda activate reasonrag
  cd /home/mayi/ReasonRAG
  CUDA_VISIBLE_DEVICES=0 python export_evidence_decisions_v2.py --num_examples 3
"""

import os
import sys
import json
import re
import time
import argparse

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
from flashrag.dataset.utils import get_batch_dataset, merge_batch_dataset
from flashrag.evaluator.metrics import F1_Score
from flashrag.pipeline import BasicPipeline
from pipeline.reasonrag_pipeline import ReasonRAGPipeline

# ── CLI ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num_examples", type=int, default=3)
args = parser.parse_args()

SLICE_SIZE = args.num_examples
OUTPUT_DIR = os.path.join(str(REPO_ROOT), "04_experiments", "logs", "20260530_evidence_decision_top10_queryfix")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evidence_decision_points.jsonl")

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
    "retrieval_topk": 10,
    "metrics": ["em", "f1", "acc"],
    "save_intermediate_data": True,
    "save_note": "evidence_decision_top10_queryfix",
    "save_dir": OUTPUT_DIR,
    "seed": 2024,
    "disable_save": True,
    "test_sample_num": None,
    "random_sample": False,
}

print("=" * 70)
print("Evidence Decision Point Exporter v2 (query-fixed)")
print("=" * 70)
print("Examples  : {}".format(SLICE_SIZE))
print("top_k     : 10")
print("max_iter  : 8")
print("Output    : {}".format(OUTPUT_FILE))
print("=" * 70)

# ── Build config + load data ───────────────────────────────────
print("\n[1/3] Building config, loading data ...")
config = Config(config_dict=config_dict)
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


# ── Subquery extraction from thoughts ──────────────────────────
def infer_subquery(question, thoughts_list):
    """Extract the implicit subquery from concatenated previous thoughts.

    Tries multiple patterns in order:
    1. "So the next query is ..."  (ReasonRAG standard)
    2. <query>...</query>           (alternative tag format)
    3. "Determine ..." / "Find ..." / "Search for ..." / "Look up ..."
    4. Numbered sub-questions: "1. Determine X's nationality"
    5. Fallback: return the original question
    """
    full_text = " ".join(thoughts_list)

    # Pattern 1: "So the next query is ..."
    m = re.search(r'So the next query is\s*(.*?)(?:\.|$)', full_text, re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip()

    # Pattern 2: <query>...</query>
    m = re.findall(r'<query>(.*?)</query>', full_text)
    if m and m[-1].strip():
        return m[-1].strip()

    # Pattern 3: imperative verbs
    for pattern in [
        r'(?:Determine|Find|Search for|Look up|Identify|Discover)\s+(.*?)(?:\.|$)',
    ]:
        m = re.findall(pattern, full_text, re.IGNORECASE)
        if m:
            return m[-1].strip()

    # Pattern 4: numbered sub-questions like "1. Determine X"
    m = re.findall(r'\d+\.\s+(.*?[?\.»])', full_text)
    if m:
        return m[-1].strip()

    # Fallback: original question
    return question


def format_docs(docs, scores):
    """Format retrieved docs with title extraction from corpus 'contents' field."""
    formatted = []
    for d, s in zip(docs, scores):
        raw = d.get("contents", d.get("text", ""))
        parts = raw.split("\n", 1)
        title = parts[0].strip().strip('"') if parts else ""
        text = parts[1].strip() if len(parts) > 1 else raw
        formatted.append({
            "title": title,
            "text": text[:500],
            "score": float(s) if s is not None else None,
        })
    return formatted


# ── Instrumented run_batch ─────────────────────────────────────
# We override run_batch to capture retrieval context at each step boundary.

decision_points = []  # global accumulator


def make_instrumented_run_batch(orig_pipeline):
    """Return a patched run_batch that captures retrieval boundary data."""
    orig_run_batch = orig_pipeline.run_batch
    retriever = orig_pipeline.retriever
    max_iter = orig_pipeline.max_iter
    stop_tokens = orig_pipeline.stop_tokens
    begin_reasoning_prompt = orig_pipeline.begin_reasoning_prompt
    document_analysis_prompt = orig_pipeline.document_analysis_prompt
    reasoning_prompt = orig_pipeline.reasoning_prompt
    answer_generation_prompt = orig_pipeline.answer_generation_prompt
    generator = orig_pipeline.generator

    def get_flags(responses):
        flags = []
        for r in responses:
            if "So the next query is" in r:
                flags.append("query")
            elif "So the answer is" in r:
                flags.append("answer")
            elif "<evidence>" in r:
                flags.append("evidence")
            else:
                flags.append("None")
        return flags

    def extract_query(response):
        m = re.search(r'So the next query is\s*(.*?)(?=\n|$)', response, re.IGNORECASE | re.DOTALL)
        if not m:
            return ""
        text = m.group(1).strip()
        text = re.sub(r'</?(answer|query|evidence)>', '', text)
        return text.strip()

    def extract_answer(response):
        prefix = "So the answer is"
        if prefix in response:
            pred = response.split(prefix)[1].strip()
        else:
            pred = response
        answer_matches = re.findall(r'<answer>(.*?)</answer>', pred)
        pred = answer_matches[-1] if answer_matches else pred
        pred = re.sub(r'<answer.*?>.*?</answer>|<query.*?>.*?</query>|answer>|<answer', '', pred, flags=re.DOTALL)
        if '.' in pred:
            pred = pred.split('.')[0].strip()
        else:
            pred = pred.strip()
        return pred

    def instrumented_run_batch(dataset):
        for item in dataset:
            item.update_output('finish_flag', False)
            item.update_output('iteration_count', 0)
            item.update_output('previous_thoughts', [])
            item.update_output('flag', None)
            item.update_output('query', None)
            item.update_output('answer', None)

        # Step 0: begin reasoning
        input_prompts = [begin_reasoning_prompt.get_string(question=item.question) for item in dataset]
        responses = generator.generate(input_prompts, stop=stop_tokens)

        for i, item in enumerate(dataset):
            response = responses[i]
            item.previous_thoughts.append(response)
            item.flag = get_flags([response])[0]
            item.query = extract_query(response)
            item.answer = extract_answer(response)

            node_dict = {
                "action_name": "begin_reasoning",
                "input_prompt": input_prompts[i],
                "response": response,
                "query": item.query,
                "answer": item.answer,
            }
            item.update_output("intermediate_node_0", node_dict)
            if item.flag in ["finish", "answer"]:
                item.finish_flag = True

        # Iterative steps
        for step in range(1, max_iter + 1):
            exist_items = [item for item in dataset if not item.finish_flag]
            if not exist_items:
                break

            active_questions = [item.question for item in exist_items]
            active_previous_thoughts = [item.previous_thoughts for item in exist_items]
            active_flags = [item.flag for item in exist_items]
            active_querys = [item.query for item in exist_items]

            # === CAPTURE RETRIEVAL BOUNDARY ===
            result = retriever._batch_search(active_querys, num=10, return_score=True)
            docs_list, scores_list = result

            # Store decision points for this retrieval step
            for i, item in enumerate(exist_items):
                pipeline_query = active_querys[i] if isinstance(active_querys[i], str) else ""
                inferred = infer_subquery(active_questions[i], active_previous_thoughts[i])
                dp = {
                    "item_id": item.data.get("id", ""),
                    "item_index": dataset.data.index(item),
                    "step": step,
                    "action": "retrieval_boundary",
                    "original_question": active_questions[i],
                    "golden_answers": item.data.get("golden_answers", []),
                    "supporting_facts": item.data.get("metadata", {}).get("supporting_facts", {}),
                    "history_thoughts": [t[:200] for t in active_previous_thoughts[i]],
                    "pipeline_query": pipeline_query,          # what pipeline sent (may be "")
                    "inferred_subquery": inferred,              # extracted from thoughts
                    "retrieval_top10": format_docs(docs_list[i], scores_list[i]),
                }
                decision_points.append(dp)

            # Continue pipeline normally
            retrieval_results = docs_list  # original format for model input

            question_thoughts_list = [
                q + "\nPrevious Thoughts: " + " ".join(thoughts)
                for q, thoughts in zip(active_questions, active_previous_thoughts)
            ]

            input_prompts = []
            for i, item in enumerate(exist_items):
                if item.iteration_count >= max_iter - 1:
                    input_prompts.append(answer_generation_prompt.get_string(question=question_thoughts_list[i]))
                elif "query" in item.flag:
                    input_prompts.append(document_analysis_prompt.get_string(
                        question=question_thoughts_list[i], retrieval_result=retrieval_results[i]
                    ))
                elif "evidence" in item.flag:
                    input_prompts.append(reasoning_prompt.get_string(question=question_thoughts_list[i]))
                else:
                    input_prompts.append(answer_generation_prompt.get_string(question=question_thoughts_list[i]))

            responses = generator.generate(input_prompts, stop=stop_tokens)
            for i, item in enumerate(exist_items):
                response = responses[i]
                item.previous_thoughts.append(response)
                item.iteration_count += 1
                item.flag = get_flags([response])[0]
                item.query = extract_query(response)
                item.answer = extract_answer(response)

                node_dict = {
                    "action_name": "unknown",
                    "input_prompt": input_prompts[i],
                    "response": response,
                    "query": item.query,
                    "answer": item.answer,
                }
                if "evidence" in item.flag:
                    node_dict["action_name"] = "document_analysis"
                    node_dict["retrieval_result"] = retrieval_results[i]
                elif "query" in item.flag:
                    node_dict["action_name"] = "query_generation"
                elif "answer" in item.flag:
                    node_dict["action_name"] = "answer_generation"
                elif "finish" in item.flag:
                    node_dict["action_name"] = "finish"

                item.update_output("intermediate_node_{}".format(item.iteration_count), node_dict)
                if item.flag in ["finish", "answer"] or item.iteration_count >= max_iter:
                    item.finish_flag = True

        for i in range(len(dataset)):
            dataset[i].pred = dataset[i].answer

        return dataset

    return instrumented_run_batch


# ── Init pipeline + patch ──────────────────────────────────────
print("\n[2/3] Initializing pipeline ...")
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

# Patch run_batch with instrumented version
pipeline.run_batch = make_instrumented_run_batch(pipeline)

# ── Run ────────────────────────────────────────────────────────
print("\n[3/3] Running inference ...")
t0 = time.time()
output_dataset = pipeline.run(sliced_data, batch_size=1, do_eval=True)
t_run = time.time() - t0
print("  Inference done ({:.1f}s)".format(t_run))

# ── Add summary points + write ────────────────────────────────
print("\n[Results] Finalizing ...")
for item_idx, item in enumerate(output_dataset.data):
    out = item.to_dict()
    ms = out.get("output", {}).get("metric_score", {})
    decision_points.append({
        "item_id": out.get("id", ""),
        "item_index": item_idx,
        "step": -1,
        "action": "summary",
        "original_question": out.get("question", ""),
        "golden_answers": out.get("golden_answers", []),
        "supporting_facts": out.get("metadata", {}).get("supporting_facts", {}),
        "final_pred": out.get("output", {}).get("pred", ""),
        "total_steps": out.get("output", {}).get("iteration_count", 0),
        "flag": out.get("output", {}).get("flag"),
        "em": ms.get("em", 0),
        "f1": ms.get("f1", 0),
    })

with open(OUTPUT_FILE, "w") as f:
    for dp in decision_points:
        f.write(json.dumps(dp, ensure_ascii=False) + "\n")

# Statistics
retrieval_steps = [dp for dp in decision_points if dp["step"] > 0]
summaries = [dp for dp in decision_points if dp["step"] == -1]
pq_empty = sum(1 for dp in retrieval_steps if not dp.get("pipeline_query", "").strip())
iq_nonempty = sum(1 for dp in retrieval_steps if dp.get("inferred_subquery", "").strip())

print("  Decision points: {}".format(len(decision_points)))
print("  Retrieval steps: {}".format(len(retrieval_steps)))
print("  Summaries: {}".format(len(summaries)))
print("  pipeline_query empty: {}/{}".format(pq_empty, len(retrieval_steps)))
print("  inferred_subquery non-empty: {}/{}".format(iq_nonempty, len(retrieval_steps)))

if summaries:
    all_em = [s["em"] for s in summaries]
    print("  EM: {:.4f} ({}/{})".format(sum(all_em)/len(all_em), int(sum(all_em)), len(all_em)))

# Schema preview
for dp in decision_points:
    if dp["step"] > 0:
        preview = {k: v for k, v in dp.items() if k != "retrieval_top10"}
        preview["retrieval_top10"] = "[{} docs, first: title={}, score={:.4f}]".format(
            len(dp["retrieval_top10"]),
            dp["retrieval_top10"][0]["title"][:40] if dp["retrieval_top10"] else "N/A",
            dp["retrieval_top10"][0]["score"] if dp["retrieval_top10"] else 0,
        )
        print("\n=== SCHEMA PREVIEW ===")
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        break

print("\n" + "=" * 70)
print("DONE — Output: {}".format(OUTPUT_FILE))
print("=" * 70)
