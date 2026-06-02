#!/usr/bin/env python3
"""
Minimal-intrusive SAPR-E rerank end-to-end ablation.

This keeps ReasonRAG's original batch state machine, prompt routing, logging,
and stopping behavior. The only intended behavior change is the retrieval step:
retrieve top-10 candidates, apply the SAPR-E v0 state-aware heuristic, and pass
the selected top-3 documents into the original document-analysis prompt.
"""

import argparse
import datetime
import json
import os
import re
import sys
import time

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
from flashrag.dataset.dataset import Dataset as FlashRAGDataset
from flashrag.utils import get_dataset
from pipeline.reasonrag_pipeline import ReasonRAGPipeline


def word_set(text):
    return set(re.findall(r"\b[a-z]{3,}\b", str(text).lower()))


def extract_entities(texts):
    pattern = r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b"
    entities = set()
    for text in texts:
        entities.update(re.findall(pattern, str(text)))
    return set(entity.lower() for entity in entities)


def normalize_title(title):
    title = re.sub(r"[^a-z0-9\s]", "", str(title).strip().lower())
    return re.sub(r"\s+", " ", title).strip()


def doc_to_parts(doc):
    raw = doc if isinstance(doc, str) else doc.get("contents", doc.get("text", ""))
    parts = str(raw).split("\n", 1)
    title = parts[0].strip().strip('"') if parts else ""
    text = parts[1].strip() if len(parts) > 1 else str(raw)
    return {"title": title, "text": text[:500], "raw": doc}


def sapr_e_score(question, history_thoughts, subquery, doc):
    score = 0.0
    doc_text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
    doc_words = word_set(doc_text)

    subquery_words = word_set(subquery)
    if subquery_words:
        score += 2.0 * len(subquery_words & doc_words) / len(subquery_words)

    question_words = word_set(question)
    if question_words:
        score += 1.0 * len(question_words & doc_words) / len(question_words)

    entities = extract_entities([question, subquery])
    if entities:
        hits = sum(1 for entity in entities if entity in doc_text)
        score += 1.5 * hits / len(entities)

    history_words = word_set(" ".join(history_thoughts))
    if doc_words and history_words:
        score += 0.5 * len(doc_words - history_words) / len(doc_words)

    title = normalize_title(doc.get("title", ""))
    for entity in entities:
        if all(piece in title for piece in entity.split()):
            score += 1.0
            break

    return score


def infer_subquery(question, thoughts):
    full_text = " ".join(thoughts)
    match = re.search(r"So the next query is\s*(.*?)(?:\.|$)", full_text, re.IGNORECASE)
    if match and match.group(1).strip():
        return match.group(1).strip()
    matches = re.findall(r"<query>(.*?)</query>", full_text, flags=re.DOTALL)
    if matches and matches[-1].strip():
        return matches[-1].strip()
    matches = re.findall(
        r"(?:Determine|Find|Search for|Look up|Identify)\s+(.*?)(?:\.|$)",
        full_text,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"\d+\.\s+(.*?[?\.])", full_text)
    if matches:
        return matches[-1].strip()
    return question


def select_top3(question, thoughts, subquery, docs):
    formatted = [doc_to_parts(doc) for doc in docs]
    scored = [(sapr_e_score(question, thoughts, subquery, doc), doc) for doc in formatted]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc["raw"] for _, doc in scored[:3]]


def build_config(args, output_dir):
    index_path = args.index_path or str(REASONRAG_ROOT / "indexes/bge_extended/bge_Flat.index")
    return {
        "data_dir": str(REASONRAG_ROOT / "dataset/"),
        "dataset_name": "hotpotqa",
        "split": ["dev", "test"],
        "index_path": index_path,
        "retrieval_method": "bge",
        "corpus_path": args.corpus_path,
        "faiss_gpu": False,
        "model2path": {
            "bge": args.bge_path,
            "qwen2.5-instruct-ReasonRAG-lora": args.generator_path,
        },
        "model2pooling": {"bge": "cls"},
        "method2index": {"bge": None},
        "generator_model": "qwen2.5-instruct-ReasonRAG-lora",
        "generator_batch_size": 1,
        "tensor_parallel_size": 1,
        "framework": "vllm",
        "gpu_id": args.gpu_id,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "generator_max_input_len": 8192,
        "generation_params": {"do_sample": False, "max_tokens": args.max_tokens},
        "retrieval_topk": 3,
        "metrics": ["em", "f1", "acc"],
        "save_intermediate_data": True,
        "save_note": "sapr_e_minimal_rerank",
        "save_dir": output_dir,
        "seed": 2024,
        "disable_save": False,
        "test_sample_num": None,
        "random_sample": False,
    }


def patch_minimal_rerank_run_batch(pipeline):
    def minimal_rerank_run_batch(dataset):
        for item in dataset:
            item.update_output("finish_flag", False)
            item.update_output("iteration_count", 0)
            item.update_output("previous_thoughts", [])
            item.update_output("flag", None)
            item.update_output("query", None)
            item.update_output("answer", None)

        input_prompts = [pipeline.begin_reasoning_prompt.get_string(question=item.question) for item in dataset]
        responses = pipeline.generator.generate(input_prompts, stop=pipeline.stop_tokens)

        for i, item in enumerate(dataset):
            response = responses[i]
            item.previous_thoughts.append(response)
            item.flag = pipeline.get_flags([response])[0]
            item.query = pipeline.get_querys([response])[0]
            item.answer = pipeline.get_answers([response])[0]
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

        for _step in range(pipeline.max_iter):
            exist_items = [item for item in dataset if not item.finish_flag]
            if not exist_items:
                break

            active_questions = [item.question for item in exist_items]
            active_previous_thoughts = [item.previous_thoughts for item in exist_items]
            active_querys = [item.query for item in exist_items]

            question_thoughts_list = [
                q + "\nPrevious Thoughts: " + " ".join(thoughts)
                for q, thoughts in zip(active_questions, active_previous_thoughts)
            ]

            top10_docs, _scores = pipeline.retriever._batch_search(
                active_querys,
                num=10,
                return_score=True,
            )
            retrieval_results = []
            for question, thoughts, query, docs in zip(
                active_questions,
                active_previous_thoughts,
                active_querys,
                top10_docs,
            ):
                subquery = query.strip() if query and query.strip() else infer_subquery(question, thoughts)
                retrieval_results.append(select_top3(question, thoughts, subquery, docs))

            input_prompts = []
            for i, item in enumerate(exist_items):
                if item.iteration_count >= pipeline.max_iter - 1:
                    input_prompts.append(
                        pipeline.answer_generation_prompt.get_string(question=question_thoughts_list[i])
                    )
                elif "query" in item.flag:
                    input_prompts.append(
                        pipeline.document_analysis_prompt.get_string(
                            question=question_thoughts_list[i],
                            retrieval_result=retrieval_results[i],
                        )
                    )
                elif "evidence" in item.flag:
                    input_prompts.append(pipeline.reasoning_prompt.get_string(question=question_thoughts_list[i]))
                else:
                    input_prompts.append(
                        pipeline.answer_generation_prompt.get_string(question=question_thoughts_list[i])
                    )

            responses = pipeline.generator.generate(input_prompts, stop=pipeline.stop_tokens)
            for i, item in enumerate(exist_items):
                response = responses[i]
                item.previous_thoughts.append(response)
                item.iteration_count += 1
                item.flag = pipeline.get_flags([response])[0]
                item.query = pipeline.get_querys([response])[0]
                item.answer = pipeline.get_answers([response])[0]

                node_dict = {
                    "action_name": "unknown",
                    "input_prompt": input_prompts[i],
                    "response": response,
                    "query": item.query,
                    "answer": item.answer,
                    "sapr_e_minimal_rerank": True,
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

                item.update_output(f"intermediate_node_{item.iteration_count}", node_dict)

                if item.flag in ["finish", "answer"] or item.iteration_count >= pipeline.max_iter:
                    item.finish_flag = True

        for i in range(len(dataset)):
            dataset[i].pred = dataset[i].answer

        return dataset

    pipeline.run_batch = minimal_rerank_run_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--index_path", default=None)
    parser.add_argument("--corpus_path", default=str(WIKI_CORPUS_PATH))
    parser.add_argument("--bge_path", default=str(BGE_MODEL_PATH))
    parser.add_argument(
        "--generator_path",
        default=str(LORA_MODEL_PATH),
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    args = parser.parse_args()

    run_id = args.run_id or "{}_sapr_e_minimal_rerank_{}samples".format(
        datetime.datetime.now().strftime("%Y%m%d"),
        args.num_examples,
    )
    output_dir = os.path.join(str(REPO_ROOT), "04_experiments/logs", run_id, "sapr_e_minimal_rerank")
    os.makedirs(output_dir, exist_ok=True)

    config = Config(config_dict=build_config(args, output_dir))
    dev_data = get_dataset(config)["dev"]
    sliced_data = FlashRAGDataset(
        config=config,
        dataset_path=dev_data.dataset_path,
        data=dev_data.data[: args.num_examples],
        sample_num=None,
        random_sample=False,
    )

    print("=" * 70)
    print("SAPR-E minimal rerank E2E ablation")
    print("run_id:", run_id)
    print("examples:", len(sliced_data))
    print("max_tokens:", args.max_tokens)
    print("output:", output_dir)
    print("=" * 70)

    pipeline = ReasonRAGPipeline(
        config,
        prompt_template=None,
        answer_format="answer",
        max_iter=8,
        max_children=2,
        max_rollouts=64,
    )
    patch_minimal_rerank_run_batch(pipeline)

    start = time.time()
    dataset = pipeline.run(sliced_data, batch_size=1, do_eval=True)
    runtime = time.time() - start
    print("Done ({:.1f}s, {:.1f}min)".format(runtime, runtime / 60))

    meta = {
        "run_id": run_id,
        "mode": "sapr_e_minimal_rerank",
        "num_examples": len(dataset.data),
        "max_tokens": args.max_tokens,
        "candidate_topk": 10,
        "selected_topk": 3,
        "runtime_s": round(runtime, 1),
        "label": "debug_result",
    }
    path = os.path.join(output_dir, "metrics_meta.json")
    with open(path, "w") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print("Saved:", path)


if __name__ == "__main__":
    main()
