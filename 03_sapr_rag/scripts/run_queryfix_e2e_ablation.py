#!/usr/bin/env python3
"""
Query-fix end-to-end ablation for ReasonRAG.

This script keeps ReasonRAG's batch state machine intact and changes only one
retrieval behavior: when the generated query is empty, infer a subquery from the
existing thoughts and use that for top-3 retrieval. It is meant to isolate
whether the earlier SAPR-E end-to-end drop came from query repair or from
evidence reranking.
"""

import argparse
import datetime
import json
import os
import re
import sys
import time

REASONRAG_ROOT = os.environ.get("REASONRAG_ROOT", "/home/mayi/ReasonRAG")
RESEARCH_ROOT = os.environ.get("RESEARCH_ROOT", "/home/mayi/RAG/agentic-rag-process-optimization")
sys.path.insert(0, REASONRAG_ROOT)

from flashrag.config import Config
from flashrag.dataset.dataset import Dataset as FlashRAGDataset
from flashrag.utils import get_dataset
from pipeline.reasonrag_pipeline import ReasonRAGPipeline


def infer_subquery(question, thoughts_list):
    full_text = " ".join(thoughts_list)
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


def build_config(args, output_dir):
    index_path = args.index_path or os.path.join(REASONRAG_ROOT, "indexes/bge_extended/bge_Flat.index")
    return {
        "data_dir": os.path.join(REASONRAG_ROOT, "dataset/"),
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
        "save_note": "queryfix_e2e_top3",
        "save_dir": output_dir,
        "seed": 2024,
        "disable_save": False,
        "test_sample_num": None,
        "random_sample": False,
    }


def patch_queryfix_run_batch(pipeline):
    def queryfix_run_batch(dataset):
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
            item.update_output(
                "intermediate_node_0",
                {
                    "action_name": "begin_reasoning",
                    "input_prompt": input_prompts[i],
                    "response": response,
                    "query": item.query,
                    "answer": item.answer,
                    "queryfix_used": False,
                },
            )
            if item.flag in ["finish", "answer"]:
                item.finish_flag = True

        for _step in range(pipeline.max_iter):
            exist_items = [item for item in dataset if not item.finish_flag]
            if not exist_items:
                break

            active_questions = [item.question for item in exist_items]
            active_previous_thoughts = [item.previous_thoughts for item in exist_items]
            active_querys = [item.query for item in exist_items]
            search_querys = []
            queryfix_used = []
            for question, thoughts, query in zip(active_questions, active_previous_thoughts, active_querys):
                fixed = False
                if not query or not query.strip():
                    query = infer_subquery(question, thoughts)
                    fixed = True
                search_querys.append(query)
                queryfix_used.append(fixed)

            question_thoughts_list = [
                q + "\nPrevious Thoughts: " + " ".join(thoughts)
                for q, thoughts in zip(active_questions, active_previous_thoughts)
            ]
            retrieval_results = pipeline.retriever.batch_search(search_querys)

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
                    "search_query": search_querys[i],
                    "queryfix_used": queryfix_used[i],
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
                item.update_output(f"intermediate_node_{item.iteration_count}", node_dict)

                if item.flag in ["finish", "answer"] or item.iteration_count >= pipeline.max_iter:
                    item.finish_flag = True

        for item in dataset:
            item.pred = item.answer
        return dataset

    pipeline.run_batch = queryfix_run_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--index_path", default=None)
    parser.add_argument("--corpus_path", default="/nas/mayi/RAG/corpus/wiki18_extended.jsonl")
    parser.add_argument("--bge_path", default="/nas/mayi/RAG/retrievers/bge-base-en-v1.5")
    parser.add_argument(
        "--generator_path",
        default="/home/mayi/LLaMA-Factory/examples/merge_lora/output/qwen2.5-7B-lora-dpo-RAG-ProGuide",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    args = parser.parse_args()

    run_id = args.run_id or "{}_queryfix_top3_{}samples".format(
        datetime.datetime.now().strftime("%Y%m%d"), args.num_examples
    )
    output_dir = os.path.join(RESEARCH_ROOT, "04_experiments/logs", run_id, "queryfix_top3")
    os.makedirs(output_dir, exist_ok=True)

    config_dict = build_config(args, output_dir)
    assert not os.path.abspath(output_dir).startswith(os.path.join(REASONRAG_ROOT, "output"))
    config = Config(config_dict=config_dict)
    all_split = get_dataset(config)
    dev_data = all_split["dev"]
    sliced_data = FlashRAGDataset(
        config=config,
        dataset_path=dev_data.dataset_path,
        data=dev_data.data[: args.num_examples],
        sample_num=None,
        random_sample=False,
    )

    print("=" * 70)
    print("Queryfix top-3 E2E ablation")
    print("run_id:", run_id)
    print("examples:", len(sliced_data))
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
    patch_queryfix_run_batch(pipeline)

    t0 = time.time()
    dataset = pipeline.run(sliced_data, batch_size=1, do_eval=True)
    runtime = time.time() - t0
    print("Done ({:.1f}s, {:.1f}min)".format(runtime, runtime / 60))

    result = {
        "run_id": run_id,
        "mode": "queryfix_top3",
        "num_examples": len(dataset.data),
        "max_tokens": args.max_tokens,
        "retrieval_topk": 3,
        "runtime_s": round(runtime, 1),
        "label": "debug_result",
    }
    with open(os.path.join(output_dir, "metrics_meta.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("Saved:", os.path.join(output_dir, "metrics_meta.json"))


if __name__ == "__main__":
    main()
