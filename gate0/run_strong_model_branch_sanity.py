#!/usr/bin/env python3
"""Small GPT-4o sibling-generation sanity check.

This script sends only selected questions to the model. It does not send
golden answers, existing trajectories, or retrieval results.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path


BEGIN_REASONING_SYSTEM = """You are an assistant for question answering with access to a retrieval tool. Upon receiving a question, your task is to:
* Analyze and Decompose the Question: Break the question into smaller, manageable sub-questions to ensure all aspects are addressed.
* Evaluate Your Knowledge: Assess each sub-question or component:
- Identify parts you can confidently answer based on your existing knowledge.
- Pinpoint parts that require additional information or verification through retrieval tools.
* Conciseness: Ensure both queries and answers are concise, using nouns or short phrases whenever possible.
* Respond Format:
If your knowledge is sufficient to answer the question, conclude with:
"So the answer is <answer>answer</answer>"
If retrieval is necessary to provide a complete answer, conclude with:
"So the next query is <query>query</query>"
"""


def normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def similarity(left: str, right: str) -> float:
    return round(difflib.SequenceMatcher(None, normalize(left), normalize(right)).ratio(), 4)


def extract_tag(text: str, tag: str) -> str:
    matches = re.findall(fr"<{tag}>(.*?)</{tag}>", text or "", flags=re.DOTALL)
    return matches[-1].strip() if matches else ""


def action_type(text: str) -> str:
    if "<answer>" in (text or ""):
        return "answer"
    if "<query>" in (text or ""):
        return "query"
    return "other"


def select_questions(key_branches_path: Path, n_questions: int) -> list[dict]:
    with open(key_branches_path, encoding="utf-8") as f:
        branches = json.load(f)

    root_branches = [b for b in branches if b.get("parent_step") == 0]
    if len(root_branches) < n_questions:
        root_branches = branches

    selected = []
    used_questions = set()

    def add(branch: dict) -> None:
        question = branch.get("question", "")
        if question and question not in used_questions and len(selected) < n_questions:
            selected.append(branch)
            used_questions.add(question)

    for branch in root_branches:
        if branch.get("typed_failure_diff") or branch.get("typed_score_diff"):
            add(branch)

    for branch in sorted(root_branches, key=lambda b: b.get("response_similarity", 1.0)):
        add(branch)

    for branch in sorted(root_branches, key=lambda b: b.get("response_similarity", 0.0), reverse=True):
        add(branch)

    return [
        {
            "traj_idx": b.get("traj_idx"),
            "bp_idx": b.get("bp_idx"),
            "question": b.get("question"),
            "original_response_similarity": b.get("response_similarity"),
            "original_typed_scores": b.get("typed_scores"),
            "original_typed_failures": b.get("typed_failure_types"),
        }
        for b in selected
    ]


def make_client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Need openai package in the current Python environment.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY before running.")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://www.dmxapi.cn/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def generate_once(client, model: str, question: str, temperature: float, max_tokens: int) -> tuple[str, dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": BEGIN_REASONING_SYSTEM},
            {"role": "user", "content": f"Question: {question}"},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    usage = {
        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
    }
    return text, usage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny strong-model branch sanity check")
    parser.add_argument(
        "--key-branches",
        type=Path,
        default=Path(__file__).resolve().parent
        / "data/branch_quality_offline/q_same_content_diff_branches.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data/strong_model_branch_sanity",
    )
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--children", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    questions = select_questions(args.key_branches, args.n_questions)
    client = make_client()

    results = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for item in questions:
        generations = []
        for child_idx in range(args.children):
            text, usage = generate_once(
                client,
                args.model,
                item["question"],
                args.temperature,
                args.max_tokens,
            )
            total_usage["prompt_tokens"] += usage["prompt_tokens"]
            total_usage["completion_tokens"] += usage["completion_tokens"]
            generations.append(
                {
                    "child_idx": child_idx,
                    "response": text,
                    "response_norm": normalize(text),
                    "action_type": action_type(text),
                    "answer": extract_tag(text, "answer"),
                    "query": extract_tag(text, "query"),
                    "usage": usage,
                }
            )
            time.sleep(args.sleep)
        pair_similarity = similarity(generations[0]["response"], generations[1]["response"])
        results.append(
            {
                **item,
                "generations": generations,
                "pair_similarity": pair_similarity,
                "exact_duplicate": generations[0]["response_norm"] == generations[1]["response_norm"],
                "same_action": generations[0]["action_type"] == generations[1]["action_type"],
                "same_answer": generations[0]["answer"] == generations[1]["answer"],
                "same_query": generations[0]["query"] == generations[1]["query"],
            }
        )

    metrics = {
        "label": "debug_result",
        "description": "Tiny GPT-4o sibling-generation sanity check. Sends questions only.",
        "model": args.model,
        "n_questions": len(results),
        "children_per_question": args.children,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "exact_duplicate_questions": sum(r["exact_duplicate"] for r in results),
        "near_duplicate_0_95_questions": sum(r["pair_similarity"] >= 0.95 for r in results),
        "same_action_questions": sum(r["same_action"] for r in results),
        "same_answer_questions": sum(r["same_answer"] for r in results),
        "same_query_questions": sum(r["same_query"] for r in results),
        "avg_pair_similarity": round(sum(r["pair_similarity"] for r in results) / max(len(results), 1), 4),
        "total_usage": total_usage,
        "total_tokens": total_usage["prompt_tokens"] + total_usage["completion_tokens"],
    }

    (args.out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"results: {args.out_dir / 'results.json'}")
    print(f"metrics: {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
