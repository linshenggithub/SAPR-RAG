#!/usr/bin/env python3
"""Offline branch-quality audit for Gate 0.

This script does not call any API and does not use retrieval. It audits whether
ReasonRAG MCTS sibling branches are genuinely different, then applies the
existing rule-based typed transition evaluator to content-different branches.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from typed_eval import evaluate_transition, extract_evidence, state_from_question


DATA_DIR = Path(__file__).resolve().parent / "data"


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def content_signature(node: dict) -> str:
    return "|".join(
        [
            normalize_text(node.get("action_name")),
            normalize_text(node.get("query")),
            normalize_text(node.get("response")),
        ]
    )


def response_similarity(children: list[dict]) -> float:
    if len(children) < 2:
        return 1.0
    ratios = []
    for i, left in enumerate(children):
        for right in children[i + 1 :]:
            ratios.append(
                difflib.SequenceMatcher(
                    None,
                    normalize_text(left.get("response")),
                    normalize_text(right.get("response")),
                ).ratio()
            )
    return min(ratios) if ratios else 1.0


def load_reward_data(reward_dir: Path) -> list[dict]:
    data = []
    for idx in range(4):
        path = reward_dir / f"reward_data{idx}.json"
        if not path.exists():
            print(f"[WARN] missing {path}")
            continue
        with open(path, encoding="utf-8") as f:
            part = json.load(f)
        print(f"loaded {path.name}: {len(part)} trajectories")
        data.extend(part)
    return data


def parse_node(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    return {
        "Q": value.get("Q"),
        "reward": value.get("reward"),
        "step": value.get("step"),
        "N": value.get("N"),
        "parent_id": value.get("parent_id"),
        "children_ids": value.get("children_ids", []),
        "action_name": value.get("action_name"),
        "query": value.get("query", "") or "",
        "response": value.get("response", "") or "",
        "input_prompt": value.get("input_prompt", "") or "",
        "retrieval_result": value.get("retrieval_result", []),
    }


def extract_tree(output: dict) -> dict[int, dict]:
    nodes: dict[int, dict] = {}
    for key, value in output.items():
        if not key.startswith("intermediate_node_"):
            continue
        try:
            node_id = int(key.split("_")[-1])
        except ValueError:
            continue
        node = parse_node(value)
        if node is not None:
            node["node_id"] = node_id
            nodes[node_id] = node
    return nodes


def find_branch_points(nodes: dict[int, dict]) -> list[dict]:
    branch_points = []
    for node_id, node in nodes.items():
        children = [nodes[cid] for cid in node.get("children_ids", []) if cid in nodes]
        if len(children) < 2:
            continue
        q_values = [child.get("Q") for child in children if child.get("Q") is not None]
        branch_points.append(
            {
                "parent_node_id": node_id,
                "parent_step": node.get("step"),
                "parent_action_name": node.get("action_name"),
                "parent_response": node.get("response", "") or "",
                "children": children,
                "q_all_same": len(set(q_values)) == 1 if len(q_values) >= 2 else False,
                "child_q_values": q_values,
                "content_all_same": len({content_signature(child) for child in children}) == 1,
                "response_similarity": response_similarity(children),
            }
        )
    return branch_points


def get_path_to_node(nodes: dict[int, dict], node_id: int) -> list[dict]:
    path = []
    seen = set()
    current = node_id
    while current in nodes and current not in seen:
        seen.add(current)
        node = nodes[current]
        path.append(node)
        current = node.get("parent_id", -1)
    return list(reversed(path))


def has_non_empty_evidence(text: str) -> bool:
    evidence = extract_evidence(text or "")
    if not evidence:
        return False
    return evidence.strip().lower() not in {"none", "n/a", "no relevant evidence"}


def infer_open_gaps(question: str, parent_path: list[dict], child: dict) -> list[str]:
    response = child.get("response", "") or ""
    has_prior_evidence = any(
        has_non_empty_evidence(node.get("response", "") or "") for node in parent_path
    )
    if "<answer>" in response and has_prior_evidence:
        return []
    return [question]


def extract_query_from_response(response: str) -> str:
    matches = re.findall(r"<query>(.*?)</query>", response or "", flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


def extract_history_queries(parent_path: list[dict]) -> list[str]:
    queries = []
    for node in parent_path:
        query = node.get("query", "") or extract_query_from_response(node.get("response", ""))
        if query:
            queries.append(query)
    return queries


def typed_score(result: dict) -> float:
    phi_s = result["phi_s"]
    phi_s_norm = 1.0 if phi_s > 0 else (0.5 if phi_s == 0 else 0.0)
    return round((result["phi_q"] + result["phi_c"] + phi_s_norm) / 3.0, 4)


def evaluate_child(question: str, parent_path: list[dict], child: dict) -> dict:
    state = state_from_question(question, open_gaps=infer_open_gaps(question, parent_path, child))
    query = child.get("query", "") or extract_query_from_response(child.get("response", ""))
    result = evaluate_transition(
        question=question,
        state=state,
        action_name=child.get("action_name", "") or "",
        response=child.get("response", "") or "",
        query=query,
        history_queries=extract_history_queries(parent_path),
    ).to_dict()
    result["typed_score"] = typed_score(result)
    return result


def snippet(text: str, limit: int = 320) -> str:
    clean = " ".join((text or "").split())
    return clean[:limit]


def branch_record(
    traj_idx: int,
    bp_idx: int,
    traj: dict,
    nodes: dict[int, dict],
    branch: dict,
) -> dict:
    question = traj.get("question", "")
    parent_path = get_path_to_node(nodes, branch["parent_node_id"])
    children = []
    for child in branch["children"]:
        eval_result = evaluate_child(question, parent_path, child)
        children.append(
            {
                "node_id": child.get("node_id"),
                "step": child.get("step"),
                "action_name": child.get("action_name"),
                "Q_llama": child.get("Q"),
                "query": child.get("query", ""),
                "response_len": len(child.get("response", "") or ""),
                "response_snippet": snippet(child.get("response", "")),
                "typed_eval": eval_result,
            }
        )
    typed_scores = [child["typed_eval"]["typed_score"] for child in children]
    failure_types = [child["typed_eval"]["failure_type"] for child in children]
    return {
        "traj_idx": traj_idx,
        "bp_idx": bp_idx,
        "question": question,
        "golden_answers": traj.get("golden_answers", []),
        "parent_node_id": branch["parent_node_id"],
        "parent_step": branch.get("parent_step"),
        "parent_action_name": branch.get("parent_action_name"),
        "q_all_same": branch["q_all_same"],
        "child_q_values": branch["child_q_values"],
        "content_all_same": branch["content_all_same"],
        "response_similarity": round(branch["response_similarity"], 4),
        "near_duplicate_0_95": branch["response_similarity"] >= 0.95,
        "typed_score_diff": len(set(typed_scores)) > 1,
        "typed_failure_diff": len(set(failure_types)) > 1,
        "typed_scores": typed_scores,
        "typed_failure_types": failure_types,
        "children": children,
    }


def pct(num: int, denom: int) -> float:
    return round(num / denom, 4) if denom else 0.0


def write_markdown_summary(path: Path, metrics: dict, key_branches: list[dict]) -> None:
    lines = [
        "# Gate 0 Offline Branch Quality Audit",
        "",
        "## Core Counts",
        "",
        f"- Total trajectories: {metrics['total_trajectories']}",
        f"- Trajectories with branch: {metrics['trajectories_with_branch']}",
        f"- Total branch points: {metrics['branch_points']}",
        f"- Exact duplicate branch points: {metrics['content_same']}",
        f"- Content-different branch points: {metrics['content_diff']}",
        f"- Q-same and content-different branch points: {metrics['q_same_content_diff']}",
        f"- Q-different and content-different branch points: {metrics['q_diff_content_diff']}",
        "",
        "## Typed Eval on Q-Same Content-Different Branches",
        "",
        f"- Branches: {metrics['key_set']['count']}",
        f"- Near-duplicate by response similarity >= 0.95: {metrics['key_set']['near_duplicate_0_95']}",
        f"- Typed score differs: {metrics['key_set']['typed_score_diff']}",
        f"- Typed failure type differs: {metrics['key_set']['typed_failure_diff']}",
        "",
        "## First Key Examples",
        "",
    ]
    for branch in key_branches[:10]:
        lines.extend(
            [
                f"### traj={branch['traj_idx']} bp={branch['bp_idx']}",
                "",
                f"- Question: {branch['question']}",
                f"- Llama Q: {branch['child_q_values']}",
                f"- Response similarity: {branch['response_similarity']}",
                f"- Typed scores: {branch['typed_scores']}",
                f"- Typed failures: {branch['typed_failure_types']}",
                "",
            ]
        )
        for child in branch["children"]:
            lines.extend(
                [
                    f"Child {child['node_id']}:",
                    f"- action: {child['action_name']}",
                    f"- query: {child['query']}",
                    f"- response: {child['response_snippet']}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline audit for MCTS branch quality")
    parser.add_argument("--reward-data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "branch_quality_offline")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_data = load_reward_data(args.reward_data_dir)

    stats = Counter()
    action_counts = Counter()
    step_counts = Counter()
    q_same_content_diff: list[dict] = []
    q_diff_content_diff: list[dict] = []
    all_content_diff: list[dict] = []

    for traj_idx, traj in enumerate(all_data):
        nodes = extract_tree(traj.get("output", {}))
        if not nodes:
            continue
        branches = find_branch_points(nodes)
        if branches:
            stats["trajectories_with_branch"] += 1
        for bp_idx, branch in enumerate(branches):
            stats["branch_points"] += 1
            stats["q_same" if branch["q_all_same"] else "q_diff"] += 1
            stats["content_same" if branch["content_all_same"] else "content_diff"] += 1
            if branch["response_similarity"] >= 0.95:
                stats["near_duplicate_0_95"] += 1
            action_counts[branch.get("parent_action_name") or "unknown"] += 1
            step_counts[str(branch.get("parent_step"))] += 1

            if branch["q_all_same"] and branch["content_all_same"]:
                stats["q_same_content_same"] += 1
            elif branch["q_all_same"] and not branch["content_all_same"]:
                stats["q_same_content_diff"] += 1
                record = branch_record(traj_idx, bp_idx, traj, nodes, branch)
                q_same_content_diff.append(record)
                all_content_diff.append(record)
            elif not branch["q_all_same"] and branch["content_all_same"]:
                stats["q_diff_content_same"] += 1
            else:
                stats["q_diff_content_diff"] += 1
                record = branch_record(traj_idx, bp_idx, traj, nodes, branch)
                q_diff_content_diff.append(record)
                all_content_diff.append(record)

    key_typed_score_diff = sum(1 for row in q_same_content_diff if row["typed_score_diff"])
    key_typed_failure_diff = sum(1 for row in q_same_content_diff if row["typed_failure_diff"])
    key_near_dup = sum(1 for row in q_same_content_diff if row["near_duplicate_0_95"])

    diff_typed_score_diff = sum(1 for row in all_content_diff if row["typed_score_diff"])
    diff_typed_failure_diff = sum(1 for row in all_content_diff if row["typed_failure_diff"])
    diff_near_dup = sum(1 for row in all_content_diff if row["near_duplicate_0_95"])

    metrics = {
        "label": "real_result",
        "description": "Offline audit. No API calls and no retrieval.",
        "total_trajectories": len(all_data),
        "trajectories_with_branch": stats["trajectories_with_branch"],
        "branch_points": stats["branch_points"],
        "q_same": stats["q_same"],
        "q_diff": stats["q_diff"],
        "content_same": stats["content_same"],
        "content_diff": stats["content_diff"],
        "q_same_content_same": stats["q_same_content_same"],
        "q_same_content_diff": stats["q_same_content_diff"],
        "q_diff_content_same": stats["q_diff_content_same"],
        "q_diff_content_diff": stats["q_diff_content_diff"],
        "content_diff_rate": pct(stats["content_diff"], stats["branch_points"]),
        "q_same_content_diff_rate_among_q_same": pct(stats["q_same_content_diff"], stats["q_same"]),
        "near_duplicate_0_95_all_branches": stats["near_duplicate_0_95"],
        "parent_action_counts": dict(action_counts),
        "parent_step_counts": dict(step_counts),
        "key_set": {
            "definition": "q_same_content_diff",
            "count": len(q_same_content_diff),
            "near_duplicate_0_95": key_near_dup,
            "typed_score_diff": key_typed_score_diff,
            "typed_failure_diff": key_typed_failure_diff,
            "typed_score_diff_rate": pct(key_typed_score_diff, len(q_same_content_diff)),
            "typed_failure_diff_rate": pct(key_typed_failure_diff, len(q_same_content_diff)),
        },
        "all_content_diff_set": {
            "count": len(all_content_diff),
            "near_duplicate_0_95": diff_near_dup,
            "typed_score_diff": diff_typed_score_diff,
            "typed_failure_diff": diff_typed_failure_diff,
            "typed_score_diff_rate": pct(diff_typed_score_diff, len(all_content_diff)),
            "typed_failure_diff_rate": pct(diff_typed_failure_diff, len(all_content_diff)),
        },
    }

    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "q_same_content_diff_branches.json").write_text(
        json.dumps(q_same_content_diff, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "q_diff_content_diff_branches.json").write_text(
        json.dumps(q_diff_content_diff, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown_summary(args.out_dir / "summary.md", metrics, q_same_content_diff)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"metrics: {args.out_dir / 'metrics.json'}")
    print(f"summary: {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
