#!/usr/bin/env python3
"""
Gate 0 Step 2: Offline typed transition audit.

The historical filename is kept for compatibility, but this script now computes
rule-based typed transition evaluation instead of query-only phi_q.
"""

import json
from collections import Counter
from pathlib import Path

from typed_eval import evaluate_transition, extract_evidence, state_from_question


# 仓库内路径：相对于本脚本所在 gate0/ 目录，跨机器自动适配
DATA_DIR = Path(__file__).resolve().parent / "data"


def has_non_empty_evidence(text: str) -> bool:
    evidence = extract_evidence(text or "")
    if not evidence:
        return False
    return evidence.strip().lower() not in {"none", "n/a", "no relevant evidence"}


def infer_open_gaps(question: str, bp: dict, child: dict) -> list:
    """Weak offline state estimate from saved tree context.

    If previous steps found some non-empty evidence, allow an answer to be
    treated as potentially valid; otherwise keep the original question open.
    This is intentionally conservative for Gate 0 and will be replaced by a
    learned/prompted state extractor in later phases.
    """
    response = child.get("response", "") or ""
    parent_responses = bp.get("history_responses", []) or []
    parent_responses.append(bp.get("parent_response", "") or "")
    has_prior_evidence = any(has_non_empty_evidence(r) for r in parent_responses)

    if "<answer>" in response and has_prior_evidence:
        return []
    return [question]


def typed_score(eval_dict: dict) -> float:
    """Scalar view for ranking children in the offline audit."""
    phi_s = eval_dict["phi_s"]
    phi_s_norm = 1.0 if phi_s > 0 else (0.5 if phi_s == 0 else 0.0)
    return round((eval_dict["phi_q"] + eval_dict["phi_c"] + phi_s_norm) / 3.0, 4)


def child_signature(child: dict) -> str:
    """Compact behavioral signature used to detect duplicated siblings."""
    action = child.get("action_name", "") or ""
    query = " ".join((child.get("query", "") or "").lower().split())
    response = " ".join((child.get("response", "") or "").lower().split())
    return f"{action}|{query}|{response}"


def child_eval(question: str, bp: dict, child: dict) -> dict:
    open_gaps = infer_open_gaps(question, bp, child)
    state = state_from_question(question, open_gaps=open_gaps)
    history_queries = bp.get("history_queries", []) or []

    result = evaluate_transition(
        question=question,
        state=state,
        action_name=child.get("action_name", "") or "",
        response=child.get("response", "") or "",
        query=child.get("query", "") or "",
        history_queries=history_queries,
    )

    result_dict = result.to_dict()
    result_dict["typed_score"] = typed_score(result_dict)
    return result_dict


def rank_index(values):
    if not values:
        return None
    return values.index(max(values))


def main():
    print("=== Gate 0 Step 2: Offline Typed Transition Audit ===\n")

    input_path = DATA_DIR / "sampled_trajectories.json"
    with open(input_path) as f:
        trajectories = json.load(f)
    print(f"Loaded {len(trajectories)} sampled trajectories")

    results = {
        "q_same_branches": [],
        "q_diff_branches": [],
    }
    stats = {
        "total_branch_points": 0,
        "total_children": 0,
        "q_same_total": 0,
        "q_same_content_diff": 0,
        "q_same_duplicate_content": 0,
        "q_same_failure_diff": 0,
        "q_same_score_diff": 0,
        "q_diff_total": 0,
        "q_diff_typed_agree": 0,
        "q_diff_typed_disagree": 0,
        "empty_query_children": 0,
        "none_evidence_children": 0,
    }
    failure_counts = Counter()
    action_counts = Counter()

    for traj_idx, traj in enumerate(trajectories):
        question = traj["question"]

        for bp_idx, bp in enumerate(traj["branch_points"]):
            stats["total_branch_points"] += 1
            child_rows = []

            for child in bp["children"]:
                stats["total_children"] += 1
                action_counts[child.get("action_name", "unknown")] += 1

                eval_result = child_eval(question, bp, child)
                failure_counts[eval_result["failure_type"]] += 1

                if eval_result["query_applicable"] and not eval_result["details"].get("query"):
                    stats["empty_query_children"] += 1
                evidence_info = eval_result["details"].get("claim_quality", {})
                if evidence_info.get("evidence_is_none"):
                    stats["none_evidence_children"] += 1

                child_rows.append({
                    "node_id": child.get("node_id"),
                    "Q": child.get("Q"),
                    "reward": child.get("reward"),
                    "step": child.get("step"),
                    "action_name": child.get("action_name"),
                    "query": child.get("query", ""),
                    "response": child.get("response", "")[:300],
                    **eval_result,
                })

            failure_types = [row["failure_type"] for row in child_rows]
            typed_scores = [row["typed_score"] for row in child_rows]
            q_values = [row["Q"] for row in child_rows if row.get("Q") is not None]
            signatures = [child_signature(child) for child in bp["children"]]

            failure_diff = len(set(failure_types)) > 1
            score_diff = len(set(typed_scores)) > 1
            content_diff = len(set(signatures)) > 1

            bp_result = {
                "traj_idx": traj_idx,
                "bp_idx": bp_idx,
                "question": question[:160],
                "parent_node_id": bp.get("parent_node_id"),
                "parent_step": bp.get("parent_step"),
                "q_all_same": bp.get("q_all_same"),
                "content_diff": content_diff,
                "failure_diff": failure_diff,
                "score_diff": score_diff,
                "children": child_rows,
            }

            if bp.get("q_all_same"):
                stats["q_same_total"] += 1
                if content_diff:
                    stats["q_same_content_diff"] += 1
                else:
                    stats["q_same_duplicate_content"] += 1
                if failure_diff:
                    stats["q_same_failure_diff"] += 1
                if score_diff:
                    stats["q_same_score_diff"] += 1
                results["q_same_branches"].append(bp_result)
            else:
                stats["q_diff_total"] += 1
                q_best = rank_index(q_values)
                typed_best = rank_index(typed_scores)
                if q_best is not None and typed_best is not None:
                    if q_best == typed_best:
                        stats["q_diff_typed_agree"] += 1
                    else:
                        stats["q_diff_typed_disagree"] += 1
                results["q_diff_branches"].append(bp_result)

    stats["failure_counts"] = dict(failure_counts)
    stats["action_counts"] = dict(action_counts)
    stats["q_same_failure_diff_rate"] = round(
        stats["q_same_failure_diff"] / max(stats["q_same_total"], 1), 4
    )
    stats["q_same_score_diff_rate"] = round(
        stats["q_same_score_diff"] / max(stats["q_same_total"], 1), 4
    )
    stats["q_same_content_diff_rate"] = round(
        stats["q_same_content_diff"] / max(stats["q_same_total"], 1), 4
    )

    print("\n=== Results ===")
    print(f"Total branch points analyzed: {stats['total_branch_points']}")
    print(f"Total children analyzed: {stats['total_children']}")
    print("\n--- Q-same branch points (scalar PRM cannot distinguish) ---")
    print(f"  Total: {stats['q_same_total']}")
    print(
        f"  Non-identical child content: {stats['q_same_content_diff']} "
        f"({stats['q_same_content_diff_rate'] * 100:.1f}%)"
    )
    print(f"  Duplicate child content: {stats['q_same_duplicate_content']}")
    print(
        f"  Failure type differs: {stats['q_same_failure_diff']} "
        f"({stats['q_same_failure_diff_rate'] * 100:.1f}%)"
    )
    print(
        f"  Typed score differs: {stats['q_same_score_diff']} "
        f"({stats['q_same_score_diff_rate'] * 100:.1f}%)"
    )

    print("\n--- Failure type distribution ---")
    for failure_type, count in failure_counts.most_common():
        print(f"  {failure_type}: {count}")

    print("\n--- Data quality diagnostics ---")
    print(f"  Empty query children: {stats['empty_query_children']}")
    print(f"  None/empty evidence children: {stats['none_evidence_children']}")
    print(f"  Action counts: {dict(action_counts)}")

    examples = [
        row for row in results["q_same_branches"]
        if row["failure_diff"] or row["score_diff"]
    ]
    print("\n=== Example: Q-same but typed-different ===")
    for ex in examples[:5]:
        print(f"\n  Q: {ex['question']}")
        for child in ex["children"]:
            print(
                f"    Node {child['node_id']}: Q={child['Q']}, "
                f"typed={child['typed_score']}, failure={child['failure_type']}, "
                f"phi=({child['phi_q']}, {child['phi_c']}, {child['phi_s']}), "
                f"action={child['action_name']}"
            )
            if child.get("query"):
                print(f"      query: {child['query'][:100]}")

    out_path = DATA_DIR / "typed_eval_results.json"
    with open(out_path, "w") as f:
        json.dump({"stats": stats, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")

    # Compatibility summary for existing notes/scripts that expect phi_q_stats.json.
    stats_path = DATA_DIR / "phi_q_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
