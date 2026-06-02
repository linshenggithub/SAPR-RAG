#!/usr/bin/env python3
"""
Gate 0 Step 1: Parse MCTS trees from ReasonRAG reward_data.
Extract branch points and node information for typed evaluation.
"""

import json
import sys
import random
from pathlib import Path
from collections import defaultdict

# 让脚本能直接 `python gate0/sample_branch_points.py` 运行：把仓库根加进 sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.paths import REASONRAG_OUTPUT_DIR  # noqa: E402

# ---- Config ----
RANDOM_SEED = 42
SAMPLE_SIZE = 50  # number of trajectories to analyze
MIN_BRANCH_CHILDREN = 2  # minimum children to be a branch point

# Paths
# 仓库外路径：从 config/paths.py 读取（可被 SAPR_REASONRAG_OUTPUT_DIR 环境变量覆盖）
REWARD_DATA_DIR = REASONRAG_OUTPUT_DIR
# 仓库内路径：相对于本脚本所在 gate0/ 目录，跨机器自动适配
OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def load_reward_data():
    """Load all reward_data files."""
    all_data = []
    for i in range(4):
        fpath = REWARD_DATA_DIR / f"reward_data{i}.json"
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
                print(f"  Loaded {fpath.name}: {len(data)} trajectories")
                all_data.extend(data)
    return all_data


def parse_node(node_dict):
    """Extract relevant fields from a node."""
    if not isinstance(node_dict, dict):
        return None
    return {
        "Q": node_dict.get("Q"),
        "reward": node_dict.get("reward"),
        "step": node_dict.get("step"),
        "N": node_dict.get("N"),
        "parent_id": node_dict.get("parent_id"),
        "children_ids": node_dict.get("children_ids", []),
        "action_name": node_dict.get("action_name"),
        "query": node_dict.get("query", ""),
        "response": node_dict.get("response", ""),
    }


def node_signature(node):
    """Behavioral signature for duplicate sibling detection."""
    parts = []
    for key in ("action_name", "query", "response"):
        parts.append(" ".join(str(node.get(key, "") or "").lower().split()))
    return "|".join(parts)


def extract_tree(output):
    """Extract the full MCTS tree from a trajectory output."""
    nodes = {}
    for key, value in output.items():
        if key.startswith("intermediate_node_"):
            node_id = int(key.split("_")[-1])
            parsed = parse_node(value)
            if parsed:
                parsed["node_id"] = node_id
                # Also keep full prompt/response for later analysis
                parsed["input_prompt"] = value.get("input_prompt", "")[:500] if isinstance(value, dict) else ""
                parsed["retrieval_result"] = value.get("retrieval_result", []) if isinstance(value, dict) else []
                nodes[node_id] = parsed
    return nodes


def find_branch_points(nodes):
    """Find all nodes that have multiple children."""
    branch_points = []
    for nid, node in nodes.items():
        children = node.get("children_ids", [])
        if len(children) >= MIN_BRANCH_CHILDREN:
            child_nodes = []
            for cid in children:
                if cid in nodes:
                    child_nodes.append(nodes[cid])
            if len(child_nodes) >= MIN_BRANCH_CHILDREN:
                # Check if Q values are identical (scalar PRM can't distinguish)
                child_Qs = [c["Q"] for c in child_nodes if c["Q"] is not None]
                q_all_same = len(set(child_Qs)) == 1 if len(child_Qs) >= 2 else False
                content_all_same = len({node_signature(c) for c in child_nodes}) == 1
                branch_points.append({
                    "parent_node_id": nid,
                    "parent_Q": node["Q"],
                    "children": child_nodes,
                    "n_children": len(child_nodes),
                    "q_all_same": q_all_same,
                    "content_all_same": content_all_same,
                    "child_Qs": child_Qs,
                })
    return branch_points


def get_path_to_node(nodes, node_id):
    """Return root-to-node path using parent_id links."""
    path = []
    seen = set()
    current_id = node_id
    while current_id in nodes and current_id not in seen:
        seen.add(current_id)
        node = nodes[current_id]
        path.append(node)
        current_id = node.get("parent_id", -1)
    return list(reversed(path))


def extract_query_history(path_nodes):
    """Extract previous queries along a root-to-parent path."""
    queries = []
    for node in path_nodes:
        query = node.get("query", "")
        if query:
            queries.append(query)
        else:
            response = node.get("response", "")
            if "<query>" in response:
                start = response.rfind("<query>") + len("<query>")
                end = response.rfind("</query>")
                if end > start:
                    queries.append(response[start:end].strip())
    return queries


def analyze_trajectory(traj_data):
    """Analyze a single trajectory's MCTS tree."""
    output = traj_data.get("output", {})
    question = traj_data.get("question", "")
    golden_answers = traj_data.get("golden_answers", [])

    nodes = extract_tree(output)
    if not nodes:
        return None

    branch_points = find_branch_points(nodes)

    return {
        "question": question,
        "golden_answers": golden_answers,
        "n_nodes": len(nodes),
        "n_branch_points": len(branch_points),
        "branch_points": branch_points,
        "nodes": nodes,
    }


def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Gate 0 Step 1: Parsing MCTS Trees ===\n")

    # Load data
    print("Loading reward data...")
    all_data = load_reward_data()
    print(f"Total trajectories loaded: {len(all_data)}\n")

    # Analyze all trajectories
    print("Analyzing MCTS trees...")
    results = []
    stats = {
        "total": 0,
        "has_branches": 0,
        "total_branch_points": 0,
        "q_same_branches": 0,
        "q_diff_branches": 0,
    }

    for traj in all_data:
        analyzed = analyze_trajectory(traj)
        if analyzed is None:
            continue
        stats["total"] += 1
        if analyzed["n_branch_points"] > 0:
            stats["has_branches"] += 1
            stats["total_branch_points"] += analyzed["n_branch_points"]
            for bp in analyzed["branch_points"]:
                if bp["q_all_same"]:
                    stats["q_same_branches"] += 1
                else:
                    stats["q_diff_branches"] += 1
                if bp["q_all_same"] and not bp.get("content_all_same", True):
                    stats["q_same_content_diff_branches"] = stats.get("q_same_content_diff_branches", 0) + 1
        results.append(analyzed)

    print(f"Analyzed: {stats['total']} trajectories")
    print(f"  With branches: {stats['has_branches']} ({stats['has_branches']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Total branch points: {stats['total_branch_points']}")
    print(f"  Q-all-same: {stats['q_same_branches']} ({stats['q_same_branches']/max(stats['total_branch_points'],1)*100:.1f}%)")
    print(f"  Q-different: {stats['q_diff_branches']} ({stats['q_diff_branches']/max(stats['total_branch_points'],1)*100:.1f}%)")
    print(f"  Q-same non-duplicate content: {stats.get('q_same_content_diff_branches', 0)}")

    # Sample trajectories for detailed analysis
    # Prefer bridge-type questions with many Q-same branch points
    candidates = [r for r in results if r["n_branch_points"] > 0]
    # Sort by number of Q-same branch points (most interesting for our analysis)
    candidates.sort(key=lambda r: sum(1 for bp in r["branch_points"] if bp["q_all_same"]), reverse=True)

    # Sample: prefer trajectories containing Q-same but non-duplicate sibling content.
    top_candidates = sorted(
        candidates,
        key=lambda r: (
            sum(1 for bp in r["branch_points"] if bp["q_all_same"] and not bp.get("content_all_same", True)),
            sum(1 for bp in r["branch_points"] if bp["q_all_same"]),
        ),
        reverse=True,
    )[:30]
    remaining = [c for c in candidates if c not in top_candidates]
    random_sample = random.sample(remaining, min(20, len(remaining)))
    sampled = top_candidates + random_sample

    print(f"\nSampled {len(sampled)} trajectories for Gate 0 analysis")
    print(f"  Top (most Q-same branches): {len(top_candidates)}")
    print(f"  Random: {len(random_sample)}")

    # Save sampled trajectories (simplified - only essential info)
    sampled_output = []
    for r in sampled:
        # Simplify: only save branch points with minimal info
        traj_info = {
            "question": r["question"],
            "golden_answers": r["golden_answers"],
            "n_nodes": r["n_nodes"],
            "n_branch_points": r["n_branch_points"],
            "branch_points": [],
        }
        for bp in r["branch_points"]:
            parent_node = r["nodes"].get(bp["parent_node_id"], {})
            parent_path = get_path_to_node(r["nodes"], bp["parent_node_id"])
            history_queries = extract_query_history(parent_path)
            bp_info = {
                "parent_node_id": bp["parent_node_id"],
                "parent_Q": bp["parent_Q"],
                "parent_step": parent_node.get("step"),
                "parent_action_name": parent_node.get("action_name"),
                "parent_query": parent_node.get("query", ""),
                "parent_response": parent_node.get("response", "")[:1000] if parent_node.get("response") else "",
                "history_queries": history_queries,
                "history_responses": [
                    n.get("response", "")[:1000] for n in parent_path if n.get("response")
                ],
                "q_all_same": bp["q_all_same"],
                "content_all_same": bp.get("content_all_same", False),
                "n_children": bp["n_children"],
                "children": [],
            }
            for c in bp["children"]:
                child_info = {
                    "node_id": c["node_id"],
                    "Q": c["Q"],
                    "reward": c["reward"],
                    "step": c["step"],
                    "action_name": c["action_name"],
                    "query": c["query"],
                    "response": c["response"][:1000] if c["response"] else "",
                    "retrieval_result": c.get("retrieval_result", []),
                }
                bp_info["children"].append(child_info)
            traj_info["branch_points"].append(bp_info)
        sampled_output.append(traj_info)

    # Save
    out_path = OUTPUT_DIR / "sampled_trajectories.json"
    with open(out_path, "w") as f:
        json.dump(sampled_output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")

    # Also save full stats
    stats_path = OUTPUT_DIR / "tree_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
