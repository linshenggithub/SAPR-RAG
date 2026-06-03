#!/usr/bin/env python3
"""分析 ReasonRAG original GPT-4o MCTS sanity 输出。"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "gate0" / "data" / "reasonrag_original_gpt4o_mcts_sanity"


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_nodes(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    nodes: dict[int, dict[str, Any]] = {}
    for key, value in record.get("output", {}).items():
        if not key.startswith("intermediate_node_"):
            continue
        try:
            node_id = int(key.rsplit("_", 1)[1])
        except ValueError:
            continue
        if isinstance(value, dict):
            nodes[node_id] = value
    return nodes


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    nodes = extract_nodes(record)
    branch_points = []
    for parent_id in sorted({node.get("parent_id") for node in nodes.values()}):
        children = [(node_id, node) for node_id, node in nodes.items() if node.get("parent_id") == parent_id]
        if len(children) < 2:
            continue
        texts = [normalize_text(node.get("response")) for _, node in children]
        pair_similarity = None
        exact_same = None
        if len(texts) >= 2:
            pair_similarity = difflib.SequenceMatcher(None, texts[0], texts[1]).ratio()
            exact_same = texts[0] == texts[1]
        branch_points.append(
            {
                "parent_id": parent_id,
                "children_ids": [node_id for node_id, _ in children],
                "children_q": [node.get("Q") for _, node in children],
                "children_n": [node.get("N") for _, node in children],
                "exact_same_first_pair": exact_same,
                "similarity_first_pair": pair_similarity,
                "responses": [text[:500] for text in texts],
            }
        )

    root_branch = next((item for item in branch_points if item["parent_id"] == 0), None)
    top_nodes = sorted(nodes.items(), key=lambda item: item[1].get("N") or 0, reverse=True)[:5]
    return {
        "id": record.get("id"),
        "question": record.get("question"),
        "golden_answers": record.get("golden_answers"),
        "status": record.get("status"),
        "error": record.get("error"),
        "elapsed_sec": record.get("elapsed_sec"),
        "n_nodes": len(nodes),
        "n_branch_points": len(branch_points),
        "root_branch": root_branch,
        "branch_points": branch_points,
        "top_nodes": [
            {
                "node_id": node_id,
                "parent_id": node.get("parent_id"),
                "action_name": node.get("action_name"),
                "Q": node.get("Q"),
                "N": node.get("N"),
                "reward": node.get("reward"),
                "answer": node.get("answer") or node.get("pred"),
            }
            for node_id, node in top_nodes
        ],
    }


def analyze_run(run_dir: Path) -> dict[str, Any] | None:
    progress_path = run_dir / "progress.json"
    summary_path = run_dir / "summary.json"
    if not progress_path.exists():
        return None
    progress = load_json(progress_path)
    summary = load_json(summary_path) if summary_path.exists() else {}
    records = [analyze_record(record) for record in progress]
    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "records": records,
    }


def write_markdown(path: Path, analyses: list[dict[str, Any]]) -> None:
    lines = [
        "# ReasonRAG original GPT-4o MCTS sanity 分析",
        "",
        "| run | id | status | sec | nodes | branches | root sim | root exact | root Q | top answers |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for analysis in analyses:
        run_name = Path(analysis["run_dir"]).name
        for record in analysis["records"]:
            root = record.get("root_branch") or {}
            root_sim = root.get("similarity_first_pair")
            root_exact = root.get("exact_same_first_pair")
            root_q = root.get("children_q")
            answers = [str(item.get("answer")) for item in record.get("top_nodes", []) if item.get("answer")]
            lines.append(
                "| {run} | {id} | {status} | {sec} | {nodes} | {branches} | {sim} | {exact} | {q} | {answers} |".format(
                    run=run_name,
                    id=record.get("id"),
                    status=record.get("status"),
                    sec=record.get("elapsed_sec"),
                    nodes=record.get("n_nodes"),
                    branches=record.get("n_branch_points"),
                    sim="" if root_sim is None else f"{root_sim:.4f}",
                    exact=root_exact,
                    q=root_q,
                    answers=", ".join(answers[:3]),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = sorted(path for path in args.runs_dir.glob("2026*") if path.is_dir())
    analyses = [item for run in runs if (item := analyze_run(run)) is not None]
    out_json = args.runs_dir / "analysis_summary.json"
    out_md = args.runs_dir / "analysis_summary.md"
    out_json.write_text(json.dumps(analyses, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(out_md, analyses)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
