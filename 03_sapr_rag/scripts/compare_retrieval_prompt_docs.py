#!/usr/bin/env python3
"""Compare retrieval documents fed to ReasonRAG prompts between two runs."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def load_items(path):
    with open(path) as handle:
        return {item["id"]: item for item in json.load(handle)}


def ordered_nodes(item):
    output = item.get("output") or {}
    if not isinstance(output, dict):
        return []
    nodes = []
    for key, value in output.items():
        if not str(key).startswith("intermediate_node_") or not isinstance(value, dict):
            continue
        try:
            index = int(str(key).split("_")[-1])
        except ValueError:
            index = 10**9
        nodes.append((index, value))
    return [node for _, node in sorted(nodes)]


def title_of(doc):
    raw = doc if isinstance(doc, str) else doc.get("contents", doc.get("text", ""))
    title = str(raw).split("\n", 1)[0].strip().strip('"')
    return title[:160]


def retrieval_titles(node):
    return [title_of(doc) for doc in node.get("retrieval_result") or []]


def answer_of(item):
    output = item.get("output") or {}
    if isinstance(output, dict):
        for key in ["pred", "answer", "prediction", "final_answer"]:
            if key in output:
                return str(output[key])
    return ""


def normalize(text):
    text = str(text or "").lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def exact_match(prediction, golds):
    prediction = normalize(prediction)
    return int(any(prediction == normalize(gold) for gold in golds or []))


def gold_titles(item):
    metadata = item.get("metadata") or {}
    facts = metadata.get("supporting_facts") or {}
    return [str(title) for title in facts.get("title") or []]


def title_hit(titles, golds):
    norm_titles = {normalize(title) for title in titles}
    return int(any(normalize(gold) in norm_titles for gold in golds))


def doc_nodes(item):
    return [
        node
        for node in ordered_nodes(item)
        if node.get("action_name") == "document_analysis" and node.get("retrieval_result")
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    left = load_items(Path(args.left))
    right = load_items(Path(args.right))
    ids = [item_id for item_id in left.keys() if item_id in right]
    if args.limit is not None:
        ids = ids[: args.limit]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    item_rows = []
    missing_node_items = 0
    for item_id in ids:
        li = left[item_id]
        ri = right[item_id]
        left_nodes = doc_nodes(li)
        right_nodes = doc_nodes(ri)
        if not left_nodes or not right_nodes:
            missing_node_items += 1

        max_steps = max(len(left_nodes), len(right_nodes))
        golds = gold_titles(li)
        item_changed = False
        item_left_gold_hit = False
        item_right_gold_hit = False
        for step in range(max_steps):
            left_titles = retrieval_titles(left_nodes[step]) if step < len(left_nodes) else []
            right_titles = retrieval_titles(right_nodes[step]) if step < len(right_nodes) else []
            overlap = len(set(left_titles) & set(right_titles))
            left_gold_hit = title_hit(left_titles, golds)
            right_gold_hit = title_hit(right_titles, golds)
            item_left_gold_hit = item_left_gold_hit or bool(left_gold_hit)
            item_right_gold_hit = item_right_gold_hit or bool(right_gold_hit)
            changed = left_titles != right_titles
            item_changed = item_changed or changed
            rows.append(
                {
                    "id": item_id,
                    "step": step,
                    "question": li.get("question"),
                    "gold_answers": li.get("golden_answers"),
                    "gold_titles": golds,
                    "left_query": left_nodes[step].get("query") if step < len(left_nodes) else None,
                    "right_query": right_nodes[step].get("query") if step < len(right_nodes) else None,
                    "left_titles": left_titles,
                    "right_titles": right_titles,
                    "overlap": overlap,
                    "changed": changed,
                    "left_gold_hit": left_gold_hit,
                    "right_gold_hit": right_gold_hit,
                }
            )

        left_answer = answer_of(li)
        right_answer = answer_of(ri)
        gold_answers = li.get("golden_answers") or []
        item_rows.append(
            {
                "id": item_id,
                "question": li.get("question"),
                "gold_answers": gold_answers,
                "gold_titles": golds,
                "left_answer": left_answer,
                "right_answer": right_answer,
                "left_em": exact_match(left_answer, gold_answers),
                "right_em": exact_match(right_answer, gold_answers),
                "left_doc_steps": len(left_nodes),
                "right_doc_steps": len(right_nodes),
                "retrieval_changed": item_changed,
                "left_any_gold_title_hit": item_left_gold_hit,
                "right_any_gold_title_hit": item_right_gold_hit,
            }
        )

    overlap_counts = Counter(row["overlap"] for row in rows)
    changed_rows = [row for row in rows if row["changed"]]
    summary = {
        "left_name": args.left_name,
        "right_name": args.right_name,
        "n_items": len(ids),
        "paired_doc_steps": len(rows),
        "missing_node_items": missing_node_items,
        "changed_doc_steps": len(changed_rows),
        "changed_doc_step_rate": len(changed_rows) / len(rows) if rows else 0.0,
        "avg_title_overlap": sum(row["overlap"] for row in rows) / len(rows) if rows else 0.0,
        "overlap_counts": dict(sorted(overlap_counts.items())),
        "left_gold_hit_steps": sum(row["left_gold_hit"] for row in rows),
        "right_gold_hit_steps": sum(row["right_gold_hit"] for row in rows),
        "left_em_count": sum(row["left_em"] for row in item_rows),
        "right_em_count": sum(row["right_em"] for row in item_rows),
        "retrieval_changed_items": sum(row["retrieval_changed"] for row in item_rows),
        "left_any_gold_title_hit_items": sum(row["left_any_gold_title_hit"] for row in item_rows),
        "right_any_gold_title_hit_items": sum(row["right_any_gold_title_hit"] for row in item_rows),
    }

    with open(outdir / "summary.json", "w") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(outdir / "per_step.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(outdir / "per_item.jsonl", "w") as handle:
        for row in item_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def esc(value, limit=150):
        return str(value).replace("|", "/").replace("\n", " ")[:limit]

    md = ["# Retrieval Prompt Document Diff", "", "## Summary", ""]
    for key, value in summary.items():
        md.append(f"- {key}: {value}")
    md += ["", "## Changed Step Examples", ""]
    md.append("| id | step | overlap | gold_titles | left_titles | right_titles |")
    md.append("|---|---:|---:|---|---|---|")
    for row in changed_rows[:40]:
        md.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                row["id"],
                row["step"],
                row["overlap"],
                esc(row["gold_titles"]),
                esc(row["left_titles"]),
                esc(row["right_titles"]),
            )
        )
    with open(outdir / "doc_diff.md", "w") as handle:
        handle.write("\n".join(md))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
