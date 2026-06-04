#!/usr/bin/env python3
"""Case analysis for baseline vs minimal SAPR-E rerank."""

import json
import re
import sys
import argparse
from collections import Counter
from pathlib import Path

# 让脚本能直接 `python 03_sapr_rag/scripts/xxx.py` 运行：把仓库根加进 sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.paths import REPO_ROOT  # noqa: E402


ROOT = REPO_ROOT


def normalize(text):
    text = str(text or "").lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def exact_match(prediction, golds):
    prediction = normalize(prediction)
    return int(any(prediction == normalize(gold) for gold in golds or []))


def f1_score(prediction, golds):
    pred_tokens = normalize(prediction).split()
    best = 0.0
    for gold in golds or []:
        gold_tokens = normalize(gold).split()
        if not pred_tokens and not gold_tokens:
            best = max(best, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap:
            precision = overlap / len(pred_tokens)
            recall = overlap / len(gold_tokens)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def answer(item):
    output = item.get("output") or {}
    if isinstance(output, dict):
        for key in ["pred", "answer", "prediction", "final_answer"]:
            if key in output:
                return str(output[key])
    return ""


def ordered_nodes(item):
    output = item.get("output") or {}
    nodes = []
    if isinstance(output, dict):
        for key, value in output.items():
            if not str(key).startswith("intermediate_node_") or not isinstance(value, dict):
                continue
            try:
                index = int(str(key).split("_")[-1])
            except ValueError:
                index = 10**9
            nodes.append((index, value))
    return [node for _, node in sorted(nodes)]


def doc_nodes(item):
    return [
        node
        for node in ordered_nodes(item)
        if node.get("action_name") == "document_analysis" and node.get("retrieval_result")
    ]


def title_of(doc):
    if isinstance(doc, dict) and doc.get("title"):
        return str(doc.get("title")).strip().strip('"')[:160]
    raw = doc if isinstance(doc, str) else doc.get("contents", doc.get("text", ""))
    return str(raw).split("\n", 1)[0].strip().strip('"')[:160]


def content_of(doc):
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        return str(doc.get("contents", doc.get("text", "")))
    return str(doc or "")


def titles(node):
    return [title_of(doc) for doc in node.get("retrieval_result") or []]


def contents(node):
    return [content_of(doc) for doc in node.get("retrieval_result") or []]


def gold_titles(item):
    facts = ((item.get("metadata") or {}).get("supporting_facts") or {})
    return [str(title) for title in facts.get("title") or []]


def hit_titles(candidate_titles, golds):
    normalized_candidates = {normalize(title) for title in candidate_titles}
    return [gold for gold in golds if normalize(gold) in normalized_candidates]


def hit_contents(candidate_contents, values):
    normalized_contents = [normalize(content) for content in candidate_contents]
    hits = []
    for value in values or []:
        normalized_value = normalize(value)
        if normalized_value and any(normalized_value in content for content in normalized_contents):
            hits.append(value)
    return hits


def last_action(item):
    nodes = ordered_nodes(item)
    return nodes[-1].get("action_name") if nodes else None


def last_tail(item):
    nodes = ordered_nodes(item)
    if not nodes:
        return ""
    return str(nodes[-1].get("response", "")).replace("\n", " ")[-220:]


def esc(value, limit=180):
    return str(value).replace("|", "/").replace("\n", " ")[:limit]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_path",
        type=Path,
        default=ROOT
        / "04_experiments/logs/20260531_sapr_e_e2e_200_maxtok256_seq_gpu0/"
        / "baseline/hotpotqa_2026_05_31_23_13_sapr_e_e2e_baseline/intermediate_data.json",
    )
    parser.add_argument(
        "--rerank_path",
        type=Path,
        default=ROOT
        / "04_experiments/logs/20260601_sapr_e_minimal_rerank_50_v1/"
        / "sapr_e_minimal_rerank/hotpotqa_2026_06_01_08_36_sapr_e_minimal_rerank/intermediate_data.json",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=ROOT / "04_experiments/metrics/20260601_sapr_e_minimal_rerank_50_case_analysis",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline_data = json.load(open(args.baseline_path))
    rerank_data = json.load(open(args.rerank_path))
    if args.limit:
        baseline_data = baseline_data[: args.limit]
        rerank_data = rerank_data[: args.limit]
    baseline = {item["id"]: item for item in baseline_data}
    rerank = {item["id"]: item for item in rerank_data}

    cases = []
    steps = []
    for item_id, baseline_item in baseline.items():
        if item_id not in rerank:
            continue
        rerank_item = rerank[item_id]
        gold_answers = baseline_item.get("golden_answers") or []
        support_titles = gold_titles(baseline_item)
        baseline_answer = answer(baseline_item)
        rerank_answer = answer(rerank_item)
        baseline_em = exact_match(baseline_answer, gold_answers)
        rerank_em = exact_match(rerank_answer, gold_answers)

        if baseline_em and not rerank_em:
            category = "regression"
        elif not baseline_em and rerank_em:
            category = "improvement"
        elif baseline_em and rerank_em:
            category = "both_correct"
        else:
            category = "both_wrong"

        baseline_nodes = doc_nodes(baseline_item)
        rerank_nodes = doc_nodes(rerank_item)
        max_steps = max(len(baseline_nodes), len(rerank_nodes))
        baseline_any_gold = False
        rerank_any_gold = False
        retrieval_changed = False
        case_steps = []

        for step in range(max_steps):
            baseline_titles = titles(baseline_nodes[step]) if step < len(baseline_nodes) else []
            rerank_titles = titles(rerank_nodes[step]) if step < len(rerank_nodes) else []
            baseline_contents = contents(baseline_nodes[step]) if step < len(baseline_nodes) else []
            rerank_contents = contents(rerank_nodes[step]) if step < len(rerank_nodes) else []
            baseline_gold_hit = hit_titles(baseline_titles, support_titles)
            rerank_gold_hit = hit_titles(rerank_titles, support_titles)
            baseline_content_gold_title_hit = hit_contents(baseline_contents, support_titles)
            rerank_content_gold_title_hit = hit_contents(rerank_contents, support_titles)
            baseline_content_gold_answer_hit = hit_contents(baseline_contents, gold_answers)
            rerank_content_gold_answer_hit = hit_contents(rerank_contents, gold_answers)
            baseline_any_gold = baseline_any_gold or bool(baseline_gold_hit)
            rerank_any_gold = rerank_any_gold or bool(rerank_gold_hit)
            retrieval_changed = retrieval_changed or baseline_titles != rerank_titles
            row = {
                "id": item_id,
                "category": category,
                "step": step,
                "gold_titles": support_titles,
                "baseline_titles": baseline_titles,
                "rerank_titles": rerank_titles,
                "overlap": len(set(baseline_titles) & set(rerank_titles)),
                "baseline_gold_hit": baseline_gold_hit,
                "rerank_gold_hit": rerank_gold_hit,
                "baseline_content_gold_title_hit": baseline_content_gold_title_hit,
                "rerank_content_gold_title_hit": rerank_content_gold_title_hit,
                "baseline_content_gold_answer_hit": baseline_content_gold_answer_hit,
                "rerank_content_gold_answer_hit": rerank_content_gold_answer_hit,
                "baseline_query": baseline_nodes[step].get("query") if step < len(baseline_nodes) else None,
                "rerank_query": rerank_nodes[step].get("query") if step < len(rerank_nodes) else None,
            }
            steps.append(row)
            case_steps.append(row)

        cases.append(
            {
                "id": item_id,
                "category": category,
                "question": baseline_item.get("question"),
                "gold_answers": gold_answers,
                "gold_titles": support_titles,
                "baseline_answer": baseline_answer,
                "rerank_answer": rerank_answer,
                "baseline_em": baseline_em,
                "rerank_em": rerank_em,
                "baseline_f1": f1_score(baseline_answer, gold_answers),
                "rerank_f1": f1_score(rerank_answer, gold_answers),
                "baseline_doc_steps": len(baseline_nodes),
                "rerank_doc_steps": len(rerank_nodes),
                "retrieval_changed": retrieval_changed,
                "baseline_any_gold_title_hit": baseline_any_gold,
                "rerank_any_gold_title_hit": rerank_any_gold,
                "baseline_last_action": last_action(baseline_item),
                "rerank_last_action": last_action(rerank_item),
                "baseline_last_tail": last_tail(baseline_item),
                "rerank_last_tail": last_tail(rerank_item),
                "steps": case_steps,
            }
        )

    summary = {
        "n": len(cases),
        "category_counts": dict(Counter(case["category"] for case in cases)),
        "baseline_em_count": sum(case["baseline_em"] for case in cases),
        "rerank_em_count": sum(case["rerank_em"] for case in cases),
        "baseline_avg_f1": sum(case["baseline_f1"] for case in cases) / len(cases),
        "rerank_avg_f1": sum(case["rerank_f1"] for case in cases) / len(cases),
        "retrieval_changed_items": sum(case["retrieval_changed"] for case in cases),
        "baseline_any_gold_items": sum(case["baseline_any_gold_title_hit"] for case in cases),
        "rerank_any_gold_items": sum(case["rerank_any_gold_title_hit"] for case in cases),
        "step_count": len(steps),
        "baseline_gold_hit_steps": sum(1 for step in steps if step["baseline_gold_hit"]),
        "rerank_gold_hit_steps": sum(1 for step in steps if step["rerank_gold_hit"]),
        "rerank_added_gold_steps": sum(
            1 for step in steps if not step["baseline_gold_hit"] and step["rerank_gold_hit"]
        ),
        "rerank_removed_gold_steps": sum(
            1 for step in steps if step["baseline_gold_hit"] and not step["rerank_gold_hit"]
        ),
        "both_gold_steps": sum(1 for step in steps if step["baseline_gold_hit"] and step["rerank_gold_hit"]),
        "baseline_content_gold_title_steps": sum(1 for step in steps if step["baseline_content_gold_title_hit"]),
        "rerank_content_gold_title_steps": sum(1 for step in steps if step["rerank_content_gold_title_hit"]),
        "baseline_content_gold_answer_steps": sum(1 for step in steps if step["baseline_content_gold_answer_hit"]),
        "rerank_content_gold_answer_steps": sum(1 for step in steps if step["rerank_content_gold_answer_hit"]),
    }

    json.dump(summary, open(args.out_dir / "summary.json", "w"), ensure_ascii=False, indent=2)
    with open(args.out_dir / "cases.jsonl", "w") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    with open(args.out_dir / "steps.jsonl", "w") as handle:
        for step in steps:
            handle.write(json.dumps(step, ensure_ascii=False) + "\n")

    md = ["# SAPR-E Minimal Rerank Case Analysis", "", "## Summary", ""]
    for key, value in summary.items():
        md.append(f"- {key}: {value}")

    for category in ["improvement", "regression", "both_wrong", "both_correct"]:
        md += ["", f"## {category} cases", ""]
        md.append("| id | gold | baseline -> rerank | doc steps | gold item-hit | question |")
        md.append("|---|---|---|---|---|---|")
        for case in [item for item in cases if item["category"] == category][:20]:
            md.append(
                "| {} | {} | {} -> {} | {}/{} | {}/{} | {} |".format(
                    case["id"],
                    esc(case["gold_answers"], 80),
                    esc(case["baseline_answer"], 80),
                    esc(case["rerank_answer"], 80),
                    case["baseline_doc_steps"],
                    case["rerank_doc_steps"],
                    case["baseline_any_gold_title_hit"],
                    case["rerank_any_gold_title_hit"],
                    esc(case["question"], 140),
                )
            )
            for step in case["steps"][:3]:
                if step["baseline_titles"] != step["rerank_titles"]:
                    md.append(
                        "| | | step {} overlap {} | B: {} | R: {} | gold B/R: {}/{} |".format(
                            step["step"],
                            step["overlap"],
                            esc(step["baseline_titles"], 120),
                            esc(step["rerank_titles"], 120),
                            esc(step["baseline_gold_hit"], 60),
                            esc(step["rerank_gold_hit"], 60),
                        )
                    )

    with open(args.out_dir / "case_analysis.md", "w") as handle:
        handle.write("\n".join(md))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("OUT", args.out_dir)


if __name__ == "__main__":
    main()
