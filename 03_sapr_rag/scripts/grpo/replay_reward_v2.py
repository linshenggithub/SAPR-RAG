#!/usr/bin/env python3
"""离线重算 Reward-v2 / 新增证据奖励。

输入 `run_direct_rollout_eval.py` 产出的轨迹 JSONL，结合 dev JSONL 中的
gold supporting facts，检查 reward 是否能区分有效查询和重复查询。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


def normalize_answer(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def normalize_text(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_query(query: Any) -> str:
    query = re.sub(r"[^\w]+", " ", str(query or "").lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", query).strip()


def f1_score(pred: Any, golds: list[str]) -> float:
    pred_tokens = normalize_answer(pred).split()
    best = 0.0
    for gold in golds:
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        same = sum(common.values())
        if same == 0:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_gold(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    metadata = row.get("metadata") or {}
    facts = metadata.get("supporting_facts") or {}
    titles = facts.get("title") or []
    sent_ids = facts.get("sent_id") or []
    context = metadata.get("context") or {}
    ctx_titles = context.get("title") or []
    ctx_sents = context.get("sentences", context.get("text", context.get("content", []))) or []
    title_to_sents = {
        normalize_title(title): sents
        for title, sents in zip(ctx_titles, ctx_sents)
    }

    evidence = []
    title_to_item = {}
    for title, sent_id in zip(titles, sent_ids):
        title = str(title).strip()
        if not title:
            continue
        key = normalize_title(title)
        item = title_to_item.get(key)
        if item is None:
            item = {"title": title, "sentences": []}
            title_to_item[key] = item
            evidence.append(item)
        sents = title_to_sents.get(key)
        if isinstance(sents, list) and isinstance(sent_id, int) and 0 <= sent_id < len(sents):
            sent = str(sents[sent_id]).strip()
            if sent and sent not in item["sentences"]:
                item["sentences"].append(sent)
    return (
        [item["title"] for item in evidence],
        ["\n".join(item["sentences"]) for item in evidence],
    )


def support_sentence_variants(value: Any) -> list[str]:
    variants = []
    if isinstance(value, (list, tuple)):
        for item in value:
            variants.extend(support_sentence_variants(item))
    else:
        variants.extend(part.strip() for part in str(value or "").splitlines() if part.strip())
    return variants


def gold_hits_for_docs(docs: list[dict[str, Any]], gold_titles: list[str], gold_sents: list[str]) -> set[int]:
    retrieved_titles = {normalize_title(doc.get("title")) for doc in docs}
    retrieved_texts = [normalize_text(doc.get("text")) for doc in docs]
    hits = set()
    for idx, title in enumerate(gold_titles):
        hit = normalize_title(title) in retrieved_titles
        if not hit and idx < len(gold_sents):
            sentences = [normalize_text(sent) for sent in support_sentence_variants(gold_sents[idx])]
            hit = any(
                sent and any(sent in text for text in retrieved_texts)
                for sent in sentences
            )
        if hit:
            hits.add(idx)
    return hits


def get_steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    for item in row.get("trace") or []:
        info = item.get("rollout_infos") or {}
        steps = info.get("retrieved_steps")
        if isinstance(steps, list):
            return steps
    return []


def replay_row(row: dict[str, Any], gold_row: dict[str, Any], gamma: float) -> dict[str, Any]:
    gold_titles, gold_sents = extract_gold(gold_row)
    gold_answers = gold_row.get("golden_answers") or row.get("gold") or []
    steps = get_steps(row)

    seen_queries = set()
    covered = set()
    final_hits = set()
    query_stats = []
    repeat_count = 0
    intercepted_count = 0
    marginal = 0.0
    queries_after_full = 0

    for turn_idx, step in enumerate(steps, start=1):
        query = step.get("query", "")
        normalized_query = step.get("normalized_query") or normalize_query(query)
        text_repeat = bool(normalized_query and normalized_query in seen_queries)
        exact_duplicate = bool(step.get("exact_duplicate") or text_repeat)
        search_executed = bool(step.get("search_executed", True))
        docs = step.get("docs") or []
        hits = gold_hits_for_docs(docs, gold_titles, gold_sents)
        new_hits = hits - covered
        if exact_duplicate:
            repeat_count += 1
        if exact_duplicate and not search_executed:
            intercepted_count += 1
        if gold_titles and len(covered) == len(gold_titles):
            queries_after_full += 1
        if gold_titles:
            marginal += (gamma ** (turn_idx - 1)) * (len(new_hits) / len(gold_titles))
        covered |= hits
        final_hits |= hits
        query_stats.append({
            "turn": turn_idx,
            "query": query,
            "exact_duplicate": exact_duplicate,
            "search_executed": search_executed,
            "new_hits": len(new_hits),
            "hit_count": len(hits),
            "is_effective": len(new_hits) > 0,
        })
        if normalized_query:
            seen_queries.add(normalized_query)

    num_gold = len(gold_titles)
    final_relevance = len(final_hits) / num_gold if num_gold else 0.0
    answer = row.get("answer")
    f1 = f1_score(answer, list(gold_answers))
    fmt = 1.0 if answer else 0.0
    query_count = len(steps)
    turn_cost = -max(0, query_count - 1)
    repeat_penalty = -min(repeat_count, 3)
    max_turn_penalty = -1.0 if row.get("error") == "max_turns_exceeded" or row.get("behavior", {}).get("finish_reason") == "max_turns_exceeded" else 0.0
    total_v2 = (
        f1
        + 0.15 * final_relevance
        + 0.05 * fmt
        + 0.02 * turn_cost
        + 0.15 * repeat_penalty
        + 0.50 * max_turn_penalty
    )
    total_marginal = (
        f1
        + 0.15 * marginal
        + 0.05 * fmt
        + 0.02 * turn_cost
        + 0.15 * repeat_penalty
        + 0.50 * max_turn_penalty
        - 0.10 * queries_after_full
    )

    return {
        "id": row.get("id"),
        "answer": answer,
        "f1": f1,
        "format": fmt,
        "query_count": query_count,
        "final_relevance": final_relevance,
        "marginal_relevance": marginal,
        "repeat_count": repeat_count,
        "intercepted_count": intercepted_count,
        "turn_cost": turn_cost,
        "max_turn_penalty": max_turn_penalty,
        "queries_after_full": queries_after_full,
        "total_v2": total_v2,
        "total_marginal": total_marginal,
        "query_stats": query_stats,
    }


def mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None and not math.isnan(value)]
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    query_stats = [query for row in rows for query in row["query_stats"]]
    effective = [query for query in query_stats if query["is_effective"]]
    duplicate = [query for query in query_stats if query["exact_duplicate"]]
    nonduplicate = [query for query in query_stats if not query["exact_duplicate"]]
    ineffective_nondup = [
        query for query in query_stats
        if not query["exact_duplicate"] and not query["is_effective"]
    ]

    def query_block(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(items),
            "rate": round(len(items) / len(query_stats), 6) if query_stats else 0.0,
            "mean_new_hits": mean([item["new_hits"] for item in items]) or 0.0,
            "mean_hit_count": mean([item["hit_count"] for item in items]) or 0.0,
        }

    return {
        "n": len(rows),
        "mean_total_v2": mean([row["total_v2"] for row in rows]),
        "mean_total_marginal": mean([row["total_marginal"] for row in rows]),
        "mean_f1": mean([row["f1"] for row in rows]),
        "mean_final_relevance": mean([row["final_relevance"] for row in rows]),
        "mean_marginal_relevance": mean([row["marginal_relevance"] for row in rows]),
        "mean_repeat_count": mean([row["repeat_count"] for row in rows]),
        "repeat_sample_rate": sum(1 for row in rows if row["repeat_count"] > 0) / len(rows) if rows else 0.0,
        "max_turn_rate": sum(1 for row in rows if row["max_turn_penalty"] < 0) / len(rows) if rows else 0.0,
        "query_stats": {
            "all": query_block(query_stats),
            "effective_new_evidence": query_block(effective),
            "exact_duplicate": query_block(duplicate),
            "nonduplicate": query_block(nonduplicate),
            "ineffective_nonduplicate": query_block(ineffective_nondup),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--gamma", type=float, default=0.9)
    args = parser.parse_args()

    results = load_jsonl(Path(args.results))
    gold_rows = {str(row.get("id", idx)): row for idx, row in enumerate(load_jsonl(Path(args.gold)))}
    replayed = []
    missing = []
    for idx, row in enumerate(results):
        key = str(row.get("id", idx))
        gold_row = gold_rows.get(key)
        if gold_row is None:
            missing.append(key)
            continue
        replayed.append(replay_row(row, gold_row, args.gamma))

    report = {
        "results": args.results,
        "gold": args.gold,
        "gamma": args.gamma,
        "missing_gold_rows": missing[:20],
        "summary": summarize(replayed),
        "rows": replayed,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
