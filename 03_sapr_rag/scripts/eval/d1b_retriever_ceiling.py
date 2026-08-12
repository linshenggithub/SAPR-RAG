#!/usr/bin/env python3
"""D1b：区分检索器上限与模型 query 生成质量。

通过已在线的 retrieval_daemon 查询同一套 BGE + FAISS 索引：
1. 用原始问题直接检索；
2. 用每个去重后的 gold title 自身直接检索（oracle query）。

三种模式都统计 top-3 / top-5 的 gold 标题平均覆盖率与完全召回率：
1. 原始问题；
2. SFT 轨迹实际生成的全部子查询；
3. gold title oracle query。
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path


TOP_KS = (3, 5, 10, 20)


def normalize_title(value):
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def load_cases(path):
    cases = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata", {})
            supporting = metadata.get("supporting_facts", {})
            context = metadata.get("context", {})
            context_by_title = dict(
                zip(context.get("title", []), context.get("sentences", []))
            )

            titles = supporting.get("title", []) or []
            unique_titles = []
            seen_titles = set()
            for title in titles:
                normalized = normalize_title(title)
                if normalized and normalized not in seen_titles:
                    seen_titles.add(normalized)
                    unique_titles.append(normalized)

            if unique_titles:
                cases.append(
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "gold_titles": unique_titles,
                    }
                )
    return cases


def load_model_queries(path):
    query_by_id = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            queries = []
            seen = set()
            for trace in row.get("trace", []):
                if not isinstance(trace, dict):
                    continue
                info = trace.get("rollout_infos", {}) or {}
                for step in info.get("retrieved_steps", []) or []:
                    query = str(step.get("query", "")).strip()
                    normalized = re.sub(r"\s+", " ", query.lower()).strip()
                    if query and normalized not in seen:
                        seen.add(normalized)
                        queries.append(query)
            query_by_id[row["id"]] = queries
    return query_by_id


def post_json(url, payload, timeout=300):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def batch_search(server, queries, top_k, batch_size):
    outputs = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        response = post_json(
            f"{server.rstrip('/')}/search_batch",
            {"queries": batch, "top_k": top_k},
        )
        results = response["results"]
        if len(results) != len(batch):
            raise RuntimeError(
                f"检索返回数量异常：请求 {len(batch)}，返回 {len(results)}"
            )
        outputs.extend(results)
        print(
            f"[search] {min(start + batch_size, len(queries))}/{len(queries)}",
            flush=True,
        )
    return outputs


def hit_titles(docs, gold_titles, top_k):
    retrieved = {
        normalize_title(doc.get("title", "")) for doc in docs[:top_k]
    }
    return sorted(set(gold_titles) & retrieved)


def summarize(details, mode, top_k):
    coverages = []
    full = 0
    for row in details:
        gold = row["gold_titles"]
        hits = row[f"{mode}_top{top_k}_hits"]
        coverage = len(hits) / len(gold)
        coverages.append(coverage)
        full += len(hits) == len(gold)
    n = len(details)
    return {
        "n": n,
        "average_gold_coverage": sum(coverages) / n,
        "full_recall_count": full,
        "full_recall_rate": full / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8100")
    parser.add_argument(
        "--dev",
        type=Path,
        required=True,
        help="HotpotQA JSONL used to define the evaluation cohort.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        help="Model trajectory JSONL for the same cohort.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with urllib.request.urlopen(
        f"{args.server.rstrip('/')}/health", timeout=10
    ) as response:
        health = json.load(response)
    if health.get("status") != "ok":
        raise RuntimeError(f"检索服务不健康：{health}")
    print(f"[health] {json.dumps(health, ensure_ascii=False)}", flush=True)

    cases = load_cases(args.dev)
    model_query_by_id = load_model_queries(args.trajectory)
    question_queries = [row["question"] for row in cases]
    oracle_queries = []
    oracle_ranges = []
    for row in cases:
        start = len(oracle_queries)
        oracle_queries.extend(row["gold_titles"])
        oracle_ranges.append((start, len(oracle_queries)))
    model_queries = []
    model_ranges = []
    for row in cases:
        start = len(model_queries)
        model_queries.extend(model_query_by_id.get(row["id"], []))
        model_ranges.append((start, len(model_queries)))

    print(
        f"[data] cases={len(cases)}, model_queries={len(model_queries)}, "
        f"oracle_queries={len(oracle_queries)}",
        flush=True,
    )
    question_results = batch_search(
        args.server,
        question_queries,
        top_k=max(TOP_KS),
        batch_size=args.batch_size,
    )
    oracle_results = batch_search(
        args.server,
        oracle_queries,
        top_k=max(TOP_KS),
        batch_size=args.batch_size,
    )
    model_results = batch_search(
        args.server,
        model_queries,
        top_k=max(TOP_KS),
        batch_size=args.batch_size,
    )

    details = []
    for case, question_docs, (oracle_start, oracle_end), (
        model_start,
        model_end,
    ) in zip(
        cases, question_results, oracle_ranges, model_ranges
    ):
        oracle_docs = oracle_results[oracle_start:oracle_end]
        model_docs = model_results[model_start:model_end]
        detail = {
            "id": case["id"],
            "question": case["question"],
            "gold_titles": case["gold_titles"],
            "gold_title_count": len(case["gold_titles"]),
        }
        for top_k in TOP_KS:
            detail[f"question_top{top_k}_hits"] = hit_titles(
                question_docs, case["gold_titles"], top_k
            )
            oracle_hits = set()
            for docs in oracle_docs:
                oracle_hits.update(
                    hit_titles(docs, case["gold_titles"], top_k)
                )
            detail[f"gold_title_top{top_k}_hits"] = sorted(oracle_hits)
            model_hits = set()
            for docs in model_docs:
                model_hits.update(
                    hit_titles(docs, case["gold_titles"], top_k)
                )
            detail[f"model_query_top{top_k}_hits"] = sorted(model_hits)
        details.append(detail)

    summary = {}
    for mode in ("question", "model_query", "gold_title"):
        for top_k in TOP_KS:
            summary[f"{mode}_top{top_k}"] = summarize(
                details, mode, top_k
            )

    artifact = {
        "description": "D1b retriever ceiling on HotpotQA dev first 200",
        "server_health": health,
        "dev_path": str(args.dev),
        "trajectory_path": str(args.trajectory),
        "summary": summary,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
    )

    print("\n=== D1b 检索器理论召回上限 ===")
    for key, metrics in summary.items():
        print(
            f"{key:>13}: 平均覆盖={metrics['average_gold_coverage']:.1%}, "
            f"完全召回={metrics['full_recall_rate']:.1%} "
            f"({metrics['full_recall_count']}/{metrics['n']})"
        )
    print(f"\n[output] {args.output}")


if __name__ == "__main__":
    main()
