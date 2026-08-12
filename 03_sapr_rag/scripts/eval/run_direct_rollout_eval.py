#!/usr/bin/env python3
"""Batch-evaluate a multi-turn SAPR scheduler through the rollout HTTP API."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from agent_infer import REASONING_SYSTEM


RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def normalize_query(query: str) -> str:
    query = re.sub(r"[^\w]+", " ", str(query).lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", query).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--rollout_url", default="http://127.0.0.1:8001")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_answer(text: str) -> str | None:
    match = RE_ANSWER.search(text or "")
    if not match:
        return None
    answer = match.group(1).strip()
    return answer or None


def output_text(item: dict[str, Any]) -> str:
    for message in reversed(item.get("messages") or []):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    choices = ((item.get("response") or {}).get("choices") or [])
    if choices:
        return str((choices[0].get("message") or {}).get("content") or "")
    return ""


def history_from_infos(info: dict[str, Any]) -> list[dict[str, Any]]:
    history = []
    for step in info.get("retrieved_steps", []) or []:
        docs = step.get("docs") or []
        if "evidence" in step:
            evidence = str(step.get("evidence") or "None").strip()
        else:
            evidence = " ".join(
                f"{doc.get('title', '')}. {doc.get('text', '')}" for doc in docs[:1]
            ).strip()
        history.append({
            "query": str(step.get("query", "")),
            "evidence": evidence,
            "exact_duplicate": bool(step.get("exact_duplicate")),
            "search_executed": bool(step.get("search_executed", True)),
        })
    return history


def behavior_from_infos(info: dict[str, Any], max_turns: int = 6) -> dict[str, Any]:
    steps = info.get("retrieved_steps", []) or []
    queries = [normalize_query(step.get("query", "")) for step in steps]
    queries = [query for query in queries if query]
    exact_duplicate_count = sum(1 for step in steps if step.get("exact_duplicate"))
    intercepted_repeat_count = sum(
        1
        for step in steps
        if step.get("exact_duplicate") and not step.get("search_executed", True)
    )
    actual_search_count = sum(1 for step in steps if step.get("search_executed", True))
    repeat_count_from_text = len(queries) - len(set(queries))
    recorded_turns = info.get("num_turns")
    num_turns = int(recorded_turns or len(steps) or 0)
    if recorded_turns is None:
        exhausted = len(steps) >= max(0, max_turns - 1)
    else:
        exhausted = num_turns >= max_turns
    return {
        "num_turns": num_turns,
        "num_queries": len(steps),
        "actual_search_count": actual_search_count,
        "exact_duplicate_count": exact_duplicate_count,
        "intercepted_repeat_count": intercepted_repeat_count,
        "repeat_count_from_text": repeat_count_from_text,
        "has_exact_duplicate": exact_duplicate_count > 0 or repeat_count_from_text > 0,
        "finish_reason": "max_turns_exceeded" if exhausted else "stopped",
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_output(path: Path, resume: bool) -> tuple[set[Any], str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not resume or not path.exists():
        return set(), "w"

    clean_rows = []
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            clean_rows.append(row)
            done.add(row.get("id"))

    with path.open("w", encoding="utf-8") as handle:
        for row in clean_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return done, "a"


def request_batch(
    session: requests.Session,
    rollout_url: str,
    batch: list[tuple[int, dict[str, Any]]],
    max_tokens: int,
    timeout: float,
) -> list[dict[str, Any]]:
    payload = {
        "infer_requests": [
            {
                "messages": [
                    {"role": "system", "content": REASONING_SYSTEM},
                    {"role": "user", "content": f"Question: {row['question']}"},
                ],
                "uuid": str(row.get("id", index)),
            }
            for index, row in batch
        ],
        "request_config": {
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "return_details": True,
        },
        "use_tqdm": False,
    }
    response = session.post(
        f"{rollout_url.rstrip('/')}/infer/",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    outputs = response.json()
    if not isinstance(outputs, list) or len(outputs) != len(batch):
        raise RuntimeError(
            f"rollout returned {type(outputs).__name__} with "
            f"{len(outputs) if isinstance(outputs, list) else 'unknown'} rows; "
            f"expected {len(batch)}"
        )
    return outputs


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.input_jsonl))
    output_path = Path(args.output_jsonl)
    done, mode = prepare_output(output_path, args.resume)
    pending = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get("id", index) not in done
    ]

    session = requests.Session()
    health = session.get(
        f"{args.rollout_url.rstrip('/')}/health/",
        timeout=min(args.timeout, 60.0),
    )
    health.raise_for_status()
    print(f"[direct_rollout] health={health.json()}", flush=True)

    started = time.time()
    with output_path.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            batch_started = time.time()
            try:
                outputs = request_batch(
                    session,
                    args.rollout_url,
                    batch,
                    args.max_tokens,
                    args.timeout,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                outputs = [{"error": error} for _ in batch]

            batch_seconds = time.time() - batch_started
            for (index, row), item in zip(batch, outputs):
                text = output_text(item)
                info = item.get("rollout_infos") or {}
                error = item.get("error")
                behavior = behavior_from_infos(info, max_turns=args.max_turns)
                answer = None if error else parse_answer(text)
                if not error and answer is None and behavior["finish_reason"] == "max_turns_exceeded":
                    error = "max_turns_exceeded"
                record = {
                    "id": row.get("id", index),
                    "question": row["question"],
                    "gold": row.get("answer") or row.get("golden_answers"),
                    "answer": None if error else answer,
                    "raw_output": text,
                    "history": history_from_infos(info),
                    "behavior": behavior,
                    "trace": [{
                        "messages": item.get("messages"),
                        "rollout_infos": info,
                    }],
                    "latency_s": round(batch_seconds / max(len(batch), 1), 3),
                }
                if error:
                    record["error"] = error
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed = start + len(batch)
            print(
                f"[direct_rollout] {completed}/{len(pending)} "
                f"batch_dt={batch_seconds:.1f}s elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
