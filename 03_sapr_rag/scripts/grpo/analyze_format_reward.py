#!/usr/bin/env python3
"""Compare old/new SaprFormatORM rules on saved GRPO completions.

Usage:
  python analyze_format_reward.py /path/to/completions.jsonl
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


RE_QUERY = re.compile(r"<query>(.*?)</query>", re.DOTALL)
RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def old_format_reward(completion: str) -> float:
    """Old rule: any previous <query> makes the sample invalid."""
    completion = completion or ""
    has_answer = bool(RE_ANSWER.search(completion))
    has_query = bool(RE_QUERY.search(completion))
    return 1.0 if has_answer and not has_query else 0.0


def new_format_reward(completion: str) -> float:
    """New rule: the last protocol tag must be a non-empty <answer>."""
    completion = completion or ""
    events: List[Tuple[int, str, str]] = []
    events.extend((m.start(), "query", m.group(1).strip()) for m in RE_QUERY.finditer(completion))
    events.extend((m.start(), "answer", m.group(1).strip()) for m in RE_ANSWER.finditer(completion))
    events.sort(key=lambda x: x[0])

    if not events:
        return 0.0

    _, last_kind, last_text = events[-1]
    return 1.0 if last_kind == "answer" and bool(last_text) else 0.0


def classify_completion(completion: str) -> str:
    """Summarize protocol shape for aggregate diagnostics."""
    completion = completion or ""
    queries = list(RE_QUERY.finditer(completion))
    answers = list(RE_ANSWER.finditer(completion))
    if answers and not queries:
        return "answer_only"
    if queries and not answers:
        return "query_only"
    if not queries and not answers:
        return "neither"

    events = [(m.start(), "query") for m in queries] + [(m.start(), "answer") for m in answers]
    events.sort(key=lambda x: x[0])
    last_kind = events[-1][1]
    return "query_then_answer" if last_kind == "answer" else "answer_then_query"


def iter_rows(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e


def flatten_completions(path: Path) -> List[Dict]:
    """Read ms-swift completions.jsonl.

    Each JSONL row is usually a table containing list-valued columns:
    step/prompt/completion/SaprFormatORM/...
    """
    flat = []
    for table_idx, row in enumerate(iter_rows(path), start=1):
        completions = row.get("completion")
        if isinstance(completions, list):
            n = len(completions)
            for i, completion in enumerate(completions):
                item = {
                    "table_idx": table_idx,
                    "row_idx": i,
                    "step": _list_get(row.get("step"), i),
                    "completion": completion or "",
                    "logged_format": _list_get(row.get("SaprFormatORM"), i),
                    "f1": _list_get(row.get("SaprF1ORM"), i),
                    "relevance": _list_get(row.get("SaprRelevanceORM"), i),
                }
                flat.append(item)
        elif "completion" in row:
            flat.append({
                "table_idx": table_idx,
                "row_idx": 0,
                "step": row.get("step"),
                "completion": completions or "",
                "logged_format": row.get("SaprFormatORM"),
                "f1": row.get("SaprF1ORM"),
                "relevance": row.get("SaprRelevanceORM"),
            })
    return flat


def _list_get(value, idx):
    if isinstance(value, list):
        return value[idx] if idx < len(value) else None
    return value


def pct(n: int, total: int) -> str:
    return f"{n / total * 100:.2f}%" if total else "nan"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("completions_jsonl", type=Path)
    ap.add_argument("--examples", type=int, default=5, help="number of changed examples to print")
    args = ap.parse_args()

    items = flatten_completions(args.completions_jsonl)
    total = len(items)
    if total == 0:
        raise SystemExit(f"No completions found in {args.completions_jsonl}")

    old_pass = 0
    new_pass = 0
    logged_pass = 0
    old_by_shape = Counter()
    new_by_shape = Counter()
    logged_by_shape = Counter()
    changed_by_shape = Counter()
    by_step = defaultdict(lambda: {"total": 0, "old": 0, "new": 0, "logged": 0})
    changed_examples = []

    for item in items:
        completion = item["completion"]
        shape = classify_completion(completion)
        old = old_format_reward(completion)
        new = new_format_reward(completion)
        logged = item.get("logged_format")

        old_pass += int(old == 1.0)
        new_pass += int(new == 1.0)
        logged_pass += int(logged == 1.0)
        old_by_shape[(shape, old)] += 1
        new_by_shape[(shape, new)] += 1
        logged_by_shape[(shape, logged)] += 1

        step = str(item.get("step"))
        by_step[step]["total"] += 1
        by_step[step]["old"] += int(old == 1.0)
        by_step[step]["new"] += int(new == 1.0)
        by_step[step]["logged"] += int(logged == 1.0)

        if old != new:
            changed_by_shape[shape] += 1
            if len(changed_examples) < args.examples:
                changed_examples.append((item, shape, old, new))

    print(f"[format] file: {args.completions_jsonl}")
    print(f"[format] completions: {total}")
    print(f"[format] logged pass: {logged_pass}/{total} ({pct(logged_pass, total)})")
    print(f"[format] old pass:    {old_pass}/{total} ({pct(old_pass, total)})")
    print(f"[format] new pass:    {new_pass}/{total} ({pct(new_pass, total)})")
    print(f"[format] delta:       +{new_pass - old_pass} ({(new_pass - old_pass) / total * 100:.2f} pp)")

    print("\n[format] shape distribution under old rule:")
    for (shape, reward), count in old_by_shape.most_common():
        print(f"  {shape:18s} reward={reward}: {count}")

    print("\n[format] shape distribution under new rule:")
    for (shape, reward), count in new_by_shape.most_common():
        print(f"  {shape:18s} reward={reward}: {count}")

    print("\n[format] changed old->new by shape:")
    for shape, count in changed_by_shape.most_common():
        print(f"  {shape:18s}: {count}")

    print("\n[format] recent steps:")
    for step in sorted(by_step, key=_step_key)[-12:]:
        stats = by_step[step]
        total_step = stats["total"]
        print(
            f"  step={step:>4s} total={total_step:3d} "
            f"logged={stats['logged']:3d} ({pct(stats['logged'], total_step)}) "
            f"old={stats['old']:3d} ({pct(stats['old'], total_step)}) "
            f"new={stats['new']:3d} ({pct(stats['new'], total_step)})"
        )

    if changed_examples:
        print("\n[format] examples changed from old=0 to new=1:")
        for item, shape, old, new in changed_examples:
            text = (item["completion"] or "").replace("\n", " ")[:600]
            print(
                f"--- step={item.get('step')} shape={shape} old={old} new={new} "
                f"f1={item.get('f1')} relevance={item.get('relevance')}"
            )
            print(text)


def _step_key(step: str) -> int:
    try:
        return int(step)
    except (TypeError, ValueError):
        return -1


if __name__ == "__main__":
    main()
