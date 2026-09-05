#!/usr/bin/env python3
"""Build state-aligned OPD data by removing all privileged teacher fields.

The output keeps the ordinary student prompt and gold columns used only by
reward functions. The external OPD teacher receives the same on-policy
messages as the student, so no teacher prompt belongs in this dataset.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


TEACHER_FIELDS = {
    "teacher_prompt",
    "teacher_prompt_version",
    "teacher_prompt_source",
    "teacher_prompt_fallback",
    "teacher_prompt_truncated",
    "teacher_prompt_tokens",
    "teacher_prompt_chars",
    "teacher_prompt_tokenizer",
    "teacher_query_prompt",
    "teacher_evidence_prompt",
    "teacher_answer_prompt",
}
REQUIRED_FIELDS = {
    "messages",
    "golden_answers",
    "gold_titles",
    "gold_sup_sents",
    "source",
}


def clean_row(row: dict, line_no: int) -> dict:
    missing = sorted(REQUIRED_FIELDS - row.keys())
    if missing:
        raise ValueError(f"line {line_no}: missing required fields: {missing}")
    cleaned = {
        key: value
        for key, value in row.items()
        if key not in TEACHER_FIELDS and not key.startswith("teacher_")
    }
    leaked = sorted(key for key in cleaned if key.startswith("teacher_"))
    if leaked:
        raise AssertionError(f"line {line_no}: teacher fields survived cleanup: {leaked}")
    return cleaned


def reservoir_sample_by_source(
    input_path: Path,
    per_source_limit: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    reservoirs: dict[str, list[dict]] = defaultdict(list)
    seen = Counter()
    with input_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = clean_row(json.loads(line), line_no)
            source = str(row["source"])
            seen[source] += 1
            bucket = reservoirs[source]
            if len(bucket) < per_source_limit:
                bucket.append(row)
                continue
            replacement = rng.randrange(seen[source])
            if replacement < per_source_limit:
                bucket[replacement] = row

    rows = [row for source in sorted(reservoirs) for row in reservoirs[source]]
    rng.shuffle(rows)
    return rows


def build_dataset(
    input_path: Path,
    output_path: Path,
    per_source_limit: int | None,
    seed: int,
) -> Counter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    if per_source_limit is not None:
        rows = reservoir_sample_by_source(input_path, per_source_limit, seed)
        with output_path.open("w", encoding="utf-8") as writer:
            for row in rows:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[str(row["source"])] += 1
        return counts

    with input_path.open(encoding="utf-8") as reader, output_path.open("w", encoding="utf-8") as writer:
        for line_no, line in enumerate(reader, 1):
            if not line.strip():
                continue
            row = clean_row(json.loads(line), line_no)
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[str(row["source"])] += 1
    return counts


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data/grpo/hotpotqa_2wiki_musique_train_multi_opsd.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data/grpo/hotpotqa_2wiki_musique_train_opd.jsonl",
    )
    parser.add_argument(
        "--per-source-limit",
        type=int,
        default=None,
        help="Reservoir-sample this many rows per source; omit to retain all rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.per_source_limit is not None and args.per_source_limit <= 0:
        raise ValueError("--per-source-limit must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--output must differ from --input")

    counts = build_dataset(
        args.input,
        args.output,
        per_source_limit=args.per_source_limit,
        seed=args.seed,
    )
    total = sum(counts.values())
    print(f"[build_opd_dataset] output={args.output}")
    print(f"[build_opd_dataset] rows={total:,} by_source={dict(sorted(counts.items()))}")
    print("[build_opd_dataset] privileged_teacher_fields=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
