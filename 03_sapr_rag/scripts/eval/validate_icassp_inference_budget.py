#!/usr/bin/env python3
"""Validate protocol and result invariants for an inference-budget run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CONFIGS = [
    "s0_k3",
    "s1_k3",
    "s2_k3",
    "s3_k3",
    "s4_k3",
    "s5_k3",
    "s6_k3",
    "s7_k3",
    "s5_k1",
    "s5_k5",
]
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]
FULL_COUNTS = {
    "hotpotqa": 7405,
    "2wikimultihopqa": 12576,
    "musique": 2417,
}
PROTOCOL = "answer_only_system_prefill_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stage", choices=["selection", "full"], required=True)
    parser.add_argument("--checkpoint-step", type=int, default=1000)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--expected-per-dataset", type=int)
    parser.add_argument("--max-failure-rate", type=float, default=0.05)
    return parser.parse_args()


def parse_config(config: str) -> tuple[int, int]:
    left, right = config.split("_")
    return int(left[1:]), int(right[1:])


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def main() -> int:
    args = parse_args()
    configs = [item for item in args.configs.split(",") if item]
    errors = []
    summary = {
        "root": str(args.root),
        "stage": args.stage,
        "checkpoint_step": args.checkpoint_step,
        "protocol": PROTOCOL,
        "configs": {},
    }
    ids_by_dataset: dict[str, set[str]] = {}

    for config in configs:
        max_searches, top_k = parse_config(config)
        config_summary = {}
        for dataset in DATASETS:
            path = (
                args.root
                / config
                / args.stage
                / f"checkpoint-{args.checkpoint_step}"
                / dataset
                / "results.jsonl"
            )
            if not path.is_file():
                errors.append(f"missing results: {path}")
                continue
            rows = load_rows(path)
            expected = (
                args.expected_per_dataset
                if args.expected_per_dataset is not None
                else FULL_COUNTS[dataset]
            )
            if len(rows) != expected:
                errors.append(
                    f"{config}/{dataset}: rows={len(rows)} expected={expected}"
                )

            ids = [str(row.get("id")) for row in rows]
            if len(set(ids)) != len(ids):
                errors.append(f"{config}/{dataset}: duplicate IDs")
            reference_ids = ids_by_dataset.setdefault(dataset, set(ids))
            if set(ids) != reference_ids:
                errors.append(f"{config}/{dataset}: ID set differs across configs")

            format_failures = 0
            service_failures = 0
            forced_answers = 0
            forced_valid = 0
            logical_total = 0
            rpc_total = 0
            for row in rows:
                sample_id = row.get("id")
                behavior = row.get("behavior") or {}
                trace = row.get("trace") or []
                info = (trace[0].get("rollout_infos") or {}) if trace else {}
                steps = info.get("retrieved_steps") or []
                logical = int(behavior.get("logical_search_count", -1))
                rpc = int(behavior.get("actual_retrieval_rpc_count", -1))
                executed = sum(bool(step.get("search_executed", True)) for step in steps)

                if behavior.get("max_searches") != max_searches:
                    errors.append(
                        f"{config}/{dataset}/{sample_id}: wrong max_searches "
                        f"{behavior.get('max_searches')}"
                    )
                if behavior.get("top_k") != top_k:
                    errors.append(
                        f"{config}/{dataset}/{sample_id}: wrong top_k "
                        f"{behavior.get('top_k')}"
                    )
                if behavior.get("forced_answer_protocol") != PROTOCOL:
                    errors.append(
                        f"{config}/{dataset}/{sample_id}: wrong protocol "
                        f"{behavior.get('forced_answer_protocol')}"
                    )
                if logical != len(steps) or logical > max_searches:
                    errors.append(
                        f"{config}/{dataset}/{sample_id}: logical={logical}, "
                        f"steps={len(steps)}, budget={max_searches}"
                    )
                if rpc != executed or rpc > logical:
                    errors.append(
                        f"{config}/{dataset}/{sample_id}: rpc={rpc}, "
                        f"executed={executed}, logical={logical}"
                    )
                if max_searches == 0 and rpc != 0:
                    errors.append(f"{config}/{dataset}/{sample_id}: S0 made {rpc} RPCs")

                for step in steps:
                    if step.get("exact_duplicate") and step.get("search_executed", True):
                        errors.append(
                            f"{config}/{dataset}/{sample_id}: duplicate query executed"
                        )
                    if step.get("search_executed", True) and not step.get("retrieval_error"):
                        docs = step.get("docs") or []
                        if len(docs) != top_k:
                            errors.append(
                                f"{config}/{dataset}/{sample_id}: "
                                f"top_k={top_k} returned {len(docs)} docs"
                            )

                forced = bool(behavior.get("forced_answer"))
                valid = bool(behavior.get("forced_answer_valid"))
                if forced:
                    forced_answers += 1
                    forced_valid += int(valid)
                    if not valid or row.get("answer") is None:
                        errors.append(
                            f"{config}/{dataset}/{sample_id}: invalid forced answer"
                        )
                if row.get("error") == "forced_answer_format_failure":
                    format_failures += 1
                if int(behavior.get("retrieval_error_count", 0)) > 0:
                    service_failures += 1

                logical_total += logical
                rpc_total += rpc

                for message in (
                    (trace[0].get("messages") or []) if trace else []
                ):
                    if any(key in message for key in ("gold", "golden_answers")):
                        errors.append(
                            f"{config}/{dataset}/{sample_id}: gold field leaked into trace"
                        )

            n = max(len(rows), 1)
            if format_failures / n > args.max_failure_rate:
                errors.append(
                    f"{config}/{dataset}: format failure rate "
                    f"{format_failures / n:.4f} exceeds {args.max_failure_rate:.4f}"
                )
            if service_failures / n > args.max_failure_rate:
                errors.append(
                    f"{config}/{dataset}: service failure rate "
                    f"{service_failures / n:.4f} exceeds {args.max_failure_rate:.4f}"
                )
            config_summary[dataset] = {
                "n": len(rows),
                "forced_answers": forced_answers,
                "forced_valid": forced_valid,
                "format_failures": format_failures,
                "service_failures": service_failures,
                "avg_logical_searches": logical_total / n,
                "avg_actual_retrieval_rpcs": rpc_total / n,
            }
        summary["configs"][config] = config_summary

    summary["status"] = "pass" if not errors else "fail"
    summary["errors"] = errors
    output = args.root / f"validation_{args.stage}.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
