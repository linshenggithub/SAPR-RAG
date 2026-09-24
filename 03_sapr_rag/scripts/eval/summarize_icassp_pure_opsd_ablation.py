#!/usr/bin/env python3
"""Summarize the matched SFT + pure OPSD Query/Answer ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from summarize_icassp_inference_budget import (
    DATASETS,
    EXPECTED_ROWS,
    SUMMARY_FIELDS as BASE_SUMMARY_FIELDS,
    paired_bootstrap,
    read_json,
    read_jsonl,
)

SUMMARY_FIELDS = [
    *BASE_SUMMARY_FIELDS,
    "avg_turns",
    "max_turns_rate",
    "empty_evidence_rate",
    "avg_latency_s",
]

MODEL_STEPS = {
    "sft": 4150,
    "query_only": 1000,
    "answer_only": 1000,
    "opsd_only": 1000,
}
COMPARISONS = [
    ("query_only", "sft"),
    ("answer_only", "sft"),
    ("opsd_only", "sft"),
    ("opsd_only", "query_only"),
    ("opsd_only", "answer_only"),
    ("query_only", "answer_only"),
]
EXPECTED_PROTOCOL = {
    "top_k": "3",
    "max_searches": "5",
    "max_turns": "6",
    "force_final_answer": "true",
    "multi_turn_scheduler": "sapr_rag_forced_answer_scheduler",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260924)
    return parser.parse_args()


def parse_config(path: Path) -> dict[str, str]:
    config = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            config[key] = value
    return config


def result_dir(root: Path, model: str, dataset: str) -> Path:
    step = MODEL_STEPS[model]
    return root / model / "full" / f"checkpoint-{step}" / dataset


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")

    rows: dict[str, dict[str, dict[str, dict]]] = {}
    summary_rows = []
    validation = {
        "status": "pass",
        "errors": [],
        "expected_protocol": EXPECTED_PROTOCOL,
        "models": {},
    }

    for model, step in MODEL_STEPS.items():
        config_path = args.root / model / "config_full.txt"
        config = parse_config(config_path)
        config_errors = [
            f"{key}={config.get(key)!r} expected={expected!r}"
            for key, expected in EXPECTED_PROTOCOL.items()
            if config.get(key) != expected
        ]
        if config_errors:
            validation["status"] = "fail"
            validation["errors"].append(
                f"{model}/config: " + ", ".join(config_errors)
            )

        rows[model] = {}
        dataset_metrics = []
        validation["models"][model] = {
            "checkpoint_step": step,
            "config": config,
            "datasets": {},
        }
        for dataset in DATASETS:
            directory = result_dir(args.root, model, dataset)
            metrics = read_json(directory / "metrics.json")
            dataset_rows = read_jsonl(directory / "results.jsonl")
            expected = EXPECTED_ROWS[dataset]
            row_errors = sum(bool(row.get("error")) for row in dataset_rows.values())
            protocol_errors = sum(
                (row.get("behavior") or {}).get("forced_answer_protocol")
                != "answer_only_system_prefill_v2"
                for row in dataset_rows.values()
            )
            budget_errors = sum(
                int((row.get("behavior") or {}).get("logical_search_count", 0)) > 5
                for row in dataset_rows.values()
            )
            model_errors = []
            if len(dataset_rows) != expected:
                model_errors.append(f"rows={len(dataset_rows)} expected={expected}")
            if int(metrics["n_total"]) != expected:
                model_errors.append(
                    f"metrics_n={metrics['n_total']} expected={expected}"
                )
            if row_errors:
                model_errors.append(f"row_errors={row_errors}")
            if protocol_errors:
                model_errors.append(f"protocol_errors={protocol_errors}")
            if budget_errors:
                model_errors.append(f"budget_errors={budget_errors}")
            for field, expected_value in (
                ("answer_rate", 1.0),
                ("forced_answer_valid_rate", 1.0),
                ("format_failure_rate", 0.0),
                ("service_failure_rate", 0.0),
            ):
                if float(metrics[field]) != expected_value:
                    model_errors.append(f"{field}={metrics[field]}")
            if model_errors:
                validation["status"] = "fail"
                validation["errors"].append(
                    f"{model}/{dataset}: " + ", ".join(model_errors)
                )

            rows[model][dataset] = dataset_rows
            dataset_metrics.append(metrics)
            summary_rows.append({
                "model": model,
                "scope": "dataset",
                "dataset": dataset,
                **metrics,
            })
            validation["models"][model]["datasets"][dataset] = {
                "rows": len(dataset_rows),
                "row_errors": row_errors,
                "protocol_errors": protocol_errors,
                "budget_errors": budget_errors,
                "answer_rate": metrics["answer_rate"],
                "forced_answer_valid_rate": metrics["forced_answer_valid_rate"],
                "format_failure_rate": metrics["format_failure_rate"],
                "service_failure_rate": metrics["service_failure_rate"],
            }

        macro = {
            field: round(
                sum(float(metrics[field]) for metrics in dataset_metrics)
                / len(dataset_metrics),
                6,
            )
            for field in SUMMARY_FIELDS
        }
        macro["n_total"] = sum(
            int(metrics["n_total"]) for metrics in dataset_metrics
        )
        summary_rows.append({
            "model": model,
            "scope": "macro",
            "dataset": "macro",
            **macro,
        })

    for dataset in DATASETS:
        id_sets = {model: set(rows[model][dataset]) for model in MODEL_STEPS}
        if len({frozenset(ids) for ids in id_sets.values()}) != 1:
            validation["status"] = "fail"
            validation["errors"].append(
                f"{dataset}: ID mismatch "
                + ", ".join(
                    f"{model}={len(ids)}" for model, ids in id_sets.items()
                )
            )

    (args.root / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if validation["status"] != "pass":
        raise ValueError(f"validation failed: {validation['errors']}")

    summary = {
        "model_steps": MODEL_STEPS,
        "datasets": DATASETS,
        "expected_rows": EXPECTED_ROWS,
        "macro_aggregation": "unweighted_mean_over_datasets",
        "models": {},
    }
    for model in MODEL_STEPS:
        model_rows = [row for row in summary_rows if row["model"] == model]
        summary["models"][model] = {
            "datasets": {
                dataset: next(
                    row for row in model_rows if row["dataset"] == dataset
                )
                for dataset in DATASETS
            },
            "macro": next(
                row for row in model_rows if row["dataset"] == "macro"
            ),
        }
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    csv_fields = ["model", "scope", "dataset", "n_total", *SUMMARY_FIELDS]
    with (args.root / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=csv_fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    bootstrap_dir = args.root / "paired_bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_index = []
    for index, (candidate, baseline) in enumerate(COMPARISONS):
        result = paired_bootstrap(
            candidate,
            baseline,
            rows,
            samples=args.bootstrap_samples,
            seed=args.seed + index,
        )
        output = bootstrap_dir / f"{candidate}__vs__{baseline}.json"
        output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        bootstrap_index.append(result)
    (bootstrap_dir / "index.json").write_text(
        json.dumps(bootstrap_index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "root": str(args.root),
        "models": len(MODEL_STEPS),
        "dataset_runs": len(MODEL_STEPS) * len(DATASETS),
        "rows": sum(EXPECTED_ROWS.values()) * len(MODEL_STEPS),
        "comparisons": len(COMPARISONS),
        "bootstrap_samples": args.bootstrap_samples,
        "validation": validation["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
