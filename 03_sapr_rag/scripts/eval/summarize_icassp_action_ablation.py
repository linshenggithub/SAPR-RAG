#!/usr/bin/env python3
"""Summarize the matched GRPO + Query/Answer OPSD additive diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from summarize_icassp_inference_budget import (
    DATASETS,
    EXPECTED_ROWS,
    SUMMARY_FIELDS,
    paired_bootstrap,
    read_json,
    read_jsonl,
)


MODELS = ["outcome_only", "query_only", "answer_only", "full"]
COMPARISONS = [
    ("query_only", "outcome_only"),
    ("answer_only", "outcome_only"),
    ("full", "outcome_only"),
    ("full", "query_only"),
    ("full", "answer_only"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--full-root", required=True, type=Path)
    parser.add_argument("--checkpoint-step", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args()


def model_dir(args: argparse.Namespace, model: str, dataset: str) -> Path:
    if model == "full":
        return args.full_root / dataset
    return (
        args.root / model / "s5_k3" / "full"
        / f"checkpoint-{args.checkpoint_step}" / dataset
    )


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")

    rows = {}
    summary_rows = []
    validation = {"status": "pass", "errors": [], "models": {}}

    for model in MODELS:
        rows[model] = {}
        dataset_metrics = []
        validation["models"][model] = {}
        for dataset in DATASETS:
            directory = model_dir(args, model, dataset)
            metrics = read_json(directory / "metrics.json")
            dataset_rows = read_jsonl(directory / "results.jsonl")
            expected = EXPECTED_ROWS[dataset]
            errors = sum("error" in row for row in dataset_rows.values())
            model_errors = []
            if len(dataset_rows) != expected:
                model_errors.append(
                    f"rows={len(dataset_rows)} expected={expected}"
                )
            if errors:
                model_errors.append(f"row_errors={errors}")
            if metrics["n_total"] != expected:
                model_errors.append(
                    f"metrics_n={metrics['n_total']} expected={expected}"
                )
            if metrics["answer_rate"] != 1.0:
                model_errors.append(
                    f"answer_rate={metrics['answer_rate']}"
                )
            if metrics["format_failure_rate"] != 0.0:
                model_errors.append(
                    f"format_failure_rate={metrics['format_failure_rate']}"
                )
            if metrics["service_failure_rate"] != 0.0:
                model_errors.append(
                    f"service_failure_rate={metrics['service_failure_rate']}"
                )
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
            validation["models"][model][dataset] = {
                "rows": len(dataset_rows),
                "row_errors": errors,
                "answer_rate": metrics["answer_rate"],
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
        id_sets = {model: set(rows[model][dataset]) for model in MODELS}
        if len({frozenset(value) for value in id_sets.values()}) != 1:
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
        "checkpoint_step": args.checkpoint_step,
        "datasets": DATASETS,
        "expected_rows": EXPECTED_ROWS,
        "macro_aggregation": "unweighted_mean_over_datasets",
        "models": {},
    }
    for model in MODELS:
        model_rows = [
            row for row in summary_rows if row["model"] == model
        ]
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

    csv_fields = [
        "model", "scope", "dataset", "n_total", *SUMMARY_FIELDS
    ]
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
        "models": len(MODELS),
        "dataset_runs": len(MODELS) * len(DATASETS),
        "new_rows": sum(EXPECTED_ROWS.values()) * 3,
        "comparisons": len(COMPARISONS),
        "validation": validation["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
