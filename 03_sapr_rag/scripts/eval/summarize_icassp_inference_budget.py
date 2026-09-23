#!/usr/bin/env python3
"""Summarize the fixed-checkpoint ICASSP inference-budget ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from score import cover_em_score, em_score, f1_score


CONFIGS = [
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
EXPECTED_ROWS = {
    "hotpotqa": 7405,
    "2wikimultihopqa": 12576,
    "musique": 2417,
}
METRIC_FNS: dict[str, Callable[[str | None, list[str]], float]] = {
    "em": em_score,
    "f1": f1_score,
    "cover_em": cover_em_score,
}
SUMMARY_FIELDS = [
    "em",
    "f1",
    "cover_em",
    "answer_rate",
    "forced_answer_rate",
    "forced_answer_valid_rate",
    "forced_answer_em",
    "forced_answer_f1",
    "search_budget_hit_rate",
    "avg_logical_searches",
    "avg_actual_retrieval_rpcs",
    "avg_reasoner_tokens",
    "avg_evidence_tokens",
    "avg_forced_answer_tokens",
    "repeat_query_rate",
    "format_failure_rate",
    "service_failure_rate",
    "latency_p50_s",
    "latency_p95_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint-step", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def parse_config(config: str) -> tuple[int, int]:
    left, right = config.split("_")
    return int(left[1:]), int(right[1:])


def result_dir(root: Path, config: str, step: int, dataset: str) -> Path:
    return root / config / "full" / f"checkpoint-{step}" / dataset


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("id"))
            if sample_id == "None":
                raise ValueError(f"{path}:{line_no}: missing id")
            if sample_id in rows:
                raise ValueError(f"{path}:{line_no}: duplicate id {sample_id}")
            rows[sample_id] = row
    return rows


def gold_answers(row: dict) -> list[str]:
    gold = row.get("gold") or []
    if isinstance(gold, str):
        return [gold]
    return [str(item) for item in gold]


def score_matrix(ids: list[str], rows: dict[str, dict]) -> np.ndarray:
    result = np.empty((len(ids), len(METRIC_FNS)), dtype=np.float64)
    for row_index, sample_id in enumerate(ids):
        row = rows[sample_id]
        gold = gold_answers(row)
        for metric_index, metric in enumerate(METRIC_FNS.values()):
            result[row_index, metric_index] = metric(row.get("answer"), gold)
    return result


def tail_probability(values: np.ndarray, threshold: float, side: str) -> float:
    if side == "le":
        count = int(np.count_nonzero(values <= threshold))
    elif side == "ge":
        count = int(np.count_nonzero(values >= threshold))
    else:
        raise ValueError(f"unknown side: {side}")
    return (count + 1) / (values.size + 1)


def summarize_bootstrap(values: np.ndarray, observed: np.ndarray) -> dict[str, dict]:
    result = {}
    for index, metric in enumerate(METRIC_FNS):
        metric_values = values[:, index]
        p_le_zero = tail_probability(metric_values, 0.0, "le")
        p_ge_zero = tail_probability(metric_values, 0.0, "ge")
        result[metric] = {
            "paired_diff": round(float(observed[index]), 6),
            "paired_diff_ci95": [
                round(float(value), 6)
                for value in np.quantile(metric_values, [0.025, 0.975])
            ],
            "paired_p_one_sided_gt": round(p_le_zero, 6),
            "paired_p_two_sided": round(
                min(1.0, 2.0 * min(p_le_zero, p_ge_zero)), 6
            ),
        }
    return result


def paired_bootstrap(
    candidate: str,
    baseline: str,
    rows: dict[str, dict[str, dict[str, dict]]],
    samples: int,
    seed: int,
) -> dict:
    boot_by_dataset: dict[str, np.ndarray] = {}
    observed_by_dataset: dict[str, np.ndarray] = {}
    result = {
        "candidate": candidate,
        "baseline": baseline,
        "bootstrap_samples": samples,
        "seed": seed,
        "datasets": {},
    }

    for dataset_index, dataset in enumerate(DATASETS):
        candidate_rows = rows[candidate][dataset]
        baseline_rows = rows[baseline][dataset]
        candidate_ids = set(candidate_rows)
        baseline_ids = set(baseline_rows)
        if candidate_ids != baseline_ids:
            raise ValueError(
                f"ID mismatch for {candidate} vs {baseline}, {dataset}: "
                f"candidate={len(candidate_ids)} baseline={len(baseline_ids)}"
            )

        ids = sorted(candidate_ids)
        candidate_scores = score_matrix(ids, candidate_rows)
        baseline_scores = score_matrix(ids, baseline_rows)
        differences = candidate_scores - baseline_scores
        observed = differences.mean(axis=0)
        observed_by_dataset[dataset] = observed

        bootstrap = np.empty((samples, len(METRIC_FNS)), dtype=np.float64)
        pair_seed = int.from_bytes(
            hashlib.sha256(
                f"{seed}:{candidate}:{baseline}:{dataset}".encode()
            ).digest()[:8],
            byteorder="big",
        )
        rng = np.random.default_rng(pair_seed)
        for start in range(0, samples, 128):
            stop = min(start + 128, samples)
            indices = rng.integers(
                0, len(ids), size=(stop - start, len(ids)), dtype=np.int32
            )
            bootstrap[start:stop] = differences[indices].mean(axis=1)
        boot_by_dataset[dataset] = bootstrap
        result["datasets"][dataset] = {
            "n": len(ids),
            "metrics": summarize_bootstrap(bootstrap, observed),
        }

    macro_bootstrap = np.mean(
        np.stack([boot_by_dataset[dataset] for dataset in DATASETS], axis=1),
        axis=1,
    )
    macro_observed = np.mean(
        np.stack([observed_by_dataset[dataset] for dataset in DATASETS], axis=0),
        axis=0,
    )
    result["macro"] = {
        "aggregation": "unweighted_mean_over_datasets_with_stratified_resampling",
        "metrics": summarize_bootstrap(macro_bootstrap, macro_observed),
    }
    return result


def load_runtime(root: Path, exact_rpc_count: int) -> dict:
    config_runs = []
    for config in CONFIGS:
        path = root / config / "runtime.tsv"
        with path.open(encoding="utf-8", newline="") as handle:
            config_runs.extend(csv.DictReader(handle, delimiter="\t"))

    gpu_peaks: dict[str, dict[str, float]] = {}
    gpu_path = root / "gpu_metrics.csv"
    if gpu_path.exists():
        with gpu_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                gpu = row["gpu_index"].strip()
                current = gpu_peaks.setdefault(
                    gpu, {"memory_used_mib": 0.0, "utilization_gpu_pct": 0.0}
                )
                current["memory_used_mib"] = max(
                    current["memory_used_mib"], float(row["memory_used_mib"])
                )
                current["utilization_gpu_pct"] = max(
                    current["utilization_gpu_pct"],
                    float(row["utilization_gpu_pct"]),
                )

    manifest_path = root / "orchestrator_full.txt"
    manifest = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            manifest[key] = value
    started_raw = manifest.get("started_at")
    if started_raw is None:
        started_raw = min(row["started_at"] for row in config_runs)
        manifest["started_at"] = started_raw
        manifest["started_at_source"] = "min_config_runtime"

    completed_raw = manifest.get("completed_at")
    if completed_raw is None:
        completed_raw = max(row["completed_at"] for row in config_runs)
        manifest["completed_at"] = completed_raw
        manifest["completed_at_source"] = "max_config_runtime"

    started = datetime.fromisoformat(started_raw)
    completed = datetime.fromisoformat(completed_raw)
    wall_seconds = (completed - started).total_seconds()

    return {
        "orchestrator_wall_seconds": wall_seconds,
        "sum_config_gpu_hours": round(
            sum(float(row["wall_seconds"]) for row in config_runs) / 3600.0, 4
        ),
        "retrieval_rpc_count": exact_rpc_count,
        "retrieval_qps_over_orchestrator_wall": round(
            exact_rpc_count / wall_seconds, 4
        ) if wall_seconds > 0 else None,
        "config_runs": config_runs,
        "gpu_peaks": gpu_peaks,
        "retrieval_health": read_json(root / "retrieval_health.json"),
        "input_sha256": (root / "input_sha256.txt").read_text(
            encoding="utf-8"
        ).splitlines(),
        "orchestrator": manifest,
    }


def write_plots(root: Path, macro_by_config: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth_configs = [f"s{value}_k3" for value in range(8)]
    x_depth = list(range(8))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for metric, label, color in [
        ("em", "EM", "#0072B2"),
        ("f1", "F1", "#D55E00"),
        ("cover_em", "Cover-EM", "#009E73"),
    ]:
        ax.plot(
            x_depth,
            [macro_by_config[config][metric] for config in depth_configs],
            marker="o",
            label=label,
            color=color,
        )
    ax.set_xlabel("Maximum logical searches")
    ax.set_ylabel("Macro score")
    ax.set_xticks(x_depth)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "max_searches_quality.png", dpi=180)
    plt.close(fig)

    fig, ax_left = plt.subplots(figsize=(7.2, 4.4))
    ax_right = ax_left.twinx()
    ax_left.plot(
        x_depth,
        [macro_by_config[config]["avg_logical_searches"] for config in depth_configs],
        marker="o",
        label="Logical searches",
        color="#0072B2",
    )
    ax_left.plot(
        x_depth,
        [macro_by_config[config]["avg_actual_retrieval_rpcs"] for config in depth_configs],
        marker="s",
        label="Retrieval RPCs",
        color="#009E73",
    )
    ax_right.plot(
        x_depth,
        [macro_by_config[config]["latency_p95_s"] for config in depth_configs],
        marker="^",
        label="Latency P95",
        color="#D55E00",
    )
    ax_left.set_xlabel("Maximum logical searches")
    ax_left.set_ylabel("Average count")
    ax_right.set_ylabel("Latency P95 (seconds)")
    ax_left.set_xticks(x_depth)
    ax_left.grid(axis="y", alpha=0.25)
    lines = ax_left.lines + ax_right.lines
    ax_left.legend(lines, [line.get_label() for line in lines], loc="upper left")
    fig.tight_layout()
    fig.savefig(root / "max_searches_cost.png", dpi=180)
    plt.close(fig)

    topk_configs = ["s5_k1", "s5_k3", "s5_k5"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for metric, label, color in [
        ("em", "EM", "#0072B2"),
        ("f1", "F1", "#D55E00"),
        ("cover_em", "Cover-EM", "#009E73"),
    ]:
        ax.plot(
            [1, 3, 5],
            [macro_by_config[config][metric] for config in topk_configs],
            marker="o",
            label=label,
            color=color,
        )
    ax.set_xlabel("Top-k documents")
    ax.set_ylabel("Macro score")
    ax.set_xticks([1, 3, 5])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "top_k_quality.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")

    summary_rows = []
    rows: dict[str, dict[str, dict[str, dict]]] = {}
    macro_by_config = {}
    exact_rpc_count = 0

    for config in CONFIGS:
        max_searches, top_k = parse_config(config)
        rows[config] = {}
        dataset_metrics = []
        for dataset in DATASETS:
            directory = result_dir(args.root, config, args.checkpoint_step, dataset)
            metrics = read_json(directory / "metrics.json")
            dataset_rows = read_jsonl(directory / "results.jsonl")
            if len(dataset_rows) != EXPECTED_ROWS[dataset]:
                raise ValueError(
                    f"{config}/{dataset}: got {len(dataset_rows)} rows, "
                    f"expected {EXPECTED_ROWS[dataset]}"
                )
            rows[config][dataset] = dataset_rows
            exact_rpc_count += sum(
                int((row.get("behavior") or {}).get("actual_retrieval_rpc_count", 0))
                for row in dataset_rows.values()
            )
            record = {
                "config": config,
                "max_searches": max_searches,
                "top_k": top_k,
                "scope": "dataset",
                "dataset": dataset,
                **metrics,
            }
            summary_rows.append(record)
            dataset_metrics.append(metrics)

        macro = {
            field: round(
                sum(float(metrics[field]) for metrics in dataset_metrics)
                / len(dataset_metrics),
                6,
            )
            for field in SUMMARY_FIELDS
        }
        macro["n_total"] = sum(int(metrics["n_total"]) for metrics in dataset_metrics)
        macro_by_config[config] = macro
        summary_rows.append({
            "config": config,
            "max_searches": max_searches,
            "top_k": top_k,
            "scope": "macro",
            "dataset": "macro",
            **macro,
        })

    summary = {
        "checkpoint_step": args.checkpoint_step,
        "datasets": DATASETS,
        "expected_rows": EXPECTED_ROWS,
        "macro_aggregation": "unweighted_mean_over_datasets",
        "configs": {
            config: {
                "max_searches": parse_config(config)[0],
                "top_k": parse_config(config)[1],
                "datasets": {
                    dataset: next(
                        row for row in summary_rows
                        if row["config"] == config and row["dataset"] == dataset
                    )
                    for dataset in DATASETS
                },
                "macro": macro_by_config[config],
            }
            for config in CONFIGS
        },
    }
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    csv_fields = [
        "config",
        "max_searches",
        "top_k",
        "scope",
        "dataset",
        "n_total",
        *SUMMARY_FIELDS,
    ]
    with (args.root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    comparisons = [("vs_s5_k3", config, "s5_k3") for config in CONFIGS if config != "s5_k3"]
    comparisons.extend(
        ("adjacent_search", f"s{value}_k3", f"s{value - 1}_k3")
        for value in range(1, 8)
    )
    bootstrap_dir = args.root / "paired_bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_index = []
    for comparison_index, (family, candidate, baseline) in enumerate(comparisons):
        result = paired_bootstrap(
            candidate,
            baseline,
            rows,
            samples=args.bootstrap_samples,
            seed=args.seed + comparison_index,
        )
        result["family"] = family
        output = bootstrap_dir / f"{family}__{candidate}__vs__{baseline}.json"
        output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        bootstrap_index.append(result)
    (bootstrap_dir / "index.json").write_text(
        json.dumps(bootstrap_index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    runtime = load_runtime(args.root, exact_rpc_count)
    (args.root / "runtime_manifest.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_plots(args.root, macro_by_config)
    print(
        json.dumps(
            {
                "root": str(args.root),
                "configs": len(CONFIGS),
                "dataset_runs": len(CONFIGS) * len(DATASETS),
                "rows": sum(EXPECTED_ROWS.values()) * len(CONFIGS),
                "comparisons": len(comparisons),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
