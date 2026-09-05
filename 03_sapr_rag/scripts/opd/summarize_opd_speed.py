#!/usr/bin/env python3
"""Summarize OPD speed benchmark logs and GPU telemetry."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import statistics
from pathlib import Path


METRIC_PATTERN = re.compile(r"\{[^\n]*'global_step/max_steps': '[^']+'[^\n]*\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--warmup-steps", type=int, default=1)
    return parser.parse_args()


def parse_train_metrics(path: Path) -> list[dict]:
    rows = []
    text = path.read_text(errors="replace")
    for match in METRIC_PATTERN.findall(text):
        try:
            row = ast.literal_eval(match)
        except (SyntaxError, ValueError):
            continue
        if row.get("step_time") is not None:
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    variants = list(csv.DictReader((args.result_dir / "summary.tsv").open(), delimiter="\t"))
    gpu_rows = list(csv.DictReader((args.result_dir / "gpu_metrics.csv").open()))
    result = {}

    for variant in variants:
        name = variant["variant"]
        run_name = f"{args.prefix}_{name}"
        log_dir = args.log_root / run_name
        train_log = log_dir / "train.log"
        rows = parse_train_metrics(train_log)
        measured = rows[args.warmup_steps:]
        step_times = [float(row["step_time"]) for row in measured]
        variant_gpu_rows = [row for row in gpu_rows if row["variant"] == name]
        per_gpu = {}
        for index in sorted({row["index"].strip() for row in variant_gpu_rows}, key=int):
            selected = [row for row in variant_gpu_rows if row["index"].strip() == index]
            utils = [float(row["utilization_gpu_pct"]) for row in selected]
            memory = [float(row["memory_used_mib"]) for row in selected]
            per_gpu[index] = {
                "mean_utilization_pct": statistics.mean(utils) if utils else None,
                "peak_memory_mib": max(memory) if memory else None,
            }
        result[name] = {
            **variant,
            "logged_steps": len(rows),
            "measured_steps": len(measured),
            "mean_step_time_s": statistics.mean(step_times) if step_times else None,
            "median_step_time_s": statistics.median(step_times) if step_times else None,
            "mean_completion_length": (
                statistics.mean(float(row["completions/mean_length"]) for row in measured)
                if measured else None
            ),
            "mean_loss": (
                statistics.mean(float(row["loss"]) for row in measured)
                if measured else None
            ),
            "mean_teacher_kl": (
                statistics.mean(float(row["teacher_kl"]) for row in measured)
                if measured else None
            ),
            "max_clipped_ratio": (
                max(float(row["completions/clipped_ratio"]) for row in measured)
                if measured else None
            ),
            "gpu": per_gpu,
        }

    output_path = args.result_dir / "summary.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
