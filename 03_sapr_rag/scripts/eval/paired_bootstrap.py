#!/usr/bin/env python3
"""Paired bootstrap comparison for two SAPR-RAG evaluation result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from score import cover_em_score, em_score, f1_score


MetricFn = Callable[[str | None, list[str]], float]
METRICS: dict[str, MetricFn] = {
    "em": em_score,
    "f1": f1_score,
    "cover_em": cover_em_score,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--reasonrag_em", type=float)
    parser.add_argument("--reasonrag_f1", type=float)
    parser.add_argument("--reasonrag_cover_em", type=float)
    return parser.parse_args()


def load_unique_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("id"))
            if sample_id == "None":
                raise ValueError(f"{path}:{line_number}: missing id")
            if sample_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate id {sample_id}")
            rows[sample_id] = row
    return rows


def gold_answers(row: dict) -> list[str]:
    gold = row.get("gold") or []
    if isinstance(gold, str):
        return [gold]
    return [str(answer) for answer in gold]


def score_rows(
    ids: list[str],
    rows: dict[str, dict],
    metric: MetricFn,
) -> np.ndarray:
    return np.asarray(
        [
            metric(rows[sample_id].get("answer"), gold_answers(rows[sample_id]))
            for sample_id in ids
        ],
        dtype=np.float64,
    )


def tail_probability(values: np.ndarray, threshold: float, side: str) -> float:
    if side == "le":
        count = int(np.count_nonzero(values <= threshold))
    elif side == "ge":
        count = int(np.count_nonzero(values >= threshold))
    else:
        raise ValueError(f"unknown side: {side}")
    return (count + 1) / (values.size + 1)


def bootstrap_means(
    candidate: np.ndarray,
    baseline: np.ndarray,
    samples: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = candidate.size
    candidate_boot = np.empty(samples, dtype=np.float64)
    diff_boot = np.empty(samples, dtype=np.float64)
    diff = candidate - baseline

    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(0, n, size=(stop - start, n), dtype=np.int32)
        candidate_boot[start:stop] = candidate[indices].mean(axis=1)
        diff_boot[start:stop] = diff[indices].mean(axis=1)
    return candidate_boot, diff_boot


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    candidate_rows = load_unique_rows(Path(args.candidate))
    baseline_rows = load_unique_rows(Path(args.baseline))
    candidate_ids = set(candidate_rows)
    baseline_ids = set(baseline_rows)
    if candidate_ids != baseline_ids:
        only_candidate = sorted(candidate_ids - baseline_ids)[:10]
        only_baseline = sorted(baseline_ids - candidate_ids)[:10]
        raise ValueError(
            "candidate/baseline ID sets differ: "
            f"candidate={len(candidate_ids)} baseline={len(baseline_ids)} "
            f"only_candidate={only_candidate} only_baseline={only_baseline}"
        )

    ids = sorted(candidate_ids)
    reasonrag_values = {
        "em": args.reasonrag_em,
        "f1": args.reasonrag_f1,
        "cover_em": args.reasonrag_cover_em,
    }
    result = {
        "candidate_path": args.candidate,
        "baseline_path": args.baseline,
        "n": len(ids),
        "bootstrap_samples": args.samples,
        "seed": args.seed,
        "metrics": {},
    }

    for metric_index, (name, metric) in enumerate(METRICS.items()):
        candidate_scores = score_rows(ids, candidate_rows, metric)
        baseline_scores = score_rows(ids, baseline_rows, metric)
        candidate_boot, diff_boot = bootstrap_means(
            candidate_scores,
            baseline_scores,
            samples=args.samples,
            seed=args.seed + metric_index,
            batch_size=args.batch_size,
        )
        p_le_zero = tail_probability(diff_boot, 0.0, "le")
        p_ge_zero = tail_probability(diff_boot, 0.0, "ge")
        metric_result = {
            "candidate": round(float(candidate_scores.mean()), 6),
            "local_baseline": round(float(baseline_scores.mean()), 6),
            "paired_diff": round(
                float((candidate_scores - baseline_scores).mean()),
                6,
            ),
            "candidate_ci95": [
                round(float(value), 6)
                for value in np.quantile(candidate_boot, [0.025, 0.975])
            ],
            "paired_diff_ci95": [
                round(float(value), 6)
                for value in np.quantile(diff_boot, [0.025, 0.975])
            ],
            "paired_p_one_sided_gt": round(p_le_zero, 6),
            "paired_p_two_sided": round(min(1.0, 2.0 * min(p_le_zero, p_ge_zero)), 6),
        }

        reasonrag = reasonrag_values[name]
        if reasonrag is not None:
            metric_result.update({
                "reasonrag_paper": reasonrag,
                "vs_reasonrag_diff": round(
                    float(candidate_scores.mean()) - reasonrag,
                    6,
                ),
                "candidate_bootstrap_p_le_reasonrag": round(
                    tail_probability(candidate_boot, reasonrag, "le"),
                    6,
                ),
                "candidate_ci_entirely_above_reasonrag": bool(
                    np.quantile(candidate_boot, 0.025) > reasonrag
                ),
            })
        result["metrics"][name] = metric_result

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
