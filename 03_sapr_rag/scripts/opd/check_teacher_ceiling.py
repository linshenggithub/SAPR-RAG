#!/usr/bin/env python3
"""Compare teacher and SFT results on identical IDs and enforce the OPD ceiling gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_score_module(project_root: Path):
    path = project_root / "03_sapr_rag/scripts/eval/score.py"
    spec = importlib.util.spec_from_file_location("sapr_score", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def align_baseline(candidate: list[dict], baseline: list[dict]) -> list[dict]:
    by_id = {str(row.get("id")): row for row in baseline}
    aligned = []
    missing = []
    for row in candidate:
        key = str(row.get("id"))
        if key not in by_id:
            missing.append(key)
        else:
            aligned.append(by_id[key])
    if missing:
        raise ValueError(f"baseline is missing {len(missing)} candidate ids; examples={missing[:5]}")
    return aligned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", action="append", required=True, help="NAME=results.jsonl")
    parser.add_argument("--baseline", action="append", required=True, help="NAME=results.jsonl")
    parser.add_argument("--min-macro-f1-delta", type=float, default=0.05)
    parser.add_argument("--max-answer-rate-drop", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        result[name] = Path(path)
    return result


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    score = load_score_module(project_root)
    teachers = parse_mapping(args.teacher)
    baselines = parse_mapping(args.baseline)
    if set(teachers) != set(baselines):
        raise ValueError("teacher and baseline dataset names must match")

    report = {"datasets": {}, "min_macro_f1_delta": args.min_macro_f1_delta}
    teacher_f1 = []
    baseline_f1 = []
    answer_rate_ok = True
    positive_datasets = 0
    for name in sorted(teachers):
        teacher_rows = load_rows(teachers[name])
        baseline_rows = align_baseline(teacher_rows, load_rows(baselines[name]))
        teacher_metrics = score.evaluate(teacher_rows)
        baseline_metrics = score.evaluate(baseline_rows)
        delta_f1 = teacher_metrics["f1"] - baseline_metrics["f1"]
        teacher_answer_rate = teacher_metrics["n_answered"] / max(teacher_metrics["n_total"], 1)
        baseline_answer_rate = baseline_metrics["n_answered"] / max(baseline_metrics["n_total"], 1)
        teacher_f1.append(teacher_metrics["f1"])
        baseline_f1.append(baseline_metrics["f1"])
        positive_datasets += int(delta_f1 > 0)
        answer_rate_ok &= teacher_answer_rate >= baseline_answer_rate - args.max_answer_rate_drop
        report["datasets"][name] = {
            "teacher": teacher_metrics,
            "sft_baseline": baseline_metrics,
            "f1_delta": round(delta_f1, 6),
            "answer_rate_delta": round(teacher_answer_rate - baseline_answer_rate, 6),
        }

    macro_teacher = sum(teacher_f1) / len(teacher_f1)
    macro_baseline = sum(baseline_f1) / len(baseline_f1)
    macro_delta = macro_teacher - macro_baseline
    passed = (
        macro_delta >= args.min_macro_f1_delta
        and positive_datasets >= 2
        and answer_rate_ok
    )
    report["summary"] = {
        "macro_teacher_f1": round(macro_teacher, 6),
        "macro_sft_f1": round(macro_baseline, 6),
        "macro_f1_delta": round(macro_delta, 6),
        "positive_f1_datasets": positive_datasets,
        "answer_rate_gate": answer_rate_ok,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
