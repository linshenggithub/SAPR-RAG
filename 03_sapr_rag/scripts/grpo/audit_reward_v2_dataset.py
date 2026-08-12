#!/usr/bin/env python3
"""审计 Reward-v2 训练数据。

重点检查真实训练 JSONL，而不是 mock：
- 训练/开发集问题是否重叠
- gold title / supporting sentence 是否为空或错位
- 2Wiki evidence 是否仍为空
- 是否误带 teacher_prompt
- gold title 在 wiki18 corpus 中是否可达
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJ_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJ_ROOT / "data/grpo/hotpotqa_2wiki_train_reward_v2.jsonl"
DEFAULT_CORPUS = PROJ_ROOT / "data/corpus/wiki18_extended.jsonl"
DEFAULT_DEV_PATHS = [
    PROJ_ROOT / "data/eval/hotpotqa/dev.jsonl",
    PROJ_ROOT / "data/eval/2wikimultihopqa/dev.jsonl",
]


def norm_text(value):
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def norm_title(value):
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_question(row):
    if row.get("question"):
        return str(row["question"]).strip()
    messages = row.get("messages") or []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content") or "")
        m = re.search(r"Question:\s*(.*)", content)
        if m:
            return m.group(1).strip()
    return ""


def load_dev_questions(paths):
    questions = set()
    ids = set()
    loaded = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            loaded[str(path)] = {"exists": False, "rows": 0}
            continue
        rows = 0
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                question = extract_question(row)
                if question:
                    questions.add(norm_text(question))
                if row.get("id") is not None:
                    ids.add(str(row["id"]))
        loaded[str(path)] = {"exists": True, "rows": rows}
    return questions, ids, loaded


def sentence_nonempty(value):
    if isinstance(value, (list, tuple)):
        return any(sentence_nonempty(v) for v in value)
    return bool(str(value or "").strip())


def iter_jsonl(path):
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            yield line_no, json.loads(line)


def scan_corpus_titles(corpus_path, needed_titles):
    reachable = set()
    if not corpus_path or not Path(corpus_path).exists():
        return reachable, {"exists": False, "rows_scanned": 0}

    needed_titles = set(needed_titles)
    rows = 0
    with open(corpus_path) as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = row.get("contents", "") or ""
            title = raw.split("\n", 1)[0].strip().strip('"')
            key = norm_title(title)
            if key in needed_titles:
                reachable.add(key)
                if len(reachable) == len(needed_titles):
                    break
    return reachable, {"exists": True, "rows_scanned": rows}


def audit(args):
    dataset = Path(args.dataset)
    dev_questions, dev_ids, dev_loaded = load_dev_questions(args.dev_paths)

    source_counts = Counter()
    question_counts = Counter()
    id_counts = Counter()
    needed_titles = set()
    rows = []

    report = {
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "dev_paths": dev_loaded,
        "total_rows": 0,
        "source_counts": {},
        "duplicate_questions": 0,
        "duplicate_ids": 0,
        "train_dev_question_overlap": 0,
        "train_dev_id_overlap": 0,
        "gold_titles_empty": 0,
        "title_sentence_length_mismatch": 0,
        "teacher_prompt_rows": 0,
        "by_source": defaultdict(lambda: {
            "rows": 0,
            "gold_titles_empty": 0,
            "title_sentence_length_mismatch": 0,
            "all_gold_sup_sents_empty": 0,
            "any_gold_sup_sent_empty": 0,
        }),
    }

    overlap_questions = []
    overlap_ids = []
    examples = defaultdict(list)

    for line_no, row in iter_jsonl(dataset):
        report["total_rows"] += 1
        source = str(row.get("source") or "unknown")
        source_counts[source] += 1
        report["by_source"][source]["rows"] += 1
        rows.append(row)

        question = extract_question(row)
        qkey = norm_text(question)
        if qkey:
            question_counts[qkey] += 1
            if qkey in dev_questions:
                report["train_dev_question_overlap"] += 1
                if len(overlap_questions) < 5:
                    overlap_questions.append({"line": line_no, "source": source, "question": question})

        if row.get("id") is not None:
            rid = str(row["id"])
            id_counts[rid] += 1
            if rid in dev_ids:
                report["train_dev_id_overlap"] += 1
                if len(overlap_ids) < 5:
                    overlap_ids.append({"line": line_no, "source": source, "id": rid})

        gold_titles = row.get("gold_titles") or []
        gold_sup_sents = row.get("gold_sup_sents") or []
        if not gold_titles:
            report["gold_titles_empty"] += 1
            report["by_source"][source]["gold_titles_empty"] += 1
            if len(examples["gold_titles_empty"]) < 5:
                examples["gold_titles_empty"].append({"line": line_no, "source": source, "question": question})

        if len(gold_titles) != len(gold_sup_sents):
            report["title_sentence_length_mismatch"] += 1
            report["by_source"][source]["title_sentence_length_mismatch"] += 1
            if len(examples["title_sentence_length_mismatch"]) < 5:
                examples["title_sentence_length_mismatch"].append({
                    "line": line_no,
                    "source": source,
                    "titles": len(gold_titles),
                    "sentences": len(gold_sup_sents),
                    "question": question,
                })

        if gold_sup_sents and not any(sentence_nonempty(v) for v in gold_sup_sents):
            report["by_source"][source]["all_gold_sup_sents_empty"] += 1
        if gold_sup_sents and any(not sentence_nonempty(v) for v in gold_sup_sents):
            report["by_source"][source]["any_gold_sup_sent_empty"] += 1

        if "teacher_prompt" in row:
            report["teacher_prompt_rows"] += 1
            if len(examples["teacher_prompt_rows"]) < 5:
                examples["teacher_prompt_rows"].append({"line": line_no, "source": source, "question": question})

        for title in gold_titles:
            key = norm_title(title)
            if key:
                needed_titles.add(key)

    report["source_counts"] = dict(source_counts)
    report["duplicate_questions"] = sum(count - 1 for count in question_counts.values() if count > 1)
    report["duplicate_ids"] = sum(count - 1 for count in id_counts.values() if count > 1)
    report["examples"] = dict(examples)
    report["overlap_question_examples"] = overlap_questions
    report["overlap_id_examples"] = overlap_ids

    reachable, corpus_report = scan_corpus_titles(args.corpus, needed_titles)
    unreachable = sorted(needed_titles - reachable)
    report["corpus"] = corpus_report
    report["gold_unique_titles"] = len(needed_titles)
    report["gold_titles_reachable"] = len(reachable)
    report["gold_titles_unreachable"] = len(unreachable)
    report["gold_title_reachability"] = round(len(reachable) / len(needed_titles), 6) if needed_titles else 0.0
    report["unreachable_title_examples"] = unreachable[:20]

    by_source = {}
    for source, values in report["by_source"].items():
        rows_count = values["rows"] or 1
        item = dict(values)
        item["all_gold_sup_sents_empty_rate"] = round(values["all_gold_sup_sents_empty"] / rows_count, 6)
        item["any_gold_sup_sent_empty_rate"] = round(values["any_gold_sup_sent_empty"] / rows_count, 6)
        by_source[source] = item
    report["by_source"] = by_source

    expected = {}
    if args.expected_source_count:
        for item in args.expected_source_count:
            source, count = item.split("=", 1)
            expected[source] = int(count)
    gate_failures = []
    if report["train_dev_question_overlap"] != 0 or report["train_dev_id_overlap"] != 0:
        gate_failures.append("train/dev overlap != 0")
    if report["gold_titles_empty"] != 0:
        gate_failures.append("gold_titles empty != 0")
    if report["title_sentence_length_mismatch"] != 0:
        gate_failures.append("title/sentence length mismatch != 0")
    if report["teacher_prompt_rows"] != 0:
        gate_failures.append("teacher_prompt rows != 0")
    for source, count in expected.items():
        if report["source_counts"].get(source, 0) != count:
            gate_failures.append(f"source count mismatch: {source}")
    if report["by_source"].get("2wiki", {}).get("all_gold_sup_sents_empty", 0) != 0:
        gate_failures.append("2Wiki all-sentence-empty != 0")

    report["expected_source_counts"] = expected
    report["gate_passed"] = not gate_failures
    report["gate_failures"] = gate_failures
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--dev_paths", nargs="*", default=[str(p) for p in DEFAULT_DEV_PATHS])
    parser.add_argument("--expected_source_count", action="append", default=[],
                        help="例如 --expected_source_count hotpotqa=3660")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail_on_gate", action="store_true")
    args = parser.parse_args()

    report = audit(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")
    if args.fail_on_gate and not report["gate_passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
