"""从推理结果里抽 badcase，按失败模式分桶。

badcase 定义：cover_em == 0（与 score.py 主指标对齐）。
分桶：
  - max_turns_exceeded: 跑满 max_turns，answer 为 None
  - no_query_or_answer: 模型输出格式崩，既没 <query> 也没 <answer>
  - other_error: 其它错误（异常等）
  - repeated_query: history 里出现重复 query（死循环模式）
  - empty_evidence_heavy: evidence=None 占 history >= 50%（检索失败主导）
  - wrong_answer: 模型答了但 cover_em=0（语义错误，最值得人工看）

每条 case 可能命中多个桶（如 max_turns + repeated_query），用 categories 列表记录。

用法：
  python extract_badcases.py \
    --input data/eval_results/hotpotqa/20260608_175824/merged.jsonl \
    --out_dir data/eval_results/hotpotqa/20260608_175824/badcases
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

# 复用 score.py 口径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import normalize_answer, cover_em_score, f1_score


def gold_list(r):
    g = r.get("gold") or []
    if isinstance(g, str):
        g = [g]
    return g


def categorize(r):
    """返回 (cover_em, categories: list[str])。"""
    cats = []
    pred = r.get("answer")
    gold = gold_list(r)
    err = r.get("error")
    history = r.get("history", [])

    cem = cover_em_score(pred, gold) if pred is not None else 0.0

    # 1. 错误类型
    if err == "max_turns_exceeded":
        cats.append("max_turns_exceeded")
    elif err == "no_query_or_answer":
        cats.append("no_query_or_answer")
    elif err is not None:
        cats.append("other_error")

    # 2. 答了但答错
    if pred is not None and cem == 0.0:
        cats.append("wrong_answer")

    # 3. 死循环（history 中 query 重复）
    qs = [h.get("query", "") for h in history]
    if qs and len(qs) != len(set(qs)):
        cats.append("repeated_query")

    # 4. evidence 抽取失败主导
    if history:
        empty = sum(1 for h in history
                    if (h.get("evidence") or "").strip().lower() in ("none", ""))
        if empty / len(history) >= 0.5:
            cats.append("empty_evidence_heavy")

    return cem, cats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="merged.jsonl")
    p.add_argument("--out_dir", required=True, help="输出目录")
    p.add_argument("--samples_per_cat", type=int, default=20,
                   help="每类抽多少条到 samples/<cat>.jsonl 便于人工看")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    samples_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    n_total = 0
    n_bad = 0
    cat_counter = Counter()
    cat_bucket = defaultdict(list)  # cat -> list of case (用于采样)
    primary_counter = Counter()     # 互斥主类别统计

    badcases_path = os.path.join(args.out_dir, "badcases.jsonl")
    with open(args.input) as fi, open(badcases_path, "w") as fo:
        for line in fi:
            if not line.strip():
                continue
            r = json.loads(line)
            n_total += 1
            cem, cats = categorize(r)
            if cem >= 1.0:
                continue
            n_bad += 1

            # 主类别：按优先级取一个，便于互斥分桶统计
            primary = "wrong_answer"
            for c in ["no_query_or_answer", "other_error",
                      "max_turns_exceeded", "wrong_answer"]:
                if c in cats:
                    primary = c
                    break
            primary_counter[primary] += 1

            for c in cats:
                cat_counter[c] += 1
                cat_bucket[c].append(r["id"] if "id" in r else None)

            # 落盘 badcase（重排字段顺序：把人最关心的放最前面，trace 放最后）
            r_out = {
                "id": r.get("id"),
                "question": r.get("question"),
                "gold": r.get("gold"),
                "answer": r.get("answer"),
                "_badcase_primary": primary,
                "_badcase_categories": cats,
                "_cover_em": cem,
                "error": r.get("error"),
                "history": r.get("history", []),
                "latency_s": r.get("latency_s"),
                "trace": r.get("trace", []),
            }
            fo.write(json.dumps(r_out, ensure_ascii=False) + "\n")

    # 每类抽 N 条到独立 samples 文件
    # 重新扫一遍，避免一次性把 badcases 吃进内存
    seen_per_cat = Counter()
    cat_samples_files = {
        c: open(os.path.join(samples_dir, f"{c}.jsonl"), "w")
        for c in cat_counter
    }
    with open(badcases_path) as f:
        for line in f:
            r = json.loads(line)
            for c in r["_badcase_categories"]:
                if seen_per_cat[c] < args.samples_per_cat:
                    cat_samples_files[c].write(line)
                    seen_per_cat[c] += 1
    for fo in cat_samples_files.values():
        fo.close()

    # 精简 csv：question/gold/answer/分类，便于电子表格扫读
    csv_path = os.path.join(args.out_dir, "badcases_brief.csv")
    with open(badcases_path) as fi, open(csv_path, "w", newline="") as fc:
        w = csv.writer(fc)
        w.writerow(["id", "primary", "categories", "n_turns",
                    "question", "gold", "answer"])
        for line in fi:
            r = json.loads(line)
            gold = r.get("gold")
            if isinstance(gold, list):
                gold = " | ".join(map(str, gold))
            w.writerow([
                r.get("id"),
                r.get("_badcase_primary"),
                ";".join(r.get("_badcase_categories", [])),
                len(r.get("history", [])),
                (r.get("question") or "").replace("\n", " "),
                (gold or "").replace("\n", " "),
                (r.get("answer") or "").replace("\n", " "),
            ])

    # summary
    summary = {
        "input": args.input,
        "n_total": n_total,
        "n_badcase": n_bad,
        "badcase_rate": round(n_bad / n_total, 4) if n_total else 0.0,
        "primary_category_counts": dict(primary_counter),
        "category_counts_overlapping": dict(cat_counter),
        "samples_per_cat": args.samples_per_cat,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[done] badcases  -> {badcases_path}")
    print(f"[done] brief csv -> {csv_path}")
    print(f"[done] samples   -> {samples_dir}/<category>.jsonl")
    print(f"[done] summary   -> {summary_path}")


if __name__ == "__main__":
    main()
