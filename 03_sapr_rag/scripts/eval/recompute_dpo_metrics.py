#!/usr/bin/env python3
"""用 SAPR-RAG 的 cover_em / EM / F1 口径重新计算 ReasonRAG DPO 结果。

ReasonRAG 的 intermediate_data.json 里每条样本有 output.pred / output.answer，
但原 pipeline 只算了 em/f1/acc。本脚本对齐 SAPR-RAG score.py 的归一化与 cover_em 逻辑，
并额外输出 merged.jsonl（供 llm_judge_deepseek.py 使用）。

用法：
    python recompute_dpo_metrics.py --input intermediate_data.json --output_dir .
"""
import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path


# ──── 与 SAPR-RAG score.py 完全一致的归一化 ────
def normalize_answer(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def em_score(pred, golds):
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def _tokens_contain(hay_toks, needle_toks):
    if not needle_toks:
        return False
    n, m = len(hay_toks), len(needle_toks)
    for i in range(n - m + 1):
        if hay_toks[i:i + m] == needle_toks:
            return True
    return False


def cover_em_score(pred, golds):
    p_toks = normalize_answer(pred).split()
    for g in golds:
        g_toks = normalize_answer(g).split()
        if g_toks and _tokens_contain(p_toks, g_toks):
            return 1.0
    return 0.0


def f1_score_fn(pred, golds):
    p_toks = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        g_toks = normalize_answer(g).split()
        if not p_toks or not g_toks:
            best = max(best, float(p_toks == g_toks))
            continue
        common = Counter(p_toks) & Counter(g_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(p_toks)
        recall = num_same / len(g_toks)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def extract_answer(item):
    """从 ReasonRAG intermediate_data 的单个样本里抽取预测答案。
    优先 output.pred，其次 output.answer；都为空则返回空串。
    """
    out = item.get("output", {})
    pred = out.get("pred")
    if pred is None or str(pred).strip() == "":
        pred = out.get("answer", "")
    return str(pred) if pred is not None else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="intermediate_data.json")
    ap.add_argument("--output_dir", default=".", help="输出目录")
    ap.add_argument("--dataset", default="", help="数据集名（仅用于日志）")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_tag = args.dataset or Path(args.input).parent.name

    n = len(data)
    em_sum = cover_sum = f1_sum = 0.0
    n_answered = 0
    n_finish_true = 0
    turns_sum = 0
    merged_rows = []

    for item in data:
        q = item.get("question", "")
        golds = item.get("golden_answers", [])
        if isinstance(golds, str):
            golds = [golds]
        golds = [g for g in golds if g is not None]

        out = item.get("output", {})
        pred = extract_answer(item)
        finish = out.get("finish_flag", False)
        iter_count = out.get("iteration_count", 0)

        if str(pred).strip():
            n_answered += 1
            em_sum += em_score(pred, golds)
            cover_sum += cover_em_score(pred, golds)
            f1_sum += f1_score_fn(pred, golds)
        if finish:
            n_finish_true += 1

        # ReasonRAG iteration_count 近似 turns（+1 因为 begin_reasoning 占一轮）
        turns_sum += int(iter_count)

        merged_rows.append({
            "id": item.get("id", ""),
            "question": q,
            "gold": golds,
            "answer": str(pred) if pred is not None else "",
            "finish_flag": finish,
            "iteration_count": iter_count,
        })

    metrics = {
        "dataset": dataset_tag,
        "source": str(args.input),
        "n_total": n,
        "n_answered": n_answered,
        "n_finish_true": n_finish_true,
        "cover_em": round(cover_sum / n, 4) if n else 0.0,
        "em": round(em_sum / n, 4) if n else 0.0,
        "f1": round(f1_sum / n, 4) if n else 0.0,
        "avg_turns": round(turns_sum / n, 3) if n else 0.0,
        "note": "cover_em/em/f1 全部用 SAPR-RAG score.py 口径归一化计算",
    }

    metrics_path = out_dir / "metrics_sapr.json"
    merged_path = out_dir / "merged.jsonl"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(merged_path, "w") as f:
        for r in merged_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nmetrics -> {metrics_path}")
    print(f"merged  -> {merged_path}  ({len(merged_rows)} rows, 供 llm_judge_deepseek.py 使用)")


if __name__ == "__main__":
    main()
