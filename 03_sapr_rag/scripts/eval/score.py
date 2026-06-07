"""SAPR-RAG 推理结果评估。

读 agent_infer.py 产出的 results.jsonl，输出 EM / F1 / 行为指标。
EM/F1 normalize 与 SQuAD/HotpotQA 官方一致。

用法：
  python score.py --input results.jsonl
  python score.py --input results.jsonl --output metrics.json
"""

import argparse
import json
import re
import string
from collections import Counter


# ─────────── SQuAD/HotpotQA normalize ───────────
def normalize_answer(s):
    """lowercase + 去标点 + 去冠词 + 折叠空格"""
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


def f1_score(pred, golds):
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


# ─────────── 评估主体 ───────────
def evaluate(results):
    n = len(results)
    n_answered = 0
    n_max_turns = 0
    em_sum = 0.0
    f1_sum = 0.0
    turns_sum = 0
    empty_ev_count = 0
    total_ev = 0
    latency_sum = 0.0

    for r in results:
        pred = r.get("answer")
        gold = r.get("gold") or []
        if isinstance(gold, str):
            gold = [gold]

        # answer 行为指标
        if r.get("error") == "max_turns_exceeded":
            n_max_turns += 1
        if pred is not None:
            n_answered += 1
            em_sum += em_score(pred, gold)
            f1_sum += f1_score(pred, gold)

        # turns / evidence 行为指标
        history = r.get("history", [])
        turns_sum += len(history)
        for h in history:
            total_ev += 1
            if (h.get("evidence") or "").strip().lower() in ("none", ""):
                empty_ev_count += 1

        latency_sum += r.get("latency_s", 0.0)

    return {
        "n_total": n,
        "n_answered": n_answered,
        "n_max_turns_exceeded": n_max_turns,
        # 答案质量（分母 = 全部题，不是 n_answered，因为 None 计 0 分）
        "em": round(em_sum / n, 4) if n else 0.0,
        "f1": round(f1_sum / n, 4) if n else 0.0,
        # 行为指标
        "avg_turns": round(turns_sum / n, 3) if n else 0.0,
        "max_turns_rate": round(n_max_turns / n, 4) if n else 0.0,
        "empty_evidence_rate": round(empty_ev_count / total_ev, 4) if total_ev else 0.0,
        "avg_latency_s": round(latency_sum / n, 2) if n else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="agent_infer.py 产出的 results.jsonl")
    p.add_argument("--output", default=None, help="可选，把 metrics 写到 json")
    args = p.parse_args()

    with open(args.input) as f:
        results = [json.loads(l) for l in f if l.strip()]

    metrics = evaluate(results)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
