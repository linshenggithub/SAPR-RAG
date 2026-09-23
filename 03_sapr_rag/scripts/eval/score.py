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


def _tokens_contain(hay_toks, needle_toks):
    """needle_toks 作为连续子序列出现在 hay_toks 中（token 级，避免 no 命中 not）"""
    if not needle_toks:
        return False
    n, m = len(hay_toks), len(needle_toks)
    for i in range(n - m + 1):
        if hay_toks[i:i + m] == needle_toks:
            return True
    return False


def cover_em_score(pred, golds):
    """cover-EM / acc：normalize 后 gold 作为连续 token 子序列出现在 pred 中。
    针对 SFT 长句答案的标准口径（与 ReasonRAG/FlashRAG 对齐）。"""
    p_toks = normalize_answer(pred).split()
    for g in golds:
        g_toks = normalize_answer(g).split()
        if g_toks and _tokens_contain(p_toks, g_toks):
            return 1.0
    return 0.0


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
    n_forced_answer = 0
    n_forced_answer_valid = 0
    n_search_budget_hit = 0
    n_format_failures = 0
    n_service_failures = 0
    em_sum = 0.0
    cover_em_sum = 0.0
    f1_sum = 0.0
    forced_em_sum = 0.0
    forced_f1_sum = 0.0
    turns_sum = 0
    logical_searches_sum = 0
    actual_retrieval_rpcs_sum = 0
    reasoner_tokens_sum = 0
    evidence_tokens_sum = 0
    forced_answer_tokens_sum = 0
    duplicate_query_count = 0
    total_query_count = 0
    empty_ev_count = 0
    total_ev = 0
    latency_sum = 0.0
    latencies = []

    for r in results:
        pred = r.get("answer")
        gold = r.get("gold") or []
        if isinstance(gold, str):
            gold = [gold]

        # answer 行为指标
        if r.get("error") == "max_turns_exceeded":
            n_max_turns += 1
        behavior = r.get("behavior") or {}
        forced_answer = bool(behavior.get("forced_answer", False))
        if forced_answer:
            n_forced_answer += 1
            n_forced_answer_valid += int(bool(behavior.get("forced_answer_valid", False)))
        if behavior.get("force_reason") == "search_budget":
            n_search_budget_hit += 1
        if r.get("error") == "forced_answer_format_failure":
            n_format_failures += 1
        retrieval_errors = int(behavior.get("retrieval_error_count", 0))
        n_service_failures += int(retrieval_errors > 0)
        if pred is not None:
            n_answered += 1
            em = em_score(pred, gold)
            cover_em = cover_em_score(pred, gold)
            f1 = f1_score(pred, gold)
            em_sum += em
            cover_em_sum += cover_em
            f1_sum += f1
            if forced_answer:
                forced_em_sum += em
                forced_f1_sum += f1

        # turns / evidence 行为指标
        history = r.get("history", [])
        turns_sum += int(behavior.get("num_turns", len(history)))
        logical_searches_sum += int(behavior.get("logical_search_count", len(history)))
        actual_retrieval_rpcs_sum += int(behavior.get(
            "actual_retrieval_rpc_count",
            behavior.get("actual_search_count", len(history)),
        ))
        reasoner_tokens_sum += int(behavior.get("reasoner_token_count", 0))
        evidence_tokens_sum += int(behavior.get("evidence_token_count", 0))
        forced_answer_tokens_sum += int(behavior.get("forced_answer_token_count", 0))
        duplicate_query_count += int(behavior.get(
            "exact_duplicate_count",
            behavior.get("repeat_count_from_text", 0),
        ))
        total_query_count += int(behavior.get("num_queries", len(history)))
        for h in history:
            total_ev += 1
            if (h.get("evidence") or "").strip().lower() in ("none", ""):
                empty_ev_count += 1

        latency = float(r.get("latency_s", 0.0))
        latency_sum += latency
        latencies.append(latency)

    sorted_latencies = sorted(latencies)

    def percentile(values, q):
        if not values:
            return 0.0
        index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
        return values[index]

    return {
        "n_total": n,
        "n_answered": n_answered,
        "n_max_turns_exceeded": n_max_turns,
        "n_forced_answer": n_forced_answer,
        "n_forced_answer_valid": n_forced_answer_valid,
        "n_search_budget_hit": n_search_budget_hit,
        "n_format_failures": n_format_failures,
        "n_service_failures": n_service_failures,
        # 答案质量（分母 = 全部题，不是 n_answered，因为 None 计 0 分）
        "em": round(em_sum / n, 4) if n else 0.0,
        "cover_em": round(cover_em_sum / n, 4) if n else 0.0,
        "f1": round(f1_sum / n, 4) if n else 0.0,
        # 行为指标
        "answer_rate": round(n_answered / n, 4) if n else 0.0,
        "forced_answer_rate": round(n_forced_answer / n, 4) if n else 0.0,
        "forced_answer_valid_rate": (
            round(n_forced_answer_valid / n_forced_answer, 4) if n_forced_answer else 0.0
        ),
        "forced_answer_em": round(forced_em_sum / n_forced_answer, 4) if n_forced_answer else 0.0,
        "forced_answer_f1": round(forced_f1_sum / n_forced_answer, 4) if n_forced_answer else 0.0,
        "search_budget_hit_rate": round(n_search_budget_hit / n, 4) if n else 0.0,
        "avg_turns": round(turns_sum / n, 3) if n else 0.0,
        "avg_logical_searches": round(logical_searches_sum / n, 3) if n else 0.0,
        "avg_actual_retrieval_rpcs": round(actual_retrieval_rpcs_sum / n, 3) if n else 0.0,
        "avg_reasoner_tokens": round(reasoner_tokens_sum / n, 3) if n else 0.0,
        "avg_evidence_tokens": round(evidence_tokens_sum / n, 3) if n else 0.0,
        "avg_forced_answer_tokens": round(forced_answer_tokens_sum / n, 3) if n else 0.0,
        "repeat_query_rate": (
            round(duplicate_query_count / total_query_count, 4) if total_query_count else 0.0
        ),
        "format_failure_rate": round(n_format_failures / n, 4) if n else 0.0,
        "service_failure_rate": round(n_service_failures / n, 4) if n else 0.0,
        "max_turns_rate": round(n_max_turns / n, 4) if n else 0.0,
        "empty_evidence_rate": round(empty_ev_count / total_ev, 4) if total_ev else 0.0,
        "avg_latency_s": round(latency_sum / n, 2) if n else 0.0,
        "latency_p50_s": round(percentile(sorted_latencies, 0.50), 3),
        "latency_p95_s": round(percentile(sorted_latencies, 0.95), 3),
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
