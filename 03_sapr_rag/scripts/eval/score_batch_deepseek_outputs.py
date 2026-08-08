#!/usr/bin/env python3
"""给 batch_deepseek_api.py 输出打 EM / Cover EM / F1。

要求输出 JSONL 中每行包含：
  - content: API 预测
  - input.gold: gold answer list

示例：
  python 03_sapr_rag/scripts/eval/score_batch_deepseek_outputs.py \
    --input outputs.jsonl \
    --output metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Iterable, List


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
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.input)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    scored = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in rows:
        pred = row.get("content", "")
        inp = row.get("input", {}) or {}
        gold = inp.get("gold", [])
        if isinstance(gold, str):
            gold = [gold]
        u = row.get("usage", {}) or {}
        for k in usage:
            usage[k] += int(u.get(k) or 0)
        scored.append({
            "key": row.get("key"),
            "id": row.get("id"),
            "ok": bool(row.get("ok")),
            "prediction": pred,
            "gold": gold,
            "em": em_score(pred, gold),
            "cover_em": cover_em_score(pred, gold),
            "f1": f1_score(pred, gold),
        })

    n = len(scored)
    ok = sum(1 for x in rows if x.get("ok"))
    metrics = {
        "n_total": n,
        "n_ok": ok,
        "em": round(sum(x["em"] for x in scored) / n, 4) if n else 0.0,
        "cover_em": round(sum(x["cover_em"] for x in scored) / n, 4) if n else 0.0,
        "f1": round(sum(x["f1"] for x in scored) / n, 4) if n else 0.0,
        **usage,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
