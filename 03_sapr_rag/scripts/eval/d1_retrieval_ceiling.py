#!/usr/bin/env python3
"""D1 检索召回上限统计。

目的：判断 F1 的天花板是被检索器卡住，还是被模型利用/推理能力卡住。

做法：对已有的 dev 轨迹（results.raw.jsonl，history[].evidence 里是每轮检索回的文档文本），
使用可严格复核的命中判定（去重 gold title 与检索文档 title 精确匹配），统计：
  - per-question gold 覆盖率（整条轨迹所有检索合起来，覆盖了多少比例的 gold 证据）
  - full_recall_rate：gold 被完全检索到的题目比例（= 检索召回上限）
  - 交叉分析：在 gold 被完全检索到的题目里，模型答对(cover-EM)的比例
    → 若召回高但答对低，瓶颈在利用；若召回本身低，瓶颈在检索。
"""
import argparse
import json
import re
from pathlib import Path


def norm_text(s: str) -> str:
    s = re.sub(r"[^\w]+", " ", str(s).lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def load_gold(dev_path: Path):
    """返回 {id: {"titles": [str, ...], "n_gold": int}}。"""
    gold = {}
    for line in dev_path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        meta = r.get("metadata", {})
        sf = meta.get("supporting_facts", {})
        titles = sf.get("title", []) or []
        unique_titles = []
        seen = set()
        for title in titles:
            normalized = norm_title(title)
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_titles.append(normalized)
        gold[r["id"]] = {
            "titles": unique_titles,
            "n_gold": len(unique_titles),
        }
    return gold


def collect_retrieved_docs(row):
    docs = []
    for trace in row.get("trace", []):
        if not isinstance(trace, dict):
            continue
        info = trace.get("rollout_infos", {}) or {}
        for step in info.get("retrieved_steps", []) or []:
            docs.extend(step.get("docs", []) or [])
    return docs


def gold_hits_for_docs(docs, g):
    retrieved_titles = {
        norm_title(doc.get("title", "")) for doc in docs or []
    }
    return {
        index
        for index, title in enumerate(g["titles"])
        if title in retrieved_titles
    }


def analyze(results_path: Path, gold):
    rows = [json.loads(l) for l in results_path.open() if l.strip()]
    n = len(rows)
    per_cov = []
    full_recall = 0
    full_and_correct = 0
    full_but_wrong = 0
    notfull_and_correct = 0
    notfull_and_wrong = 0
    n_gold_avail = 0
    for r in rows:
        rid = r["id"]
        g = gold.get(rid)
        if not g or g["n_gold"] == 0:
            continue
        n_gold_avail += 1
        hits = gold_hits_for_docs(collect_retrieved_docs(r), g)
        cov = len(hits) / g["n_gold"]
        per_cov.append(cov)
        is_full = len(hits) == g["n_gold"]
        # cover-EM: gold answer 作为子串出现在预测里
        pred = norm_text(r.get("answer", ""))
        golds = r.get("gold", []) or []
        correct = any(norm_text(gg) and norm_text(gg) in pred for gg in golds)
        if is_full:
            full_recall += 1
            if correct:
                full_and_correct += 1
            else:
                full_but_wrong += 1
        else:
            if correct:
                notfull_and_correct += 1
            else:
                notfull_and_wrong += 1
    avg_cov = sum(per_cov) / len(per_cov) if per_cov else 0.0
    return {
        "n": n,
        "n_gold_avail": n_gold_avail,
        "avg_gold_coverage": avg_cov,
        "full_recall_rate": full_recall / n_gold_avail if n_gold_avail else 0,
        "full_recall_count": full_recall,
        "full_and_correct": full_and_correct,
        "full_but_wrong": full_but_wrong,
        "notfull_and_correct": notfull_and_correct,
        "notfull_and_wrong": notfull_and_wrong,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Result JSONL to analyze; repeat for multiple runs.",
    )
    args = parser.parse_args()

    targets = []
    for item in args.result:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            parser.error(f"invalid --result {item!r}; expected NAME=PATH")
        targets.append((name, Path(raw_path)))

    gold = load_gold(args.dev)
    print(f"[gold] loaded {len(gold)} questions from {args.dev}\n")
    for name, p in targets:
        if not p.exists():
            print(f"[skip] {name}: not found {p}")
            continue
        s = analyze(p, gold)
        print(f"=== {name} ===")
        print(f"  样本数(有gold): {s['n_gold_avail']}/{s['n']}")
        print(f"  平均gold覆盖率: {s['avg_gold_coverage']*100:.1f}%")
        print(f"  完全召回率(检索上限): {s['full_recall_rate']*100:.1f}%  ({s['full_recall_count']}/{s['n_gold_avail']})")
        print(f"  交叉分析(cover-EM口径):")
        print(f"    gold全召回 & 答对 : {s['full_and_correct']}")
        print(f"    gold全召回 & 答错 : {s['full_but_wrong']}  <- 利用/推理损失")
        print(f"    gold未全召回 & 答对: {s['notfull_and_correct']}")
        print(f"    gold未全召回 & 答错: {s['notfull_and_wrong']}  <- 检索损失")
        fr = s['full_and_correct'] + s['full_but_wrong']
        if fr:
            print(f"    → 在gold全召回题中，答对率: {s['full_and_correct']/fr*100:.1f}%")
        print()


if __name__ == "__main__":
    main()
