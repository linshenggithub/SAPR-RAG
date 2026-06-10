#!/usr/bin/env python3
"""统计 SFT 推理结果对 HotpotQA gold supporting_facts 的检索命中率。

两个对齐口径：
  1) title 精确匹配（归一化后）：检索到的 doc.title 是否覆盖 gold supporting title
  2) text fallback：应对 corpus 标题错位脏数据——gold 句子文本是否出现在某个检索 doc 正文里

输出题级覆盖率（所有 gold 标题是否都命中）和文档级命中率，并按 cover_em 对错分层。
"""
import json
import re
import os
import sys
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import cover_em_score


def norm_title(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_gold(path):
    gold = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            sf = d.get("metadata", {}).get("supporting_facts", {})
            titles = sf.get("title", []) or []
            ctx = d.get("metadata", {}).get("context", {})
            ctx_titles = ctx.get("title", []) or []
            ctx_sents = ctx.get("sentences", ctx.get("text", [])) or []
            # 抽出 supporting 句子文本（按 title+sent_id 定位）
            sent_ids = sf.get("sent_id", []) or []
            sup_sents = []
            title2sents = {t: s for t, s in zip(ctx_titles, ctx_sents)}
            for t, sid in zip(titles, sent_ids):
                sents = title2sents.get(t)
                if isinstance(sents, list) and 0 <= sid < len(sents):
                    sup_sents.append(sents[sid])
            gold[d["id"]] = {
                "titles": titles,
                "sup_sents": sup_sents,
            }
    return gold


def collect_retrieved(rec):
    """返回该题所有检索到的 (title, text) 列表（跨 turn 去重）。"""
    docs = []
    seen = set()
    for t in rec.get("trace", []):
        if t.get("stage") == "retrieve":
            for d in t.get("docs", []):
                key = (d.get("title", ""), d.get("text", "")[:80])
                if key in seen:
                    continue
                seen.add(key)
                docs.append((d.get("title", ""), d.get("text", "")))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/mlx_devbox/users/mayi.summer/playground/SAPR-RAG/data/eval_results/hotpotqa/20260608_175824/merged.jsonl")
    ap.add_argument("--gold", default="/mlx_devbox/users/mayi.summer/playground/SAPR-RAG/data/eval/hotpotqa/dev.jsonl")
    args = ap.parse_args()

    gold = load_gold(args.gold)

    # 累计器：分 cover_em 正确/错误两层 + 总体
    buckets = {"all": defaultdict(float), "correct": defaultdict(float), "wrong": defaultdict(float)}

    n = 0
    n_with_cem = 0
    for line in open(args.results):
        rec = json.loads(line)
        rid = rec.get("id")
        if rid not in gold:
            continue
        g = gold[rid]
        gtitles = [norm_title(t) for t in g["titles"]]
        if not gtitles:
            continue
        n += 1

        docs = collect_retrieved(rec)
        rtitles = set(norm_title(t) for t, _ in docs)
        rtexts_norm = [norm_text(tx) for _, tx in docs]

        # title 命中
        title_hits = sum(1 for gt in gtitles if gt in rtitles)
        # text fallback 命中（gold 句子是否出现在任一 doc 正文）
        text_hits = 0
        for gs in g["sup_sents"]:
            gsn = norm_text(gs)
            if not gsn:
                continue
            if any(gsn in tx for tx in rtexts_norm):
                text_hits += 1
        # 联合命中（title 命中 或 对应句子文本命中）——按 title 个数上限
        # 简化：union 命中数 = max(title_hits, text_hits) 不精确，改为逐 gold 判定
        union_hits = 0
        for i, gt in enumerate(gtitles):
            hit = gt in rtitles
            if not hit and i < len(g["sup_sents"]):
                gsn = norm_text(g["sup_sents"][i])
                hit = bool(gsn) and any(gsn in tx for tx in rtexts_norm)
            union_hits += 1 if hit else 0

        ng = len(gtitles)
        full_title = 1.0 if title_hits == ng else 0.0
        full_union = 1.0 if union_hits == ng else 0.0

        # cover_em：结果文件没存，用 answer vs gold 现算
        cem = cover_em_score(rec.get("answer", ""), rec.get("gold", []))
        n_with_cem += 1
        layer = "correct" if cem >= 1.0 else "wrong"

        for b in ["all"] + ([layer] if layer else []):
            buckets[b]["n"] += 1
            buckets[b]["title_doc_hits"] += title_hits
            buckets[b]["union_doc_hits"] += union_hits
            buckets[b]["gold_docs"] += ng
            buckets[b]["full_title"] += full_title
            buckets[b]["full_union"] += full_union

    def report(name):
        b = buckets[name]
        nn = b["n"]
        if nn == 0:
            print(f"[{name}] no data")
            return
        print(f"[{name}] n={int(nn)}")
        print(f"    文档级命中率 (title精确):  {b['title_doc_hits']/b['gold_docs']:.3f}")
        print(f"    文档级命中率 (title+text): {b['union_doc_hits']/b['gold_docs']:.3f}")
        print(f"    题级全覆盖率 (title精确):  {b['full_title']/nn:.3f}")
        print(f"    题级全覆盖率 (title+text): {b['full_union']/nn:.3f}")

    print(f"总题数(有gold supporting): {n} | 含cover_em字段: {n_with_cem}\n")
    report("all")
    print()
    if buckets["correct"]["n"] or buckets["wrong"]["n"]:
        report("correct")
        print()
        report("wrong")


if __name__ == "__main__":
    main()
