#!/usr/bin/env python3
"""dev.jsonl → ms-swift GRPO 训练 jsonl。

每行输出：
  - messages: [{system: REASONING_SYSTEM}, {user: "Question: {q}"}]（首轮 prompt）
  - golden_answers / gold_titles / gold_sup_sents：透传列，供 reward 用
    （均为该行独立 list，满足 ms-swift rows_to_batched 要求）

可选 --corpus 做"gold 全部不可达"预过滤（grpo_plan §2.0c 的 78 题）：
  扫一遍 wiki18_extended.jsonl 建归一化 title 集合，剔除 gold title 全不在集合的题。
  不传 --corpus 则跳过预过滤（reward 端对 num_gold=0 有兜底）。

用法：
  python build_grpo_dataset.py \
    --gold  data/eval/hotpotqa/dev.jsonl \
    --output data/grpo/hotpotqa_train.jsonl \
    [--corpus data/corpus/wiki18_extended.jsonl]
"""
import argparse
import json
import os
import re
from pathlib import Path

PROJ_ROOT = Path("/mlx_devbox/users/mayi.summer/playground/SAPR-RAG")

# 与 agent_infer.REASONING_SYSTEM 完全一致（内联复制，避免 import 重依赖）
REASONING_SYSTEM = (
    "You are an assistant for question answering with access to a retrieval tool. "
    "Upon receiving a question, your task is to:\n"
    "* Analyze and Decompose the Question: Break the question into smaller, manageable "
    "sub-questions to ensure all aspects are addressed.\n"
    "* Evaluate Your Knowledge: Assess each sub-question or component:\n"
    "- Identify parts you can confidently answer based on your existing knowledge.\n"
    "- Pinpoint parts that require additional information or verification through retrieval tools.\n"
    "* Conciseness: Ensure both queries and answers are concise, using nouns or short "
    "phrases whenever possible.\n"
    "* Respond Format:\n"
    "If your knowledge is sufficient to answer the question, conclude with:\n"
    '"So the answer is <answer>answer</answer>"\n'
    "If retrieval is necessary to provide a complete answer, conclude with:\n"
    '"So the next query is <query>query</query>"\n'
)


def norm_title(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_gold(d):
    """从一条 dev 记录抽 gold_titles 与 gold_sup_sents（复用 retrieval_recall.load_gold 逻辑）。"""
    sf = d.get("metadata", {}).get("supporting_facts", {})
    titles = sf.get("title", []) or []
    sent_ids = sf.get("sent_id", []) or []
    ctx = d.get("metadata", {}).get("context", {})
    ctx_titles = ctx.get("title", []) or []
    ctx_sents = ctx.get("sentences", ctx.get("text", [])) or []
    title2sents = {t: s for t, s in zip(ctx_titles, ctx_sents)}
    sup_sents = []
    for t, sid in zip(titles, sent_ids):
        sents = title2sents.get(t)
        if isinstance(sents, list) and 0 <= sid < len(sents):
            sup_sents.append(sents[sid])
    # gold_titles 去重保序
    seen = set()
    uniq_titles = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            uniq_titles.append(t)
    return uniq_titles, sup_sents


def load_corpus_titles(corpus_path):
    """扫 corpus 建归一化 title 集合（contents 首行为 title）。"""
    titles = set()
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = item.get("contents", "") or ""
            first = raw.split("\n", 1)[0].strip().strip('"')
            titles.add(norm_title(first))
    return titles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(PROJ_ROOT / "data/eval/hotpotqa/dev.jsonl"))
    ap.add_argument("--output", default=str(PROJ_ROOT / "data/grpo/hotpotqa_train.jsonl"))
    ap.add_argument("--corpus", default=None,
                    help="可选，传入则做 gold 全不可达预过滤")
    args = ap.parse_args()

    corpus_titles = None
    if args.corpus:
        print(f"[build] scanning corpus titles ({args.corpus}) ...")
        corpus_titles = load_corpus_titles(args.corpus)
        print(f"[build] corpus titles: {len(corpus_titles)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    n_in = n_out = n_skip_unreach = n_skip_empty = 0
    with open(args.gold) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n_in += 1
            question = d.get("question", "")
            golden = d.get("golden_answers", []) or []
            gold_titles, gold_sup_sents = extract_gold(d)

            if not gold_titles:
                n_skip_empty += 1
                continue

            # 预过滤：gold 全部不可达
            if corpus_titles is not None:
                reachable = any(norm_title(t) in corpus_titles for t in gold_titles)
                if not reachable:
                    n_skip_unreach += 1
                    continue

            row = {
                "messages": [
                    {"role": "system", "content": REASONING_SYSTEM},
                    {"role": "user", "content": f"Question: {question}"},
                ],
                "golden_answers": golden,
                "gold_titles": gold_titles,
                "gold_sup_sents": gold_sup_sents,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[build] in={n_in} out={n_out} "
          f"skip_empty_gold={n_skip_empty} skip_unreachable={n_skip_unreach}")
    print(f"[build] -> {args.output}")


if __name__ == "__main__":
    main()
