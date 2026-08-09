#!/usr/bin/env python3
"""HotpotQA + 2Wiki train 混合 → ms-swift GRPO 训练 jsonl。

每行输出（与 build_grpo_dataset.py 兼容）：
  - messages: [{system: REASONING_SYSTEM}, {user: "Question: {q}"}]
  - golden_answers / gold_titles / gold_sup_sents：透传列，供 reward 用
  - source: "hotpotqa" / "2wiki"  ← 新增，便于训练分析

设计：
  1. 所有源都需要 supporting_facts（否则 SaprRelevanceORM 算不了）→ MuSiQue 不参与 GRPO 训练
  2. cap 总数 = MAX_TOTAL（默认 7321，与单 HotpotQA SFT 子集大小对齐，跑得完 4 天）
  3. 按比例采样：[hotpotqa, 2wiki]，默认 [0.5, 0.5]，保证两源平衡
  4. 可选 --corpus 做 gold 全不可达预过滤（剔除 corpus 里完全找不到 gold title 的题）

用法：
  python build_grpo_dataset_mixed.py \
    --output data/grpo/hotpotqa_2wiki_train.jsonl \
    --max_total 7321 \
    --corpus data/corpus/wiki18_extended.jsonl

输出会自动 shuffle，避免训练时按数据集顺序聚集（影响 loss 曲线观察）。
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

PROJ_ROOT = Path(os.environ.get("SAPR_RAG_ROOT", Path(__file__).resolve().parents[3]))

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
    "* Retrieval Discipline: The retrieval system is deterministic; do not repeat a "
    "previous query because the same query returns the same documents. If a query did "
    "not provide enough evidence, ask a new query targeting a different entity, "
    "relation, or missing fact; otherwise answer with the available evidence.\n"
    "* Respond Format:\n"
    "If your knowledge is sufficient to answer the question, conclude with:\n"
    '"So the answer is <answer>answer</answer>"\n'
    "If retrieval is necessary to provide a complete answer, conclude with:\n"
    '"So the next query is <query>query</query>"\n'
)

SOURCES = {
    "hotpotqa": PROJ_ROOT / "data/raw/hotpotqa/train.jsonl",
    "2wiki":    PROJ_ROOT / "data/raw/2wikimultihopqa/train.jsonl",
}


def norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def extract_gold(d):
    """提取按 unique title 对齐的 gold evidence。"""
    sf = (d.get("metadata") or {}).get("supporting_facts", {}) or {}
    titles = sf.get("title", []) or []
    sent_ids = sf.get("sent_id", []) or []
    ctx = (d.get("metadata") or {}).get("context", {}) or {}
    ctx_titles = ctx.get("title", []) or []
    ctx_sents = ctx.get("sentences", ctx.get("text", ctx.get("content", []))) or []
    title2sents = {
        norm_title(str(title)): sentences
        for title, sentences in zip(ctx_titles, ctx_sents)
    }

    evidence = []
    title_to_item = {}
    for title, sent_id in zip(titles, sent_ids):
        title = str(title).strip()
        if not title:
            continue
        title_key = norm_title(title)
        item = title_to_item.get(title_key)
        if item is None:
            item = {"title": title, "sentences": []}
            title_to_item[title_key] = item
            evidence.append(item)

        sentences = title2sents.get(title_key)
        if isinstance(sentences, list) and isinstance(sent_id, int) and 0 <= sent_id < len(sentences):
            sentence = str(sentences[sent_id]).strip()
            if sentence and sentence not in item["sentences"]:
                item["sentences"].append(sentence)

    return (
        [item["title"] for item in evidence],
        ["\n".join(item["sentences"]) for item in evidence],
    )


def load_corpus_titles(corpus_path):
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


def load_source(name, path, corpus_titles):
    """读取一个源的训练集，做基本过滤（gold 非空 + 可达），返回 list of dict。"""
    rows = []
    n_in = n_skip_empty = n_skip_unreach = 0
    with open(path) as f:
        for line in f:
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
            if corpus_titles is not None:
                reachable = any(norm_title(t) in corpus_titles for t in gold_titles)
                if not reachable:
                    n_skip_unreach += 1
                    continue
            rows.append({
                "messages": [
                    {"role": "system", "content": REASONING_SYSTEM},
                    {"role": "user", "content": f"Question: {question}"},
                ],
                "golden_answers": golden,
                "gold_titles": gold_titles,
                "gold_sup_sents": gold_sup_sents,
                "source": name,
            })
    print(f"  [{name}] in={n_in} kept={len(rows)} "
          f"skip_empty_gold={n_skip_empty} skip_unreachable={n_skip_unreach}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(PROJ_ROOT / "data/grpo/hotpotqa_2wiki_train.jsonl"))
    ap.add_argument("--max_total", type=int, default=7321,
                    help="总样本数上限（与单源 SFT 子集对齐）")
    ap.add_argument("--ratios", default="0.5,0.5",
                    help="逗号分隔的采样比例 [hotpotqa,2wiki]")
    ap.add_argument("--corpus", default=None,
                    help="可选：传入则做 gold 全不可达预过滤")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    assert len(ratios) == 2 and abs(sum(ratios) - 1.0) < 1e-6, f"--ratios 必须 2 个数且和为 1: {ratios}"

    # 检查源文件
    for name, p in SOURCES.items():
        if not p.exists():
            raise FileNotFoundError(f"源文件不存在: {p}\n请先准备好 {name} 训练集。")

    # 预加载 corpus 可达性
    corpus_titles = None
    if args.corpus:
        print(f"[build] scanning corpus titles ({args.corpus}) ...")
        corpus_titles = load_corpus_titles(args.corpus)
        print(f"[build] corpus titles: {len(corpus_titles)}")

    # 各源加载
    all_rows = {}
    for name, p in SOURCES.items():
        print(f"[build] loading {name} from {p} ...")
        all_rows[name] = load_source(name, p, corpus_titles)

    # 按比例 cap 采样
    rng = random.Random(args.seed)
    out_rows = []
    for name, ratio in zip(["hotpotqa", "2wiki"], ratios):
        target = int(args.max_total * ratio)
        rows = all_rows[name]
        if len(rows) <= target:
            print(f"  [{name}] kept {len(rows)} (less than target {target})")
            out_rows.extend(rows)
        else:
            sampled = rng.sample(rows, target)
            print(f"  [{name}] sampled {target} from {len(rows)}")
            out_rows.extend(sampled)

    # shuffle 防止聚集
    rng.shuffle(out_rows)
    print(f"[build] total kept: {len(out_rows)}")
    print(f"[build] composition: " +
          ", ".join(f"{n}={sum(1 for r in out_rows if r['source'] == n)}"
                    for n in ["hotpotqa", "2wiki"]))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fout:
        for row in out_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build] -> {args.output}  ({os.path.getsize(args.output)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
