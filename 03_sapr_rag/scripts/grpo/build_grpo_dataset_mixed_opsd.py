#!/usr/bin/env python3
"""HotpotQA + 2Wiki train 混合 → ms-swift GRPO 训练 jsonl。

每行输出（与 build_grpo_dataset.py 兼容）：
  - messages: [{system: REASONING_SYSTEM}, {user: "Question: {q}"}]
  - golden_answers / gold_titles / gold_sup_sents：透传列，供 reward 用
  - source: "hotpotqa" / "2wiki"  ← 新增，便于训练分析
  - 可选 teacher_prompt 及审计元数据：仅 --teacher_prompt_mode gold 时输出

设计：
  1. 所有源都需要 supporting_facts（否则 SaprRelevanceORM 算不了）→ MuSiQue 不参与 GRPO 训练
  2. cap 总数 = MAX_TOTAL（默认 7321，与单 HotpotQA SFT 子集大小对齐，跑得完 4 天）
  3. 按比例采样：[hotpotqa, 2wiki]，默认 [0.5, 0.5]，保证两源平衡
  4. 可选 --corpus 做 gold 全不可达预过滤（剔除 corpus 里完全找不到 gold title 的题）
  5. gold teacher prompt 在采样后构建，避免为未入选样本做 tokenizer 编码

用法：
  python build_grpo_dataset_mixed.py \
    --output data/grpo/hotpotqa_2wiki_train.jsonl \
    --max_total 7321 \
    --corpus data/corpus/wiki18_extended.jsonl

  python build_grpo_dataset_mixed.py \
    --output data/grpo/hotpotqa_2wiki_train_opsd.jsonl \
    --teacher_prompt_mode gold \
    --teacher_prompt_max_tokens 1536

输出会自动 shuffle，避免训练时按数据集顺序聚集（影响 loss 曲线观察）。
"""
import argparse
import json
import random
import re
from pathlib import Path

import os

PROJ_ROOT = Path(os.environ.get("SAPR_RAG_ROOT", Path(__file__).resolve().parents[3]))
DEFAULT_TOKENIZER = PROJ_ROOT / "03_sapr_rag/models/Qwen2.5-7B-Instruct"
TEACHER_PROMPT_VERSION = "sapr-gold-v1"
TEACHER_PROMPT_SOURCE = "gold_supporting_facts"

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
    """提取按 unique title 对齐的 gold evidence。

    同一 title 的多条 supporting sentence 用换行连接，保留一项 title 对应
    一项 sentence evidence 的不变量。
    """
    evidence, _ = extract_teacher_evidence(d)
    return (
        [item["title"] for item in evidence],
        ["\n".join(item["sentences"]) for item in evidence],
    )


def extract_teacher_evidence(d):
    """按 supporting facts 原顺序聚合 title 与去重后的句子。"""
    sf = (d.get("metadata") or {}).get("supporting_facts", {}) or {}
    titles = sf.get("title", []) or []
    sent_ids = sf.get("sent_id", []) or []
    ctx = (d.get("metadata") or {}).get("context", {}) or {}
    ctx_titles = ctx.get("title", []) or []
    ctx_sents = ctx.get("sentences", ctx.get("text", ctx.get("content", []))) or []
    title2sents = {norm_title(str(t)): s for t, s in zip(ctx_titles, ctx_sents)}

    evidence = []
    title_to_item = {}
    missing_sentence = False
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

        sentence = ""
        sentences = title2sents.get(title_key)
        if isinstance(sentences, list) and isinstance(sent_id, int) and 0 <= sent_id < len(sentences):
            sentence = str(sentences[sent_id]).strip()
        if sentence:
            sentence_key = re.sub(r"\s+", " ", sentence).lower()
            existing = {re.sub(r"\s+", " ", s).lower() for s in item["sentences"]}
            if sentence_key not in existing:
                item["sentences"].append(sentence)
        else:
            missing_sentence = True

    return evidence, missing_sentence


class PromptTokenCounter:
    """优先使用 Qwen tokenizer；加载失败时按 UTF-8 字节保守计数。"""

    def __init__(self, tokenizer_path):
        self.tokenizer = None
        self.name = "utf8_bytes_conservative"
        self.error = None
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                local_files_only=True,
                trust_remote_code=True,
            )
            self.name = str(tokenizer_path)
        except Exception as exc:
            self.error = str(exc)

    def count(self, text):
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # 任意 tokenizer 的 token 数不会超过 UTF-8 byte 数；该 fallback 会过度截断但不会超预算。
        return len(text.encode("utf-8"))


def _as_text_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    result = []
    seen = set()
    for item in values:
        text = str(item).strip()
        key = re.sub(r"\s+", " ", text).lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _format_evidence_block(index, item):
    lines = [f"{index}. Title: {item['title']}"]
    if item["sentences"]:
        lines.append("   Supporting facts:")
        lines.extend(f"   - {sentence}" for sentence in item["sentences"])
    return "\n".join(lines)


def build_gold_teacher_prompt(row, token_counter, max_tokens):
    """构建保留 question/answer、按 gold fact 顺序填充的 privileged prompt。"""
    question = row.pop("_teacher_question")
    evidence = row.pop("_teacher_evidence")
    missing_sentence = row.pop("_teacher_missing_sentence")
    answers = _as_text_list(row.get("golden_answers"))
    answer_text = " | ".join(answers) if answers else "(not provided)"

    prefix = (
        f"Question: {question}\n\n"
        "Privileged gold evidence follows. Use it to plan concise retrieval queries and "
        "derive the answer; do not copy this instruction into the response.\n"
        "Gold evidence:\n"
    )
    suffix = (
        f"\n\nGold answer(s): {answer_text}\n\n"
        "Reason step by step over the same response protocol as the student. "
        "When more retrieval is needed, end with <query>query</query>. "
        "When the answer is supported, end with <answer>answer</answer>."
    )
    base_prompt = prefix + suffix
    base_tokens = token_counter.count(base_prompt)
    if base_tokens > max_tokens:
        raise ValueError(
            "teacher prompt budget cannot retain question and gold answer: "
            f"required={base_tokens}, budget={max_tokens}, question={question[:80]!r}"
        )

    blocks = []
    truncated = False
    for index, item in enumerate(evidence, 1):
        block = _format_evidence_block(index, item)
        candidate = prefix + "\n".join(blocks + [block]) + suffix
        if token_counter.count(candidate) > max_tokens:
            truncated = True
            break
        blocks.append(block)

    if len(blocks) < len(evidence):
        truncated = True
    prompt = prefix + "\n".join(blocks) + suffix
    prompt_tokens = token_counter.count(prompt)
    assert prompt_tokens <= max_tokens, (prompt_tokens, max_tokens)

    fallback = missing_sentence or not evidence or any(not item["sentences"] for item in evidence)
    row.update({
        "teacher_prompt": prompt,
        "teacher_prompt_version": TEACHER_PROMPT_VERSION,
        "teacher_prompt_source": TEACHER_PROMPT_SOURCE,
        "teacher_prompt_fallback": fallback,
        "teacher_prompt_truncated": truncated,
        "teacher_prompt_tokens": prompt_tokens,
        "teacher_prompt_chars": len(prompt),
        "teacher_prompt_tokenizer": token_counter.name,
    })
    return fallback, truncated


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


def load_source(name, path, corpus_titles, teacher_prompt_mode):
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
            row = {
                "messages": [
                    {"role": "system", "content": REASONING_SYSTEM},
                    {"role": "user", "content": f"Question: {question}"},
                ],
                "golden_answers": golden,
                "gold_titles": gold_titles,
                "gold_sup_sents": gold_sup_sents,
                "source": name,
            }
            if teacher_prompt_mode == "gold":
                evidence, missing_sentence = extract_teacher_evidence(d)
                row.update({
                    "_teacher_question": question,
                    "_teacher_evidence": evidence,
                    "_teacher_missing_sentence": missing_sentence,
                })
            rows.append(row)
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
    ap.add_argument("--teacher_prompt_mode", choices=["none", "gold"], default="none",
                    help="none=保持原始 GRPO 数据；gold=增加 OPSD privileged teacher_prompt")
    ap.add_argument("--teacher_prompt_max_tokens", type=int, default=1536,
                    help="teacher_prompt 内容 token 上限（默认 1536）")
    ap.add_argument("--teacher_tokenizer", default=str(DEFAULT_TOKENIZER),
                    help="teacher prompt 计数使用的本地 tokenizer；失败时按 UTF-8 字节保守计数")
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    assert len(ratios) == 2 and abs(sum(ratios) - 1.0) < 1e-6, f"--ratios 必须 2 个数且和为 1: {ratios}"
    if args.teacher_prompt_max_tokens <= 0:
        raise ValueError("--teacher_prompt_max_tokens 必须为正整数")

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
        all_rows[name] = load_source(name, p, corpus_titles, args.teacher_prompt_mode)

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

    if args.teacher_prompt_mode == "gold":
        token_counter = PromptTokenCounter(args.teacher_tokenizer)
        if token_counter.error:
            print("[build] WARNING: Qwen tokenizer 加载失败，使用 UTF-8 字节保守计数: "
                  f"{token_counter.error}")
        fallback_count = truncated_count = 0
        for row in out_rows:
            fallback, truncated = build_gold_teacher_prompt(
                row, token_counter, args.teacher_prompt_max_tokens)
            fallback_count += int(fallback)
            truncated_count += int(truncated)
        token_lengths = [row["teacher_prompt_tokens"] for row in out_rows]
        print(f"[build] teacher_prompt mode=gold version={TEACHER_PROMPT_VERSION} "
              f"tokenizer={token_counter.name}")
        print(f"[build] teacher_prompt coverage={len(out_rows)}/{len(out_rows)} "
              f"fallback={fallback_count} truncated={truncated_count} "
              f"tokens_min={min(token_lengths, default=0)} "
              f"tokens_max={max(token_lengths, default=0)} "
              f"tokens_avg={sum(token_lengths) / len(token_lengths) if token_lengths else 0:.1f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fout:
        for row in out_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build] -> {args.output}  ({os.path.getsize(args.output)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
