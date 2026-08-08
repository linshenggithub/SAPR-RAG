#!/usr/bin/env python3
"""多跳 QA 天花板诊断实验。

用 DeepSeek API 在三种设置下评测：
  1. closed_book  闭卷：只给问题，测模型参数知识上限
  2. oracle       开卷+完整上下文：给数据集自带的 distractor 全部段落（distractor 设置上限）
                  - hotpotqa/2wikimultihopqa: metadata.context 中的全部段落
                  - musique: question_decomposition 中的所有 support_paragraph
  3. retrieval    开卷+我们的检索：给 FAISS 检索 top-k 文档（需指定索引路径）

输出每个数据集的 EM / F1 / cover_em，以及三层对比分析。

用法：
  export DEEPSEEK_API_KEY=sk-xxx
  # 闭卷测试，先跑 100 条试水
  python ceiling_diagnostic.py --dataset 2wikimultihopqa --setting closed_book --max_samples 100

  # Oracle 测试，全量
  python ceiling_diagnostic.py --dataset hotpotqa --setting oracle

  # 检索设置（需要 corpus + index）
  python ceiling_diagnostic.py --dataset musique --setting retrieval \
      --corpus_path data/corpus/wiki18_extended.jsonl \
      --index_path data/index/bge_extended_Flat.index \
      --top_k 5
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-pro"
TIMEOUT_S = 120
MAX_RETRIES = 5

SYSTEM_PROMPT = """You are a helpful QA assistant. Answer the question based on the provided context (if any).
Rules:
- If context is provided, answer ONLY based on the context. Do not use external knowledge.
- Keep your answer as concise as possible — usually a short phrase or entity name.
- If you cannot find the answer in the context, say "I don't know".
- Do not include any explanation, just the answer."""


def call_deepseek(api_key, model, user_prompt, temperature=0.0, max_tokens=2048):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    backoff = 2
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=TIMEOUT_S)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            else:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(backoff)
        backoff *= 2
    return f"[ERROR] {last_err}"


def cache_key(model, setting, dataset, qid, question, extra=""):
    s = f"{model}|{setting}|{dataset}|{qid}|{question}|{extra}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


# ─────────── 数据集加载 & gold context 提取 ───────────

def load_dataset(dataset_name, data_dir="data/eval"):
    path = Path(data_dir) / dataset_name / "dev.jsonl"
    if not path.exists():
        sys.exit(f"[FATAL] 找不到数据集: {path}")
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_oracle_context(sample, dataset_name):
    """提取 oracle 上下文。

    - hotpotqa / 2wikimultihopqa：用 metadata.context 中的全部段落（distractor 设置）
    - musique：用 question_decomposition 中每跳的 support_paragraph
    """
    meta = sample.get("metadata", {})

    if dataset_name in ("hotpotqa", "2wikimultihopqa"):
        ctx = meta.get("context", {})
        titles = ctx.get("title", [])
        if dataset_name == "hotpotqa":
            sentences_list = ctx.get("sentences", [])
        else:
            content_list = ctx.get("content", [])
            sentences_list = [[c] for c in content_list]

        paragraphs = []
        for i, title in enumerate(titles):
            sents = sentences_list[i] if i < len(sentences_list) else []
            if dataset_name == "hotpotqa":
                para_text = " ".join(sents).strip()
            else:
                para_text = sents[0] if sents else ""
            if para_text:
                paragraphs.append(f"[{title}]\n{para_text}")

        return "\n\n".join(paragraphs)

    elif dataset_name == "musique":
        decomp = meta.get("question_decomposition", [])
        paragraphs = []
        seen_titles = set()
        for hop in decomp:
            sp = hop.get("support_paragraph", {})
            title = sp.get("title", "")
            text = sp.get("paragraph_text", "")
            if title and title not in seen_titles and text:
                seen_titles.add(title)
                paragraphs.append(f"[{title}]\n{text}")
        return "\n\n".join(paragraphs)

    else:
        return ""


# ─────────── 检索器（用于 retrieval 设置）───────────

class FAISSRetriever:
    def __init__(self, corpus_path, index_path, bge_model_name="BAAI/bge-base-en-v1.5"):
        import faiss
        from sentence_transformers import SentenceTransformer

        print(f"[retriever] loading corpus from {corpus_path} ...", flush=True)
        self.corpus = []
        with open(corpus_path) as f:
            for line in f:
                try:
                    self.corpus.append(json.loads(line))
                except Exception:
                    continue
        print(f"[retriever] {len(self.corpus)} docs loaded", flush=True)

        print(f"[retriever] loading FAISS index from {index_path} ...", flush=True)
        self.index = faiss.read_index(index_path)
        print(f"[retriever] index loaded, dim={self.index.d}, ntotal={self.index.ntotal}", flush=True)

        print(f"[retriever] loading encoder {bge_model_name} ...", flush=True)
        self.encoder = SentenceTransformer(bge_model_name)
        print("[retriever] ready", flush=True)

    def search(self, query, top_k=5):
        import numpy as np
        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.corpus):
                doc = self.corpus[idx]
                results.append({
                    "score": float(scores[0][i]),
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                })
        return results


# ─────────── prompt 构造 ───────────

def build_prompt(setting, question, context=None):
    if setting == "closed_book":
        return f"Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
    else:
        return (
            f"Answer the question based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )


# ─────────── 评估指标（复用 score.py 逻辑）───────────

import re
import string
from collections import Counter


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
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


# ─────────── 主流程 ───────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["hotpotqa", "2wikimultihopqa", "musique"])
    p.add_argument("--setting", required=True, choices=["closed_book", "oracle", "retrieval"])
    p.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek 模型名")
    p.add_argument("--concurrency", type=int, default=32, help="API 并发数")
    p.add_argument("--max_samples", type=int, default=0, help="0=全跑，>0=只跑前 N 条")
    p.add_argument("--output_dir", default="data/eval_results/ceiling")
    p.add_argument("--cache_dir", default="data/ceiling_cache")
    p.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")

    p.add_argument("--corpus_path", default=None, help="retrieval 设置需要")
    p.add_argument("--index_path", default=None, help="retrieval 设置需要")
    p.add_argument("--top_k", type=int, default=5, help="retrieval top-k")
    p.add_argument("--bge_model", default="BAAI/bge-base-en-v1.5")

    p.add_argument("--data_dir", default="data/eval")
    args = p.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.exit(f"[FATAL] 环境变量 {args.api_key_env} 未设置")

    if args.setting == "retrieval":
        if not args.corpus_path or not args.index_path:
            sys.exit("[FATAL] retrieval 设置需要 --corpus_path 和 --index_path")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{args.model}_{args.dataset}_{args.setting}_cache.jsonl"

    out_file = out_dir / f"{args.model}_{args.dataset}_{args.setting}_results.jsonl"
    metrics_file = out_dir / f"{args.model}_{args.dataset}_{args.setting}_metrics.json"

    print(f"=== Ceiling Diagnostic ===", flush=True)
    print(f"  dataset:  {args.dataset}", flush=True)
    print(f"  setting:  {args.setting}", flush=True)
    print(f"  model:    {args.model}", flush=True)

    samples = load_dataset(args.dataset, args.data_dir)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"  samples:  {len(samples)}", flush=True)

    retriever = None
    if args.setting == "retrieval":
        retriever = FAISSRetriever(args.corpus_path, args.index_path, args.bge_model)

    cache = {}
    if cache_file.exists():
        with open(cache_file) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    cache[obj["key"]] = obj["prediction"]
                except Exception:
                    continue
    print(f"  cache:    {len(cache)} hits", flush=True)

    cache_fp = open(cache_file, "a", buffering=1)

    # 构造任务
    tasks = []
    for i, sample in enumerate(samples):
        qid = sample.get("id", str(i))
        question = sample.get("question", "")
        gold = sample.get("golden_answers", [])
        if isinstance(gold, str):
            gold = [gold]

        context = None
        extra_for_cache = ""

        if args.setting == "oracle":
            context = extract_oracle_context(sample, args.dataset)
            extra_for_cache = context[:200]
        elif args.setting == "retrieval":
            docs = retriever.search(question, top_k=args.top_k)
            context_parts = []
            for d in docs:
                title = d.get("title", "")
                content = d.get("content", "")
                context_parts.append(f"[{title}]\n{content}")
            context = "\n\n".join(context_parts)
            extra_for_cache = f"top{args.top_k}_" + context[:200]

        key = cache_key(args.model, args.setting, args.dataset, qid, question, extra_for_cache)
        tasks.append({"idx": i, "qid": qid, "question": question, "gold": gold,
                      "context": context, "key": key})

    n_cached_initial = sum(1 for t in tasks if t["key"] in cache)
    n_to_call = len(tasks) - n_cached_initial
    print(f"  concurrency: {args.concurrency}", flush=True)
    print(f"  to_call:    {n_to_call} (cached {n_cached_initial})", flush=True)

    results = [None] * len(tasks)
    n_api_calls = 0
    n_done = 0
    t0 = time.time()
    lock_progress = [0]

    def worker(t):
        key = t["key"]
        if key in cache:
            return t, cache[key], True
        prompt = build_prompt(args.setting, t["question"], t["context"])
        prediction = call_deepseek(api_key, args.model, prompt)
        return t, prediction, False

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            t, prediction, was_cached = fut.result()
            if not was_cached:
                cache[t["key"]] = prediction
                cache_fp.write(json.dumps({"key": t["key"], "prediction": prediction}, ensure_ascii=False) + "\n")
                n_api_calls += 1

            em = em_score(prediction, t["gold"])
            cover_em = cover_em_score(prediction, t["gold"])
            f1 = f1_score(prediction, t["gold"])

            results[t["idx"]] = {
                "id": t["qid"],
                "question": t["question"],
                "gold": t["gold"],
                "prediction": prediction,
                "em": em,
                "cover_em": cover_em,
                "f1": f1,
            }

            lock_progress[0] += 1
            done = lock_progress[0]
            if done % 100 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = n_api_calls / elapsed if elapsed > 0 and n_api_calls > 0 else 0
                avg_em = sum(r["em"] for r in results if r is not None) / done
                avg_cover = sum(r["cover_em"] for r in results if r is not None) / done
                avg_f1 = sum(r["f1"] for r in results if r is not None) / done
                print(f"  [{done}/{len(tasks)}] api={n_api_calls} rate={rate:.1f}/s  "
                      f"EM={avg_em:.3f} CoverEM={avg_cover:.3f} F1={avg_f1:.3f}", flush=True)

    cache_fp.close()

    n = len(results)
    metrics = {
        "dataset": args.dataset,
        "setting": args.setting,
        "model": args.model,
        "n_total": n,
        "n_api_calls": n_api_calls,
        "em": round(sum(r["em"] for r in results) / n, 4),
        "cover_em": round(sum(r["cover_em"] for r in results) / n, 4),
        "f1": round(sum(r["f1"] for r in results) / n, 4),
    }

    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n=== DONE ===", flush=True)
    print(f"  results -> {out_file}", flush=True)
    print(f"  metrics -> {metrics_file}", flush=True)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
