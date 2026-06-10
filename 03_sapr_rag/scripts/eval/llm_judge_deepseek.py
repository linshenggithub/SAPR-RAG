#!/usr/bin/env python3
"""DeepSeek-V3 LLM-as-judge 评估器。

读 merged.jsonl 中每条 (question, gold, answer)，用 DeepSeek 二分类判定
模型答案是否事实等价于 gold answers，得到 llm_acc_deepseek 指标。

特性：
- HTTP cache（按 (question, gold, answer) 三元组 sha256），断点续算
- 并发 N 并行 API 调用
- 二分类输出：correct / incorrect / unjudgeable
- 输出 metrics_llm.json + per-question judgments.jsonl

用法：
    export DEEPSEEK_API_KEY=sk-xxx
    python llm_judge_deepseek.py --merged path/to/merged.jsonl
    # 默认输出到 merged 同目录的 metrics_llm.json + judgments.jsonl
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
DEFAULT_MODEL = "deepseek-chat"  # DeepSeek-V3
TIMEOUT_S = 60
MAX_RETRIES = 4

JUDGE_SYSTEM = """You are a strict but fair fact-checking judge for QA tasks.
Given a question, the ground-truth answer(s), and a model's predicted answer, decide whether the prediction is **factually equivalent** to any gold answer.

Equivalence rules:
- Different surface forms with same fact = correct (e.g., "WWII" vs "World War II"; "Stephen" vs "Steve"; "USA" vs "United States").
- Subset / partial answers are correct ONLY if they uniquely identify the gold (e.g., "Wozniak" alone may be ambiguous if multiple Wozniaks exist; usually treat as correct unless clearly ambiguous).
- Predictions that contain extra wrong facts beyond the gold are still correct if the gold is clearly stated.
- Empty / "I don't know" / no-answer = incorrect.
- Use your world knowledge to allow obvious paraphrases.

Reply with ONE WORD ONLY: `correct` or `incorrect`. No explanation."""


def sha256_key(question: str, gold, answer: str) -> str:
    g = json.dumps(gold, ensure_ascii=False, sort_keys=True) if isinstance(gold, list) else str(gold)
    s = f"{question}\n--GOLD--\n{g}\n--PRED--\n{answer}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


def make_user_prompt(question: str, gold, answer: str) -> str:
    if isinstance(gold, list):
        gold_str = " | ".join(str(g) for g in gold)
    else:
        gold_str = str(gold)
    return (
        f"Question: {question}\n\n"
        f"Gold answer(s): {gold_str}\n\n"
        f"Model prediction: {answer}\n\n"
        f"Is the model prediction factually equivalent to any gold answer? Reply `correct` or `incorrect`."
    )


def call_deepseek(api_key: str, model: str, user_prompt: str, retries: int = MAX_RETRIES) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 5,
    }
    backoff = 2
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=TIMEOUT_S)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip().lower()
                # 容错解析
                if "correct" in content and "incorrect" not in content:
                    return "correct"
                if "incorrect" in content:
                    return "incorrect"
                return "unjudgeable"
            else:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(backoff)
        backoff *= 2
    return f"ERROR:{last_err}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged", required=True, help="merged.jsonl 路径（含 question/gold/answer）")
    p.add_argument("--output_dir", default=None, help="输出目录，默认 merged 同目录")
    p.add_argument("--cache_dir", default=None, help="cache 目录，默认 SAPR-RAG/data/llm_judge_cache")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max_samples", type=int, default=0, help="0=全跑，>0=只跑前 N 条做 sanity")
    p.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.exit(f"[FATAL] 环境变量 {args.api_key_env} 未设置")

    merged_path = Path(args.merged)
    if not merged_path.exists():
        sys.exit(f"[FATAL] 找不到 {merged_path}")

    out_dir = Path(args.output_dir) if args.output_dir else merged_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        Path("/mlx_devbox/users/mayi.summer/playground/SAPR-RAG/data/llm_judge_cache")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "deepseek_judge_cache.jsonl"

    judgments_path = out_dir / "judgments.jsonl"
    metrics_path = out_dir / "metrics_llm.json"

    # 加载 cache
    cache = {}
    if cache_file.exists():
        with open(cache_file) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    cache[obj["key"]] = obj["judgment"]
                except Exception:
                    continue
    print(f"[cache] loaded {len(cache)} cached judgments from {cache_file}", flush=True)

    # 加载 merged.jsonl
    records = []
    with open(merged_path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    print(f"[input] loaded {len(records)} records from {merged_path}", flush=True)

    if args.max_samples > 0:
        records = records[: args.max_samples]
        print(f"[input] sampled {len(records)} for sanity", flush=True)

    # 构造任务
    tasks = []
    for r in records:
        q = r.get("question", "")
        g = r.get("gold", []) or r.get("golden_answers", [])
        a = r.get("answer") or ""
        # 空答案直接判 incorrect，不调用 API
        key = sha256_key(q, g, a)
        if not str(a).strip():
            cache[key] = "incorrect"
        tasks.append({"key": key, "q": q, "g": g, "a": a, "id": r.get("id", "")})

    n_cached_initial = sum(1 for t in tasks if t["key"] in cache)
    n_to_call = len(tasks) - n_cached_initial
    print(f"[plan] total={len(tasks)} cached={n_cached_initial} to_call={n_to_call}", flush=True)

    # 并发调用
    cache_fp = open(cache_file, "a", buffering=1)  # line-buffered
    n_done_now = 0
    t0 = time.time()
    lock_progress = [0]

    def worker(t):
        if t["key"] in cache:
            return t, cache[t["key"]], True  # cached
        prompt = make_user_prompt(t["q"], t["g"], t["a"])
        verdict = call_deepseek(api_key, args.model, prompt)
        return t, verdict, False

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            t, verdict, was_cached = fut.result()
            if not was_cached:
                cache[t["key"]] = verdict
                cache_fp.write(json.dumps({"key": t["key"], "judgment": verdict}, ensure_ascii=False) + "\n")
                n_done_now += 1
            lock_progress[0] += 1
            done = lock_progress[0]
            if done % 200 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = n_done_now / elapsed if elapsed > 0 and n_done_now > 0 else 0
                print(f"  progress {done}/{len(tasks)}  api_calls={n_done_now}  rate={rate:.1f}/s", flush=True)
    cache_fp.close()

    # 写 judgments.jsonl + metrics
    n_correct = 0
    n_incorrect = 0
    n_unjudgeable = 0
    n_error = 0
    with open(judgments_path, "w") as fout:
        for t in tasks:
            v = cache.get(t["key"], "ERROR")
            row = {"id": t["id"], "question": t["q"], "gold": t["g"], "answer": t["a"], "judgment": v}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if v == "correct":
                n_correct += 1
            elif v == "incorrect":
                n_incorrect += 1
            elif v == "unjudgeable":
                n_unjudgeable += 1
            else:
                n_error += 1

    n = len(tasks)
    n_judged = n_correct + n_incorrect
    metrics = {
        "n_total": n,
        "n_judged": n_judged,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_unjudgeable": n_unjudgeable,
        "n_error": n_error,
        "llm_acc_deepseek": round(n_correct / n_judged, 4) if n_judged > 0 else 0.0,
        "llm_acc_full": round(n_correct / n, 4) if n > 0 else 0.0,  # 把无法判分的都算错
        "model": args.model,
        "merged": str(merged_path),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===", flush=True)
    print(f"  judgments -> {judgments_path}")
    print(f"  metrics   -> {metrics_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
