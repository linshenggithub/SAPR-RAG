#!/usr/bin/env python3
"""DeepSeek Agentic Zeroshot RAG baseline.

流程与 SAPR-RAG agentic 推理保持同构：
  reasoning: DeepSeek 输出 <query> 或 <answer>
  retrieve : 调已有 retrieval daemon
  evidence : DeepSeek 从检索文档抽 <evidence>
  loop     : 最多 max_turns 轮

默认用于小样本成本探针，不直接全量：
  python 03_sapr_rag/scripts/eval/agentic_deepseek_zeroshot.py \
    --dataset musique --limit 50 \
    --retrieval_url http://127.0.0.1:8100 \
    --env_path 03_sapr_rag/.env \
    --output data/eval_results/deepseek_agentic/musique50.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

import requests


REASONING_SYSTEM_STRICT_XML = (
    "You are an agentic QA controller with access to a retrieval tool.\n"
    "Your job is to answer multi-hop questions by deciding whether to search or answer.\n\n"
    "Output format is STRICT. You must output exactly ONE XML block and nothing else:\n"
    "- If more information is needed, output: <query>short search query</query>\n"
    "- If the answer is known from the question and previous evidence, output: <answer>short final answer</answer>\n\n"
    "Rules:\n"
    "- Do not explain your reasoning.\n"
    "- Do not write text before or after the XML block.\n"
    "- Do not output both query and answer in the same turn.\n"
    "- Prefer retrieval for entity-specific facts, dates, places, relations, or multi-hop links unless the evidence is already in Previous Thoughts.\n"
    "- Keep queries and answers concise.\n\n"
    "Examples:\n"
    "Question: Who directed the film that won the 1999 Academy Award for Best Picture?\n"
    "<query>1999 Academy Award for Best Picture winner</query>\n\n"
    "Question: Where is Ulrich Walter's employer headquartered?\n"
    "Previous Thoughts: <evidence>Ulrich Walter is employed by the Technical University of Munich.</evidence>\n"
    "<query>Technical University of Munich headquarters</query>\n\n"
    "Question: What city is the Technical University of Munich headquartered in?\n"
    "Previous Thoughts: <evidence>The Technical University of Munich is headquartered in Munich.</evidence>\n"
    "<answer>Munich</answer>\n"
)

REASONING_SYSTEM_ANTI_REPEAT = (
    "You are an agentic QA controller with access to a deterministic retrieval tool.\n"
    "Your job is to answer multi-hop questions by decomposing them into useful search steps.\n\n"
    "Output format is STRICT. You must output exactly ONE XML block and nothing else:\n"
    "- Search action: <query>short search query</query>\n"
    "- Final action: <answer>short final answer</answer>\n\n"
    "Environment rules:\n"
    "- The retriever is deterministic: the same query will return the same documents.\n"
    "- Never repeat a previous query exactly or semantically.\n"
    "- If a previous query returned <evidence>None</evidence>, that exact query failed. Do not retry it.\n"
    "- After a failed query, generate a simpler one-hop query targeting a different entity or relation.\n"
    "- If retrieved evidence is enough, stop searching and output <answer>...</answer>.\n"
    "- If several searches fail, output the best answer you can infer instead of looping forever.\n\n"
    "Query strategy:\n"
    "- For multi-hop questions, search one hop at a time.\n"
    "- Prefer entity-specific queries such as '<entity> spouse', '<film> distributor', '<company> founder'.\n"
    "- Avoid broad combined queries that ask multiple hops at once.\n\n"
    "Do not explain your reasoning. Do not write text outside the XML block.\n\n"
    "Example after failed search:\n"
    "Question: Who founded the company that distributed the film UHF?\n"
    "Previous Thoughts:\n"
    "So the next query is <query>UHF film distributor company founder</query> Based on the query, the relevant evidence is <evidence>None</evidence>.\n"
    "<query>UHF film distributor</query>\n\n"
    "Example after useful evidence:\n"
    "Question: Who founded the company that distributed the film UHF?\n"
    "Previous Thoughts:\n"
    "So the next query is <query>UHF film distributor</query> Based on the query, the relevant evidence is <evidence>UHF was distributed by Orion Pictures.</evidence>.\n"
    "<query>Orion Pictures founders</query>\n\n"
    "Example final answer:\n"
    "Question: Who founded the company that distributed the film UHF?\n"
    "Previous Thoughts:\n"
    "So the next query is <query>UHF film distributor</query> Based on the query, the relevant evidence is <evidence>UHF was distributed by Orion Pictures.</evidence>.\n"
    "So the next query is <query>Orion Pictures founders</query> Based on the query, the relevant evidence is <evidence>Orion Pictures was founded by Arthur Krim, Robert Benjamin, and Mike Medavoy.</evidence>.\n"
    "<answer>Arthur Krim, Robert Benjamin, and Mike Medavoy</answer>\n"
)

EVIDENCE_SYSTEM = (
    "You are an evidence extraction assistant.\n"
    "Given a query and reference documents, output exactly ONE XML block and nothing else:\n"
    "<evidence>concise evidence text</evidence>\n"
    "If no relevant evidence is found, output exactly: <evidence>None</evidence>\n"
    "Do not explain."
)

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

RE_QUERY = re.compile(r"<query>(.*?)</query>", re.DOTALL)
RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
RE_EVIDENCE = re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["hotpotqa", "2wikimultihopqa", "musique"])
    p.add_argument("--data_dir", default="data/eval")
    p.add_argument("--limit", type=int, default=50, help="0=全量；默认 50 做成本探针")
    p.add_argument("--output", required=True)
    p.add_argument("--metrics", default=None)
    p.add_argument("--env_path", default=None)
    p.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    p.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    p.add_argument("--retrieval_url", required=True)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--max_turns", type=int, default=6)
    p.add_argument("--reason_max_tokens", type=int, default=128)
    p.add_argument("--evidence_max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max_retries", type=int, default=4)
    p.add_argument("--progress_every", type=int, default=10)
    p.add_argument("--prompt_variant", choices=["strict_xml", "anti_repeat"], default="strict_xml")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--input_yuan_per_mtok", type=float, default=1.0)
    p.add_argument("--output_yuan_per_mtok", type=float, default=2.0)
    return p.parse_args()


def load_dotenv(path: Optional[str]) -> None:
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_dataset(dataset: str, data_dir: Path, limit: int) -> List[Dict[str, Any]]:
    path = data_dir / dataset / "dev.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def get_gold(sample: Dict[str, Any]) -> List[str]:
    gold = sample.get("golden_answers", [])
    if isinstance(gold, str):
        return [gold]
    return [str(x) for x in gold]


def render_history(history: Sequence[Dict[str, str]]) -> str:
    parts = []
    for h in history:
        parts.append(
            f"So the next query is <query>{h['query']}</query> "
            f"Based on the query, the relevant evidence is <evidence>{h['evidence']}</evidence>."
        )
    return "\n\n".join(parts)


def build_reasoning_prompt(question: str, history: Sequence[Dict[str, str]], prompt_variant: str) -> List[Dict[str, str]]:
    user = f"Question: {question}"
    if history:
        user += "\nPrevious Thoughts:\n" + render_history(history)
        failed_queries = [
            h["query"] for h in history
            if str(h.get("evidence", "")).strip().lower() in {"none", "<none>"}
        ]
        if failed_queries and prompt_variant == "anti_repeat":
            user += "\n\nFailed queries that must NOT be repeated:\n"
            user += "\n".join(f"- {q}" for q in failed_queries)
    user += "\n\nReturn exactly one XML block: <query>...</query> or <answer>...</answer>."
    system = REASONING_SYSTEM_ANTI_REPEAT if prompt_variant == "anti_repeat" else REASONING_SYSTEM_STRICT_XML
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_evidence_prompt(query: str, docs: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    reference = " ".join(f"{d.get('title', '')}. {d.get('text', d.get('content', ''))}" for d in docs)
    user = f"Question: {query}. Reference: <reference>{reference}</reference>"
    return [{"role": "system", "content": EVIDENCE_SYSTEM}, {"role": "user", "content": user}]


def parse_action(text: str) -> Dict[str, str]:
    m = RE_ANSWER.search(text)
    if m:
        return {"type": "answer", "value": m.group(1).strip()}
    m = RE_QUERY.search(text)
    if m:
        return {"type": "query", "value": m.group(1).strip()}
    return {"type": "unknown", "value": text.strip()}


def parse_evidence(text: str) -> str:
    m = RE_EVIDENCE.search(text)
    return m.group(1).strip() if m else "None"


class Usage:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.calls = 0
        self.failed = 0

    def add(self, usage: Dict[str, int]) -> None:
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)

    def to_dict(self) -> Dict[str, int]:
        return {
            "api_calls": self.calls,
            "api_failed": self.failed,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def call_deepseek(
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float,
    max_retries: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(max_retries):
        try:
            r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                obj = r.json()
                choice = obj.get("choices", [{}])[0]
                msg = choice.get("message", {}) or {}
                usage = obj.get("usage", {}) or {}
                return {
                    "ok": True,
                    "content": msg.get("content") or "",
                    "usage": {
                        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                        "completion_tokens": int(usage.get("completion_tokens") or 0),
                        "total_tokens": int(usage.get("total_tokens") or 0),
                    },
                    "error": None,
                }
            last_error = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
        time.sleep(min(30, 1.5 * (2 ** attempt)))
    return {"ok": False, "content": "", "usage": {}, "error": last_error}


class HTTPRetriever:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        r = self.session.get(f"{self.base_url}/health", timeout=timeout)
        r.raise_for_status()
        print(f"[retriever] {r.text[:300]}", flush=True)

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        r = self.session.post(
            f"{self.base_url}/search_batch",
            json={"queries": [query], "top_k": top_k},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["results"][0]


def run_one(sample: Dict[str, Any], idx: int, args, api_key: str, retriever: HTTPRetriever) -> Dict[str, Any]:
    question = str(sample.get("question", ""))
    qid = sample.get("id", str(idx))
    history: List[Dict[str, str]] = []
    trace: List[Dict[str, Any]] = []
    usage = Usage()

    answer: Optional[str] = None
    error: Optional[str] = None

    for turn in range(args.max_turns):
        reason = call_deepseek(
            api_key, args.model, build_reasoning_prompt(question, history, args.prompt_variant),
            args.reason_max_tokens, args.temperature, args.timeout, args.max_retries,
        )
        if not reason["ok"]:
            usage.failed += 1
            error = f"reason_api_failed: {reason['error']}"
            break
        usage.add(reason["usage"])
        action = parse_action(reason["content"])
        trace.append({"turn": turn, "stage": "reason", "out": reason["content"], "parsed": action})

        if action["type"] == "answer":
            answer = action["value"]
            break
        if action["type"] != "query":
            error = "no_query_or_answer"
            break

        query = action["value"]
        docs = retriever.search(query, args.top_k)
        trace.append({"turn": turn, "stage": "retrieve", "query": query, "docs": docs})

        ev = call_deepseek(
            api_key, args.model, build_evidence_prompt(query, docs),
            args.evidence_max_tokens, args.temperature, args.timeout, args.max_retries,
        )
        if not ev["ok"]:
            usage.failed += 1
            error = f"evidence_api_failed: {ev['error']}"
            break
        usage.add(ev["usage"])
        evidence = parse_evidence(ev["content"])
        trace.append({"turn": turn, "stage": "evidence", "out": ev["content"], "parsed": evidence})
        history.append({"query": query, "evidence": evidence})

    if answer is None and error is None:
        error = "max_turns_exceeded"

    gold = get_gold(sample)
    return {
        "idx": idx,
        "id": qid,
        "dataset": args.dataset,
        "question": question,
        "gold": gold,
        "answer": answer,
        "history": history,
        "trace": trace,
        "error": error,
        **usage.to_dict(),
    }


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


def cover_em_score(pred, golds):
    p_toks = normalize_answer(pred).split()
    for g in golds:
        g_toks = normalize_answer(g).split()
        if not g_toks:
            continue
        for i in range(len(p_toks) - len(g_toks) + 1):
            if p_toks[i:i + len(g_toks)] == g_toks:
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


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_path)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics) if args.metrics else out_path.with_suffix(out_path.suffix + ".metrics.json")

    rows = load_dataset(args.dataset, Path(args.data_dir), args.limit)
    completed: Dict[int, Dict[str, Any]] = {}
    if args.resume and out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    old = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if old.get("idx") is not None:
                    completed[int(old["idx"])] = old
    todo = [(i, s) for i, s in enumerate(rows) if i not in completed]
    print(
        f"[plan] dataset={args.dataset} n={len(rows)} todo={len(todo)} "
        f"completed={len(completed)} model={args.model} max_turns={args.max_turns}",
        flush=True,
    )
    retriever = HTTPRetriever(args.retrieval_url, args.timeout)

    started = time.time()
    results: List[Optional[Dict[str, Any]]] = [completed.get(i) for i in range(len(rows))]
    lock = Lock()
    done = len(completed)

    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8", buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(run_one, s, i, args, api_key, retriever) for i, s in todo]
            for fut in as_completed(futures):
                r = fut.result()
                r["em"] = em_score(r["answer"], r["gold"])
                r["cover_em"] = cover_em_score(r["answer"], r["gold"])
                r["f1"] = f1_score(r["answer"], r["gold"])
                with lock:
                    results[r["idx"]] = r
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    done += 1
                    if done % args.progress_every == 0 or done == len(rows):
                        ok_answer = sum(1 for x in results if x and x.get("answer"))
                        print(f"[progress] {done}/{len(rows)} answered={ok_answer}", flush=True)

    final = [r for r in results if r is not None]
    elapsed = time.time() - started
    prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in final)
    completion_tokens = sum(int(r.get("completion_tokens") or 0) for r in final)
    estimated_yuan = prompt_tokens / 1_000_000 * args.input_yuan_per_mtok + completion_tokens / 1_000_000 * args.output_yuan_per_mtok
    n = len(final)
    metrics = {
        "dataset": args.dataset,
        "setting": "deepseek_agentic_zeroshot",
        "prompt_variant": args.prompt_variant,
        "model": args.model,
        "n_total": n,
        "n_answered": sum(1 for r in final if r.get("answer")),
        "n_errors": sum(1 for r in final if r.get("error")),
        "em": round(sum(r["em"] for r in final) / n, 4) if n else 0.0,
        "cover_em": round(sum(r["cover_em"] for r in final) / n, 4) if n else 0.0,
        "f1": round(sum(r["f1"] for r in final) / n, 4) if n else 0.0,
        "avg_turns": round(sum(len(r.get("history", [])) for r in final) / n, 4) if n else 0.0,
        "api_calls": sum(int(r.get("api_calls") or 0) for r in final),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "elapsed_sec": round(elapsed, 2),
        "throughput_qps": round(n / elapsed, 4) if elapsed > 0 else 0.0,
        "estimated_yuan": round(estimated_yuan, 4),
        "output": str(out_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[done]", json.dumps(metrics, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
