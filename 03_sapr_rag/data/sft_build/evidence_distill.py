#!/usr/bin/env python3
"""Distill <evidence> from R3-RAG retrieval documents using DeepSeek.

For every (subquery, retrieval_documents) pair encountered in R3 cold-start
data, call DeepSeek with ReasonRAG's DOCUMENT_ANALYSIS_PROMPT to produce a
concise <evidence>. Cache by sha256(query + reference) so re-runs / debug do
not re-spend.

Outputs one cache file:
  evidence_cache.jsonl   {"key", "query", "reference", "evidence",
                          "tokens_in", "tokens_out", "ts"}

This is the standalone "fill the cache" stage. The downstream conversion
(reasoning samples + evidence-extraction samples) reads from this cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reuse the existing DeepSeek client from the offline pipeline.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # 03_sapr_rag/
sys.path.insert(0, str(ROOT))
from utils.deepseek_client import DeepSeekClient  # noqa: E402

# --- ReasonRAG DOCUMENT_ANALYSIS_PROMPT (verbatim from pipeline) ---------
EVIDENCE_SYSTEM = (
    "You are an information retrieval assistant. Your task is to extract "
    "relevant evidence from the provided Wikipedia documents based on the "
    "latest query.\n\n"
    "Instructions:\n"
    "* Identify key terms or concepts in the query.\n"
    "* Search the documents for evidence that supports the query.\n"
    "* Be concise: ideally one or two short sentences.\n"
    "* Response format:\n"
    "If relevant evidence is found, output:\n"
    '   Based on the query, the relevant evidence is <evidence>evidence</evidence>.\n'
    "If no relevant evidence is found, output:\n"
    '   <evidence>None</evidence>.\n'
)
EVIDENCE_USER_TEMPLATE = (
    "Question: {query}. Reference: <reference>{reference}</reference>"
)

# --- R3 step parsing (same regex shape as r3_to_reasonrag_sft) -----------
STEP_RE = re.compile(r"^Step\s+\d+\s*:\s*$", re.MULTILINE)
LABELS = ("The problem analysis:", "The retrieval query:",
          "The retrieval documents:", "The final answer:")


def _grab(block, label, until_labels):
    idx = block.find(label)
    if idx < 0:
        return None
    start = idx + len(label)
    end = len(block)
    for ul in until_labels:
        j = block.find(ul, start)
        if 0 <= j < end:
            end = j
    return block[start:end].strip()


def parse_steps(text):
    parts = STEP_RE.split(text or "")
    out = []
    for chunk in parts[1:]:
        out.append({
            "query": _grab(chunk, "The retrieval query:",
                           ["The retrieval documents:", "The final answer:"]),
            "documents": _grab(chunk, "The retrieval documents:", ["The final answer:"]),
        })
    return out


def cache_key(query: str, reference: str) -> str:
    h = hashlib.sha256()
    h.update(query.encode("utf-8"))
    h.update(b"\n--R3REF--\n")
    h.update(reference.encode("utf-8"))
    return h.hexdigest()[:32]


def collect_pairs(parquet_path: str, limit_rows: int):
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    if limit_rows > 0:
        df = df.head(limit_rows)
    seen = set()
    pairs = []
    for _, r in df.iterrows():
        for src in (r.get("instruction") or "", r.get("output") or ""):
            for st in parse_steps(src):
                q = (st.get("query") or "").strip()
                d = (st.get("documents") or "").strip()
                if not q or not d:
                    continue
                k = cache_key(q, d)
                if k in seen:
                    continue
                seen.add(k)
                pairs.append({"key": k, "query": q, "reference": d})
    return pairs


def load_cache(cache_path: Path):
    cache = {}
    if not cache_path.exists():
        return cache
    with cache_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "key" in rec:
                cache[rec["key"]] = rec
    return cache


def parse_evidence(raw: str) -> str:
    """Extract the evidence text inside <evidence>...</evidence>."""
    if raw is None:
        return ""
    m = re.search(r"<evidence>(.*?)</evidence>", raw, re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/r3.parquet")
    ap.add_argument("--cache", required=True, help="output cache jsonl (append-only)")
    ap.add_argument("--limit-rows", type=int, default=0,
                    help="limit on R3 source rows scanned (0=all)")
    ap.add_argument("--limit-pairs", type=int, default=0,
                    help="cap distinct (q, ref) pairs to call (0=all uncached)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--dry-run", action="store_true",
                    help="only print plan, no API calls")
    args = ap.parse_args()

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    print(f"[cache] loaded {len(cache)} existing entries from {cache_path}",
          file=sys.stderr)

    pairs = collect_pairs(args.src, args.limit_rows)
    uncached = [p for p in pairs if p["key"] not in cache]
    todo = uncached[:args.limit_pairs] if args.limit_pairs > 0 else uncached

    print(f"[plan] R3 rows={'all' if args.limit_rows==0 else args.limit_rows}, "
          f"distinct pairs={len(pairs)}, cached={len(pairs)-len(uncached)}, "
          f"uncached={len(uncached)}, to_call={len(todo)}", file=sys.stderr)
    if args.dry_run or not todo:
        return

    # Load env for API key.
    env_path = ROOT / ".env"
    client = DeepSeekClient.from_env(env_path=env_path if env_path.exists() else None)

    # Streamed append; lock for thread-safe writes.
    write_lock = threading.Lock()
    fout = cache_path.open("a", buffering=1)
    started = time.time()
    done = 0
    errs = 0

    def _one(item):
        nonlocal done, errs
        msgs = [
            {"role": "system", "content": EVIDENCE_SYSTEM},
            {"role": "user", "content": EVIDENCE_USER_TEMPLATE.format(
                query=item["query"], reference=item["reference"])},
        ]
        before = client.stats.prompt_tokens, client.stats.completion_tokens
        try:
            raw = client.chat(msgs, max_tokens=args.max_tokens, temperature=0.0)
        except Exception as e:
            with write_lock:
                errs += 1
                done += 1
            return None, e
        ev = parse_evidence(raw)
        # Token usage delta for this call (not perfectly thread-safe, but fine
        # for aggregate accounting; per-row may drift a few tokens).
        after = client.stats.prompt_tokens, client.stats.completion_tokens
        rec = {
            "key": item["key"],
            "query": item["query"],
            "reference": item["reference"],
            "evidence_raw": raw,
            "evidence": ev,
            "tokens_in": after[0] - before[0],
            "tokens_out": after[1] - before[1],
            "ts": int(time.time()),
        }
        with write_lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            done += 1
            if done % 50 == 0 or done == len(todo):
                elapsed = time.time() - started
                rate = done / elapsed if elapsed else 0
                eta = (len(todo) - done) / rate if rate else float("inf")
                print(f"[progress] {done}/{len(todo)} ({rate:.1f}/s) "
                      f"errs={errs} eta={eta:.0f}s "
                      f"tokens_in={client.stats.prompt_tokens} "
                      f"tokens_out={client.stats.completion_tokens}",
                      file=sys.stderr)
        return rec, None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, it) for it in todo]
        for f in as_completed(futs):
            f.result()

    fout.close()
    elapsed = time.time() - started
    print(f"[done] {done} calls in {elapsed:.1f}s, errs={errs}, "
          f"tokens_in={client.stats.prompt_tokens} "
          f"tokens_out={client.stats.completion_tokens}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
