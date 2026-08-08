#!/usr/bin/env python3
"""为 DeepSeek Standard RAG baseline 生成批量 API prompt。

流程：
  Question -> BGE/FAISS top-k from wiki18_extended -> prompt JSONL

只负责检索和 prompt 构造，不调用 API。输出可直接交给
`batch_deepseek_api.py` 批式调用，调用结果中会保留 input 字段，方便后续评分。

示例：
  python 03_sapr_rag/scripts/eval/prepare_standard_rag_prompts.py \
    --dataset 2wikimultihopqa \
    --limit 200 \
    --top_k 3 \
    --output data/eval_results/deepseek_standard_rag/prompts_2wiki_top3_200.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import requests


SYSTEM_PROMPT = """You are a helpful QA assistant.
Answer the question using only the provided context.
Keep the final answer concise, usually a short phrase or entity name.
If the answer is not supported by the context, say "I don't know".
Do not include explanations."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["hotpotqa", "2wikimultihopqa", "musique"])
    p.add_argument("--data_dir", default="data/eval")
    p.add_argument("--limit", type=int, default=200, help="0=全量；默认 200 用于成本探针")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--output", required=True)
    p.add_argument("--corpus_path", default="data/corpus/wiki18_extended.jsonl")
    p.add_argument("--index_path", default="data/index/bge_extended_Flat.index")
    p.add_argument("--bge_model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--encode_batch_size", type=int, default=64)
    p.add_argument("--max_doc_chars", type=int, default=1200)
    p.add_argument("--device", default=None, help="SentenceTransformer device；默认自动选择")
    p.add_argument("--retrieval_url", default=None,
                   help="已有 retrieval daemon 地址，例如 http://127.0.0.1:8100；设置后不加载本地 BGE/FAISS")
    p.add_argument("--retrieval_batch_size", type=int, default=16)
    p.add_argument("--retrieval_timeout", type=float, default=120.0)
    return p.parse_args()


def load_dataset(dataset: str, data_dir: Path, limit: int) -> List[Dict[str, Any]]:
    path = data_dir / dataset / "dev.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
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


def encode_questions(
    questions: Sequence[str],
    model_name: str,
    batch_size: int,
    device: str | None,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embs = model.encode(
        list(questions),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embs, dtype=np.float32)


def search_index(index_path: Path, embs: np.ndarray, top_k: int) -> np.ndarray:
    import faiss

    index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    _, ids = index.search(embs, top_k)
    return ids


def iter_corpus_hits(corpus_path: Path, needed_ids: set[int]) -> Dict[int, Dict[str, str]]:
    docs: Dict[int, Dict[str, str]] = {}
    if not needed_ids:
        return docs
    max_needed = max(needed_ids)
    with corpus_path.open(encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx > max_needed and len(docs) == len(needed_ids):
                break
            if line_idx not in needed_ids:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = {}
            docs[line_idx] = {
                "title": str(obj.get("title", "")),
                "content": str(obj.get("content", "")),
            }
    return docs


def retrieve_via_daemon(
    questions: Sequence[str],
    retrieval_url: str,
    top_k: int,
    batch_size: int,
    timeout: float,
) -> List[List[Dict[str, Any]]]:
    base = retrieval_url.rstrip("/")
    health = requests.get(f"{base}/health", timeout=timeout)
    health.raise_for_status()
    print(f"[daemon] health={health.text[:300]}", flush=True)

    out: List[List[Dict[str, Any]]] = []
    for start in range(0, len(questions), batch_size):
        batch = list(questions[start:start + batch_size])
        resp = requests.post(
            f"{base}/search_batch",
            json={"queries": batch, "top_k": top_k},
            timeout=timeout,
        )
        resp.raise_for_status()
        rows = resp.json()["results"]
        for docs in rows:
            normalized = []
            for rank, d in enumerate(docs, 1):
                normalized.append({
                    "rank": rank,
                    "doc_id": d.get("doc_id", None),
                    "title": d.get("title", ""),
                    "content": truncate_text(d.get("text", d.get("content", "")), 10**9),
                    "score": d.get("score"),
                })
            out.append(normalized)
        done = min(start + batch_size, len(questions))
        print(f"[daemon] retrieved {done}/{len(questions)}", flush=True)
    return out


def truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def build_prompt(question: str, docs: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        title = d.get("title", "")
        content = d.get("content", "")
        parts.append(f"[Doc {i}: {title}]\n{content}")
    context = "\n\n".join(parts)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    data_dir = repo_root / args.data_dir
    corpus_path = repo_root / args.corpus_path
    index_path = repo_root / args.index_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    samples = load_dataset(args.dataset, data_dir, args.limit)
    questions = [str(s.get("question", "")) for s in samples]
    print(f"[load] dataset={args.dataset} n={len(samples)} top_k={args.top_k}", flush=True)

    if args.retrieval_url:
        retrieved_by_sample = retrieve_via_daemon(
            questions,
            args.retrieval_url,
            args.top_k,
            args.retrieval_batch_size,
            args.retrieval_timeout,
        )
    else:
        embs = encode_questions(questions, args.bge_model, args.encode_batch_size, args.device)
        print(f"[encode] embs={embs.shape}", flush=True)

        ids = search_index(index_path, embs, args.top_k)
        needed_ids = {int(x) for row in ids for x in row if int(x) >= 0}
        print(f"[search] unique_doc_ids={len(needed_ids)}", flush=True)

        docs_by_id = iter_corpus_hits(corpus_path, needed_ids)
        missing = needed_ids - set(docs_by_id)
        if missing:
            print(f"[warn] missing_docs={len(missing)}", file=sys.stderr, flush=True)

        retrieved_by_sample = []
        for row_ids in ids:
            retrieved = []
            for rank, doc_id in enumerate(row_ids, 1):
                doc_id = int(doc_id)
                raw = docs_by_id.get(doc_id, {"title": "", "content": ""})
                content = truncate_text(raw.get("content", ""), args.max_doc_chars)
                retrieved.append({
                    "rank": rank,
                    "doc_id": doc_id,
                    "title": raw.get("title", ""),
                    "content": content,
                })
            retrieved_by_sample.append(retrieved)

    with output_path.open("w", encoding="utf-8") as fout:
        for i, (sample, retrieved_raw) in enumerate(zip(samples, retrieved_by_sample)):
            retrieved = []
            for d in retrieved_raw:
                dd = dict(d)
                dd["content"] = truncate_text(dd.get("content", ""), args.max_doc_chars)
                retrieved.append(dd)
            qid = sample.get("id", str(i))
            question = str(sample.get("question", ""))
            row = {
                "id": f"{args.dataset}:{qid}:top{args.top_k}",
                "dataset": args.dataset,
                "qid": qid,
                "question": question,
                "gold": get_gold(sample),
                "top_k": args.top_k,
                "retrieved_docs": retrieved,
                "system": SYSTEM_PROMPT,
                "prompt": build_prompt(question, retrieved),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    meta = {
        "dataset": args.dataset,
        "n": len(samples),
        "top_k": args.top_k,
        "output": str(output_path),
        "corpus_path": str(corpus_path),
        "index_path": str(index_path),
        "bge_model": args.bge_model,
        "max_doc_chars": args.max_doc_chars,
        "retrieval_url": args.retrieval_url,
        "retrieval_batch_size": args.retrieval_batch_size if args.retrieval_url else None,
        "elapsed_sec": round(elapsed, 2),
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[done]", json.dumps(meta, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
