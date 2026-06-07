#!/usr/bin/env python3
"""SAPR-R v1 — Step 3: 用 BGE+FAISS 给每个 subquery 检索 wiki18 top-K 候选。

输入:
    {in_dir}/reasoning_steps.jsonl   step2 产出
    SAPR_BGE_MODEL_PATH / SAPR_BGE_INDEX_PATH / SAPR_WIKI_CORPUS_PATH

输出:
    {out_dir}/candidates.jsonl       每行一个 (qid, step_idx) 单元
    {out_dir}/run_meta.json          跑次元数据（耗时、命中率、device 等）

每行 schema:
    {
      "qid": "...",
      "step_idx": 0,
      "question": "...",
      "gt_answer": "...",
      "supporting_titles": ["...", "..."],
      "subquery": "...",
      "subject_entity": "...",
      "step_gold": "...",
      "prior_thoughts": ["...", "..."],   # 前 step_idx 个 thought 句
      "candidates": [
        {"doc_id": 12345, "title": "...", "text": "...", "retriever_score": 0.92},
        ...K
      ]
    }

Resumable:
    若 candidates.jsonl 已存在，按 (qid, step_idx) 复合 key 跳过已完成单元。

跑法（5090）:
    source config/env_5090.sh
    conda activate reasonrag
    CUDA_VISIBLE_DEVICES=0 python 03_sapr_rag/data/build_v1/step3_retrieve_candidates.py \
        --in-dir 03_sapr_rag/data/build_v1/out/v1_5k \
        --out-dir 03_sapr_rag/data/build_v1/out/v1_5k \
        --top-k 10

资源:
    FAISS Flat 索引约 60GB，需机器有 ≥80GB 可用内存。
    Encoder 占用 ~2GB GPU；step3 编码完即释放，索引在 CPU 内存中。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- 仓内路径派生（按 docs/coding_standard.md §2 规范）---
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]   # SAPR-RAG/
sys.path.insert(0, str(_REPO_ROOT))

from config.paths import (  # noqa: E402
    BGE_INDEX_PATH,
    BGE_MODEL_PATH,
    WIKI_CORPUS_PATH,
)


logger = logging.getLogger("sapr_v1.step3")


# 与 reretrieve_evidence_with_inferred_subquery.py / FlashRAG parse_query 保持一致
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ----------------------- IO helpers -----------------------

@dataclass
class StepUnit:
    qid: str
    step_idx: int
    question: str
    gt_answer: str
    supporting_titles: List[str]
    subquery: str
    subject_entity: str
    step_gold: str
    prior_thoughts: List[str]


def load_step2_units(in_path: Path) -> List[StepUnit]:
    """读 step2 reasoning_steps.jsonl，flatten 成 (qid, step_idx) 单元。

    丢弃 ok=False 的整条样本；prior_thoughts 取前 step_idx 个 thought 句。
    """
    units: List[StepUnit] = []
    n_records = 0
    n_skipped = 0
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_records += 1
            obj = json.loads(line)
            if not obj.get("ok"):
                n_skipped += 1
                continue
            steps = obj.get("reasoning_steps") or []
            if not steps:
                n_skipped += 1
                continue
            qid = obj.get("qid")
            question = obj.get("question", "")
            gt = obj.get("gt_answer", "")
            sup_titles = obj.get("supporting_titles") or []
            thoughts: List[str] = [s.get("thought", "") for s in steps]
            for i, step in enumerate(steps):
                units.append(StepUnit(
                    qid=qid,
                    step_idx=i,
                    question=question,
                    gt_answer=gt,
                    supporting_titles=list(sup_titles),
                    subquery=step.get("subquery", ""),
                    subject_entity=step.get("subject_entity", ""),
                    step_gold=step.get("step_gold", ""),
                    prior_thoughts=thoughts[:i],
                ))
    logger.info(
        "loaded %d records (skipped %d not-ok); flattened to %d (qid, step_idx) units",
        n_records, n_skipped, len(units),
    )
    return units


def load_completed_keys(out_path: Path) -> set:
    """断点续跑：读已存在 candidates.jsonl 收集 (qid, step_idx) 集合。"""
    if not out_path.exists():
        return set()
    keys: set = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("qid")
                step_idx = obj.get("step_idx")
                if qid is not None and step_idx is not None:
                    keys.add((qid, int(step_idx)))
            except json.JSONDecodeError:
                continue
    return keys


# ----------------------- BGE encoder -----------------------

def pick_device() -> str:
    """优先选有 ≥2GB 空闲显存的 GPU；否则 CPU。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        for gi in range(torch.cuda.device_count()):
            free_gb = torch.cuda.mem_get_info(gi)[0] / 1e9
            if free_gb > 2.0:
                return f"cuda:{gi}"
    except Exception as e:
        logger.warning("cuda probe failed: %s", e)
    return "cpu"


def encode_queries(queries: List[str], model_path: str, device: str, batch_size: int) -> "Any":
    """BGE batch encode → L2-normalized [N,768] float32 numpy."""
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    logger.info("BGE model loaded in %.1fs (device=%s)", time.time() - t0, device)

    t0 = time.time()
    chunks: List[Any] = []
    n = len(queries)
    for start in range(0, n, batch_size):
        batch = queries[start:start + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        cls = torch.nn.functional.normalize(out.last_hidden_state[:, 0], p=2, dim=1)
        chunks.append(cls.cpu().numpy())
        if (start // batch_size) % 50 == 0:
            logger.info("encode progress %d/%d", min(start + batch_size, n), n)

    embeddings = np.ascontiguousarray(np.concatenate(chunks).astype("float32"))
    logger.info("encoded %d queries in %.1fs", n, time.time() - t0)

    # 释放 encoder 以腾出 GPU 给后续工作
    del model, tokenizer
    if device != "cpu":
        torch.cuda.empty_cache()
    logger.info("encoder freed from GPU")
    return embeddings


# ----------------------- corpus fetch -----------------------

def fetch_corpus_lines(corpus_path: str, doc_ids: List[int]) -> Dict[int, Dict[str, str]]:
    """按行号读 wiki18 jsonl 抽 title/text。

    corpus 每行 JSON 含 'contents'（首行=title，其余=正文）；doc_id == 行号（0-indexed）。
    一次扫描搞定，避免 21M 文档全量加载。
    """
    needed = sorted({int(d) for d in doc_ids if d >= 0})
    logger.info("fetching %d unique doc lines from corpus", len(needed))

    result: Dict[int, Dict[str, str]] = {}
    needed_set = set(needed)
    last_log = time.time()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx in needed_set:
                d = json.loads(line)
                raw = d.get("contents", d.get("text", ""))
                first_line = raw.split("\n", 1)[0].strip().strip('"')
                text = raw[len(first_line):].strip()[:500]
                result[idx] = {"title": first_line, "text": text}
                needed_set.discard(idx)
                if not needed_set:
                    break
            if time.time() - last_log > 30:
                logger.info(
                    "  scanned %d lines, found %d/%d ...",
                    idx, len(result), len(needed),
                )
                last_log = time.time()

    logger.info("fetched %d/%d docs", len(result), len(needed))
    return result


# ----------------------- main -----------------------

@dataclass
class RunMeta:
    started_at: str = ""
    finished_at: str = ""
    elapsed_sec: float = 0.0
    n_units_total: int = 0
    n_units_done: int = 0
    n_units_skipped_resumed: int = 0
    n_units_failed: int = 0
    top_k: int = 10
    encoder_device: str = ""
    encode_batch_size: int = 64
    bge_model_path: str = ""
    bge_index_path: str = ""
    wiki_corpus_path: str = ""
    in_jsonl: str = ""
    out_jsonl: str = ""
    gold_recall_top_k: float = 0.0
    n_units_with_gold_eval: int = 0


def _normalize_title(title: str) -> str:
    import re
    t = title.strip().lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAPR-R v1 step3 — retrieve top-K candidates per subquery via BGE+FAISS",
    )
    parser.add_argument("--in-dir", type=Path, required=True,
                        help="step2 输出目录（含 reasoning_steps.jsonl）")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="step3 输出目录（默认 = in-dir）")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-debug", type=int, default=0,
                        help="只跑前 N 个 (qid, step_idx) 单元做 debug，0=全量")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    in_dir = args.in_dir.resolve()
    out_dir = (args.out_dir or in_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    in_jsonl = in_dir / "reasoning_steps.jsonl"
    out_jsonl = out_dir / "candidates.jsonl"
    meta_path = out_dir / "run_meta_step3.json"

    if not in_jsonl.exists():
        raise FileNotFoundError(f"step2 output not found: {in_jsonl}")

    bge_model = str(BGE_MODEL_PATH)
    bge_index = str(BGE_INDEX_PATH)
    corpus = str(WIKI_CORPUS_PATH)

    # --- 1. load step2 + dedup against existing output ---
    units = load_step2_units(in_jsonl)
    if args.limit_debug > 0:
        units = units[: args.limit_debug]

    completed = load_completed_keys(out_jsonl)
    if completed:
        logger.info("found %d already-completed (qid, step_idx) units; resuming", len(completed))
    todo = [u for u in units if (u.qid, u.step_idx) not in completed]
    logger.info("todo=%d, skipped_resumed=%d", len(todo), len(units) - len(todo))

    meta = RunMeta(
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        n_units_total=len(units),
        n_units_skipped_resumed=len(units) - len(todo),
        top_k=args.top_k,
        encode_batch_size=args.batch_size,
        bge_model_path=bge_model,
        bge_index_path=bge_index,
        wiki_corpus_path=corpus,
        in_jsonl=str(in_jsonl),
        out_jsonl=str(out_jsonl),
    )

    if not todo:
        logger.info("nothing to do, all units already completed")
        meta.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        meta_path.write_text(json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8")
        return 0

    started = time.time()

    # --- 2. encode subqueries ---
    queries = [BGE_QUERY_PREFIX + (u.subquery or "").strip() for u in todo]
    device = pick_device()
    meta.encoder_device = device
    embeddings = encode_queries(queries, bge_model, device, args.batch_size)

    # --- 3. FAISS search ---
    import faiss
    import numpy as np

    import os as _os
    idx_size_gb = _os.path.getsize(bge_index) / 1e9
    use_mmap = _os.environ.get("FAISS_MMAP", "1") == "1"
    logger.info("loading FAISS index (%.1f GB, mmap=%s) ...", idx_size_gb, use_mmap)
    t0 = time.time()
    if use_mmap:
        index = faiss.read_index(bge_index, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    else:
        index = faiss.read_index(bge_index)
    logger.info("FAISS loaded in %.1fs, ntotal=%d", time.time() - t0, index.ntotal)

    t0 = time.time()
    scores, indices = index.search(embeddings, args.top_k)
    logger.info("FAISS searched %d queries in %.1fs", len(todo), time.time() - t0)
    del index

    # --- 4. fetch unique corpus lines ---
    all_doc_ids = [int(x) for x in indices.flatten().tolist()]
    corpus_info = fetch_corpus_lines(corpus, all_doc_ids)

    # --- 5. write jsonl ---
    n_failed = 0
    n_gold_eval = 0
    n_gold_hit = 0
    with out_jsonl.open("a", encoding="utf-8") as fout:
        for j, u in enumerate(todo):
            cands: List[Dict[str, Any]] = []
            for doc_id, score in zip(indices[j], scores[j]):
                if doc_id < 0:
                    continue
                info = corpus_info.get(int(doc_id), {"title": "unknown", "text": ""})
                cands.append({
                    "doc_id": int(doc_id),
                    "title": info["title"],
                    "text": info["text"],
                    "retriever_score": float(score),
                })
            if not cands:
                n_failed += 1
                logger.warning("unit %s/%d got 0 candidates", u.qid, u.step_idx)

            record = {
                "qid": u.qid,
                "step_idx": u.step_idx,
                "question": u.question,
                "gt_answer": u.gt_answer,
                "supporting_titles": u.supporting_titles,
                "subquery": u.subquery,
                "subject_entity": u.subject_entity,
                "step_gold": u.step_gold,
                "prior_thoughts": u.prior_thoughts,
                "candidates": cands,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 命中率统计：只在 step_idx==0 单元做（避免重复计数同一组 supporting_titles）
            if u.step_idx == 0 and u.supporting_titles:
                n_gold_eval += len(u.supporting_titles)
                gold = {_normalize_title(t) for t in u.supporting_titles}
                ret = {_normalize_title(c["title"]) for c in cands}
                n_gold_hit += len(gold & ret)

        fout.flush()

    # --- 6. meta ---
    meta.n_units_done = len(todo) - n_failed
    meta.n_units_failed = n_failed
    meta.n_units_with_gold_eval = n_gold_eval
    meta.gold_recall_top_k = (n_gold_hit / n_gold_eval) if n_gold_eval else 0.0
    meta.elapsed_sec = round(time.time() - started, 2)
    meta.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")

    meta_path.write_text(json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "DONE  done=%d  failed=%d  gold_recall@%d=%.1f%%  elapsed=%.1fs",
        meta.n_units_done, meta.n_units_failed, args.top_k,
        100 * meta.gold_recall_top_k, meta.elapsed_sec,
    )
    logger.info("wrote: %s", out_jsonl)
    logger.info("meta:  %s", meta_path)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
