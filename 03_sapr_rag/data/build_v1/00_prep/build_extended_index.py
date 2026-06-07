"""build_extended_index.py — 用 BGE-base-en-v1.5 编码 wiki18_extended.jsonl 并构建 FAISS Flat 索引。

输入：
    data/corpus/wiki18_extended.jsonl    # build_extended_corpus.py 产出
    models/bge-base-en-v1.5/             # download_assets.sh 产出

输出：
    data/index/bge_extended_Flat.index   # FAISS IndexFlatIP（cosine via normalize + IP）
    data/index/bge_extended_Flat.meta.json

关键设计：
- **多卡数据并行**：通过 accelerate 自动切片到所有可见 GPU，每卡 encode 一个分片
  → 8 卡 H20 总耗时 ~15-30 min（22M 段，fp16）
- **chunked encode + flush**：每 chunk 落盘临时 npy，主程序最后一次性 .add 到 FAISS
  → 中途崩溃可从 last chunk 接续重跑
- **fp16 编码 + IndexFlatIP**：与 ReasonRAG 原版一致，保证 retrieval@k 完全可比
- **L2 normalize**：BGE 是 cosine 相似度，FAISS IP 检索前必须 L2 normalize embedding

用法（单卡）：
    source config/env_local.sh
    python 03_sapr_rag/data/build_v1/00_prep/build_extended_index.py \\
        --batch-size 256 --max-seq-len 256

用法（多卡，最简单的方式：torchrun）：
    source config/env_local.sh
    torchrun --nproc_per_node=4 03_sapr_rag/data/build_v1/00_prep/build_extended_index.py \\
        --batch-size 256 --max-seq-len 256

重跑安全：每 chunk 输出 chunk_{i}.npy，重跑时已存在的 chunk 跳过。最后 .add 到 FAISS 一次性完成。

预估资源（22M 段，fp16，seq_len=256）：
- 1× H20 100GB:  batch=512, ~2-3h
- 4× H20 100GB:  batch=512, ~30-60min
- 1× RTX 3090 8GB: batch=64,  ~24-48h
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# 与 step3 保持一致；编码 doc 时不加 query 前缀
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_DIM = 768


def _read_corpus_chunked(corpus_path: Path, chunk_size: int):
    """流式读 jsonl，按 chunk_size 切片，yield (chunk_idx, list[str])."""
    chunk_idx = 0
    buf: List[str] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            buf.append(obj.get("contents", obj.get("text", "")))
            if len(buf) >= chunk_size:
                yield chunk_idx, buf
                chunk_idx += 1
                buf = []
        if buf:
            yield chunk_idx, buf


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return x / norms


def _encode_chunks(
    corpus_path: Path,
    model_path: Path,
    out_dir: Path,
    chunk_size: int,
    batch_size: int,
    max_seq_len: int,
    device: str,
    dtype: str,
):
    """单卡：流式 encode 所有 chunk，落盘 chunk_{i}.npy。"""
    import torch
    from transformers import AutoModel, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading BGE encoder from %s on %s (%s)", model_path, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch_dtype).to(device).eval()

    t0 = time.time()
    n_total = 0
    last_log = t0

    for chunk_idx, texts in _read_corpus_chunked(corpus_path, chunk_size):
        chunk_path = out_dir / f"chunk_{chunk_idx:05d}.npy"
        if chunk_path.exists():
            arr = np.load(chunk_path)
            if arr.shape[0] == len(texts):
                logger.info("[skip] chunk %d already exists with %d rows", chunk_idx, arr.shape[0])
                n_total += arr.shape[0]
                continue
            logger.warning("[redo] chunk %d size mismatch (%d vs %d), re-encode",
                           chunk_idx, arr.shape[0], len(texts))

        embs: List[np.ndarray] = []
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            with torch.no_grad():
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=max_seq_len, return_tensors="pt").to(device)
                out = model(**enc)
                # BGE: cls token embedding
                emb = out.last_hidden_state[:, 0, :]  # [B, 768]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            embs.append(emb.float().cpu().numpy().astype(np.float32))

        arr = np.concatenate(embs, axis=0)
        np.save(chunk_path, arr)
        n_total += arr.shape[0]

        if time.time() - last_log > 30:
            elapsed = time.time() - t0
            speed = n_total / max(elapsed, 1)
            logger.info("  chunk %d done, total=%d, %.0f docs/s, elapsed %.0fs",
                        chunk_idx, n_total, speed, elapsed)
            last_log = time.time()

    logger.info("[encode done] total=%d docs in %.1fs", n_total, time.time() - t0)
    return n_total


def _build_faiss(out_dir: Path, index_out: Path, n_expected: int):
    """把 chunk_*.npy 顺序 add 到 IndexFlatIP，落盘。"""
    import faiss

    chunks = sorted(out_dir.glob("chunk_*.npy"))
    logger.info("building FAISS IndexFlatIP from %d chunks ...", len(chunks))

    index = faiss.IndexFlatIP(BGE_DIM)
    n_added = 0
    t0 = time.time()
    last_log = t0

    for chunk_path in chunks:
        arr = np.load(chunk_path).astype(np.float32)
        # 二次 L2 normalize 防御（万一上游 fp16 误差）
        arr = _l2_normalize(arr)
        index.add(arr)
        n_added += arr.shape[0]
        if time.time() - last_log > 30:
            logger.info("  added %d / ? from %s, elapsed %.0fs",
                        n_added, chunk_path.name, time.time() - t0)
            last_log = time.time()

    if n_expected and n_added != n_expected:
        logger.warning("[warn] added %d but expected %d", n_added, n_expected)

    index_out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_out))
    logger.info("[faiss done] index with %d vecs written to %s (%.1fs)",
                n_added, index_out, time.time() - t0)
    return n_added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", type=Path,
                        help="wiki18_extended.jsonl 路径（默认 data/corpus/）")
    parser.add_argument("--model-path", type=Path,
                        help="BGE 模型目录（默认 models/bge-base-en-v1.5/）")
    parser.add_argument("--index-out", type=Path,
                        help="FAISS index 输出（默认 data/index/bge_extended_Flat.index）")
    parser.add_argument("--chunks-dir", type=Path,
                        help="临时 chunk_*.npy 目录（默认 data/index/_chunks_extended/）")
    parser.add_argument("--chunk-size", type=int, default=50_000,
                        help="每个临时 chunk 的段数；越大文件越少但崩溃损失越大")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="BGE forward batch size；H20 100GB 建议 512")
    parser.add_argument("--max-seq-len", type=int, default=256,
                        help="BGE max sequence length；与 ReasonRAG 一致取 256")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                        help="编码精度；fp16 显存省一半，FAISS index 仍存 fp32")
    parser.add_argument("--device", default="cuda",
                        help="cuda / cuda:N / cpu；多卡走 torchrun 自动分配")
    parser.add_argument("--encode-only", action="store_true",
                        help="只 encode，不 build FAISS（多卡 run 后由 rank 0 单独 build）")
    parser.add_argument("--build-only", action="store_true",
                        help="跳过 encode，只把已有 chunk 拼成 FAISS（rank 0 调用）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")

    repo_root = Path(__file__).resolve().parents[4]
    corpus_path = args.corpus_path or (repo_root / "data" / "corpus" / "wiki18_extended.jsonl")
    model_path = args.model_path or (repo_root / "models" / "bge-base-en-v1.5")
    index_out = args.index_out or (repo_root / "data" / "index" / "bge_extended_Flat.index")
    chunks_dir = args.chunks_dir or (repo_root / "data" / "index" / "_chunks_extended")

    if not args.build_only:
        if not corpus_path.exists():
            raise FileNotFoundError(f"{corpus_path} not found; run build_extended_corpus.py first")
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found; run download_assets.sh first")

    # ---- 多卡：torchrun 启动时根据 LOCAL_RANK 各自 encode 自己那一份 chunk ----
    # 简化做法：每个 rank 处理 chunk_id % world_size == rank 的 chunk
    # （chunk 切分由 corpus_path 顺序读决定，所有 rank 看到的 chunk 序列一致）
    # 注意：这要求每个 rank 都跑 _read_corpus_chunked，IO 上有冗余，但 22M 段 5 GB jsonl
    # 顺序读 ~30s，对总时长几乎无影响。
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = args.device
    if world_size > 1 and device.startswith("cuda") and ":" not in device:
        device = f"cuda:{local_rank}"

    if world_size > 1:
        logger.info("multi-GPU mode: rank=%d, world_size=%d, device=%s", rank, world_size, device)

    # ---- encode 阶段 ----
    n_total_local = 0
    if not args.build_only:
        if world_size == 1:
            n_total_local = _encode_chunks(
                corpus_path=corpus_path, model_path=model_path, out_dir=chunks_dir,
                chunk_size=args.chunk_size, batch_size=args.batch_size,
                max_seq_len=args.max_seq_len, device=device, dtype=args.dtype,
            )
        else:
            # 各 rank 只处理 chunk_idx % world_size == rank 的 chunk
            n_total_local = _encode_chunks_distributed(
                corpus_path=corpus_path, model_path=model_path, out_dir=chunks_dir,
                chunk_size=args.chunk_size, batch_size=args.batch_size,
                max_seq_len=args.max_seq_len, device=device, dtype=args.dtype,
                rank=rank, world_size=world_size,
            )

    # ---- build 阶段：只在 rank 0 做 ----
    if args.encode_only:
        logger.info("[encode-only] skip FAISS build, exit on rank %d", rank)
        return
    if world_size > 1 and rank != 0:
        logger.info("rank %d: encode done, FAISS build is rank 0's job, exit", rank)
        return

    n_added = _build_faiss(chunks_dir, index_out, n_expected=0)

    # 写 meta
    meta_path = index_out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "n_vectors": n_added, "dim": BGE_DIM,
        "encoder_model": str(model_path),
        "corpus_path": str(corpus_path),
        "chunk_size": args.chunk_size, "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len, "dtype": args.dtype,
        "world_size": world_size,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("meta written to %s", meta_path)


def _encode_chunks_distributed(
    corpus_path: Path,
    model_path: Path,
    out_dir: Path,
    chunk_size: int,
    batch_size: int,
    max_seq_len: int,
    device: str,
    dtype: str,
    rank: int,
    world_size: int,
):
    """多卡：每 rank 只 encode chunk_idx % world_size == rank 的 chunk。"""
    import torch
    from transformers import AutoModel, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[rank %d] loading BGE on %s (%s)", rank, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch_dtype).to(device).eval()

    t0 = time.time()
    n_local = 0
    last_log = t0

    for chunk_idx, texts in _read_corpus_chunked(corpus_path, chunk_size):
        if chunk_idx % world_size != rank:
            continue
        chunk_path = out_dir / f"chunk_{chunk_idx:05d}.npy"
        if chunk_path.exists():
            arr = np.load(chunk_path)
            if arr.shape[0] == len(texts):
                n_local += arr.shape[0]
                continue

        embs: List[np.ndarray] = []
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            with torch.no_grad():
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=max_seq_len, return_tensors="pt").to(device)
                out = model(**enc)
                emb = out.last_hidden_state[:, 0, :]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            embs.append(emb.float().cpu().numpy().astype(np.float32))

        arr = np.concatenate(embs, axis=0)
        np.save(chunk_path, arr)
        n_local += arr.shape[0]

        if time.time() - last_log > 30:
            elapsed = time.time() - t0
            logger.info("[rank %d] chunk %d done, local=%d, %.0fs elapsed",
                        rank, chunk_idx, n_local, elapsed)
            last_log = time.time()

    logger.info("[rank %d] encode done: local=%d in %.1fs", rank, n_local, time.time() - t0)
    return n_local


if __name__ == "__main__":
    main()
