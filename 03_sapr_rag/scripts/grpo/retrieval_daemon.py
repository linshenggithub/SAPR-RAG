#!/usr/bin/env python
"""检索 daemon：把 BGE+FAISS 检索做成独立 HTTP 服务，供 GRPO rollout 多进程共享。

为什么要 daemon：GRPO 每个 step 对每个 prompt 采样 K 条轨迹，每条轨迹的每次检索都实时
打 FAISS。若每个训练进程各自加载 68GB 索引，必然 OOM。daemon 让索引只加载一份，rollout
scheduler 通过 HTTP 调用。

检索逻辑与 agent_infer.py 的 BGEFaissRetriever 严格 1:1（BGE query 前缀 / cls pooling +
L2 normalize / mmap 只读索引 / contents 按首行切 title / 正文 [:500]），保证
训练 / 推理 / eval 三处口径完全一致。

启动：
  CUDA_VISIBLE_DEVICES=0 python retrieval_daemon.py --port 8100
查询：
  curl -s localhost:8100/health
  curl -s -X POST localhost:8100/search_batch \
       -H 'content-type: application/json' \
       -d '{"queries": ["who founded Apple"], "top_k": 3}'
"""
import argparse
import socket
import sys
import threading
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# ─────────── 路径（与 agent_infer.py 对齐）───────────
PROJ_ROOT = Path("/mlx_devbox/users/mayi.summer/playground/SAPR-RAG")
BGE_PATH = PROJ_ROOT / "models/bge-base-en-v1.5"
INDEX_PATH = PROJ_ROOT / "data/index/bge_extended_Flat.index"
CORPUS_PATH = PROJ_ROOT / "data/corpus/wiki18_extended.jsonl"

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BGEFaissRetriever:
    """与 agent_infer.py 同款检索器，唯一区别：text 截断长度做成可配置（默认 500，与 eval 一致）。"""

    def __init__(self, bge_path, index_path, corpus_path, device="cuda:0", text_truncate=500):
        self.device = device
        self.text_truncate = text_truncate

        print(f"[retriever] loading BGE on {device} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(bge_path)
        self.model = AutoModel.from_pretrained(bge_path).to(device).eval()

        print(f"[retriever] loading FAISS index ({index_path}, mmap) ...", flush=True)
        t0 = time.time()
        self.index = faiss.read_index(
            str(index_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
        )
        print(f"[retriever] index loaded in {time.time()-t0:.1f}s, "
              f"n_vectors={self.index.ntotal}, dim={self.index.d}", flush=True)

        print(f"[retriever] loading corpus via HF datasets ({corpus_path}) ...", flush=True)
        t0 = time.time()
        import datasets
        self.corpus = datasets.load_dataset(
            "json", data_files=str(corpus_path), split="train",
        )
        print(f"[retriever] corpus loaded in {time.time()-t0:.1f}s, "
              f"n_docs={len(self.corpus)}", flush=True)

    @torch.no_grad()
    def encode(self, queries):
        prefixed = [BGE_QUERY_PREFIX + q for q in queries]
        enc = self.tokenizer(prefixed, padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(self.device)
        out = self.model(**enc)
        emb = torch.nn.functional.normalize(out.last_hidden_state[:, 0], p=2, dim=1)
        return np.ascontiguousarray(emb.cpu().numpy().astype("float32"))

    def search_batch(self, queries, top_k=3):
        if not queries:
            return []
        embs = self.encode(queries)
        scores, doc_ids = self.index.search(embs, top_k)
        results = []
        for row_ids, row_scores in zip(doc_ids, scores):
            row_ids = row_ids.tolist()
            docs = self._fetch(row_ids)
            results.append(
                [{"title": docs[d]["title"], "text": docs[d]["text"], "score": float(s)}
                 for d, s in zip(row_ids, row_scores) if d in docs]
            )
        return results

    def _fetch(self, doc_ids):
        docs = {}
        for d in doc_ids:
            if d < 0:
                continue
            item = self.corpus[int(d)]
            raw = item.get("contents", "") or ""
            parts = raw.split("\n", 1)
            first = parts[0].strip().strip('"')
            text = (parts[1] if len(parts) > 1 else "").strip()[:self.text_truncate]
            docs[d] = {"title": first, "text": text}
        return docs


# ─────────── HTTP 层 ───────────
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class SearchBatchRequest(BaseModel):
    queries: list[str]
    top_k: int = 3


def build_app(retriever: BGEFaissRetriever) -> FastAPI:
    app = FastAPI()
    # GPU 前向需串行化：多个 rollout 并发请求若同时跑 BGE encode + FAISS search，
    # 会争抢同一张卡显存 / faiss 内部状态。用一把锁把整个检索串起来，单次延迟很低。
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", "n_vectors": retriever.index.ntotal,
                "n_docs": len(retriever.corpus), "text_truncate": retriever.text_truncate}

    @app.post("/search")
    def search(req: SearchRequest):
        with lock:
            docs = retriever.search_batch([req.query], top_k=req.top_k)[0]
        return {"results": docs}

    @app.post("/search_batch")
    def search_batch(req: SearchBatchRequest):
        with lock:
            results = retriever.search_batch(req.queries, top_k=req.top_k)
        return {"results": results}

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--text_truncate", type=int, default=500,
                   help="正文截断长度，必须与 eval / SFT 训练数据口径一致（默认 500）")
    p.add_argument("--bge_path", default=str(BGE_PATH))
    p.add_argument("--index_path", default=str(INDEX_PATH))
    p.add_argument("--corpus_path", default=str(CORPUS_PATH))
    args = p.parse_args()

    # 端口预检：被占就立即退出，避免重复加载 68GB 索引
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((args.host, args.port))
        except OSError as e:
            print(f"[daemon] port {args.host}:{args.port} already in use ({e}); abort.",
                  file=sys.stderr, flush=True)
            sys.exit(2)

    retriever = BGEFaissRetriever(
        args.bge_path, args.index_path, args.corpus_path,
        device=args.device, text_truncate=args.text_truncate,
    )
    app = build_app(retriever)
    print(f"[daemon] ready, serving on {args.host}:{args.port}", flush=True)
    # workers=1：单进程持有索引；并发由 uvicorn 的 async + 内部锁处理
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")


if __name__ == "__main__":
    main()
