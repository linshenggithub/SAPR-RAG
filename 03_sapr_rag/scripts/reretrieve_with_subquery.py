#!/usr/bin/env python3
"""
Reretrieve evidence decision points using inferred_subquery.

Reads the original evidence_decision_points.jsonl (where retrieval_top10
came from empty-query searches), re-encodes each inferred_subquery with
BGE, searches the FAISS index for top-10, and replaces retrieval_top10
with the new results.

Optimized: only reads the ~1130 needed corpus lines instead of loading
all 21M documents.

Usage:
  conda activate reasonrag
  CUDA_VISIBLE_DEVICES=1 python reretrieve_with_subquery.py
"""

import os
import sys
import json
import time
import numpy as np

# 让脚本能直接 `python 03_sapr_rag/scripts/xxx.py` 运行：把仓库根加进 sys.path
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.paths import (  # noqa: E402
    REASONRAG_ROOT,
    WIKI_CORPUS_PATH,
    BGE_INDEX_PATH,
    BGE_MODEL_PATH,
)

# ── Paths ────────────────────────────────────────────────────────
INPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "../../04_experiments/logs/20260530_evidence_decision_top10_queryfix/"
    "evidence_decision_points.jsonl",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../04_experiments/logs/20260530_evidence_decision_top10_queryfix_reretrieved/",
)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evidence_decision_points.jsonl")

BGE_PATH = str(BGE_MODEL_PATH)
_INDEX_CANDIDATES = [
    str(REASONRAG_ROOT / "indexes/bge_extended/bge_Flat.index"),
    str(BGE_INDEX_PATH),
]
INDEX_PATH = next((p for p in _INDEX_CANDIDATES if os.path.exists(p)), _INDEX_CANDIDATES[0])
CORPUS_PATH = str(WIKI_CORPUS_PATH)
TOP_K = 10

# Resolve relative paths
INPUT_FILE = os.path.abspath(INPUT_FILE)
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
OUTPUT_FILE = os.path.abspath(OUTPUT_FILE)


def fetch_corpus_lines(corpus_path, doc_ids):
    """Read only the needed lines from corpus by line offset.

    corpus format: each line is a JSON object, doc_id == line number (0-indexed).
    Returns dict: doc_id -> {"title": str, "text": str}
    """
    needed = sorted(set(int(d) for d in doc_ids if d >= 0))
    print("  Fetching {} unique doc lines from corpus ...".format(len(needed)))

    result = {}
    needed_set = set(needed)
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
            if idx % 5000000 == 0 and idx > 0:
                # Progress indicator — scanning forward
                print("    scanned {}M lines, found {}/{} ..."
                      .format(idx // 1000000, len(result), len(needed)))

    print("  Fetched {}/{} docs".format(len(result), len(needed)))
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Reretrieve with inferred_subquery")
    print("=" * 70)
    print("Input : {}".format(INPUT_FILE))
    print("Output: {}".format(OUTPUT_FILE))
    print("BGE   : {}".format(BGE_PATH))
    print("Index : {}".format(INDEX_PATH))
    print("Corpus: {}".format(CORPUS_PATH))
    print("top_k : {}".format(TOP_K))
    print("=" * 70)

    # ── 1. Load original JSONL ───────────────────────────────────
    print("\n[1/5] Loading original decision points ...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    retrieval_steps = [d for d in data if d.get("step", -1) > 0]
    print("  Total points: {}, Retrieval steps: {}".format(len(data), len(retrieval_steps)))

    # Collect subqueries
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    subqueries = []
    step_indices = []
    for i, d in enumerate(data):
        if d.get("step", -1) > 0:
            sq = d.get("inferred_subquery", "").strip()
            if not sq:
                sq = d.get("original_question", "")
            # Add BGE query instruction prefix (matches FlashRAG's parse_query)
            subqueries.append(BGE_QUERY_PREFIX + sq)
            step_indices.append(i)
    print("  Subqueries to encode: {} (with BGE query prefix)".format(len(subqueries)))

    # ── 2. Encode subqueries ─────────────────────────────────────
    print("\n[2/5] Encoding subqueries with BGE ...")
    import faiss
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cpu"
    for gi in range(torch.cuda.device_count()):
        free_gb = torch.cuda.mem_get_info(gi)[0] / 1e9
        if free_gb > 2.0:
            device = "cuda:{}".format(gi)
            break
    print("  Device: {}".format(device))

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(BGE_PATH)
    model = AutoModel.from_pretrained(BGE_PATH).to(device).eval()
    print("  Model loaded ({:.1f}s)".format(time.time() - t0))

    t0 = time.time()
    all_embs = []
    bs = 64
    for start in range(0, len(subqueries), bs):
        batch = subqueries[start:start + bs]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        cls = torch.nn.functional.normalize(out.last_hidden_state[:, 0], p=2, dim=1)
        all_embs.append(cls.cpu().numpy())
    embeddings = np.ascontiguousarray(np.concatenate(all_embs).astype("float32"))
    print("  Encoded {} queries ({:.1f}s)".format(len(subqueries), time.time() - t0))

    # Free model GPU memory before loading FAISS index
    del model, tokenizer
    if device != "cpu":
        torch.cuda.empty_cache()
    print("  Encoder freed from GPU")

    # ── 3. Search FAISS ──────────────────────────────────────────
    print("\n[3/5] Searching FAISS index ...")
    idx_size_gb = os.path.getsize(INDEX_PATH) / 1e9
    print("  Loading index ({:.1f} GB) ...".format(idx_size_gb))
    t0 = time.time()
    index = faiss.read_index(INDEX_PATH)
    print("  Loaded ({:.1f}s), ntotal: {}".format(time.time() - t0, index.ntotal))

    t0 = time.time()
    scores, indices = index.search(embeddings, TOP_K)
    print("  Searched ({:.1f}s)".format(time.time() - t0))
    del index

    # ── 4. Fetch corpus lines ────────────────────────────────────
    print("\n[4/5] Fetching doc titles from corpus ...")
    all_doc_ids = indices.flatten()
    t0 = time.time()
    corpus_info = fetch_corpus_lines(CORPUS_PATH, all_doc_ids)
    print("  Done ({:.1f}s)".format(time.time() - t0))

    # ── 5. Build output JSONL ────────────────────────────────────
    print("\n[5/5] Writing reretrieved JSONL ...")
    new_data = [dict(d) for d in data]  # copy each row

    for j, data_idx in enumerate(step_indices):
        new_top10 = []
        for doc_id, score in zip(indices[j], scores[j]):
            if doc_id < 0:
                continue
            info = corpus_info.get(int(doc_id), {"title": "unknown", "text": ""})
            new_top10.append({
                "title": info["title"],
                "text": info["text"],
                "score": float(score),
            })
        new_data[data_idx]["retrieval_top10"] = new_top10
        new_data[data_idx]["reretrieved"] = True

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for d in new_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("  Written: {}".format(OUTPUT_FILE))

    # ── Quick validation ─────────────────────────────────────────
    import re
    def normalize_title(title):
        t = title.strip().lower()
        t = re.sub(r'[^a-z0-9\s]', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    n_gold_found = 0
    n_gold_total = 0
    for d in new_data:
        if d.get("step", -1) <= 0:
            continue
        sp = d.get("supporting_facts", {})
        gold = set(normalize_title(t) for t in sp.get("title", []))
        ret = set(normalize_title(doc["title"]) for doc in d.get("retrieval_top10", []))
        n_gold_found += len(gold & ret)
        n_gold_total += len(gold)

    print("\n  Gold doc recall (top-10): {}/{} ({:.1f}%)".format(
        n_gold_found, n_gold_total,
        100 * n_gold_found / n_gold_total if n_gold_total else 0))

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
