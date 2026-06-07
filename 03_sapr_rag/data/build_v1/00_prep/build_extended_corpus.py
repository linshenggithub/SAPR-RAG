"""build_extended_corpus.py — 拼接 wiki18_100w + reasonrag/RAG_extend_corpus → wiki18_extended.jsonl

输入（由 download_assets.sh 落盘）：
    data/raw/wiki18_100w.jsonl                 # FlashRAG 标准格式，每行 {id, contents}
    data/raw/RAG_extend_corpus/<parquet|jsonl>  # HuggingFace dataset，列 (id, title, contents)

输出：
    data/corpus/wiki18_extended.jsonl          # 每行 {id: <new_seq>, contents: "title\\n正文"}
                                               # doc_id == 行号（0-indexed），与 step3 fetch_corpus_lines 约定一致

设计要点：
- 不依赖 datasets 库，纯 stdlib 流式读写，避免 22M 段全量加载内存
- 自动检测 RAG_extend_corpus 子目录里的实际文件格式（parquet / jsonl 都能读）
- 输出后做完整性校验：行数 ≈ 21M + 1.34M ≈ 22.34M

用法：
    source config/env_local.sh
    python 03_sapr_rag/data/build_v1/00_prep/build_extended_corpus.py

重跑安全：会先检查输出文件是否完整（行数对得上），完整就跳过。
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# 从 wiki18 对比来看 RAG_extend_corpus 行数的合理范围（用作完整性校验，不强卡）
EXPECTED_BASE_MIN = 20_000_000
EXPECTED_EXTEND_MIN = 1_000_000


def _normalize_contents(title: str, text: str) -> str:
    """统一 FlashRAG 风格：contents = "title\\n正文"。
    step3.fetch_corpus_lines 用 split('\\n', 1) 拿首行 = title，剩下 = text。
    """
    title = (title or "").strip().strip('"').replace("\n", " ")
    text = (text or "").strip()
    if not title:
        return text
    return f"{title}\n{text}"


def _iter_wiki18_100w(path: Path):
    """yield (title, text) from FlashRAG wiki18_100w.jsonl.
    每行：{"id": "...", "contents": "title\\n正文"}
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw = obj.get("contents", obj.get("text", ""))
            head, _, body = raw.partition("\n")
            yield head.strip().strip('"'), body.strip()


def _iter_extend_corpus(extend_dir: Path):
    """yield (title, text) from reasonrag/RAG_extend_corpus.

    实际文件格式可能是 parquet（HuggingFace 默认）或 jsonl。优先 parquet 走 pyarrow，
    否则 fallback 到逐 jsonl 文件。
    Schema：(id, title, contents)；contents 已是正文（不重复 title）。
    """
    parquet_files = sorted(extend_dir.rglob("*.parquet"))
    jsonl_files = sorted(extend_dir.rglob("*.jsonl"))

    if parquet_files:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError(
                "RAG_extend_corpus is in parquet format but pyarrow is not installed. "
                "Run: pip install pyarrow"
            ) from e
        logger.info("reading %d parquet shard(s) from %s", len(parquet_files), extend_dir)
        for shard in parquet_files:
            table = pq.read_table(shard, columns=["title", "contents"])
            titles = table.column("title").to_pylist()
            contents = table.column("contents").to_pylist()
            for t, c in zip(titles, contents):
                yield t or "", c or ""
        return

    if jsonl_files:
        logger.info("reading %d jsonl file(s) from %s", len(jsonl_files), extend_dir)
        for f in jsonl_files:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    yield obj.get("title", ""), obj.get("contents", obj.get("text", ""))
        return

    raise RuntimeError(
        f"No parquet or jsonl files found under {extend_dir}. "
        f"Did download_assets.sh complete successfully?"
    )


def _count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki18-path", type=Path,
                        help="wiki18_100w.jsonl 路径（默认从仓内 data/raw 找）")
    parser.add_argument("--extend-dir", type=Path,
                        help="RAG_extend_corpus 目录（默认从仓内 data/raw 找）")
    parser.add_argument("--out-path", type=Path,
                        help="输出 wiki18_extended.jsonl 路径（默认 data/corpus/wiki18_extended.jsonl）")
    parser.add_argument("--limit-base", type=int, default=0,
                        help="只读 wiki18_100w 前 N 行做 smoke（0 = 全量）")
    parser.add_argument("--limit-extend", type=int, default=0,
                        help="只读 extend_corpus 前 N 行做 smoke（0 = 全量）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")

    repo_root = Path(__file__).resolve().parents[4]
    wiki18 = args.wiki18_path or (repo_root / "data" / "raw" / "wiki18_100w.jsonl")
    extend_dir = args.extend_dir or (repo_root / "data" / "raw" / "RAG_extend_corpus")
    out_path = args.out_path or (repo_root / "data" / "corpus" / "wiki18_extended.jsonl")

    if not wiki18.exists():
        raise FileNotFoundError(f"wiki18_100w.jsonl not found at {wiki18}; run download_assets.sh first")
    if not extend_dir.exists():
        raise FileNotFoundError(f"RAG_extend_corpus dir not found at {extend_dir}; run download_assets.sh first")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 重跑短路：输出已存在且行数 ≥ 阈值就直接复用
    if out_path.exists() and not (args.limit_base or args.limit_extend):
        existing = _count_lines(out_path)
        if existing >= EXPECTED_BASE_MIN + EXPECTED_EXTEND_MIN:
            logger.info("[skip] %s already has %d lines (>= %d), reuse",
                        out_path, existing, EXPECTED_BASE_MIN + EXPECTED_EXTEND_MIN)
            return
        logger.warning("[redo] %s has only %d lines, will overwrite", out_path, existing)

    t0 = time.time()
    n_base = 0
    n_extend = 0
    n_skipped_empty = 0
    last_log = t0

    with open(out_path, "w", encoding="utf-8") as out:
        # ---- 1. base: wiki18_100w ----
        logger.info("phase 1/2: streaming wiki18_100w from %s ...", wiki18)
        for title, text in _iter_wiki18_100w(wiki18):
            if args.limit_base and n_base >= args.limit_base:
                break
            contents = _normalize_contents(title, text)
            if not contents.strip():
                n_skipped_empty += 1
                continue
            doc_id = n_base + n_extend  # 全局递增
            out.write(json.dumps(
                {"id": str(doc_id), "contents": contents}, ensure_ascii=False) + "\n")
            n_base += 1
            if time.time() - last_log > 30:
                logger.info("  base progress: %d (elapsed %.1fs)", n_base, time.time() - t0)
                last_log = time.time()

        # ---- 2. extension: RAG_extend_corpus ----
        logger.info("phase 2/2: streaming RAG_extend_corpus from %s ...", extend_dir)
        for title, text in _iter_extend_corpus(extend_dir):
            if args.limit_extend and n_extend >= args.limit_extend:
                break
            contents = _normalize_contents(title, text)
            if not contents.strip():
                n_skipped_empty += 1
                continue
            doc_id = n_base + n_extend
            out.write(json.dumps(
                {"id": str(doc_id), "contents": contents}, ensure_ascii=False) + "\n")
            n_extend += 1
            if time.time() - last_log > 30:
                logger.info("  extend progress: %d (elapsed %.1fs)", n_extend, time.time() - t0)
                last_log = time.time()

    elapsed = time.time() - t0
    total = n_base + n_extend
    logger.info("[done] wrote %d lines to %s (base=%d, extend=%d, skipped_empty=%d, %.1fs)",
                total, out_path, n_base, n_extend, n_skipped_empty, elapsed)

    # 完整性校验
    if not (args.limit_base or args.limit_extend):
        if n_base < EXPECTED_BASE_MIN:
            logger.warning("[warn] base lines %d < expected min %d", n_base, EXPECTED_BASE_MIN)
        if n_extend < EXPECTED_EXTEND_MIN:
            logger.warning("[warn] extend lines %d < expected min %d", n_extend, EXPECTED_EXTEND_MIN)

    # 顺手写个 meta 方便审计
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "n_base": n_base, "n_extend": n_extend, "n_total": total,
        "n_skipped_empty": n_skipped_empty, "elapsed_sec": elapsed,
        "wiki18_path": str(wiki18), "extend_dir": str(extend_dir),
        "out_path": str(out_path),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("meta written to %s", meta_path)


if __name__ == "__main__":
    main()
