#!/usr/bin/env python3
"""SAPR-R v1 — Step 4: 用 DeepSeek 给每个 (state, doc) 打 cls label + evidence。

输入:
    {in_dir}/candidates.jsonl    step3 产出，每行 K 个 candidates

输出:
    {out_dir}/cls_labels.jsonl   每行一个 (qid, step_idx, doc_id) 三元组
    {out_dir}/run_meta_step4.json

每行 schema:
    {
      "qid": "...",
      "step_idx": 0,
      "doc_id": 12345,
      "doc_title": "...",
      "cls_label": 0|1,
      "evidence": "...",
      "raw_response": "...",
      "ok": true|false,
      "error": null|"..."
    }

Resumable:
    若 cls_labels.jsonl 已存在，按 (qid, step_idx, doc_id) 复合 key 跳过已完成。

跑法（5090）:
    source config/env_5090.sh
    conda activate reasonrag
    python 03_sapr_rag/data/build_v1/step4_label_cls.py \
        --in-dir 03_sapr_rag/data/build_v1/out/v1_5k \
        --max-workers 50 \
        --chunk-size 2000

成本预估:
    5k question × 平均 2.5 step × 10 doc ≈ 125k 调用
    DeepSeek-V3 ~¥150-200，~2-3 小时（50 并发）

为什么分 chunk:
    chat_batch 一次返回所有结果才能落盘；125k 调用一次进 RAM 风险大且崩溃丢全部。
    每 chunk_size 个 pair 一次 chat_batch → 写盘 → flush，崩溃最多丢一个 chunk。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

# --- 仓内路径派生（按 docs/coding_standard.md §2 规范）---
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]   # SAPR-RAG/
sys.path.insert(0, str(_REPO_ROOT))

# 包名以数字开头，无法常规 import，用 importlib 间接导入
import importlib.util


def _load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass / typing 需要从 sys.modules 反查模块
    spec.loader.exec_module(mod)
    return mod


_PROMPTS = _load_module(
    _REPO_ROOT / "03_sapr_rag" / "data" / "build_v1" / "prompts.py",
    "_v1_prompts",
)
_DEEPSEEK = _load_module(
    _REPO_ROOT / "03_sapr_rag" / "utils" / "deepseek_client.py",
    "_v1_deepseek_client",
)

build_step4_messages = _PROMPTS.build_step4_messages
DeepSeekClient = _DEEPSEEK.DeepSeekClient


logger = logging.getLogger("sapr_v1.step4")


# ----------------------- IO helpers -----------------------

@dataclass
class LabelTask:
    """一个 (state, doc) pair 待打标。"""
    qid: str
    step_idx: int
    doc_id: int
    doc_title: str
    doc_text: str
    # state context（用于构 prompt）
    question: str
    prior_thoughts: List[str]
    subquery: str
    subject_entity: str
    step_gold: str


def load_candidates(in_path: Path) -> List[LabelTask]:
    """读 step3 candidates.jsonl，flatten 成 (qid, step_idx, doc_id) tasks。"""
    tasks: List[LabelTask] = []
    n_units = 0
    n_empty_cand = 0
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n_units += 1
            cands = obj.get("candidates") or []
            if not cands:
                n_empty_cand += 1
                continue
            qid = obj.get("qid")
            step_idx = int(obj.get("step_idx", 0))
            question = obj.get("question", "")
            subquery = obj.get("subquery", "")
            subject_entity = obj.get("subject_entity", "")
            step_gold = obj.get("step_gold", "")
            prior_thoughts = obj.get("prior_thoughts") or []
            for c in cands:
                tasks.append(LabelTask(
                    qid=qid,
                    step_idx=step_idx,
                    doc_id=int(c.get("doc_id", -1)),
                    doc_title=c.get("title", ""),
                    doc_text=c.get("text", ""),
                    question=question,
                    prior_thoughts=list(prior_thoughts),
                    subquery=subquery,
                    subject_entity=subject_entity,
                    step_gold=step_gold,
                ))
    logger.info(
        "loaded %d (qid, step_idx) units (skipped %d empty); flattened to %d (qid, step_idx, doc_id) tasks",
        n_units, n_empty_cand, len(tasks),
    )
    return tasks


def load_completed_keys(out_path: Path) -> set:
    """断点续跑：读已存在 cls_labels.jsonl 收集 (qid, step_idx, doc_id) 集合。"""
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
                doc_id = obj.get("doc_id")
                if qid is not None and step_idx is not None and doc_id is not None:
                    keys.add((qid, int(step_idx), int(doc_id)))
            except json.JSONDecodeError:
                continue
    return keys


# ----------------------- output validation -----------------------

@dataclass
class LabelValidation:
    ok: bool
    reason: str = ""
    cls_label: Optional[int] = None
    evidence: str = ""


def validate_label_response(obj: Any) -> LabelValidation:
    """对 DeepSeek 返回的 JSON 做 schema 校验。

    期望: {"label": 0|1, "evidence": "..."}（evidence 可为空字符串，label=0 时尤甚）
    """
    if not isinstance(obj, dict):
        return LabelValidation(ok=False, reason="root_not_object")
    label = obj.get("label")
    if label is None or label not in (0, 1, "0", "1", True, False):
        return LabelValidation(ok=False, reason=f"bad_label={label!r}")
    label_int = int(bool(label)) if isinstance(label, bool) else int(label)
    if label_int not in (0, 1):
        return LabelValidation(ok=False, reason=f"label_out_of_range={label_int}")
    evidence = obj.get("evidence", "")
    if not isinstance(evidence, str):
        return LabelValidation(ok=False, reason=f"evidence_not_str={type(evidence).__name__}")
    return LabelValidation(ok=True, cls_label=label_int, evidence=evidence.strip())


# ----------------------- main pipeline -----------------------

@dataclass
class RunMeta:
    started_at: str = ""
    finished_at: str = ""
    elapsed_sec: float = 0.0
    n_tasks_total: int = 0
    n_tasks_processed: int = 0
    n_tasks_skipped_resumed: int = 0
    n_succeeded: int = 0
    n_failed: int = 0
    n_pos: int = 0
    n_neg: int = 0
    pos_ratio: float = 0.0
    chunk_size: int = 2000
    max_workers: int = 50
    in_jsonl: str = ""
    out_jsonl: str = ""
    deepseek_model: str = ""
    deepseek_provider: str = ""
    api_stats: Dict[str, int] = field(default_factory=dict)


def _build_chunk_prompts(chunk: List[LabelTask]) -> List[List[Dict[str, str]]]:
    return [
        build_step4_messages(
            question=t.question,
            prior_thoughts=t.prior_thoughts,
            subquery=t.subquery,
            subject_entity=t.subject_entity,
            step_gold=t.step_gold,
            doc_title=t.doc_title,
            doc_text=t.doc_text,
        )
        for t in chunk
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAPR-R v1 step4 — label cls + evidence per (state, doc) via DeepSeek",
    )
    parser.add_argument("--in-dir", type=Path, required=True,
                        help="step3 输出目录（含 candidates.jsonl）")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="step4 输出目录（默认 = in-dir）")
    parser.add_argument("--max-workers", type=int, default=50,
                        help="DeepSeek API 并发线程数")
    parser.add_argument("--chunk-size", type=int, default=2000,
                        help="每块 pair 数；一块完成即写盘 flush 一次")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prefer", type=str, default="deepseek",
                        choices=["deepseek", "dmxapi"])
    parser.add_argument("--limit-debug", type=int, default=0,
                        help="只跑前 N 个 task 做 debug，0 = 全量")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    in_dir = args.in_dir.resolve()
    out_dir = (args.out_dir or in_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    in_jsonl = in_dir / "candidates.jsonl"
    out_jsonl = out_dir / "cls_labels.jsonl"
    meta_path = out_dir / "run_meta_step4.json"

    if not in_jsonl.exists():
        raise FileNotFoundError(f"step3 output not found: {in_jsonl}")

    # --- 1. load tasks + dedup against existing output ---
    tasks = load_candidates(in_jsonl)
    if args.limit_debug > 0:
        tasks = tasks[: args.limit_debug]

    completed = load_completed_keys(out_jsonl)
    if completed:
        logger.info("found %d already-completed (qid, step_idx, doc_id); resuming", len(completed))
    todo = [t for t in tasks if (t.qid, t.step_idx, t.doc_id) not in completed]
    logger.info("todo=%d, skipped_resumed=%d", len(todo), len(tasks) - len(todo))

    # --- 2. DeepSeek client ---
    client = DeepSeekClient.from_env(
        env_path=_REPO_ROOT / "03_sapr_rag" / ".env",
        prefer=args.prefer,
    )
    logger.info(
        "DeepSeek client ready: provider_url=%s model=%s",
        client.config.base_url, client.config.model,
    )

    meta = RunMeta(
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        n_tasks_total=len(tasks),
        n_tasks_skipped_resumed=len(tasks) - len(todo),
        chunk_size=args.chunk_size,
        max_workers=args.max_workers,
        in_jsonl=str(in_jsonl),
        out_jsonl=str(out_jsonl),
        deepseek_model=client.config.model,
        deepseek_provider=client.config.base_url,
    )

    if not todo:
        logger.info("nothing to do, all tasks already completed")
        meta.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        meta_path.write_text(json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8")
        return 0

    # --- 3. chunked chat_batch + 流式落盘 ---
    started = time.time()
    write_lock = Lock()
    n_total_done = 0
    n_total_ok = 0
    n_total_fail = 0
    n_pos = 0
    n_neg = 0

    n_chunks = (len(todo) + args.chunk_size - 1) // args.chunk_size

    with out_jsonl.open("a", encoding="utf-8") as fout:
        for ci in range(n_chunks):
            chunk = todo[ci * args.chunk_size:(ci + 1) * args.chunk_size]
            prompts = _build_chunk_prompts(chunk)

            chunk_progress = {"done": 0}

            def _progress(done: int, total: int) -> None:
                chunk_progress["done"] = done

            t0 = time.time()
            raw_outputs = client.chat_batch(
                prompts,
                max_workers=args.max_workers,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                response_format={"type": "json_object"},
                on_failure="skip",
                progress_cb=_progress,
            )
            chunk_elapsed = time.time() - t0

            chunk_ok = 0
            chunk_fail = 0
            for task, raw in zip(chunk, raw_outputs):
                record: Dict[str, Any] = {
                    "qid": task.qid,
                    "step_idx": task.step_idx,
                    "doc_id": task.doc_id,
                    "doc_title": task.doc_title,
                    "cls_label": None,
                    "evidence": "",
                    "raw_response": raw or "",
                    "ok": False,
                    "error": None,
                }
                if raw is None:
                    record["error"] = "api_failed_after_retry"
                    chunk_fail += 1
                else:
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError as e:
                        record["error"] = f"json_decode: {e}"
                        chunk_fail += 1
                    else:
                        val = validate_label_response(obj)
                        if not val.ok:
                            record["error"] = f"schema: {val.reason}"
                            chunk_fail += 1
                        else:
                            record["cls_label"] = val.cls_label
                            record["evidence"] = val.evidence
                            record["ok"] = True
                            chunk_ok += 1
                            if val.cls_label == 1:
                                n_pos += 1
                            else:
                                n_neg += 1

                with write_lock:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            fout.flush()
            n_total_done += len(chunk)
            n_total_ok += chunk_ok
            n_total_fail += chunk_fail

            logger.info(
                "chunk %d/%d done in %.1fs  ok=%d fail=%d  total=%d/%d  pos_ratio=%.1f%%",
                ci + 1, n_chunks, chunk_elapsed, chunk_ok, chunk_fail,
                n_total_done, len(todo),
                100 * n_pos / max(1, n_pos + n_neg),
            )

    # --- 4. meta ---
    meta.n_tasks_processed = n_total_done
    meta.n_succeeded = n_total_ok
    meta.n_failed = n_total_fail
    meta.n_pos = n_pos
    meta.n_neg = n_neg
    meta.pos_ratio = (n_pos / (n_pos + n_neg)) if (n_pos + n_neg) else 0.0
    meta.elapsed_sec = round(time.time() - started, 2)
    meta.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
    meta.api_stats = asdict(client.stats)

    meta_path.write_text(json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "DONE  ok=%d  fail=%d  pos=%d (%.1f%%)  neg=%d  elapsed=%.1fs  prompt_tok=%d  completion_tok=%d",
        meta.n_succeeded, meta.n_failed, meta.n_pos, 100 * meta.pos_ratio, meta.n_neg,
        meta.elapsed_sec, client.stats.prompt_tokens, client.stats.completion_tokens,
    )
    logger.info("wrote: %s", out_jsonl)
    logger.info("meta:  %s", meta_path)
    return 0 if meta.n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
