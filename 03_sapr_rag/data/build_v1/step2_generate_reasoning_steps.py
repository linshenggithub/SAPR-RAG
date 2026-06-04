#!/usr/bin/env python3
"""SAPR-R v1 — Step 2: 用 DeepSeek 把 HotpotQA train question 拆解为 reasoning_steps。

输入:
    HotpotQA train jsonl（FlashRAG 格式）
    每行包含: id / question / golden_answers / metadata.supporting_facts (list[[title, sent_idx]])

输出:
    {out_dir}/reasoning_steps.jsonl     每行一个 question 的 reasoning_steps（DeepSeek 直出）
    {out_dir}/run_meta.json             跑次元数据（时间/采样/统计/预估成本）

每条输出格式:
    {
      "qid": "5a8b57f25542995d1e6f1371",
      "question": "...",
      "gt_answer": "...",
      "supporting_titles": ["...", "..."],
      "reasoning_steps": [
        {"subquery": "...", "subject_entity": "...", "thought": "...", "step_gold": "..."},
        ...
      ],
      "raw_response": <DeepSeek 原始 JSON 文本，便于复盘>,
      "ok": true|false,
      "error": null | "..."
    }

Resumable:
    若 reasoning_steps.jsonl 已存在，会读取已完成 qid 集合，跳过重复调用，
    继续追加未完成 question 的结果。

CLI:
    python -m 03_sapr_rag.data.build_v1.step2_generate_reasoning_steps \
        --n-samples 5000 --seed 42 \
        --max-workers 30 --out-dir 03_sapr_rag/data/build_v1/out/v1_5k

依赖:
    - DEEPSEEK_API_KEY 或 DMXAPI_API_KEY 已 export
    - SAPR_HOTPOTQA_TRAIN_PATH 已 export（通过 source config/env_*.sh）
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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

build_step2_messages = _PROMPTS.build_step2_messages
DeepSeekClient = _DEEPSEEK.DeepSeekClient

from config.paths import HOTPOTQA_TRAIN_PATH  # noqa: E402  (sys.path injected above)


logger = logging.getLogger("sapr_v1.step2")


# ----------------------- IO helpers -----------------------

def load_hotpotqa_train(path: Path, n_samples: int, seed: int) -> List[Dict[str, Any]]:
    """读 FlashRAG jsonl 并随机抽 n_samples 条。

    每行 schema:
      {"id": str, "question": str, "golden_answers": [str], "metadata": {...}}
    metadata.supporting_facts 通常是 list[[title, sent_idx]]
    """
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    rng = random.Random(seed)
    if n_samples >= len(items):
        logger.info(
            "n_samples=%d >= total=%d, using all", n_samples, len(items)
        )
        return items
    return rng.sample(items, n_samples)


def extract_supporting_titles(item: Dict[str, Any]) -> List[str]:
    """从 metadata.supporting_facts 抽 unique titles，保持出现顺序。"""
    metadata = item.get("metadata") or {}
    facts = metadata.get("supporting_facts") or []
    seen: set = set()
    titles: List[str] = []
    for entry in facts:
        if isinstance(entry, (list, tuple)) and entry:
            title = str(entry[0]).strip()
        elif isinstance(entry, dict):
            title = str(entry.get("title", "")).strip()
        else:
            continue
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def golden_answer(item: Dict[str, Any]) -> str:
    answers = item.get("golden_answers") or []
    if not answers:
        return ""
    return str(answers[0]).strip()


def load_completed_qids(out_path: Path) -> set:
    """断点续跑：读已存在 jsonl 收集 qid 集合。"""
    if not out_path.exists():
        return set()
    qids: set = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = obj.get("qid")
                if qid:
                    qids.add(qid)
            except json.JSONDecodeError:
                continue
    return qids


# ----------------------- output validation -----------------------

@dataclass
class StepValidation:
    ok: bool
    reason: str = ""
    cleaned_steps: List[Dict[str, Any]] = field(default_factory=list)


def _word_count(s: str) -> int:
    return len((s or "").split())


def validate_reasoning_steps(obj: Any) -> StepValidation:
    """对 DeepSeek 返回的 JSON 做 schema / 长度校验。

    放宽策略：长度超限只警告，不丢弃；schema 不全才标 ok=False。
    """
    if not isinstance(obj, dict):
        return StepValidation(ok=False, reason="root_not_object")
    steps = obj.get("reasoning_steps")
    if not isinstance(steps, list) or not steps:
        return StepValidation(ok=False, reason="missing_reasoning_steps_list")
    if len(steps) > 4:
        return StepValidation(ok=False, reason=f"too_many_steps={len(steps)}")

    cleaned: List[Dict[str, Any]] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            return StepValidation(ok=False, reason=f"step{i}_not_object")
        for k in ("subquery", "subject_entity", "thought", "step_gold"):
            v = step.get(k)
            if not isinstance(v, str) or not v.strip():
                return StepValidation(ok=False, reason=f"step{i}_missing_{k}")
        cleaned.append({
            "subquery": step["subquery"].strip(),
            "subject_entity": step["subject_entity"].strip(),
            "thought": step["thought"].strip(),
            "step_gold": step["step_gold"].strip(),
        })

    # 长度警告（不致命）
    for i, s in enumerate(cleaned):
        if _word_count(s["subquery"]) > 20:
            logger.debug("step%d subquery exceeds 15 words: %s", i, s["subquery"])
        if _word_count(s["thought"]) > 30:
            logger.debug("step%d thought exceeds 25 words: %s", i, s["thought"])
        if _word_count(s["step_gold"]) > 8:
            logger.debug("step%d step_gold exceeds 6 words: %s", i, s["step_gold"])

    return StepValidation(ok=True, cleaned_steps=cleaned)


# ----------------------- main pipeline -----------------------

@dataclass
class RunMeta:
    started_at: str = ""
    finished_at: str = ""
    elapsed_sec: float = 0.0
    n_samples_requested: int = 0
    n_samples_processed: int = 0
    n_succeeded: int = 0
    n_failed: int = 0
    n_skipped_resumed: int = 0
    seed: int = 42
    max_workers: int = 20
    out_dir: str = ""
    hotpotqa_train_path: str = ""
    deepseek_provider: str = ""
    deepseek_model: str = ""
    api_stats: Dict[str, int] = field(default_factory=dict)


def _build_inputs(
    items: List[Dict[str, Any]],
    completed: set,
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, str]]]]:
    """把 raw items 转成 (todo_items, prompts) 双列表，跳过已完成 qid。"""
    todo_items: List[Dict[str, Any]] = []
    prompts: List[List[Dict[str, str]]] = []
    for item in items:
        qid = item.get("id")
        if qid is None:
            continue
        if qid in completed:
            continue
        question = (item.get("question") or "").strip()
        if not question:
            continue
        gt = golden_answer(item)
        if not gt:
            continue
        titles = extract_supporting_titles(item)
        msgs = build_step2_messages(question, gt, titles)
        todo_items.append({
            "qid": qid,
            "question": question,
            "gt_answer": gt,
            "supporting_titles": titles,
        })
        prompts.append(msgs)
    return todo_items, prompts


def main() -> int:
    parser = argparse.ArgumentParser(description="SAPR-R v1 step2 — generate reasoning_steps with DeepSeek")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=20,
                        help="DeepSeek API 并发线程数（建议 ≤50）")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path,
                        default=_REPO_ROOT / "03_sapr_rag" / "data" / "build_v1" / "out" / "v1_default")
    parser.add_argument("--hotpotqa-train-path", type=Path, default=None,
                        help="覆盖 SAPR_HOTPOTQA_TRAIN_PATH（用于本地小批量调试）")
    parser.add_argument("--prefer", type=str, default="deepseek",
                        choices=["deepseek", "dmxapi"],
                        help="API provider 优先级")
    parser.add_argument("--limit-debug", type=int, default=0,
                        help="只跑前 N 条做 debug，0 表示按 n-samples 跑")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    train_path = args.hotpotqa_train_path or Path(str(HOTPOTQA_TRAIN_PATH))
    if not train_path.exists():
        raise FileNotFoundError(f"HotpotQA train jsonl not found: {train_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.out_dir / "reasoning_steps.jsonl"
    meta_path = args.out_dir / "run_meta.json"

    # --- 加载 + 抽样 ---
    logger.info("loading HotpotQA train from %s", train_path)
    items = load_hotpotqa_train(train_path, args.n_samples, args.seed)
    if args.limit_debug > 0:
        items = items[: args.limit_debug]
    logger.info("sampled %d items (seed=%d)", len(items), args.seed)

    # --- 断点续跑 ---
    completed = load_completed_qids(out_jsonl)
    if completed:
        logger.info("found %d already-completed qids; resuming", len(completed))

    todo_items, prompts = _build_inputs(items, completed)
    logger.info("todo=%d, skipped=%d", len(todo_items), len(items) - len(todo_items))

    # --- DeepSeek client ---
    client = DeepSeekClient.from_env(
        env_path=_REPO_ROOT / "03_sapr_rag" / ".env",
        prefer=args.prefer,
    )
    logger.info(
        "DeepSeek client ready: provider_url=%s model=%s",
        client.config.base_url, client.config.model,
    )

    # --- 调用 + 流式落盘 ---
    meta = RunMeta(
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        n_samples_requested=len(items),
        n_samples_processed=len(todo_items),
        n_skipped_resumed=len(items) - len(todo_items),
        seed=args.seed,
        max_workers=args.max_workers,
        out_dir=str(args.out_dir),
        hotpotqa_train_path=str(train_path),
        deepseek_model=client.config.model,
        deepseek_provider=client.config.base_url,
    )

    started = time.time()
    write_lock = Lock()

    if not todo_items:
        logger.info("nothing to do, all qids already completed")
        meta.elapsed_sec = 0.0
        meta.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        meta_path.write_text(json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8")
        return 0

    # 用 chat_batch（拿原始 string），自己解析 + 校验，方便保留 raw_response
    progress_state = {"done": 0, "ok": 0, "fail": 0}

    def _progress(done: int, total: int) -> None:
        progress_state["done"] = done
        if done % max(1, total // 50) == 0 or done == total:
            logger.info(
                "progress %d/%d (ok=%d fail=%d)",
                done, total, progress_state["ok"], progress_state["fail"],
            )

    raw_outputs = client.chat_batch(
        prompts,
        max_workers=args.max_workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        response_format={"type": "json_object"},
        on_failure="skip",
        progress_cb=_progress,
    )

    # --- 解析 + 写盘 ---
    with out_jsonl.open("a", encoding="utf-8") as fout:
        for item, raw in zip(todo_items, raw_outputs):
            record: Dict[str, Any] = {
                "qid": item["qid"],
                "question": item["question"],
                "gt_answer": item["gt_answer"],
                "supporting_titles": item["supporting_titles"],
                "reasoning_steps": [],
                "raw_response": raw or "",
                "ok": False,
                "error": None,
            }

            if raw is None:
                record["error"] = "api_failed_after_retry"
                progress_state["fail"] += 1
            else:
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as e:
                    record["error"] = f"json_decode: {e}"
                    progress_state["fail"] += 1
                else:
                    val = validate_reasoning_steps(obj)
                    if not val.ok:
                        record["error"] = f"schema: {val.reason}"
                        progress_state["fail"] += 1
                    else:
                        record["reasoning_steps"] = val.cleaned_steps
                        record["ok"] = True
                        progress_state["ok"] += 1

            with write_lock:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

    # --- meta + 汇总 ---
    meta.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
    meta.elapsed_sec = round(time.time() - started, 2)
    meta.n_succeeded = progress_state["ok"]
    meta.n_failed = progress_state["fail"]
    meta.api_stats = asdict(client.stats)

    meta_path.write_text(json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "DONE  ok=%d  fail=%d  elapsed=%.1fs  prompt_tok=%d  completion_tok=%d",
        meta.n_succeeded, meta.n_failed, meta.elapsed_sec,
        client.stats.prompt_tokens, client.stats.completion_tokens,
    )
    logger.info("wrote: %s", out_jsonl)
    logger.info("meta:  %s", meta_path)
    return 0 if meta.n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
