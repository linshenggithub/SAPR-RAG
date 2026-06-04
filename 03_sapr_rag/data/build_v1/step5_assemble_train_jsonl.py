#!/usr/bin/env python3
"""SAPR-R v1 — Step 5: 拼装训练 jsonl。

把 step3 candidates.jsonl（state + K 个 doc + retriever_score）
和  step4 cls_labels.jsonl（每个 doc 的 cls_label / evidence / ok）
按 (qid, step_idx, doc_id) join 起来，
按 (qid, step_idx) 分组算组内 listwise rank_target，
按 qid hash 切 train/dev，输出可直接喂 reranker 训练的 jsonl。

输入:
    {in_dir}/candidates.jsonl   step3 产出
    {in_dir}/cls_labels.jsonl   step4 产出

输出:
    {out_dir}/train.jsonl
    {out_dir}/dev.jsonl
    {out_dir}/run_meta_step5.json

每行 schema:
    {
      "qid": "...", "step_idx": 0, "split": "train"|"dev",
      "state": {
        "question": "...", "history_thoughts": [...],
        "subquery": "...", "subject_entity": "..."
      },
      "step_gold": "...",
      "candidates": [
        {"doc_id": 12345, "title": "...", "text": "...",
         "retriever_score": 0.92, "retriever_score_norm": 0.78,
         "cls_label": 1, "rank_target": 0.31},
        ... × K_kept
      ],
      "meta": {"k_raw": 10, "k_kept": 10, "n_pos": 4, "alpha": 0.7}
    }

跑法:
    python 03_sapr_rag/data/build_v1/step5_assemble_train_jsonl.py \
        --in-dir 03_sapr_rag/data/build_v1/out/v1_5k

Resumable:
    step5 是纯 join，没有外部调用。每次重跑会覆盖 train/dev jsonl。
    如想保留旧产物，请改 --out-dir 或手动备份。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("sapr_v1.step5")


# ----------------------- IO -----------------------

def load_candidates(in_path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """读 step3 candidates.jsonl，按 (qid, step_idx) 索引整 unit。"""
    units: Dict[Tuple[str, int], Dict[str, Any]] = {}
    n_dup = 0
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("qid")
            step_idx = int(obj.get("step_idx", 0))
            key = (qid, step_idx)
            if key in units:
                n_dup += 1
                continue  # 保留首次出现，后续重复忽略
            units[key] = obj
    if n_dup:
        logger.warning("candidates.jsonl: %d duplicate (qid, step_idx) ignored", n_dup)
    logger.info("loaded %d candidate units from %s", len(units), in_path.name)
    return units


def load_cls_labels(in_path: Path) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    """读 step4 cls_labels.jsonl，按 (qid, step_idx, doc_id) 索引。"""
    labels: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    n_dup = 0
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("qid")
            step_idx = int(obj.get("step_idx", 0))
            doc_id = int(obj.get("doc_id", -1))
            key = (qid, step_idx, doc_id)
            if key in labels:
                n_dup += 1
                continue
            labels[key] = obj
    if n_dup:
        logger.warning("cls_labels.jsonl: %d duplicate (qid, step_idx, doc_id) ignored", n_dup)
    logger.info("loaded %d cls labels from %s", len(labels), in_path.name)
    return labels


# ----------------------- math helpers -----------------------

def normalize_scores(scores: List[float], mode: str) -> List[float]:
    """组内归一。mode ∈ {minmax, zscore, none}。"""
    if not scores:
        return []
    if mode == "none":
        return list(scores)
    if mode == "minmax":
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            return [0.5] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]
    if mode == "zscore":
        n = len(scores)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(var)
        if std < 1e-9:
            return [0.0] * len(scores)
        return [(s - mean) / std for s in scores]
    raise ValueError(f"unknown norm mode: {mode}")


def softmax(values: List[float]) -> List[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    z = sum(exps)
    if z < 1e-12:
        return [1.0 / len(values)] * len(values)
    return [e / z for e in exps]


# ----------------------- split -----------------------

def assign_split(qid: str, dev_ratio: float, seed: int) -> str:
    """按 qid hash 决定 train/dev；同一 qid 所有 step 必落同一 split。"""
    h = hashlib.md5(f"{seed}:{qid}".encode("utf-8")).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return "dev" if frac < dev_ratio else "train"


# ----------------------- assembly -----------------------

@dataclass
class AssembleStats:
    n_units_total: int = 0
    n_units_kept: int = 0
    drop_missing_label: int = 0
    drop_label_failed: int = 0
    drop_too_few_kept: int = 0
    drop_all_negative: int = 0
    n_train: int = 0
    n_dev: int = 0
    n_pos_total: int = 0
    n_neg_total: int = 0
    n_docs_kept: int = 0
    avg_k_kept: float = 0.0
    avg_n_pos_per_group: float = 0.0
    pos_ratio: float = 0.0
    n_groups_all_negative: int = 0
    n_groups_all_positive: int = 0


def assemble_one_group(
    unit: Dict[str, Any],
    label_index: Dict[Tuple[str, int, int], Dict[str, Any]],
    alpha: float,
    norm_mode: str,
    min_k_kept: int,
    drop_all_negative: bool,
    stats: AssembleStats,
) -> Optional[Dict[str, Any]]:
    """处理一个 (qid, step_idx) 单元 → 一行训练样本（或 None 被丢弃）。"""
    qid = unit["qid"]
    step_idx = int(unit["step_idx"])
    raw_cands = unit.get("candidates") or []

    kept: List[Dict[str, Any]] = []
    for c in raw_cands:
        doc_id = int(c.get("doc_id", -1))
        lab = label_index.get((qid, step_idx, doc_id))
        if lab is None:
            stats.drop_missing_label += 1
            continue
        if not lab.get("ok"):
            stats.drop_label_failed += 1
            continue
        cls_label = lab.get("cls_label")
        if cls_label not in (0, 1):
            stats.drop_label_failed += 1
            continue
        kept.append({
            "doc_id": doc_id,
            "title": c.get("title", ""),
            "text": c.get("text", ""),
            "retriever_score": float(c.get("retriever_score", 0.0)),
            "cls_label": int(cls_label),
        })

    if len(kept) < min_k_kept:
        stats.drop_too_few_kept += 1
        return None

    n_pos = sum(d["cls_label"] for d in kept)
    if n_pos == 0:
        stats.n_groups_all_negative += 1
        if drop_all_negative:
            stats.drop_all_negative += 1
            return None
    if n_pos == len(kept):
        stats.n_groups_all_positive += 1

    rs = [d["retriever_score"] for d in kept]
    rs_norm = normalize_scores(rs, norm_mode)
    raw_target = [alpha * d["cls_label"] + (1.0 - alpha) * rs_norm[i]
                  for i, d in enumerate(kept)]
    rank_target = softmax(raw_target)

    candidates_out = []
    for i, d in enumerate(kept):
        candidates_out.append({
            "doc_id": d["doc_id"],
            "title": d["title"],
            "text": d["text"],
            "retriever_score": d["retriever_score"],
            "retriever_score_norm": round(rs_norm[i], 6),
            "cls_label": d["cls_label"],
            "rank_target": round(rank_target[i], 6),
        })

    return {
        "qid": qid,
        "step_idx": step_idx,
        "state": {
            "question": unit.get("question", ""),
            "history_thoughts": list(unit.get("prior_thoughts") or []),
            "subquery": unit.get("subquery", ""),
            "subject_entity": unit.get("subject_entity", ""),
        },
        "step_gold": unit.get("step_gold", ""),
        "candidates": candidates_out,
        "meta": {
            "k_raw": len(raw_cands),
            "k_kept": len(kept),
            "n_pos": n_pos,
            "alpha": alpha,
            "norm_mode": norm_mode,
        },
    }


# ----------------------- main -----------------------

@dataclass
class RunMeta:
    started_at: str = ""
    finished_at: str = ""
    elapsed_sec: float = 0.0
    in_dir: str = ""
    out_dir: str = ""
    candidates_jsonl: str = ""
    cls_labels_jsonl: str = ""
    train_jsonl: str = ""
    dev_jsonl: str = ""
    alpha: float = 0.7
    norm_mode: str = "minmax"
    dev_ratio: float = 0.1
    seed: int = 42
    min_k_kept: int = 2
    drop_all_negative: bool = True
    stats: Dict[str, Any] = field(default_factory=dict)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAPR-R v1 step5 — assemble train.jsonl / dev.jsonl from step3+step4 outputs",
    )
    parser.add_argument("--in-dir", type=Path, required=True,
                        help="目录含 candidates.jsonl + cls_labels.jsonl")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="train.jsonl / dev.jsonl 输出目录（默认 = in-dir）")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="rank_target 中 cls_label 的权重；retriever 权重 = 1-alpha")
    parser.add_argument("--norm-mode", type=str, default="minmax",
                        choices=["minmax", "zscore", "none"],
                        help="retriever_score 组内归一方式")
    parser.add_argument("--dev-ratio", type=float, default=0.1,
                        help="dev 切分比例，按 qid hash 划分")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-k-kept", type=int, default=2,
                        help="组内保留 doc 数下限；不足则丢整组")
    parser.add_argument("--keep-all-negative", action="store_true",
                        help="默认丢全 0 group（rank_target 退化为均匀分布无信号）；加此 flag 保留")
    parser.add_argument("--limit-debug", type=int, default=0,
                        help="只处理前 N 个 (qid, step_idx) 单元做 debug，0=全量")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    in_dir = args.in_dir.resolve()
    out_dir = (args.out_dir or in_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = in_dir / "candidates.jsonl"
    cls_labels_path = in_dir / "cls_labels.jsonl"
    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"
    meta_path = out_dir / "run_meta_step5.json"

    for p in (candidates_path, cls_labels_path):
        if not p.exists():
            raise FileNotFoundError(f"required input not found: {p}")

    started = time.time()
    cands = load_candidates(candidates_path)
    labels = load_cls_labels(cls_labels_path)

    keys = list(cands.keys())
    if args.limit_debug > 0:
        keys = keys[: args.limit_debug]

    stats = AssembleStats(n_units_total=len(keys))
    drop_all_negative = not args.keep_all_negative

    train_records: List[Dict[str, Any]] = []
    dev_records: List[Dict[str, Any]] = []
    n_pos_total = 0
    n_neg_total = 0
    k_kept_sum = 0
    n_pos_per_group_sum = 0
    split_counter: Counter = Counter()

    for key in keys:
        unit = cands[key]
        record = assemble_one_group(
            unit, labels,
            alpha=args.alpha,
            norm_mode=args.norm_mode,
            min_k_kept=args.min_k_kept,
            drop_all_negative=drop_all_negative,
            stats=stats,
        )
        if record is None:
            continue

        split = assign_split(record["qid"], args.dev_ratio, args.seed)
        record["split"] = split
        split_counter[split] += 1

        n_kept = record["meta"]["k_kept"]
        n_pos = record["meta"]["n_pos"]
        k_kept_sum += n_kept
        n_pos_per_group_sum += n_pos
        n_pos_total += n_pos
        n_neg_total += (n_kept - n_pos)

        if split == "dev":
            dev_records.append(record)
        else:
            train_records.append(record)

    stats.n_units_kept = len(train_records) + len(dev_records)
    stats.n_train = len(train_records)
    stats.n_dev = len(dev_records)
    stats.n_pos_total = n_pos_total
    stats.n_neg_total = n_neg_total
    stats.n_docs_kept = n_pos_total + n_neg_total
    stats.avg_k_kept = (k_kept_sum / stats.n_units_kept) if stats.n_units_kept else 0.0
    stats.avg_n_pos_per_group = (n_pos_per_group_sum / stats.n_units_kept) if stats.n_units_kept else 0.0
    stats.pos_ratio = (n_pos_total / stats.n_docs_kept) if stats.n_docs_kept else 0.0

    # 写盘
    with train_path.open("w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with dev_path.open("w", encoding="utf-8") as f:
        for r in dev_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = RunMeta(
        started_at=time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(started)),
        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        elapsed_sec=round(time.time() - started, 2),
        in_dir=str(in_dir),
        out_dir=str(out_dir),
        candidates_jsonl=str(candidates_path),
        cls_labels_jsonl=str(cls_labels_path),
        train_jsonl=str(train_path),
        dev_jsonl=str(dev_path),
        alpha=args.alpha,
        norm_mode=args.norm_mode,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        min_k_kept=args.min_k_kept,
        drop_all_negative=drop_all_negative,
        stats=asdict(stats),
    )
    meta_path.write_text(json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "DONE  groups: total=%d kept=%d  (drop missing=%d failed=%d too_few=%d all_neg=%d)",
        stats.n_units_total, stats.n_units_kept,
        stats.drop_missing_label, stats.drop_label_failed,
        stats.drop_too_few_kept, stats.drop_all_negative,
    )
    logger.info(
        "split  train=%d dev=%d  avg_k_kept=%.2f  avg_pos/group=%.2f  pos_ratio=%.1f%%",
        stats.n_train, stats.n_dev, stats.avg_k_kept,
        stats.avg_n_pos_per_group, 100 * stats.pos_ratio,
    )
    logger.info("wrote: %s", train_path)
    logger.info("wrote: %s", dev_path)
    logger.info("meta:  %s", meta_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
