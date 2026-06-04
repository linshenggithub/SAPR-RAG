#!/usr/bin/env python3
"""SAPR-R v1 — step5 smoke test（无外部依赖，本地秒跑）。

验证目标:
    [step5_assemble_train_jsonl.py](./step5_assemble_train_jsonl.py) 的整套 join 逻辑：
      candidates.jsonl + cls_labels.jsonl 加载 → (qid, step_idx, doc_id) join
      → 各类过滤分支 → rank_target 计算 → train/dev 切分 → 写盘 + meta 统计

构造 4 个 (qid, step_idx) unit + 对应 cls labels，每个 unit 4 个 doc，
覆盖以下分支（用 cls_label / ok 字段配置）：

    G1 = (qA, 0)：4 doc 全 ok，2 pos 2 neg          → kept
    G2 = (qA, 1)：4 doc 全 ok，0 pos 4 neg          → 默认丢（drop_all_negative）
    G3 = (qB, 0)：4 doc，1 missing label + 1 ok=false + 2 pos → kept k=2
    G4 = (qC, 0)：4 doc 全 ok 全 pos                → kept（全 1 也 OK）

跑法:
    python 03_sapr_rag/data/build_v1/test_step5_smoke.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
sys.path.insert(0, str(_REPO_ROOT))


def _load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_STEP5 = _load_module(
    _REPO_ROOT / "03_sapr_rag" / "data" / "build_v1" / "step5_assemble_train_jsonl.py",
    "_v1_step5",
)


# ----------------------- mock data -----------------------

def _mk_unit(qid: str, step_idx: int, doc_ids):
    """构 step3 candidates.jsonl 一行，每个 doc 给递增 retriever_score。"""
    return {
        "qid": qid,
        "step_idx": step_idx,
        "question": f"Q-{qid}-question",
        "gt_answer": "ans",
        "supporting_titles": [],
        "subquery": f"Q-{qid}-{step_idx}-subquery",
        "subject_entity": "EntX",
        "step_gold": f"gold-{qid}-{step_idx}",
        "prior_thoughts": [] if step_idx == 0 else [f"prior-thought-{step_idx-1}"],
        "candidates": [
            {"doc_id": d, "title": f"Title-{d}", "text": f"Text-{d}",
             "retriever_score": 0.5 + 0.1 * i}
            for i, d in enumerate(doc_ids)
        ],
    }


def _mk_label(qid, step_idx, doc_id, cls_label, ok=True):
    return {
        "qid": qid,
        "step_idx": step_idx,
        "doc_id": doc_id,
        "doc_title": f"Title-{doc_id}",
        "cls_label": cls_label if ok else None,
        "evidence": "ev" if ok and cls_label == 1 else "",
        "raw_response": "{}",
        "ok": ok,
        "error": None if ok else "schema: bad_label=None",
    }


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_inputs(in_dir: Path):
    cand_records = [
        _mk_unit("qA", 0, [101, 102, 103, 104]),  # G1
        _mk_unit("qA", 1, [201, 202, 203, 204]),  # G2 全 0
        _mk_unit("qB", 0, [301, 302, 303, 304]),  # G3 缺 label + ok=false
        _mk_unit("qC", 0, [401, 402, 403, 404]),  # G4 全 1
    ]
    write_jsonl(in_dir / "candidates.jsonl", cand_records)

    label_records = []
    # G1: 2 pos 2 neg
    label_records += [
        _mk_label("qA", 0, 101, 1),
        _mk_label("qA", 0, 102, 0),
        _mk_label("qA", 0, 103, 1),
        _mk_label("qA", 0, 104, 0),
    ]
    # G2: 4 neg
    label_records += [
        _mk_label("qA", 1, 201, 0),
        _mk_label("qA", 1, 202, 0),
        _mk_label("qA", 1, 203, 0),
        _mk_label("qA", 1, 204, 0),
    ]
    # G3: 301 missing（不写入）；302 ok=False；303、304 都 cls=1
    label_records += [
        _mk_label("qB", 0, 302, 0, ok=False),
        _mk_label("qB", 0, 303, 1),
        _mk_label("qB", 0, 304, 1),
    ]
    # G4: 全 1
    label_records += [
        _mk_label("qC", 0, 401, 1),
        _mk_label("qC", 0, 402, 1),
        _mk_label("qC", 0, 403, 1),
        _mk_label("qC", 0, 404, 1),
    ]
    write_jsonl(in_dir / "cls_labels.jsonl", label_records)


# ----------------------- assertions -----------------------

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def assert_close(a: float, b: float, eps=1e-6, msg=""):
    assert abs(a - b) < eps, f"{msg}: {a} vs {b}"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="sapr_v1_step5_smoke_"))
    try:
        in_dir = tmp / "out_smoke"
        in_dir.mkdir(parents=True)
        build_inputs(in_dir)
        train_path = in_dir / "train.jsonl"
        dev_path = in_dir / "dev.jsonl"
        meta_path = in_dir / "run_meta_step5.json"

        # --- 跑一次（默认参数）---
        old_argv = sys.argv
        try:
            sys.argv = [
                "step5", "--in-dir", str(in_dir),
                "--alpha", "0.7",
                "--norm-mode", "minmax",
                "--dev-ratio", "0.0",   # 全部进 train，断言更确定
                "--seed", "42",
                "--min-k-kept", "2",
            ]
            rc = _STEP5.main()
        finally:
            sys.argv = old_argv
        assert rc == 0, f"main returned {rc}"

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stats = meta["stats"]

        # --- 1. 总体计数 ---
        assert stats["n_units_total"] == 4, stats
        # G1 ✓ / G2 全负丢 / G3 ✓ (k_kept=2) / G4 ✓ → 3 kept
        assert stats["n_units_kept"] == 3, stats
        assert stats["drop_all_negative"] == 1, stats
        assert stats["drop_missing_label"] == 1, stats   # G3 doc=301
        assert stats["drop_label_failed"] == 1, stats    # G3 doc=302 ok=False
        assert stats["drop_too_few_kept"] == 0, stats

        # 全部进 train（dev_ratio=0）
        assert stats["n_train"] == 3
        assert stats["n_dev"] == 0
        assert dev_path.read_text() == "" or len(read_jsonl(dev_path)) == 0

        records = read_jsonl(train_path)
        assert len(records) == 3
        by_key = {(r["qid"], r["step_idx"]): r for r in records}

        # --- 2. G1 验证 ---
        g1 = by_key[("qA", 0)]
        assert g1["meta"]["k_raw"] == 4
        assert g1["meta"]["k_kept"] == 4
        assert g1["meta"]["n_pos"] == 2
        cls_g1 = [c["cls_label"] for c in g1["candidates"]]
        assert cls_g1 == [1, 0, 1, 0]
        # rank_target sum to 1
        assert_close(sum(c["rank_target"] for c in g1["candidates"]), 1.0,
                     msg="g1 rank_target sum")
        # cls=1 的 rank_target 应严格大于 cls=0
        pos_targets = [c["rank_target"] for c in g1["candidates"] if c["cls_label"] == 1]
        neg_targets = [c["rank_target"] for c in g1["candidates"] if c["cls_label"] == 0]
        assert min(pos_targets) > max(neg_targets), \
            f"g1: pos targets must dominate neg, got pos={pos_targets} neg={neg_targets}"
        # state 字段
        assert g1["state"]["question"] == "Q-qA-question"
        assert g1["state"]["history_thoughts"] == []
        assert g1["state"]["subquery"] == "Q-qA-0-subquery"

        # --- 3. G3 验证（join 后只剩 2 doc）---
        g3 = by_key[("qB", 0)]
        assert g3["meta"]["k_raw"] == 4, g3["meta"]
        assert g3["meta"]["k_kept"] == 2, g3["meta"]
        assert g3["meta"]["n_pos"] == 2, g3["meta"]
        kept_ids = sorted(c["doc_id"] for c in g3["candidates"])
        assert kept_ids == [303, 304], kept_ids
        assert_close(sum(c["rank_target"] for c in g3["candidates"]), 1.0,
                     msg="g3 rank_target sum")

        # --- 4. G4 全 1 验证（保留；rank_target 退化为 retriever_score 主导）---
        g4 = by_key[("qC", 0)]
        assert g4["meta"]["k_kept"] == 4
        assert g4["meta"]["n_pos"] == 4
        assert all(c["cls_label"] == 1 for c in g4["candidates"])
        assert_close(sum(c["rank_target"] for c in g4["candidates"]), 1.0,
                     msg="g4 rank_target sum")
        # retriever_score 高的 rank_target 也应高
        sorted_by_rs = sorted(g4["candidates"], key=lambda c: c["retriever_score"])
        sorted_by_rt = sorted(g4["candidates"], key=lambda c: c["rank_target"])
        assert [c["doc_id"] for c in sorted_by_rs] == [c["doc_id"] for c in sorted_by_rt], \
            "g4: retriever_score 单调时 rank_target 排序应一致"

        # --- 5. retriever_score_norm minmax 范围检查 ---
        for r in records:
            ns = [c["retriever_score_norm"] for c in r["candidates"]]
            assert all(-1e-6 <= n <= 1 + 1e-6 for n in ns), f"norm out of range: {ns}"
            if len(ns) >= 2:
                assert abs(min(ns)) < 1e-6, f"minmax min should be 0, got {min(ns)}"
                assert abs(max(ns) - 1.0) < 1e-6, f"minmax max should be 1, got {max(ns)}"

        # --- 6. dev split 划分（用真实 dev_ratio=1.0 全部进 dev）---
        # 清掉旧产物再跑一次
        for p in (train_path, dev_path, meta_path):
            if p.exists():
                p.unlink()
        try:
            sys.argv = [
                "step5", "--in-dir", str(in_dir),
                "--dev-ratio", "1.0", "--seed", "42",
            ]
            rc = _STEP5.main()
        finally:
            sys.argv = old_argv
        assert rc == 0
        meta2 = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta2["stats"]["n_train"] == 0
        assert meta2["stats"]["n_dev"] == 3
        dev_records = read_jsonl(dev_path)
        assert all(r["split"] == "dev" for r in dev_records)
        # 同 qid 所有 step 必同 split：qA 两个 step 现在都不会出现（因 G2 被丢，
        # 但 G1 在 dev_ratio=1.0 下是 dev），核心保证是 split 字段
        assert all(r["split"] in ("train", "dev") for r in dev_records)

        # --- 7. --keep-all-negative 行为 ---
        for p in (train_path, dev_path, meta_path):
            if p.exists():
                p.unlink()
        try:
            sys.argv = [
                "step5", "--in-dir", str(in_dir),
                "--dev-ratio", "0.0", "--seed", "42",
                "--keep-all-negative",
            ]
            rc = _STEP5.main()
        finally:
            sys.argv = old_argv
        assert rc == 0
        meta3 = json.loads(meta_path.read_text(encoding="utf-8"))
        # G2 这次也保留
        assert meta3["stats"]["n_units_kept"] == 4, meta3["stats"]
        assert meta3["stats"]["drop_all_negative"] == 0, meta3["stats"]

        print("[test] ALL OK ✓")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
