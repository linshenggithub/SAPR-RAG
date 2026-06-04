#!/usr/bin/env python3
"""SAPR-R v1 — step4 smoke test（无 API 依赖，本地秒跑）。

验证目标:
    [step4_label_cls.py](./step4_label_cls.py) 的整套数据流：
      candidates.jsonl 解析 → flatten task → chat_batch → schema 校验
      → 流式落盘 → 断点续跑 → run_meta_step4.json 统计

如何工作:
    1. 在 tmp 目录构造 5 行 mock candidates.jsonl，flatten 后正好 5 个 task
    2. monkey-patch step4 模块里的 DeepSeekClient.from_env，让它返回一个
       FakeClient；FakeClient.chat_batch 按 task 顺序返回预设响应，覆盖：
         T1: 合法 label=1 + evidence            → ok=True, cls_label=1
         T2: 合法 label=0 + evidence=""         → ok=True, cls_label=0
         T3: 非法 schema（缺 label 字段）       → ok=False, error 含 "schema"
         T4: 非法 JSON（不是合法 JSON 字符串）  → ok=False, error 含 "json_decode"
         T5: API 失败（raw=None）               → ok=False, error 含 "api_failed"
    3. 调 step4 的 main()，断言产物文件内容、统计、断点续跑

跑法:
    python 03_sapr_rag/data/build_v1/test_step4_smoke.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


# ----------------------- 路径派生 + 加载 step4 模块 -----------------------

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]   # SAPR-RAG/
sys.path.insert(0, str(_REPO_ROOT))


def _load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_STEP4 = _load_module(
    _REPO_ROOT / "03_sapr_rag" / "data" / "build_v1" / "step4_label_cls.py",
    "_v1_step4_under_test",
)


# ----------------------- mock 数据构造 -----------------------

def _make_candidate(doc_id: int, title: str, text: str, score: float = 0.9):
    return {"doc_id": doc_id, "title": title, "text": text, "retriever_score": score}


def build_mock_candidates_jsonl(out_path: Path) -> int:
    """造 mock candidates.jsonl；返回 flatten 后的 task 总数。

    单元布局：
      unit 0 (qid=q1, step_idx=0)：1 个 candidate → T1 合法 yes
      unit 1 (qid=q1, step_idx=1)：1 个 candidate → T2 合法 no
      unit 2 (qid=q2, step_idx=0)：2 个 candidate → T3 schema 失败 + T4 json 失败
      unit 3 (qid=q3, step_idx=0)：1 个 candidate → T5 API 失败
    总 task = 5。
    """
    units = [
        {
            "qid": "q1", "step_idx": 0,
            "question": "Who founded the company that owns the Oberoi Group?",
            "gt_answer": "Mohan Singh Oberoi",
            "supporting_titles": ["Oberoi Group", "Mohan Singh Oberoi"],
            "subquery": "Who founded the Oberoi Group?",
            "subject_entity": "Oberoi Group",
            "step_gold": "Mohan Singh Oberoi founded Oberoi Group",
            "prior_thoughts": [],
            "candidates": [
                _make_candidate(1001, "Oberoi Group",
                                "The Oberoi Group is an Indian hotel company founded in 1934 by Mohan Singh Oberoi."),
            ],
        },
        {
            "qid": "q1", "step_idx": 1,
            "question": "Who founded the company that owns the Oberoi Group?",
            "gt_answer": "Mohan Singh Oberoi",
            "supporting_titles": ["Oberoi Group", "Mohan Singh Oberoi"],
            "subquery": "When was Mohan Singh Oberoi born?",
            "subject_entity": "Mohan Singh Oberoi",
            "step_gold": "Mohan Singh Oberoi was born in 1898",
            "prior_thoughts": ["Mohan Singh Oberoi founded the Oberoi Group in 1934."],
            "candidates": [
                _make_candidate(2002, "Indian hotel industry",
                                "The Indian hotel industry expanded rapidly after independence."),
            ],
        },
        {
            "qid": "q2", "step_idx": 0,
            "question": "Which film did the director of Inception direct in 2008?",
            "gt_answer": "The Dark Knight",
            "supporting_titles": ["Christopher Nolan", "The Dark Knight"],
            "subquery": "Who directed Inception?",
            "subject_entity": "Inception",
            "step_gold": "Christopher Nolan directed Inception",
            "prior_thoughts": [],
            "candidates": [
                _make_candidate(3003, "Inception",
                                "Inception is a 2010 science fiction film directed by Christopher Nolan."),
                _make_candidate(3004, "Random distractor",
                                "This document is unrelated to the question."),
            ],
        },
        {
            "qid": "q3", "step_idx": 0,
            "question": "What is the capital of the country whose author wrote War and Peace?",
            "gt_answer": "Moscow",
            "supporting_titles": ["Leo Tolstoy", "Russia"],
            "subquery": "Who wrote War and Peace?",
            "subject_entity": "War and Peace",
            "step_gold": "Leo Tolstoy wrote War and Peace",
            "prior_thoughts": [],
            "candidates": [
                _make_candidate(4005, "Leo Tolstoy",
                                "Leo Tolstoy was a Russian writer who authored War and Peace."),
            ],
        },
    ]
    with out_path.open("w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    n_tasks = sum(len(u["candidates"]) for u in units)
    return n_tasks


# ----------------------- Fake DeepSeek client -----------------------

@dataclass
class _FakeStats:
    requests: int = 0
    succeeded: int = 0
    failed: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _FakeConfig:
    base_url: str = "fake://test"
    model: str = "fake-deepseek"


class FakeClient:
    """模拟 DeepSeekClient.chat_batch 行为，按 task 顺序逐条返回预设响应。"""

    def __init__(self, scripted: list):
        self._scripted = list(scripted)   # raw response or None
        self.config = _FakeConfig()
        self.stats = _FakeStats()
        self._cursor = 0

    def chat_batch(self, prompts, *, max_workers, temperature, max_tokens,
                   response_format, on_failure, progress_cb):
        # 真实 client 的接口 -> 我们简单按顺序消费 scripted
        n = len(prompts)
        out = []
        for _ in range(n):
            if self._cursor >= len(self._scripted):
                raise AssertionError(
                    f"FakeClient: scripted 用尽 (cursor={self._cursor}, n_scripted={len(self._scripted)})"
                )
            out.append(self._scripted[self._cursor])
            self._cursor += 1
        # 模拟统计
        self.stats.requests += n
        self.stats.succeeded += sum(1 for x in out if x is not None)
        self.stats.failed += sum(1 for x in out if x is None)
        self.stats.prompt_tokens += 100 * n
        self.stats.completion_tokens += 30 * n
        if progress_cb:
            progress_cb(n, n)
        return out


# 5 个 task 的预设响应（顺序与 candidates flatten 顺序一致）
SCRIPTED_RESPONSES = [
    # T1: 合法 yes
    json.dumps({"label": 1, "evidence": "founded in 1934 by Mohan Singh Oberoi"}),
    # T2: 合法 no
    json.dumps({"label": 0, "evidence": ""}),
    # T3: schema 失败（缺 label 字段）
    json.dumps({"evidence": "Christopher Nolan", "verdict": "yes"}),
    # T4: 非法 JSON
    "not a valid json {{{",
    # T5: API 失败
    None,
]


# ----------------------- 校验 -----------------------

def assert_records(out_jsonl: Path, expected_pos: int, expected_neg: int,
                   expected_failed: int, expected_total: int):
    with out_jsonl.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == expected_total, \
        f"expected {expected_total} records, got {len(records)}"

    pos = sum(1 for r in records if r["ok"] and r["cls_label"] == 1)
    neg = sum(1 for r in records if r["ok"] and r["cls_label"] == 0)
    failed = sum(1 for r in records if not r["ok"])

    assert pos == expected_pos, f"pos: expected {expected_pos}, got {pos}"
    assert neg == expected_neg, f"neg: expected {expected_neg}, got {neg}"
    assert failed == expected_failed, f"failed: expected {expected_failed}, got {failed}"

    # 逐条核对路径
    by_key = {(r["qid"], r["step_idx"], r["doc_id"]): r for r in records}

    # T1 q1/0/1001 ok yes
    r = by_key[("q1", 0, 1001)]
    assert r["ok"] is True
    assert r["cls_label"] == 1
    assert r["evidence"] == "founded in 1934 by Mohan Singh Oberoi"
    assert r["error"] is None

    # T2 q1/1/2002 ok no
    r = by_key[("q1", 1, 2002)]
    assert r["ok"] is True
    assert r["cls_label"] == 0
    assert r["evidence"] == ""

    # T3 q2/0/3003 schema 失败
    r = by_key[("q2", 0, 3003)]
    assert r["ok"] is False
    assert r["cls_label"] is None
    assert "schema" in (r["error"] or ""), f"T3 error should contain schema, got: {r['error']}"

    # T4 q2/0/3004 json 解析失败
    r = by_key[("q2", 0, 3004)]
    assert r["ok"] is False
    assert "json_decode" in (r["error"] or ""), f"T4 error: {r['error']}"

    # T5 q3/0/4005 API 失败
    r = by_key[("q3", 0, 4005)]
    assert r["ok"] is False
    assert "api_failed" in (r["error"] or ""), f"T5 error: {r['error']}"

    # raw_response 留痕（供复盘）：T1-T4 有 raw，T5 raw=""
    assert by_key[("q1", 0, 1001)]["raw_response"]
    assert by_key[("q3", 0, 4005)]["raw_response"] == ""


def assert_meta(meta_path: Path, *, expected_total, expected_succeeded,
                expected_failed, expected_pos, expected_neg):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["n_tasks_total"] == expected_total, meta
    assert meta["n_succeeded"] == expected_succeeded, meta
    assert meta["n_failed"] == expected_failed, meta
    assert meta["n_pos"] == expected_pos, meta
    assert meta["n_neg"] == expected_neg, meta
    expected_ratio = expected_pos / max(1, expected_pos + expected_neg)
    assert abs(meta["pos_ratio"] - expected_ratio) < 1e-6


# ----------------------- 跑测 -----------------------

def run_test() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step4_smoke_"))
    print(f"[test] tmp dir = {tmp}")
    try:
        in_dir = tmp / "in"
        in_dir.mkdir()
        candidates_path = in_dir / "candidates.jsonl"
        n_tasks = build_mock_candidates_jsonl(candidates_path)
        print(f"[test] mock candidates: {candidates_path} (flatten {n_tasks} tasks)")
        assert n_tasks == 5

        # --- monkey-patch from_env ---
        fake_client = FakeClient(SCRIPTED_RESPONSES)
        original_from_env = _STEP4.DeepSeekClient.from_env
        _STEP4.DeepSeekClient.from_env = classmethod(
            lambda cls, env_path=None, prefer="deepseek": fake_client
        )

        try:
            # 用 sys.argv 注入 CLI 参数；step4.main() 是 argparse 驱动
            old_argv = sys.argv
            sys.argv = [
                "step4_label_cls.py",
                "--in-dir", str(in_dir),
                "--out-dir", str(tmp / "out"),
                "--chunk-size", "3",     # 故意分两块（3+2）测多 chunk 路径
                "--max-workers", "2",
                "--log-level", "WARNING",
            ]
            try:
                rc = _STEP4.main()
            finally:
                sys.argv = old_argv

            # 有失败 → main 返回 1
            assert rc == 1, f"expected rc=1 due to failures, got {rc}"

            out_jsonl = tmp / "out" / "cls_labels.jsonl"
            meta_path = tmp / "out" / "run_meta_step4.json"
            assert out_jsonl.exists(), out_jsonl
            assert meta_path.exists(), meta_path

            assert_records(
                out_jsonl,
                expected_pos=1,
                expected_neg=1,
                expected_failed=3,
                expected_total=5,
            )
            assert_meta(
                meta_path,
                expected_total=5,
                expected_succeeded=2,
                expected_failed=3,
                expected_pos=1,
                expected_neg=1,
            )
            print("[test] PASS  records & meta 校验通过 (5 tasks, 1 pos / 1 neg / 3 failed)")

            # --- 断点续跑：再跑一次应识别全部已完成、不再调 LLM ---
            fake_client_2 = FakeClient([])  # 空 scripted；若被调用必抛 AssertionError
            _STEP4.DeepSeekClient.from_env = classmethod(
                lambda cls, env_path=None, prefer="deepseek": fake_client_2
            )
            sys.argv = [
                "step4_label_cls.py",
                "--in-dir", str(in_dir),
                "--out-dir", str(tmp / "out"),
                "--chunk-size", "3",
                "--log-level", "WARNING",
            ]
            try:
                rc2 = _STEP4.main()
            finally:
                sys.argv = old_argv

            assert rc2 == 0, f"resume run expected rc=0, got {rc2}"
            with out_jsonl.open("r", encoding="utf-8") as f:
                n_lines = sum(1 for _ in f)
            assert n_lines == 5, f"resume should not append; got {n_lines} lines"
            print("[test] PASS  断点续跑：5/5 跳过，不再调 fake client")

        finally:
            _STEP4.DeepSeekClient.from_env = original_from_env

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[test] cleaned tmp = {tmp}")

    print("\n[test] ALL OK ✓")


if __name__ == "__main__":
    run_test()
