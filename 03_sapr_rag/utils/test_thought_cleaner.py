"""thought_cleaner 单测。

跑法（任意一种）：
    cd 03_sapr_rag/utils && python test_thought_cleaner.py
    pytest 03_sapr_rag/utils/test_thought_cleaner.py -v

样本来源：
- 真实样本：从 reward_data*.json 与 ReasonRAG prompt 模板抽取的代表脏 thought
- 边界样本：嵌套 / 缺闭合 / 答案在中间 / 空串 / 仅标签
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让独立 python 直接跑也能 import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from thought_cleaner import (  # noqa: E402
    clean_subquery,
    clean_thought,
    extract_evidence,
)


# ---------------------------------------------------------------------------
# 真实样本（来自 ReasonRAG reward_data 与 prompt 模板）
# ---------------------------------------------------------------------------

CASES_REAL = [
    # case 1: begin_reasoning 输出
    # 注意：<query> 内容是 subquery，不是 thought；thought 是分析文本本身。
    # clean_thought 应剥 "To answer this question..." 前导样板 + <query> 标签 +
    # "So the next query is" 句式，剩下分析文本主体。
    (
        "To answer this question, I'll break it down into smaller parts:\n"
        "1. Identify the hotel company associated with the Oberoi family.\n"
        "So the next query is <query>head office location of The Oberoi Group</query>",
        "Identify the hotel company",
    ),
    # case 2: document_analysis 输出 with evidence
    (
        "Based on the query, the relevant evidence is "
        "<evidence>The Oberoi Group has its head office in Delhi.</evidence>",
        "The Oberoi Group has its head office in Delhi.",
    ),
    # case 3: document_analysis None evidence
    (
        "<evidence>None</evidence>",
        "",  # 没有有效 evidence，且全是 schema，clean 后空
    ),
    # case 4: reasoning meta-headed 输出
    (
        "Error Reflection: No errors found in the previous thoughts.\n"
        "Information Sufficiency: The current information is sufficient to answer.\n"
        "So the answer is <answer>Delhi</answer>",
        # 段头被去掉，固定句式被去掉，<answer> 被剥
        "No errors found in the previous thoughts.",
    ),
    # case 5: answer_generation 简短输出
    (
        "So the answer is <answer>Delhi</answer>",
        "",
    ),
]


def test_real_thoughts():
    """对 5 条真实样本，clean_thought 应去掉所有 schema 元素。"""
    for raw, expected_first_word_in in CASES_REAL:
        cleaned = clean_thought(raw)
        # 不强求字面相等，但 schema 元素必须全部消失
        assert "<answer>" not in cleaned and "</answer>" not in cleaned, cleaned
        assert "<query>" not in cleaned and "</query>" not in cleaned, cleaned
        assert "<evidence>" not in cleaned and "</evidence>" not in cleaned, cleaned
        assert "Error Reflection:" not in cleaned, cleaned
        assert "Information Sufficiency:" not in cleaned, cleaned
        assert "So the answer is" not in cleaned, cleaned
        assert "So the next query is" not in cleaned, cleaned
        assert "Based on the query, the relevant evidence is" not in cleaned, cleaned
        # 期望子串
        if expected_first_word_in:
            assert expected_first_word_in in cleaned, (
                f"expect {expected_first_word_in!r} in cleaned but got {cleaned!r}"
            )


def test_subquery_extraction():
    cases = [
        # XML 标签优先
        (
            "Some prelude. So the next query is <query>head office of Oberoi</query>",
            "head office of Oberoi",
        ),
        # 没有 XML，回退到固定句式
        (
            "So the next query is head office of Oberoi",
            "head office of Oberoi",
        ),
        # 嵌套标签 + 末尾句号
        (
            'So the next query is <query>"Tom Hanks birthplace".</query>',
            '"Tom Hanks birthplace".',
        ),
        # 都没匹配，回退到 raw
        ("plain query string", "plain query string"),
        # 空串，用 fallback
        ("", "fallback_q"),
    ]
    for raw, expected in cases:
        out = clean_subquery(raw, fallback="fallback_q")
        assert expected in out or out == expected, (raw, out)


def test_evidence_extraction():
    assert extract_evidence("<evidence>X is Y</evidence>") == "X is Y"
    assert extract_evidence("<evidence>None</evidence>") is None
    assert extract_evidence("<evidence>  </evidence>") is None
    assert extract_evidence("no tag here") is None
    # 取最后一个
    assert (
        extract_evidence("<evidence>A</evidence> ... <evidence>B</evidence>") == "B"
    )


def test_edge_cases():
    # 空串 / None
    assert clean_thought("") == ""
    assert clean_thought("", fallback="x") == "x"

    # 仅 schema 元素
    assert clean_thought("<answer>Delhi</answer>") == ""
    assert clean_thought(
        "<answer>Delhi</answer>", fallback="head office of Oberoi"
    ) == "head office of Oberoi"

    # 缺闭合标签：fragment 也应被剥
    assert "<answer" not in clean_thought("So the answer is <answer>Delhi")
    assert "</answer>" not in clean_thought("Delhi</answer>")

    # 嵌套：外层 <answer> 包内层文本
    cleaned = clean_thought("So the answer is <answer>X is <query>Y</query></answer>")
    assert "<" not in cleaned and ">" not in cleaned

    # 截断：超长输入应截到 max_words
    long_raw = " ".join(["word"] * 100)
    assert len(clean_thought(long_raw, max_words=10).split()) == 10


def test_max_words_default():
    """默认 max_words=25，对长 evidence 也应截断。"""
    long_evidence = " ".join(["evidenceword"] * 50)
    raw = f"<evidence>{long_evidence}</evidence>"
    cleaned = clean_thought(raw)
    assert len(cleaned.split()) <= 25


def test_subquery_strip_quotes_and_period():
    # 末尾点号 / 引号需剥
    out = clean_subquery('So the next query is "Tom Hanks birthplace".')
    assert out.startswith("Tom") or out == "Tom Hanks birthplace"


# ---------------------------------------------------------------------------
# 自动化覆盖率审计：所有 schema 元素必须能被剥
# ---------------------------------------------------------------------------

SCHEMA_ELEMENTS_TO_STRIP = [
    "<answer>x</answer>",
    "<query>x</query>",
    "<evidence>x</evidence>",
    "<reference>x</reference>",
    "<information>x</information>",
    "So the answer is X",
    "So the next query is X",
    "Based on the query, the relevant evidence is X",
    "Error Reflection: comment",
    "Information Sufficiency: comment",
    "Conciseness: comment",
    "Conclusion: comment",
    "Instructions: comment",
]


def test_blacklist_coverage():
    """对每个 schema 元素，clean_thought 应将其抹掉。"""
    misses = []
    for raw in SCHEMA_ELEMENTS_TO_STRIP:
        cleaned = clean_thought(raw)
        # 元素不应原样出现
        # 条件：元素中的 marker 词不应在 cleaned 中（仅 X / comment 可残留）
        forbidden_markers = [
            "<answer>", "</answer>", "<query>", "</query>",
            "<evidence>", "</evidence>", "<reference>", "</reference>",
            "<information>", "</information>",
            "So the answer is", "So the next query is",
            "Based on the query, the relevant evidence is",
            "Error Reflection:", "Information Sufficiency:",
            "Conciseness:", "Conclusion:", "Instructions:",
        ]
        for m in forbidden_markers:
            if m in cleaned:
                misses.append((raw, cleaned, m))

    if misses:
        for raw, cleaned, m in misses:
            print(f"MISS: marker={m!r} raw={raw!r} -> cleaned={cleaned!r}")
    assert not misses, f"{len(misses)} schema elements not stripped"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def _run_all():
    tests = [
        ("test_real_thoughts", test_real_thoughts),
        ("test_subquery_extraction", test_subquery_extraction),
        ("test_evidence_extraction", test_evidence_extraction),
        ("test_edge_cases", test_edge_cases),
        ("test_max_words_default", test_max_words_default),
        ("test_subquery_strip_quotes_and_period", test_subquery_strip_quotes_and_period),
        ("test_blacklist_coverage", test_blacklist_coverage),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
