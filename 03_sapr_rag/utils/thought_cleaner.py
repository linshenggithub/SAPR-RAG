"""
ReasonRAG generator raw thought 清洗工具。

用途：v1 trained reranker 推理时，generator 输出的 thought 含有 ReasonRAG prompt
显式约束的 schema 元素（XML 标签 + meta 段头）。reranker 训练分布是
DeepSeek 离线生成的干净陈述，所以在 reranker 调用入口必须把脏 thought
翻译成干净版，否则训-推分布不一致。

作用域：仅用于 reranker 输入。其他链路（generator prompt 拼接、trajectory
日志、reward 计算）保持脏不动。

黑名单来源：直接从 ReasonRAG 四个 prompt 模板枚举，不依赖经验采样。
- BEGIN_REASONING / REASONING / ANSWER_GENERATION:
    "So the answer is <answer>...</answer>"
    "So the next query is <query>...</query>"
- DOCUMENT_ANALYSIS:
    "Based on the query, the relevant evidence is <evidence>...</evidence>"
    "<evidence>None</evidence>"
- REASONING 段头:
    "Error Reflection: ..."
    "Information Sufficiency: ..."

参考：ReasonRAG/pipeline/reasonrag_pipeline.py REASONING_PROMPT 等。
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# 正则模式（顺序敏感：必须先剥外层 XML 再去段头，否则段头匹配会被标签干扰）
# ---------------------------------------------------------------------------

# 1. ReasonRAG schema 中所有 XML-style 标签（含闭合不全的残片）
_TAG_PAIRS = re.compile(
    r"<(answer|query|evidence|reference|information)[^>]*>"
    r".*?"
    r"</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAG_FRAGMENTS = re.compile(
    r"</?(answer|query|evidence|reference|information)[^>]*>?",
    re.IGNORECASE,
)

# 2. ReasonRAG prompt 显式规定的固定句式
_FIXED_PHRASES = [
    re.compile(r"So the answer is\s*", re.IGNORECASE),
    re.compile(r"So the next query is\s*", re.IGNORECASE),
    re.compile(r"So the next is\s*", re.IGNORECASE),
    re.compile(
        r"Based on the (?:query|provided documents|reference)[,]?\s*"
        r"the relevant evidence is\s*",
        re.IGNORECASE,
    ),
]

# 3. REASONING_PROMPT 三个 meta 段头：去掉段头并保留段落正文
#    形如 "Error Reflection: No errors found in the previous thoughts."
#    清洗后保留 "No errors found in the previous thoughts."
_META_SECTION_HEADERS = re.compile(
    r"(?:^|(?<=[\s\.]))"
    r"(Error Reflection|Information Sufficiency|Conciseness|Conclusion|"
    r"Analyze and Decompose the Question|Evaluate Your Knowledge|"
    r"Respond Format|Instructions|Response format)"
    r"\s*[:：]\s*",
    re.IGNORECASE,
)

# 4. "To answer this question, ..." / "To address this question, ..."
#    用于剥前导自我介绍（仅句首匹配）
_LEADING_BOILERPLATE = re.compile(
    r"^\s*(?:To answer this question|To address this question|"
    r"Let(?:'s| us) (?:break|analyze|consider))[^\.\n]*[\.\n]\s*",
    re.IGNORECASE,
)

# 5. 多余空白
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{2,}")


def _strip_xml_schema(text: str) -> str:
    """先剥成对 XML，再扫散落标签残片。"""
    text = _TAG_PAIRS.sub("", text)
    text = _TAG_FRAGMENTS.sub("", text)
    return text


def _strip_meta_phrases(text: str) -> str:
    for pat in _FIXED_PHRASES:
        text = pat.sub("", text)
    text = _META_SECTION_HEADERS.sub("", text)
    text = _LEADING_BOILERPLATE.sub("", text)
    return text


def _normalize_whitespace(text: str) -> str:
    text = _MULTI_NEWLINE.sub("\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _truncate_to_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def extract_evidence(text: str) -> Optional[str]:
    """从 raw response 抽 <evidence>...</evidence> 内部内容。

    返回 None 表示未找到 evidence 标签或内容是 None / 空串。
    """
    matches = re.findall(
        r"<evidence[^>]*>(.*?)</evidence>",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not matches:
        return None
    last = matches[-1].strip()
    if not last or last.lower() == "none":
        return None
    return last


def clean_thought(
    raw: str,
    max_words: int = 25,
    fallback: Optional[str] = None,
) -> str:
    """把 generator raw thought / response 翻译成干净陈述。

    流程：
    1. 优先抽 <evidence> 段（document_analysis 输出在这里）；
    2. 其余 action 走通用清洗：剥 XML / 去固定句式 / 去段头 / 去前导样板；
    3. 标准化空白，截断到 max_words 词；
    4. 清洗后空串则回退到 fallback（一般传当前 subquery）。

    Args:
        raw: 原始 thought / response 字符串
        max_words: 截断词数上限，默认 25
        fallback: 清洗失败时的回退字符串，None 时返回空串

    Returns:
        干净 thought 字符串
    """
    if not raw:
        return fallback or ""

    evidence = extract_evidence(raw)
    if evidence is not None:
        return _truncate_to_words(_normalize_whitespace(evidence), max_words)

    text = _strip_xml_schema(raw)
    text = _strip_meta_phrases(text)
    text = _normalize_whitespace(text)
    text = _truncate_to_words(text, max_words)
    if not text:
        return fallback or ""
    return text


def clean_subquery(raw: str, fallback: Optional[str] = None) -> str:
    """从 generator 输出中抽 subquery。

    优先级：<query>...</query> 标签 > "So the next query is ..." 后续片段 > raw。
    清洗后空串回退到 fallback。
    """
    if not raw:
        return fallback or ""

    matches = re.findall(
        r"<query[^>]*>(.*?)</query>",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    if matches:
        cand = matches[-1].strip()
        if cand:
            return _normalize_whitespace(cand)

    m = re.search(
        r"So the next query is\s*(.+?)(?:\n|$)",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        cand = _strip_xml_schema(m.group(1)).strip()
        cand = re.sub(r'^["“]|["”\.]+$', "", cand).strip()
        if cand:
            return _normalize_whitespace(cand)

    cand = _normalize_whitespace(_strip_xml_schema(raw))
    if cand:
        return cand
    return fallback or ""
