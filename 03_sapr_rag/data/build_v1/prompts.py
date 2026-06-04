"""SAPR-R v1 离线数据构造 prompt 模板。

集中托管 step2 / step4 等离线管线使用的所有 prompt，避免散落在脚本里。
所有模板都用 ``str.format(**kwargs)`` 渲染，调用方负责传齐占位符。

Naming convention:
    STEP{N}_{ROLE}_TEMPLATE  e.g. STEP2_SYSTEM_TEMPLATE / STEP2_USER_TEMPLATE
    STEP{N}_FEW_SHOTS        list[dict] of {input, output} for in-context examples
    build_step{N}_messages   helper to assemble the final messages list

Decisions captured here (与用户对齐):
    - step2 输出 schema 含四字段：subquery / subject_entity / thought / step_gold
    - step2 仅喂 supporting_titles，不喂 supporting_facts 全句
    - step4 用 "前 k-1 个 thought 句" 作为 PRIOR REASONING（不是 step_gold 片段）
      理由：thought 是完整 SVO 句，与推理时 extract_evidence 抽出的 evidence
            分布对齐；step_gold 留给"current step GT atomic fact"角色
    - step4 few-shot 含 positive / wrong-entity / repeat-history 三例
    - step4 输出 evidence 字段用于人工抽审 label noise，不进训练
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# =====================================================================
# Step 2 — Question decomposition into reasoning_steps
# =====================================================================

STEP2_SYSTEM_TEMPLATE = """\
You are a multi-hop question decomposition expert. Given a question and its
ground-truth answer, you decompose the question into a minimal sequence of
single-hop sub-questions. Each sub-question retrieves ONE atomic fact, and
those facts compose into the final answer.

Output rules — STRICT:
1. Output a JSON object with key "reasoning_steps" (a list).
2. Each step has exactly four fields:
   - "subquery":       <=15 words, natural English question or noun-phrase
                       request (e.g. "founding year of The Oberoi Group").
                       NO XML tags, NO meta phrases like "I need to find...".
   - "subject_entity": the SINGLE entity this sub-question is asking about
                       (e.g. "The Oberoi Group"). It must be a named entity
                       (person, organization, location, work-of-art, etc.)
                       that uniquely identifies what this sub-question is
                       querying. Use a noun phrase, not a sentence.
                       For comparison questions where the sub-question
                       targets one of two entities, this is the ONE that
                       sub-question is currently about.
   - "thought":        <=25 words, ONE declarative sentence stating the fact
                       this step retrieves (subject-verb-object form).
                       Format: "<subject_entity> <relation> <object>."
                       Example: "The Oberoi Group was founded in 1934."
                       NOT a plan ("I will look up..."), NOT meta reflection
                       ("I have enough info"), and NOT a question. Must be
                       PHRASEABLE as if extracted from a Wikipedia paragraph.
   - "step_gold":      one short phrase (<=6 words) — the atomic fact answer
                       to this sub-question. Usually an entity, date,
                       number, or short noun phrase. Example: "1934".
3. Steps must be ordered: each step's subquery may reference entities
   resolved by earlier step_gold. The last step's "thought" must imply the
   final answer.
4. Number of steps: 2-3 typical, MAX 4. Do NOT pad.
5. Do NOT use XML tags (<query>, <answer>, <evidence>) anywhere.
6. Do NOT use meta phrases: "Error Reflection", "Information Sufficiency",
   "Based on the query", "So the answer is", "I need to", "Let me", "First,".

Few-shot examples follow.
"""


STEP2_FEW_SHOTS: List[Dict[str, Any]] = [
    {
        "input": {
            "question": "What year was the company that owns The Oberoi Hotel founded?",
            "gt_answer": "1934",
            "supporting_titles": ["The Oberoi Hotel", "The Oberoi Group"],
        },
        "output": {
            "reasoning_steps": [
                {
                    "subquery": "Which company owns The Oberoi Hotel?",
                    "subject_entity": "The Oberoi Hotel",
                    "thought": "The Oberoi Hotel is owned by The Oberoi Group.",
                    "step_gold": "The Oberoi Group",
                },
                {
                    "subquery": "Founding year of The Oberoi Group.",
                    "subject_entity": "The Oberoi Group",
                    "thought": "The Oberoi Group was founded in 1934.",
                    "step_gold": "1934",
                },
            ],
        },
    },
    {
        "input": {
            "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
            "gt_answer": "yes",
            "supporting_titles": ["Scott Derrickson", "Ed Wood"],
        },
        "output": {
            "reasoning_steps": [
                {
                    "subquery": "Nationality of Scott Derrickson.",
                    "subject_entity": "Scott Derrickson",
                    "thought": "Scott Derrickson is American.",
                    "step_gold": "American",
                },
                {
                    "subquery": "Nationality of Ed Wood.",
                    "subject_entity": "Ed Wood",
                    "thought": "Ed Wood is American.",
                    "step_gold": "American",
                },
            ],
        },
    },
]


STEP2_USER_TEMPLATE = """\
QUESTION:
{question}

GROUND-TRUTH ANSWER:
{gt_answer}

SUPPORTING TOPICS (from HotpotQA gold annotation, only as a hint for
decomposition; do NOT copy verbatim):
{supporting_titles_joined}

Decompose this question into ordered sub-questions. Output JSON only.
"""


def build_step2_messages(
    question: str,
    gt_answer: str,
    supporting_titles: List[str],
) -> List[Dict[str, str]]:
    """生成 step2 的 OpenAI messages list（含 system + few-shot + user）。"""
    fewshot_lines: List[str] = []
    for ex in STEP2_FEW_SHOTS:
        fewshot_lines.append("EXAMPLE — input:")
        fewshot_lines.append(f"QUESTION: {ex['input']['question']}")
        fewshot_lines.append(f"GT ANSWER: {ex['input']['gt_answer']}")
        fewshot_lines.append(
            f"SUPPORTING TOPICS: {json.dumps(ex['input']['supporting_titles'])}"
        )
        fewshot_lines.append("")
        fewshot_lines.append("EXAMPLE — output:")
        fewshot_lines.append(json.dumps(ex["output"], ensure_ascii=False, indent=2))
        fewshot_lines.append("")

    system_content = STEP2_SYSTEM_TEMPLATE + "\n" + "\n".join(fewshot_lines)

    user_content = STEP2_USER_TEMPLATE.format(
        question=question.strip(),
        gt_answer=str(gt_answer).strip(),
        supporting_titles_joined=json.dumps(supporting_titles, ensure_ascii=False),
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# =====================================================================
# Step 4 — Answer-aware cls labelling
# =====================================================================
#
# 调用方：03_sapr_rag/data/build_v1/step4_label_cls.py
# 决策依据：docs/repo_overview.md §0bis.3（cls 打标 = 方案 I answer-aware）
# 配套上游：step2 必须提供 subject_entity 字段
# ---------------------------------------------------------------------

STEP4_SYSTEM_TEMPLATE = """\
You are a strict evidence verifier for a multi-hop QA dataset. Given a
sub-question, its ground-truth atomic fact, and a candidate document, you
decide whether the document EXPLICITLY states the ground-truth fact in a
form that directly answers the sub-question.

Output rules — STRICT:
1. Output a single JSON object with exactly two fields:
   - "label":    "yes" or "no"
   - "evidence": a <=20-word verbatim quote from the document that supports
                 your "yes" decision. Use "" if label is "no".
2. NO extra commentary, NO chain-of-thought, NO XML tags.

Decision rules — be strict:
A. Output "yes" ONLY IF ALL of the following hold:
   A1. The document mentions the SUBJECT ENTITY of the sub-question
       (not just a co-occurring entity from a different topic).
   A2. The document explicitly states the GROUND-TRUTH FACT (lexically or
       semantically matched, not merely implied).
   A3. The fact is stated about the correct subject — NOT about a
       differently-named entity that shares some words (e.g.
       "The Imperial Hotel" != "The Oberoi Hotel").

B. Output "no" if ANY of the following hold:
   B1. The document is about a different entity that shares a surface form
       with the subject (wrong-entity distractor).
   B2. The document is topically related but does not state the
       ground-truth fact.
   B3. The document only repeats facts already in PRIOR REASONING without
       providing the current sub-question's answer.
   B4. The document mentions the right entity AND the right value, but
       relates them through an unrelated relation (e.g. "founded in 1934"
       vs. "fired its CEO in 1934").

Borderline guidance:
   - For comparison questions where multiple entities share a label
     (e.g. both subjects are "American"), output "yes" only if the
     document is ABOUT the subject_entity (not just a brief mention as
     trivia in a different article).
   - If the document quotes a value close but not equal to step_gold
     (off by one year, etc.), output "no".

Few-shot examples follow.
"""


STEP4_FEW_SHOTS: List[Dict[str, Any]] = [
    {
        "input": {
            "question": "What year was the company that owns The Oberoi Hotel founded?",
            "prior_reasoning": ["The Oberoi Hotel is owned by The Oberoi Group."],
            "subquery": "Founding year of The Oberoi Group.",
            "subject_entity": "The Oberoi Group",
            "step_gold": "1934",
            "doc_title": "The Oberoi Group",
            "doc_text": (
                "The Oberoi Group is an Indian hotel company founded in 1934 "
                "by Mohan Singh Oberoi. Its headquarters are in Delhi."
            ),
        },
        "output": {
            "label": "yes",
            "evidence": "founded in 1934 by Mohan Singh Oberoi",
        },
    },
    {
        "input": {
            "question": "What year was the company that owns The Oberoi Hotel founded?",
            "prior_reasoning": ["The Oberoi Hotel is owned by The Oberoi Group."],
            "subquery": "Founding year of The Oberoi Group.",
            "subject_entity": "The Oberoi Group",
            "step_gold": "1934",
            "doc_title": "The Imperial Hotel",
            "doc_text": (
                "The Imperial Hotel was founded in 1931 in New Delhi by "
                "Sardar Bahadur Sir Sobha Singh."
            ),
        },
        "output": {"label": "no", "evidence": ""},
    },
    {
        "input": {
            "question": "What year was the company that owns The Oberoi Hotel founded?",
            "prior_reasoning": ["The Oberoi Hotel is owned by The Oberoi Group."],
            "subquery": "Founding year of The Oberoi Group.",
            "subject_entity": "The Oberoi Group",
            "step_gold": "1934",
            "doc_title": "The Oberoi Hotel",
            "doc_text": (
                "The Oberoi Hotel is a luxury hotel in Mumbai owned by "
                "The Oberoi Group."
            ),
        },
        "output": {"label": "no", "evidence": ""},
    },
]


STEP4_USER_TEMPLATE = """\
ORIGINAL QUESTION:
{question}

PRIOR REASONING (facts already established; may be empty for the first
sub-question):
{prior_reasoning_block}

CURRENT SUB-QUESTION:
{subquery}

SUBJECT ENTITY OF SUB-QUESTION:
{subject_entity}

GROUND-TRUTH ATOMIC FACT (the answer to the current sub-question):
{step_gold}

CANDIDATE DOCUMENT:
Title: {doc_title}
Text:
{doc_text}

Decide whether this document explicitly states the ground-truth fact about
the subject entity. Output JSON only.
"""


def _format_prior_reasoning_block(prior_thoughts: List[str]) -> str:
    """渲染 PRIOR REASONING 字段；空列表时显式提示。

    输入是前 k-1 个 thought 句子（完整 SVO 陈述句），与推理时
    extract_evidence() 抽出的 evidence 分布对齐。
    """
    cleaned = [t.strip() for t in prior_thoughts if t and t.strip()]
    if not cleaned:
        return "(none — this is the first sub-question)"
    return "\n".join(f"- {t}" for t in cleaned)


def build_step4_messages(
    question: str,
    prior_thoughts: List[str],
    subquery: str,
    subject_entity: str,
    step_gold: str,
    doc_title: str,
    doc_text: str,
) -> List[Dict[str, str]]:
    """生成 step4 的 OpenAI messages list。"""
    fewshot_lines: List[str] = []
    for ex in STEP4_FEW_SHOTS:
        inp = ex["input"]
        fewshot_lines.append("EXAMPLE — input:")
        fewshot_lines.append(f'ORIGINAL QUESTION: "{inp["question"]}"')
        fewshot_lines.append(
            f"PRIOR REASONING: {json.dumps(inp['prior_reasoning'])}"
        )
        fewshot_lines.append(f'CURRENT SUB-QUESTION: "{inp["subquery"]}"')
        fewshot_lines.append(f'SUBJECT ENTITY: "{inp["subject_entity"]}"')
        fewshot_lines.append(f'GROUND-TRUTH ATOMIC FACT: "{inp["step_gold"]}"')
        fewshot_lines.append(f'DOCUMENT TITLE: "{inp["doc_title"]}"')
        fewshot_lines.append(f"DOCUMENT TEXT: {inp['doc_text']}")
        fewshot_lines.append("")
        fewshot_lines.append("EXAMPLE — output:")
        fewshot_lines.append(json.dumps(ex["output"], ensure_ascii=False))
        fewshot_lines.append("")

    system_content = STEP4_SYSTEM_TEMPLATE + "\n" + "\n".join(fewshot_lines)

    user_content = STEP4_USER_TEMPLATE.format(
        question=question.strip(),
        prior_reasoning_block=_format_prior_reasoning_block(prior_thoughts),
        subquery=subquery.strip(),
        subject_entity=subject_entity.strip(),
        step_gold=step_gold.strip(),
        doc_title=doc_title.strip(),
        doc_text=doc_text.strip(),
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


__all__ = [
    "STEP2_SYSTEM_TEMPLATE",
    "STEP2_USER_TEMPLATE",
    "STEP2_FEW_SHOTS",
    "build_step2_messages",
    "STEP4_SYSTEM_TEMPLATE",
    "STEP4_USER_TEMPLATE",
    "STEP4_FEW_SHOTS",
    "build_step4_messages",
]
