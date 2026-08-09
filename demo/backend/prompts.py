from __future__ import annotations

import re
from typing import Any


REASONING_SYSTEM = (
    "You are an assistant for question answering with access to a retrieval tool. "
    "Upon receiving a question, your task is to:\n"
    "* Analyze and Decompose the Question: Break the question into smaller, manageable "
    "sub-questions to ensure all aspects are addressed.\n"
    "* Evaluate Your Knowledge: Assess each sub-question or component:\n"
    "- Identify parts you can confidently answer based on your existing knowledge.\n"
    "- Pinpoint parts that require additional information or verification through retrieval tools.\n"
    "* Conciseness: Ensure both queries and answers are concise, using nouns or short "
    "phrases whenever possible.\n"
    "* Respond Format:\n"
    "If your knowledge is sufficient to answer the question, conclude with:\n"
    '"So the answer is <answer>answer</answer>"\n'
    "If retrieval is necessary to provide a complete answer, conclude with:\n"
    '"So the next query is <query>query</query>"\n'
)

EVIDENCE_SYSTEM = (
    "You are an information retrieval assistant. Given a query and a reference document, "
    "extract a concise piece of evidence that directly answers the query. "
    "If no relevant evidence is found, output <evidence>None</evidence>. "
    "Otherwise, output the evidence in the format: "
    "Based on the query, the relevant evidence is <evidence>evidence_text</evidence>."
)


def render_history(history: list[dict[str, Any]]) -> str:
    parts = []
    for item in history:
        parts.append(
            f"So the next query is <query>{item['query']}</query> "
            "Based on the query, the relevant evidence is "
            f"<evidence>{item['evidence']}</evidence>."
        )
    return "\n\n".join(parts)


def build_reasoning_messages(
    question: str, history: list[dict[str, Any]]
) -> list[dict[str, str]]:
    instruction = f"Question: {question}"
    if history:
        instruction += "\nPrevious Thoughts: " + render_history(history)
    return [
        {"role": "system", "content": REASONING_SYSTEM},
        {"role": "user", "content": instruction},
    ]


def build_evidence_messages(query: str, docs: list[dict[str, Any]]) -> list[dict[str, str]]:
    reference = " ".join(f"{doc['title']}. {doc['text']}" for doc in docs)
    return [
        {"role": "system", "content": EVIDENCE_SYSTEM},
        {
            "role": "user",
            "content": f"Question: {query}. Reference: <reference>{reference}</reference>",
        },
    ]


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(fr"<{tag}>(.*?)(?:</{tag}>|$)", text, re.DOTALL)
    return match.group(1).strip() if match else None
