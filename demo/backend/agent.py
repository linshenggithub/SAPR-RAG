from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from .prompts import build_evidence_messages, build_reasoning_messages, extract_tag


def _without_partial_closing_tag(content: str, tag: str) -> str:
    closing = f"</{tag}>"
    if closing in content:
        return content.split(closing, 1)[0]
    for size in range(min(len(content), len(closing) - 1), 0, -1):
        if content.endswith(closing[:size]):
            return content[:-size]
    return content


class SAPRDemoAgent:
    def __init__(
        self,
        model: Any,
        retriever: Any,
        *,
        max_turns: int,
        top_k: int,
        max_tokens: int,
        evidence_max_tokens: int,
    ):
        self.model = model
        self.retriever = retriever
        self.max_turns = max_turns
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.evidence_max_tokens = evidence_max_tokens

    async def run(self, question: str) -> AsyncIterator[dict[str, Any]]:
        history: list[dict[str, Any]] = []

        for turn in range(self.max_turns):
            yield {"type": "status", "stage": "reason", "turn": turn}
            raw = ""
            mode: str | None = None
            emitted_answer = ""
            messages = build_reasoning_messages(question, history)

            async for delta in self.model.chat_stream(
                messages,
                max_tokens=self.max_tokens,
                stop=["</query>", "</answer>"],
            ):
                raw += delta
                if mode is None:
                    answer_pos = raw.find("<answer>")
                    query_pos = raw.find("<query>")
                    positions = [
                        (answer_pos, "answer"),
                        (query_pos, "query"),
                    ]
                    positions = [(pos, kind) for pos, kind in positions if pos >= 0]
                    if positions:
                        mode = min(positions)[1]
                        if mode == "answer":
                            yield {"type": "answer_start", "turn": turn}

                if mode == "answer":
                    content = raw.split("<answer>", 1)[1]
                    visible = _without_partial_closing_tag(content, "answer")
                    new_text = visible[len(emitted_answer) :]
                    if new_text:
                        emitted_answer = visible
                        yield {"type": "answer_delta", "delta": new_text, "turn": turn}

            if mode == "answer":
                answer = extract_tag(raw, "answer") or emitted_answer.strip()
                if not answer:
                    yield {"type": "error", "code": "empty_answer"}
                    return
                yield {
                    "type": "done",
                    "answer": answer,
                    "turns": turn + 1,
                    "history": history,
                }
                return

            query = extract_tag(raw, "query")
            if not query:
                yield {"type": "error", "code": "invalid_agent_action"}
                return

            yield {"type": "query", "query": query, "turn": turn}
            yield {"type": "status", "stage": "retrieve", "turn": turn}
            docs = await self.retriever.search(query, self.top_k)
            public_docs = [
                {
                    "title": str(doc.get("title", "Untitled")),
                    "text": str(doc.get("text", "")),
                    "score": float(doc.get("score", 0.0)),
                }
                for doc in docs
            ]
            yield {"type": "documents", "documents": public_docs, "turn": turn}

            yield {"type": "status", "stage": "evidence", "turn": turn}
            evidence_raw = await self.model.chat(
                build_evidence_messages(query, public_docs),
                max_tokens=self.evidence_max_tokens,
                stop=["</evidence>"],
            )
            evidence = extract_tag(evidence_raw, "evidence") or "None"
            history.append({"query": query, "evidence": evidence})
            yield {"type": "evidence", "evidence": evidence, "turn": turn}

        yield {
            "type": "error",
            "code": "max_turns_exceeded",
            "message": f"The agent did not finish within {self.max_turns} turns.",
        }
