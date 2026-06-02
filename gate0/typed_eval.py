#!/usr/bin/env python3
"""
Rule-based typed transition evaluation for Gate 0.

This module is intentionally model-free. It provides a stable first-pass
implementation of query / claim / stop diagnostics so the MCTS experiments do
not depend on an LLM judge for typed evaluation.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "in",
    "on", "at", "by", "for", "from", "to", "with", "without", "about",
    "into", "over", "under", "is", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "has", "have", "had", "who", "what",
    "when", "where", "which", "why", "how", "this", "that", "these",
    "those", "it", "its", "his", "her", "their", "there", "as", "than",
    "first", "second", "third", "last", "year", "name", "person", "people",
    "city", "country", "state", "university", "school", "film", "book",
    "album", "song", "actor", "actress", "writer", "president",
}


@dataclass
class Claim:
    text: str
    evidence_refs: List[str] = field(default_factory=list)
    status: str = "unknown"


@dataclass
class State:
    entities: List[str] = field(default_factory=list)
    open_gaps: List[str] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)


@dataclass
class TransitionEval:
    phi_q: float
    phi_c: float
    phi_s: float
    failure_type: str
    query_applicable: bool
    claim_applicable: bool
    stop_applicable: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    return " ".join(text.translate(str.maketrans("", "", string.punctuation)).split())


def token_set(text: str) -> Set[str]:
    return {
        tok for tok in normalize_text(text).split()
        if tok and tok not in STOPWORDS and len(tok) > 1
    }


def simple_ner(text: str) -> Set[str]:
    """Small deterministic entity extractor used for Gate 0 diagnostics."""
    if not text:
        return set()

    entities: Set[str] = set()

    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
        entities.add(match.group(1).strip())

    for match in re.finditer(r"\b([A-Z][a-z]+)\b", text):
        word = match.group(1).strip()
        if word.lower() not in STOPWORDS:
            entities.add(word)

    for match in re.finditer(r"\b(\d{3,4})\b", text):
        entities.add(match.group(1))

    for match in re.finditer(r'"([^"]+)"', text):
        quoted = match.group(1).strip()
        if quoted:
            entities.add(quoted)

    return entities


def extract_query(response: str) -> str:
    matches = re.findall(r"<query>(.*?)</query>", response or "", flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


def extract_answer(response: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", response or "", flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


def extract_evidence(response: str) -> str:
    matches = re.findall(r"<evidence>(.*?)</evidence>", response or "", flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    if "relevant evidence is" in (response or "").lower():
        return response.strip()
    return ""


def has_none_evidence(evidence: str, response: str = "") -> bool:
    text = normalize_text(evidence or response)
    return bool(re.search(r"\bnone\b|\bno documents found\b|\bno relevant evidence\b", text))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap(source: Set[str], target: Set[str]) -> float:
    if not source:
        return 1.0 if not target else 0.0
    if not target:
        return 0.0
    return len(source & target) / len(source)


def _query_quality(
    question: str,
    state: State,
    query: str,
    history_queries: List[str],
) -> Dict[str, Any]:
    if not query:
        return {
            "score": 0.0,
            "gap_targeting": 0.0,
            "bridge_preservation": 0.0,
            "non_redundancy": 1.0,
            "query_entities": [],
            "target_entities": [],
            "max_history_similarity": 0.0,
        }

    query_entities = simple_ner(query)
    question_entities = set(state.entities) or simple_ner(question)
    gap_text = " ".join(state.open_gaps) if state.open_gaps else question
    target_entities = simple_ner(gap_text) or question_entities

    query_tokens = token_set(query)
    question_tokens = token_set(question)
    gap_tokens = token_set(gap_text)

    entity_gap = overlap(query_entities, target_entities)
    token_gap = jaccard(query_tokens, gap_tokens or question_tokens)
    gap_targeting = max(entity_gap, token_gap)

    if question_entities:
        retained = len(query_entities & question_entities)
        bridge_preservation = min(1.0, retained / min(len(question_entities), 2))
    else:
        bridge_preservation = jaccard(query_tokens, question_tokens)

    max_sim = 0.0
    for previous_query in history_queries:
        previous_tokens = token_set(previous_query)
        max_sim = max(max_sim, jaccard(query_tokens, previous_tokens))
    non_redundancy = 1.0 - max_sim

    score = 0.45 * gap_targeting + 0.35 * bridge_preservation + 0.20 * non_redundancy
    if max_sim >= 0.85:
        score = min(score, 0.20)

    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "gap_targeting": round(gap_targeting, 4),
        "bridge_preservation": round(bridge_preservation, 4),
        "non_redundancy": round(non_redundancy, 4),
        "query_entities": sorted(query_entities),
        "target_entities": sorted(target_entities),
        "max_history_similarity": round(max_sim, 4),
    }


def _claim_quality(question: str, query: str, response: str) -> Dict[str, Any]:
    evidence = extract_evidence(response)
    if not evidence:
        return {
            "score": 0.0,
            "evidence": "",
            "evidence_is_none": True,
            "empty_evidence": True,
            "support_overlap": 0.0,
        }
    if has_none_evidence(evidence, response):
        return {
            "score": 0.0,
            "evidence": evidence,
            "evidence_is_none": True,
            "empty_evidence": True,
            "support_overlap": 0.0,
        }

    evidence_tokens = token_set(evidence)
    target_tokens = token_set(question) | token_set(query)
    support_overlap = jaccard(evidence_tokens, target_tokens)

    # Weak Gate 0 heuristic: non-empty evidence gets partial credit, and direct
    # lexical support increases confidence. This is the future NLI replacement seam.
    score = 0.55 + 0.45 * support_overlap
    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "evidence": evidence,
        "evidence_is_none": False,
        "empty_evidence": False,
        "support_overlap": round(support_overlap, 4),
    }


def _stop_quality(state: State, response: str) -> Dict[str, Any]:
    answer = extract_answer(response)
    if not answer:
        return {
            "score": 0.0,
            "answer": "",
            "has_open_gaps": bool(state.open_gaps),
        }

    has_open_gaps = bool(state.open_gaps)
    return {
        "score": -1.0 if has_open_gaps else 1.0,
        "answer": answer,
        "has_open_gaps": has_open_gaps,
    }


def _infer_query_applicable(action_name: str, response: str, query: str) -> bool:
    if action_name == "document_analysis":
        return False
    if action_name in {"query_rewrite", "force_continue"}:
        return True
    if query or "<query>" in (response or ""):
        return True
    return action_name in {"begin_reasoning", "reasoning"} and not extract_answer(response)


def _infer_claim_applicable(action_name: str, response: str) -> bool:
    return action_name == "document_analysis" or "<evidence>" in (response or "")


def evaluate_transition(
    question: str,
    state: Optional[State] = None,
    action_name: str = "",
    response: str = "",
    query: str = "",
    history_queries: Optional[List[str]] = None,
    theta_q: float = 0.45,
    theta_c: float = 0.45,
) -> TransitionEval:
    """Evaluate one parent -> child transition with model-free heuristics."""
    state = state or State(entities=sorted(simple_ner(question)), open_gaps=[question])
    history_queries = history_queries or []
    query = query or extract_query(response)

    query_applicable = _infer_query_applicable(action_name, response, query)
    claim_applicable = _infer_claim_applicable(action_name, response)
    stop_applicable = bool(extract_answer(response))

    query_result = _query_quality(question, state, query, history_queries) if query_applicable else {"score": 1.0}
    claim_result = _claim_quality(question, query, response) if claim_applicable else {"score": 1.0}
    stop_result = _stop_quality(state, response)

    phi_q = float(query_result["score"])
    phi_c = float(claim_result["score"])
    phi_s = float(stop_result["score"])

    query_bad = query_applicable and phi_q < theta_q
    claim_bad = claim_applicable and phi_c < theta_c
    stop_bad = stop_applicable and phi_s < 0
    empty_evidence_bad = (
        claim_applicable
        and bool(claim_result.get("empty_evidence"))
        and action_name == "document_analysis"
    )

    failures = []
    if query_bad or empty_evidence_bad:
        failures.append("query_fail")
    if claim_bad and not empty_evidence_bad:
        failures.append("claim_fail")
    if stop_bad:
        failures.append("stop_fail")

    if not failures:
        failure_type = "success"
    elif len(failures) > 1:
        failure_type = "mixed"
    else:
        failure_type = failures[0]

    return TransitionEval(
        phi_q=round(phi_q, 4),
        phi_c=round(phi_c, 4),
        phi_s=round(phi_s, 4),
        failure_type=failure_type,
        query_applicable=query_applicable,
        claim_applicable=claim_applicable,
        stop_applicable=stop_applicable,
        details={
            "query": query,
            "query_quality": query_result,
            "claim_quality": claim_result,
            "stop_quality": stop_result,
            "thresholds": {"theta_q": theta_q, "theta_c": theta_c},
        },
    )


def state_from_question(
    question: str,
    open_gaps: Optional[List[str]] = None,
    claims: Optional[List[Claim]] = None,
) -> State:
    return State(
        entities=sorted(simple_ner(question)),
        open_gaps=open_gaps if open_gaps is not None else [question],
        claims=claims or [],
    )
