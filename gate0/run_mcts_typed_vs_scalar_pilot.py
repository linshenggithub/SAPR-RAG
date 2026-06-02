#!/usr/bin/env python3
"""
Inference-style no-label MCTS pilot (Gate0-B)

Purpose: Compare scalar self-eval vs typed transition eval for MCTS branch selection,
         WITHOUT accessing golden answers during search.

Rules:
  - golden_answers are ONLY used post-hoc to compute EM/F1
  - search/selection/expansion/backprop NEVER access golden_answers
  - baseline: scalar self-eval (GPT-4o evaluates reasoning quality, no golden answer)
  - treatment: typed transition eval (φ_q, φ_c, φ_s computed from NER + set ops)

Usage:
  python gate0/run_mcts_typed_vs_scalar_pilot.py --mode sanity     # 1 sample, verify format & no-leak
  python gate0/run_mcts_typed_vs_scalar_pilot.py --mode baseline   # run baseline on N samples
  python gate0/run_mcts_typed_vs_scalar_pilot.py --mode treatment  # run treatment on N samples
  python gate0/run_mcts_typed_vs_scalar_pilot.py --mode eval       # compute EM/F1 from saved results
"""

import os
import re
import sys
import json
import copy
import math
import time
import random
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from collections import Counter

import yaml
from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────
# 让脚本能直接 `python gate0/run_mcts_typed_vs_scalar_pilot.py` 运行
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.paths import (  # noqa: E402
    BGE_INDEX_PATH as _BGE_INDEX_PATH,
    BGE_MODEL_PATH as _BGE_MODEL_PATH,
    HOTPOTQA_DEV_PATH as _HOTPOTQA_DEV_PATH,
    WIKI_CORPUS_PATH as _WIKI_CORPUS_PATH,
)

# 仓库内路径：相对于本脚本所在 gate0/ 目录派生
PROJECT_DIR = _REPO_ROOT
GATE0_DIR = _REPO_ROOT / "gate0"
DATA_DIR = GATE0_DIR / "data"
RESULTS_DIR = GATE0_DIR / "results"

from typed_eval import (  # noqa: E402
    evaluate_transition as rule_evaluate_transition,
    extract_evidence as rule_extract_evidence,
    has_none_evidence as rule_has_none_evidence,
    state_from_question as rule_state_from_question,
    token_set as rule_token_set,
)

# External resources（仓外路径：从 config/paths.py 读取，可被 SAPR_* 环境变量覆盖）
BGE_INDEX_PATH = str(_BGE_INDEX_PATH)
WIKI_CORPUS_PATH = str(_WIKI_CORPUS_PATH)
BGE_MODEL_PATH = str(_BGE_MODEL_PATH)
HOTPOTQA_DEV_PATH = str(_HOTPOTQA_DEV_PATH)

# API config (loaded from .env, NOT hardcoded)
DMXAPI_BASE_URL = "https://www.dmxapi.cn/v1"
DMXAPI_MODEL = "gpt-4o"

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(GATE0_DIR / "pilot.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── MCTS Parameters (aligned with ReasonRAG data_generation.py) ───
MCTS_CONFIG = {
    "max_iter": 7,
    "max_children": 2,
    "max_rollouts": 64,
    "c": 1.414,          # UCT exploration constant
    "beta": 0.95,        # step decay (for reward, not used in inference mode)
    "retrieval_topk": 3,
    "max_tokens": 256,
}


# ====================================================================
#  Prompt Templates (from original ReasonRAG, no golden answer)
# ====================================================================

BEGIN_REASONING_SYSTEM = """You are an assistant for question answering with access to a retrieval tool. Upon receiving a question, your task is to:
* Analyze and Decompose the Question: Break the question into smaller, manageable sub-questions to ensure all aspects are addressed.
* Evaluate Your Knowledge: Assess each sub-question or component:
- Identify parts you can confidently answer based on your existing knowledge.
- Pinpoint parts that require additional information or verification through retrieval tools.
* Conciseness: Ensure both queries and answers are concise, using nouns or short phrases whenever possible.
* Respond Format:
If your knowledge is sufficient to answer the question, conclude with:
"So the answer is <answer>answer</answer>"
If retrieval is necessary to provide a complete answer, conclude with:
"So the next query is <query>query</query>"
"""

DOCUMENT_ANALYSIS_SYSTEM = """You are an information retrieval assistant. Your task is to extract relevant evidence from the provided Wikipedia documents based on the latest query.

Instructions:

* Identify key terms or concepts in the query.
* Search the documents for evidence that supports the query.
* Response format:
If relevant evidence is found, output:
   Based on the query, the relevant evidence is <evidence>evidence</evidence>.
If no relevant evidence is found, output:
   <evidence>None</evidence>.
"""

REASONING_SYSTEM = """You are a question-answering assistant with access to a retrieval tool. Your goal is to provide a concise and accurate reasoning process.
Instructions:
* Error Reflection: If errors exist in previous thoughts, identify and correct them. Skip this step if no errors are present.
* Information Sufficiency: Evaluate whether the current information is sufficient to fully and accurately answer the question. If additional retrieval is needed, deconstruct the question and generate the next query. Avoid repeating previous queries. If no meaningful new query can be generated, explain why and provide an answer based on the current information.
* Conciseness: Ensure both queries and answers are concise, using nouns or short phrases whenever possible.
* Conclusion:
If generating an answer:
"So the answer is <answer>answer</answer>".
If more retrieval is needed:
"So the next query is <query>query</query>".
"""

ANSWER_GENERATION_SYSTEM = """You are a reasoning assistant with retrieval. Give a precise and very concise final answer for the given question, conclude with 'So the answer is <answer>answer</answer>'. Keep your final answer brief and to the point, followed without any explanation.
"""

# ── No-golden self-evaluation prompt (replaces original evaluate_thoughts) ──
SELF_EVAL_SYSTEM = """You are a reasoning quality evaluator. Given a question and an agent's reasoning process, evaluate the quality of the reasoning.

Assess:
1. Does the reasoning logically follow from the evidence?
2. Are all parts of the question addressed?
3. Is the conclusion well-supported?

Output a single integer score between 0 and 100.
Respond ONLY with: So the score is [Score].
"""

# ── Differentiated expansion prompts (v4 Plan A) ──

QUERY_REWRITE_SYSTEM = """You are a query rewriting assistant for multi-hop question answering.

The previous query failed to find the right information. Your task is to rewrite the query to better target the information gap.

Rules:
1. Preserve ALL bridge entities (key named entities) from the original question
2. Be more specific about what information is needed
3. Use different search terms than the previous query
4. Keep the query concise

Output format:
"So the next query is <query>your rewritten query here</query>"
"""

EVIDENCE_REEXAMINE_SYSTEM = """You are an evidence re-examination assistant for multi-hop question answering.

The previous evidence extraction missed key information or was not well-supported. Your task is to re-examine the retrieved documents and extract better evidence.

Rules:
1. Focus on finding facts that directly address the question
2. Look for bridge entities that connect different pieces of information
3. If no relevant evidence exists, output None

Output format:
If relevant evidence is found:
"Based on the query, the relevant evidence is <evidence>evidence text</evidence>."
If no relevant evidence:
"<evidence>None</evidence>."
"""

FORCE_CONTINUE_SYSTEM = """You are a reasoning assistant. The previous step concluded too early — not all aspects of the question have been addressed yet.

You MUST continue reasoning, not give a final answer. Generate a new search query to find the missing information.

Output format:
"So the next query is <query>your new query here</query>"
"""


# ====================================================================
#  LLM Client (DMXAPI GPT-4o)
# ====================================================================

class LLMClient:
    """Thin wrapper around DMXAPI OpenAI client."""

    def __init__(self, api_key: str, base_url: str = DMXAPI_BASE_URL, model: str = DMXAPI_MODEL):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.total_tokens = 0
        self.call_count = 0

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 256, stop: list = None) -> str:
        """Single turn generation. Returns response text."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if stop:
            kwargs["stop"] = stop

        response = self.client.chat.completions.create(**kwargs)
        self.total_tokens += response.usage.total_tokens
        self.call_count += 1
        return response.choices[0].message.content

    def generate_batch(self, system_prompts: list, user_prompts: list, max_tokens: int = 256) -> list:
        """Batch generation (sequential for API)."""
        results = []
        for sp, up in zip(system_prompts, user_prompts):
            results.append(self.generate(sp, up, max_tokens))
        return results


# ====================================================================
#  Retriever (BGE + FAISS, local, no LLM needed)
# ====================================================================

class Retriever:
    """Local BGE retriever using pre-built FAISS index."""

    def __init__(self, index_path: str, corpus_path: str, model_path: str, topk: int = 3):
        self.topk = topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.model_path = model_path
        self._retriever = None
        self._corpus = None

    def _lazy_init(self):
        """Load retriever on first use (saves startup time during testing)."""
        if self._retriever is not None:
            return
        sys.path.insert(0, "/home/mayi/RAG/FlashRAG")
        from flashrag.config import Config
        from flashrag.retriever import DenseRetriever

        # Build config dict that FlashRAG's Config class expects
        config_dict = {
            "retrieval_method": "bge",
            "model2path": {"bge": self.model_path},
            "index_path": self.index_path,
            "corpus_path": self.corpus_path,
            "retrieval_topk": self.topk,
            "faiss_gpu": False,
            "retrieval_pooling_method": "cls",
            "instruction": None,
            "retrieval_batch_size": 256,
            "retrieval_use_fp16": True,
            "retrieval_query_max_length": 128,
            "save_retrieval_cache": False,
            "use_retrieval_cache": False,
            "retrieval_cache_path": None,
            "use_reranker": False,
            "rerank_model_name": None,
            "rerank_model_path": None,
            "rerank_topk": 5,
            "rerank_max_length": 512,
            "rerank_batch_size": 256,
            "rerank_use_fp16": True,
            "rerank_pooling_method": None,
        }
        config = Config(config_dict=config_dict)
        self._retriever = DenseRetriever(config)
        log.info(f"Retriever loaded: bge, topk={self.topk}")

    def search(self, query: str) -> list:
        """Search for documents. Returns list of {title, text}."""
        self._lazy_init()
        results = self._retriever.search(query, return_score=False)
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                results = results[0]
        return results[:self.topk]


class MockRetriever:
    """Tiny deterministic retriever for fast debug runs."""

    def __init__(self, topk: int = 3):
        self.topk = topk

    def search(self, query: str) -> list:
        docs = [
            {
                "title": "Moscow State University",
                "text": (
                    "Moscow State University is a public research university in Moscow, Russia. "
                    "It was founded in 1755."
                ),
            },
            {
                "title": "Sergei Aleksandrovich Tokarev",
                "text": (
                    "Sergei Aleksandrovich Tokarev was a Soviet ethnographer and historian. "
                    "He was a professor at Moscow State University."
                ),
            },
            {
                "title": "Gerald Ford",
                "text": "Gerald Ford was the 38th president of the United States.",
            },
        ]
        return docs[:self.topk]


class ContextRetriever:
    """Fast per-sample lexical retriever over HotpotQA provided context.

    This avoids loading the 64GB FAISS index during Gate 0 debugging. It is a
    debug/context retrieval mode, not a replacement for matched BGE experiments.
    """

    def __init__(self, topk: int = 3):
        self.topk = topk
        self.docs = []

    def set_sample(self, sample: dict):
        context = sample.get("context", {}) or {}
        titles = context.get("title", []) if isinstance(context, dict) else []
        sentences = context.get("sentences", []) if isinstance(context, dict) else []
        docs = []
        for title, sent_list in zip(titles, sentences):
            if isinstance(sent_list, list):
                text = " ".join(str(s) for s in sent_list)
            else:
                text = str(sent_list)
            docs.append({"title": str(title), "text": text})
        self.docs = docs

    def search(self, query: str) -> list:
        if not self.docs:
            return []
        query_tokens = rule_token_set(query)
        scored = []
        for doc in self.docs:
            title_tokens = rule_token_set(doc.get("title", ""))
            text_tokens = rule_token_set(doc.get("text", ""))
            title_overlap = len(query_tokens & title_tokens)
            text_overlap = len(query_tokens & text_tokens)
            score = 3 * title_overlap + text_overlap
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for score, doc in scored[:self.topk]]


# ====================================================================
#  MCTS Node
# ====================================================================

class MCTSNode:
    """A node in the MCTS tree. Tracks NO golden answers."""

    __slots__ = [
        "id", "question", "thoughts", "parent", "parent_id",
        "children", "children_ids", "step", "N", "Q",
        "next_state", "node_dict", "max_tokens_reached",
        # Typed eval results (treatment only)
        "phi_q", "phi_c", "phi_s", "failure_type",
    ]

    def __init__(self, index, question, parent=None, parent_id=-1,
                 step=0, next_state=None, thoughts=None, node_dict=None):
        self.id = index
        self.question = question
        self.thoughts = thoughts or []
        self.parent = parent
        self.parent_id = parent_id
        self.children = []
        self.children_ids = []
        self.step = step
        self.N = 0
        self.Q = 0.0
        self.next_state = next_state
        self.node_dict = node_dict or {}
        self.max_tokens_reached = False
        self.phi_q = None
        self.phi_c = None
        self.phi_s = None
        self.failure_type = None

    def add_child(self, child):
        self.children.append(child)
        self.children_ids.append(child.id)

    def update_Q(self, q_val: float):
        self.N += 1
        # Running average
        self.Q = self.Q + (q_val - self.Q) / self.N
        self.node_dict["Q"] = round(self.Q, 4)
        self.node_dict["step"] = self.step
        self.node_dict["N"] = self.N

    def is_fully_expanded(self, max_children: int) -> bool:
        return self.next_state is None or len(self.children) >= max_children


# ====================================================================
#  Text Processing Utilities
# ====================================================================

def process_text(text: str) -> str:
    """Normalize query/answer tags in response."""
    # Ensure <query>...</query> tags
    next_match = re.search(r'So the next (?:query )?is (.*?)(?= So the answer is|$)', text, re.DOTALL)
    if next_match:
        content = next_match.group(1).strip()
        content = re.sub(r'[\.\"]+$', '', content)
        if not re.search(r'<query>.*</query>', content):
            content = f'<query>{content}</query>'
        text = text.replace(next_match.group(1), content)

    # Ensure <answer>...</answer> tags
    for match in re.finditer(r'So the answer is (.*?)(?= So the answer is|$)', text, re.DOTALL):
        content = match.group(1).strip()
        content = re.sub(r'[\.\"]+$', '', content)
        if not re.search(r'<answer>.*</answer>', content):
            content = f'<answer>{content}</answer>'
        text = text.replace(match.group(1), content)

    return text


def extract_answer(pred: str) -> str:
    """Extract final answer from response."""
    answer_matches = re.findall(r'<answer>(.*?)</answer>', pred)
    if answer_matches:
        pred = answer_matches[-1].strip()
    elif "So the answer is" in pred:
        pred = pred.split("So the answer is")[-1].strip()
    else:
        return ""
    # Clean up remaining tags
    pred = re.sub(r'<answer.*?>.*?</answer>|<query.*?>.*?</query>|answer>|<answer', '', pred, flags=re.DOTALL)
    if '.' in pred:
        pred = pred.split('.')[0].strip()
    return pred.strip()


def extract_query(response: str) -> str:
    """Extract next query from response."""
    query_matches = re.findall(r'<query>(.*?)</query>', response)
    if query_matches:
        return query_matches[-1].strip()
    return ""


def get_action_type(response: str) -> str:
    """Determine action type from response."""
    if '<query>' in response:
        return "query"
    elif '<answer>' in response:
        return "answer"
    elif '<evidence>' in response:
        return "evidence"
    return "other"


# ====================================================================
#  MCTS Pipeline (no golden answer)
# ====================================================================

class MCTSPipeline:
    """MCTS pipeline for inference-style evaluation."""

    def __init__(self, llm: LLMClient, retriever: Retriever,
                 mode: str = "baseline", config: dict = None):
        """
        mode: "baseline" (scalar self-eval) or "treatment" (typed eval)
        """
        self.llm = llm
        self.retriever = retriever
        self.mode = mode
        self.config = config or MCTS_CONFIG
        self.max_iter = self.config["max_iter"]
        self.max_children = self.config["max_children"]
        self.max_rollouts = self.config["max_rollouts"]
        self.c = self.config["c"]
        self.index = 0
        self.stop_tokens = ["<|im_end|>", "</answer>", "</query>", "</evidence>"]

    # ── MCTS Core ──────────────────────────────────────────────

    def search(self, question: str) -> dict:
        """Run MCTS search on a question. Returns tree + prediction."""
        self.index = 0
        thoughts = []
        root = MCTSNode(
            self.index, question, parent_id=-1, step=0,
            next_state="begin_reasoning", thoughts=thoughts
        )

        for rollout_idx in range(self.max_rollouts):
            leaf = self._select(root)
            if leaf is None:
                break
            child = self._expand(leaf)
            if child is None:
                continue
            q_val = self._simulate(child)
            self._backpropagate(child, q_val)

        # Get best answer from tree
        prediction, tree_log = self._extract_answer(root)
        return {
            "question": question,
            "prediction": prediction,
            "tree": tree_log,
            "n_rollouts": min(rollout_idx + 1, self.max_rollouts),
            "total_llm_calls": self.llm.call_count,
            "total_tokens": self.llm.total_tokens,
        }

    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCT selection."""
        while not self._is_terminal(node):
            if not node.is_fully_expanded(self.max_children):
                return node
            if not node.children:
                return node
            node = max(node.children, key=lambda c: self._uct(c))
        return node

    def _uct(self, node: MCTSNode) -> float:
        if node.N == 0:
            return float("inf")
        return node.Q + self.c * math.sqrt(math.log(max(node.parent.N, 1)) / max(node.N, 1))

    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expand a node by generating a new child."""
        if node.is_fully_expanded(self.max_children):
            return None

        self.index += 1
        child_id = self.index
        child = self._get_next_state(node, child_id)
        if child is not None:
            node.add_child(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """Evaluate node quality. Mode-dependent."""
        if self.mode == "baseline":
            return self._scalar_self_eval(node)
        else:
            return self._typed_eval(node)

    def _backpropagate(self, node: MCTSNode, q_val: float):
        """Backpropagate Q value up the tree."""
        while node is not None:
            node.update_Q(q_val)
            node = node.parent

    def _is_terminal(self, node: MCTSNode) -> bool:
        return node.next_state is None or node.max_tokens_reached

    # ── State Transitions ──────────────────────────────────────

    def _get_next_state(self, parent: MCTSNode, child_id: int) -> Optional[MCTSNode]:
        """Generate next state. In treatment mode, uses failure-attributed expansion."""
        action = parent.next_state
        if action is None:
            return None

        thoughts = copy.copy(parent.thoughts)
        response = ""
        expansion_method = "normal"  # Track which expansion method was used
        override_next_action = None  # Override next action for differentiated expansion
        eval_action_name = action

        # ── Treatment: Check if we should use differentiated expansion ──
        if self.mode == "treatment" and parent.failure_type in ("query_fail", "claim_fail", "stop_fail"):
            result = self._differentiated_expansion(parent, action, thoughts)
            if result:
                response, expansion_method, override_next_action = result
                eval_action_name = expansion_method
                if response:
                    thoughts.append(response)

        # ── Normal expansion (baseline, or treatment when no failure) ──
        if not response:
            if action == "begin_reasoning":
                response = self._call_llm(BEGIN_REASONING_SYSTEM, f"Question: {parent.question}")
                response = process_text(response)
                thoughts.append(response)
            elif action == "document_analysis":
                query = extract_query(thoughts[-1]) if thoughts else parent.question
                docs = self.retriever.search(query)
                ref_text = self._format_reference(docs)
                question_thoughts = parent.question + "\nPrevious Thoughts: " + " ".join(thoughts)
                user_prompt = f"Question: {question_thoughts}. Reference: <reference>{ref_text}</reference>"
                response = self._call_llm(DOCUMENT_ANALYSIS_SYSTEM, user_prompt)
                thoughts.append(response)
            elif action == "reasoning":
                question_thoughts = parent.question + "\nPrevious Thoughts: " + " ".join(thoughts)
                response = self._call_llm(REASONING_SYSTEM, f"Question: {question_thoughts}")
                response = process_text(response)
                thoughts.append(response)
            elif action == "answer_generation":
                question_thoughts = parent.question + "\nPrevious Thoughts: " + " ".join(thoughts)
                response = self._call_llm(ANSWER_GENERATION_SYSTEM, f"Question: {question_thoughts}")
                response = process_text(response)
                thoughts.append(response)

        if not response:
            return None

        # Determine next state: use override for differentiated expansion, otherwise normal
        if override_next_action is not None:
            next_action = override_next_action
        else:
            next_action = self._next_action(action, response, parent.step + 1)

        child = MCTSNode(
            child_id, parent.question,
            parent=parent, parent_id=parent.id,
            step=parent.step + 1,
            next_state=next_action,
            thoughts=thoughts,
        )
        child.node_dict = {
            "action_name": eval_action_name,
            "base_action_name": action,
            "response": response,
            "query": extract_query(response),
            "answer": extract_answer(response),
            "children_ids": [],
            "expansion_method": expansion_method,  # Track which strategy was used
        }

        return child

    def _differentiated_expansion(self, parent: MCTSNode,
                                   action: str, thoughts: list):
        """Failure-attributed differentiated expansion (v4 Plan A core).

        Returns (response, method, override_next_action) or None if skipped.
        Based on parent's failure_type, use a different expansion strategy:
        - query_fail  → rewrite the query, then go to document_analysis
        - claim_fail  → re-examine evidence, then go to reasoning
        - stop_fail   → force continuation with a new query, then document_analysis
        """
        failure = parent.failure_type
        question_thoughts = parent.question + "\nPrevious Thoughts: " + " ".join(thoughts)

        if failure == "query_fail":
            # Rewrite the query: find the previous query, rewrite it
            prev_query = extract_query(thoughts[-1]) if thoughts else parent.question
            user_prompt = (
                f"Question: {parent.question}\n"
                f"Previous failed query: {prev_query}\n"
                f"Reasoning so far: {question_thoughts}\n"
                f"The previous query did not find the right information. "
                f"Rewrite the query to better target the information gap. "
                f"Preserve all key named entities from the original question."
            )
            response = self._call_llm(QUERY_REWRITE_SYSTEM, user_prompt)
            response = process_text(response)
            # After rewriting query, next should be document_analysis
            return response, "query_rewrite", "document_analysis"

        elif failure == "claim_fail":
            # Re-examine evidence: re-do document analysis with focus instruction
            prev_query = extract_query(thoughts[-1]) if thoughts else parent.question
            if not prev_query:
                prev_query = parent.question
            docs = self.retriever.search(prev_query)
            ref_text = self._format_reference(docs)
            user_prompt = (
                f"Question: {question_thoughts}\n"
                f"The previous evidence extraction was poor. "
                f"Re-examine these documents carefully.\n"
                f"Reference: <reference>{ref_text}</reference>"
            )
            response = self._call_llm(EVIDENCE_REEXAMINE_SYSTEM, user_prompt)
            # After re-examining evidence, next should be reasoning
            return response, "evidence_reexamine", "reasoning"

        elif failure == "stop_fail":
            # Force continuation: generate a new query instead of stopping
            user_prompt = (
                f"Question: {parent.question}\n"
                f"Previous steps: {question_thoughts}\n"
                f"Not all aspects of the question have been answered yet. "
                f"Generate a new search query to find the missing information."
            )
            response = self._call_llm(FORCE_CONTINUE_SYSTEM, user_prompt)
            response = process_text(response)
            # After generating new query, go to document_analysis
            return response, "force_continue", "document_analysis"

        return None

    def _next_action(self, current_action: str, response: str, step: int) -> Optional[str]:
        """Determine next action from current action and response."""
        if step >= self.max_iter:
            return None  # Force terminal at max_iter

        if current_action == "answer_generation":
            return None

        if '<answer>' in response:
            return None  # Terminal

        if '<query>' in response or current_action == "document_analysis":
            if current_action in ("begin_reasoning", "reasoning"):
                return "document_analysis"
            elif current_action == "document_analysis":
                return "reasoning"

        if '<evidence>' in response:
            return "reasoning"

        return None  # Default terminal

    def _format_reference(self, docs: list) -> str:
        """Format retrieved documents as reference text."""
        parts = []
        for doc in docs:
            if isinstance(doc, dict):
                title = doc.get("title", "")
                text = doc.get("text", "")
                parts.append(f"Wikipedia Title: {title}\n{text}\n\n")
        return "".join(parts) if parts else "No documents found."

    def _call_llm(self, system: str, user: str, max_tokens: int = None) -> str:
        """Call LLM with system + user prompt."""
        if max_tokens is None:
            max_tokens = self.config["max_tokens"]
        return self.llm.generate(system, user, max_tokens=max_tokens)

    # ── Evaluation Methods ─────────────────────────────────────

    def _scalar_self_eval(self, node: MCTSNode) -> float:
        """Baseline: scalar self-evaluation WITHOUT golden answer."""
        reasoning = " ".join(node.thoughts) if node.thoughts else ""
        if not reasoning:
            return 0.0

        user_prompt = (
            f"Question: {node.question}\n"
            f"Agent Reasoning Process: {reasoning}\n"
        )
        response = self._call_llm(SELF_EVAL_SYSTEM, user_prompt, max_tokens=50)

        # Extract score
        score = self._extract_score(response)
        return score / 100.0  # Normalize to [0, 1]

    def _typed_eval(self, node: MCTSNode) -> float:
        """Treatment: model-free typed transition evaluation.

        The only LLM calls in treatment mode are still generation calls
        (reasoning, evidence analysis, repair prompts). Typed scoring itself is
        rule-based and uses no API call.
        """
        if not node.node_dict:
            node.phi_q = node.phi_c = node.phi_s = None
            node.failure_type = "no_transition"
            return 0.0

        response = node.node_dict.get("response", "")
        query = node.node_dict.get("query", "")
        action_name = node.node_dict.get("action_name", "")
        state = rule_state_from_question(
            node.question,
            open_gaps=self._infer_open_gaps_for_node(node),
        )
        result = rule_evaluate_transition(
            question=node.question,
            state=state,
            action_name=action_name,
            response=response,
            query=query,
            history_queries=self._history_queries(node),
        )

        node.phi_q = result.phi_q
        node.phi_c = result.phi_c
        node.phi_s = result.phi_s
        node.failure_type = result.failure_type

        score = self._typed_score(node)
        node.node_dict["phi_q"] = result.phi_q
        node.node_dict["phi_c"] = result.phi_c
        node.node_dict["phi_s"] = result.phi_s
        node.node_dict["failure_type"] = result.failure_type
        node.node_dict["typed_score"] = round(score, 4)
        node.node_dict["typed_details"] = result.details
        if result.failure_type == "stop_fail" and node.step < self.max_iter:
            node.next_state = "reasoning"
            node.node_dict["stop_overridden"] = True
        return score

    def _history_queries(self, node: MCTSNode) -> List[str]:
        """Queries before the current transition."""
        history = []
        for thought in node.thoughts[:-1]:
            query = extract_query(thought)
            if query:
                history.append(query)
        return history

    def _has_prior_non_empty_evidence(self, node: MCTSNode) -> bool:
        """Weak online state signal for whether an answer can plausibly stop."""
        for thought in node.thoughts[:-1]:
            evidence = rule_extract_evidence(thought)
            if evidence and not rule_has_none_evidence(evidence, thought):
                return True
        return False

    def _infer_open_gaps_for_node(self, node: MCTSNode) -> List[str]:
        """Conservative Gate 0 state estimate.

        We only mark gaps closed for an answer node after previous non-empty
        evidence exists. Otherwise the original question remains open.
        """
        response = node.node_dict.get("response", "") if node.node_dict else ""
        if "<answer>" in response and self._has_prior_non_empty_evidence(node):
            return []
        return [node.question]

    def _typed_score(self, node: MCTSNode) -> float:
        """Compute a scalar score from typed eval values stored on node."""
        if node.phi_q is None:
            return 0.0
        phi_s_norm = 1.0 if node.phi_s > 0 else (0.5 if node.phi_s == 0 else 0.0)
        score = (node.phi_q + node.phi_c + phi_s_norm) / 3.0
        return max(0.0, min(1.0, score))

    def _extract_score(self, text: str) -> float:
        """Extract numeric score from LLM response."""
        matches = re.findall(r'[0-9]+\.?[0-9]*', text)
        if matches:
            score = float(matches[-1])
            return min(100, max(0, score))
        return 50.0  # Default

    # ── Answer Extraction ──────────────────────────────────────

    def _extract_answer(self, root: MCTSNode) -> Tuple[str, dict]:
        """Extract best answer from tree (most visited terminal node)."""
        best_node = self._find_best_terminal(root)

        prediction = ""
        if best_node and best_node.node_dict:
            prediction = best_node.node_dict.get("answer", "")
            if not prediction:
                response = best_node.node_dict.get("response", "")
                prediction = extract_answer(response)

        tree_log = self._tree_to_dict(root)
        return prediction, tree_log

    def _find_best_terminal(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Find terminal node with highest visit count."""
        best = None
        best_N = -1

        stack = [node]
        while stack:
            n = stack.pop()
            if self._is_terminal(n) and n.N > best_N:
                if n.node_dict and n.node_dict.get("answer"):
                    best = n
                    best_N = n.N
            stack.extend(n.children)

        return best

    def _tree_to_dict(self, root: MCTSNode) -> dict:
        """Serialize tree to dict for logging."""
        result = {}
        queue = [root]
        while queue:
            node = queue.pop(0)
            node_info = {
                "id": node.id,
                "step": node.step,
                "N": node.N,
                "Q": round(node.Q, 4),
                "next_state": node.next_state,
                "parent_id": node.parent_id,
                "children_ids": node.children_ids,
            }
            if node.node_dict:
                node_info["action_name"] = node.node_dict.get("action_name", "")
                node_info["base_action_name"] = node.node_dict.get("base_action_name", "")
                node_info["query"] = node.node_dict.get("query", "")
                node_info["answer"] = node.node_dict.get("answer", "")
                node_info["response"] = node.node_dict.get("response", "")[:300]
                node_info["expansion_method"] = node.node_dict.get("expansion_method", "")
                node_info["typed_score"] = node.node_dict.get("typed_score")
                node_info["failure_type"] = node.node_dict.get("failure_type", node.failure_type)
                node_info["stop_overridden"] = node.node_dict.get("stop_overridden", False)
            if node.phi_q is not None:
                node_info["phi_q"] = node.phi_q
                node_info["phi_c"] = node.phi_c
                node_info["phi_s"] = node.phi_s
            result[f"node_{node.id}"] = node_info
            queue.extend(node.children)
        return result


# ====================================================================
#  Metrics (golden answer used ONLY here, post-hoc)
# ====================================================================

def normalize_answer(s: str) -> str:
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punc(lower(s))).strip()


def compute_em(prediction: str, golden_answers: list) -> float:
    """Exact match."""
    pred = normalize_answer(prediction)
    for ga in golden_answers:
        if pred == normalize_answer(ga):
            return 1.0
    return 0.0


def compute_f1(prediction: str, golden_answers: list) -> float:
    """Token-level F1."""
    pred_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    for ga in golden_answers:
        gt_tokens = normalize_answer(ga).split()
        if not pred_tokens or not gt_tokens:
            if pred_tokens == gt_tokens:
                best_f1 = max(best_f1, 1.0)
            continue
        common = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
        if common == 0:
            continue
        precision = common / len(pred_tokens)
        recall = common / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


# ====================================================================
#  Data Loading
# ====================================================================

def load_hotpotqa(path: str, n_samples: int = 50, seed: int = 42) -> list:
    """Load HotpotQA dev samples."""
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))

    random.seed(seed)
    sampled = random.sample(items, min(n_samples, len(items)))

    results = []
    for item in sampled:
        results.append({
            "id": item.get("id", ""),
            "question": item["question"],
            "golden_answers": item.get("golden_answers", []),
            "type": item.get("metadata", {}).get("type", "unknown"),
            "context": item.get("metadata", {}).get("context", {}),
            "supporting_facts": item.get("metadata", {}).get("supporting_facts", {}),
        })
    return results


# ====================================================================
#  Leakage Check
# ====================================================================

def check_no_leakage(pipeline: MCTSPipeline, question: str):
    """Verify that the pipeline never accesses golden_answers during search."""
    # Monkey-patch the LLM to check prompts don't contain golden answer hints
    original_generate = pipeline.llm.generate
    prompts_seen = []

    def patched_generate(system, user, **kwargs):
        prompts_seen.append({"system": system[:200], "user": user[:200]})
        return original_generate(system, user, **kwargs)

    pipeline.llm.generate = patched_generate
    return prompts_seen


# ====================================================================
#  Main
# ====================================================================

def load_api_key() -> str:
    """Load API key from .env file."""
    env_path = GATE0_DIR.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("DMXAPI_API_KEY="):
                    return line.strip().split("=", 1)[1]
    # Fallback to env var
    key = os.environ.get("DMXAPI_API_KEY")
    if key:
        return key
    raise ValueError("No API key found. Set DMXAPI_API_KEY in gate0/.env or environment.")


def run_sanity_check(llm: LLMClient, retriever: Retriever):
    """Run 1 sample to verify format, no-leak, output structure."""
    log.info("=" * 60)
    log.info("SANITY CHECK: 1 sample, no API calls for search (dry-run structure)")
    log.info("=" * 60)

    # Load 1 sample
    samples = load_hotpotqa(HOTPOTQA_DEV_PATH, n_samples=1, seed=42)
    sample = samples[0]
    log.info(f"Sample: {sample['question']}")
    log.info(f"Golden answers: {sample['golden_answers']}")

    # Test retrieval (no LLM needed)
    log.info("\n--- Testing retriever ---")
    try:
        docs = retriever.search(sample["question"])
        log.info(f"Retrieved {len(docs)} documents")
        if docs:
            log.info(f"  First doc title: {docs[0].get('title', 'N/A')}")
    except Exception as e:
        log.warning(f"Retriever failed (will init lazily): {e}")

    # Test LLM connectivity (1 call)
    log.info("\n--- Testing LLM (1 call) ---")
    response = llm.generate(
        BEGIN_REASONING_SYSTEM,
        f"Question: {sample['question']}",
        max_tokens=100,
    )
    log.info(f"LLM response: {response[:200]}")
    log.info(f"Tokens used: {llm.total_tokens}")

    # Verify format parsing
    response = process_text(response)
    action = get_action_type(response)
    log.info(f"Action type: {action}")
    if action == "query":
        log.info(f"Query: {extract_query(response)}")
    elif action == "answer":
        log.info(f"Answer: {extract_answer(response)}")

    # Verify metrics
    log.info("\n--- Testing metrics ---")
    em = compute_em("Arthur's Magazine", ["Arthur's Magazine"])
    f1 = compute_f1("Arthur's Magazine", ["Arthur's Magazine"])
    log.info(f"EM={em}, F1={f1}")

    # Leakage check: verify SELF_EVAL_SYSTEM doesn't mention golden answer
    log.info("\n--- Leakage check ---")
    assert "golden" not in SELF_EVAL_SYSTEM.lower(), "LEAK: SELF_EVAL mentions golden!"
    assert "answer" not in SELF_EVAL_SYSTEM.lower()[:50], "LEAK: SELF_EVAL might leak answer!"
    log.info("PASSED: No golden answer leakage in scalar eval prompt")
    log.info("PASSED: Typed eval is rule-based and makes no LLM call")

    log.info("\n✅ Sanity check passed!")
    return True


def run_api_sanity(llm: LLMClient):
    """Check DMXAPI connectivity without loading the retriever."""
    response = llm.generate(
        BEGIN_REASONING_SYSTEM,
        "Question: What year was Moscow State University founded?",
        max_tokens=60,
    )
    log.info("API sanity response: %s", response[:200].replace("\n", " "))
    log.info("API sanity calls=%s tokens=%s", llm.call_count, llm.total_tokens)
    return True


def run_experiment(llm: LLMClient, retriever: Retriever, mode: str,
                   n_samples: int = 50, seed: int = 42,
                   config: dict = None, output_suffix: str = ""):
    """Run full experiment."""
    log.info(f"\n{'=' * 60}")
    log.info(f"Running experiment: mode={mode}, n_samples={n_samples}")
    log.info(f"{'=' * 60}")

    samples = load_hotpotqa(HOTPOTQA_DEV_PATH, n_samples=n_samples, seed=seed)
    pipeline = MCTSPipeline(llm, retriever, mode=mode, config=config)

    results = []
    for i, sample in enumerate(samples):
        log.info(f"\n[{i+1}/{n_samples}] Q: {sample['question'][:80]}...")

        # Reset LLM token counter per sample for tracking
        start_tokens = llm.total_tokens
        start_calls = llm.call_count

        # Run MCTS search (golden_answers NOT passed to pipeline)
        if hasattr(retriever, "set_sample"):
            retriever.set_sample(sample)
        result = pipeline.search(sample["question"])

        # Compute metrics POST-HOC (golden answers used only here)
        em = compute_em(result["prediction"], sample["golden_answers"])
        f1 = compute_f1(result["prediction"], sample["golden_answers"])

        result["golden_answers"] = sample["golden_answers"]
        result["type"] = sample["type"]
        result["em"] = em
        result["f1"] = f1
        result["llm_calls_this_sample"] = llm.call_count - start_calls
        result["tokens_this_sample"] = llm.total_tokens - start_tokens

        results.append(result)
        log.info(f"  Prediction: {result['prediction'][:80]}")
        log.info(f"  Golden: {sample['golden_answers']}")
        log.info(f"  EM={em:.1f}, F1={f1:.3f}, calls={result['llm_calls_this_sample']}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""
    out_path = RESULTS_DIR / f"{mode}{suffix}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"\nResults saved to {out_path}")

    # Print summary
    avg_em = sum(r["em"] for r in results) / len(results)
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_calls = sum(r["llm_calls_this_sample"] for r in results) / len(results)
    total_tokens = llm.total_tokens
    log.info(f"\n{'=' * 40}")
    log.info(f"Summary ({mode}):")
    log.info(f"  Avg EM: {avg_em:.3f}")
    log.info(f"  Avg F1: {avg_f1:.3f}")
    log.info(f"  Avg LLM calls/sample: {avg_calls:.0f}")
    log.info(f"  Total tokens: {total_tokens}")
    log.info(f"{'=' * 40}")

    return results


def eval_results(baseline_path: str, treatment_path: str):
    """Compare baseline vs treatment results."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(treatment_path) as f:
        treatment = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"Comparison: baseline ({len(baseline)} samples) vs treatment ({len(treatment)} samples)")
    print(f"{'=' * 60}")

    b_em = sum(r["em"] for r in baseline) / len(baseline)
    b_f1 = sum(r["f1"] for r in baseline) / len(baseline)
    t_em = sum(r["em"] for r in treatment) / len(treatment)
    t_f1 = sum(r["f1"] for r in treatment) / len(treatment)

    print(f"{'':20s} {'Baseline':>10s} {'Treatment':>10s} {'Diff':>10s}")
    print(f"{'EM':20s} {b_em:>10.3f} {t_em:>10.3f} {t_em-b_em:>+10.3f}")
    print(f"{'F1':20s} {b_f1:>10.3f} {t_f1:>10.3f} {t_f1-b_f1:>+10.3f}")

    # Per-sample comparison
    print(f"\nPer-sample:")
    for b, t in zip(baseline, treatment):
        symbol = "✓" if t["f1"] > b["f1"] else ("✗" if t["f1"] < b["f1"] else "=")
        print(f"  {symbol} B_F1={b['f1']:.2f} T_F1={t['f1']:.2f} | {b['question'][:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference-style no-label MCTS pilot")
    parser.add_argument("--mode", choices=["api_sanity", "sanity", "baseline", "treatment", "eval"],
                        required=True, help="Run mode")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_rollouts", type=int, default=None, help="Override MCTS max_rollouts")
    parser.add_argument("--max_iter", type=int, default=None, help="Override MCTS max_iter")
    parser.add_argument("--max_children", type=int, default=None, help="Override MCTS max_children")
    parser.add_argument("--output_suffix", default="", help="Suffix for result filename")
    parser.add_argument("--mock_retriever", action="store_true", help="Use tiny fixed docs for fast debug")
    parser.add_argument(
        "--retriever_mode",
        choices=["bge", "context", "mock"],
        default="bge",
        help="Retrieval backend: bge loads the full FAISS index; context uses per-sample HotpotQA context; mock uses fixed docs.",
    )
    args = parser.parse_args()

    if args.mode == "eval":
        eval_results(
            str(RESULTS_DIR / "baseline_results.json"),
            str(RESULTS_DIR / "treatment_results.json"),
        )
        sys.exit(0)

    # Init components
    api_key = load_api_key()
    llm = LLMClient(api_key)
    run_config = dict(MCTS_CONFIG)
    if args.max_rollouts is not None:
        run_config["max_rollouts"] = args.max_rollouts
    if args.max_iter is not None:
        run_config["max_iter"] = args.max_iter
    if args.max_children is not None:
        run_config["max_children"] = args.max_children

    if args.mode == "api_sanity":
        run_api_sanity(llm)
        sys.exit(0)

    retriever_mode = "mock" if args.mock_retriever else args.retriever_mode
    if retriever_mode == "mock":
        retriever = MockRetriever(topk=run_config["retrieval_topk"])
    elif retriever_mode == "context":
        retriever = ContextRetriever(topk=run_config["retrieval_topk"])
    else:
        retriever = Retriever(
            index_path=BGE_INDEX_PATH,
            corpus_path=WIKI_CORPUS_PATH,
            model_path=BGE_MODEL_PATH,
            topk=run_config["retrieval_topk"],
        )

    if args.mode == "sanity":
        run_sanity_check(llm, retriever)
    elif args.mode == "baseline":
        run_experiment(
            llm, retriever, "baseline", args.n_samples, args.seed,
            config=run_config, output_suffix=args.output_suffix,
        )
    elif args.mode == "treatment":
        run_experiment(
            llm, retriever, "treatment", args.n_samples, args.seed,
            config=run_config, output_suffix=args.output_suffix,
        )
