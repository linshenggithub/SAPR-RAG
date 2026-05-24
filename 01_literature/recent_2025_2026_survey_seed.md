# Recent Literature Seed: Agentic RAG Process Optimization

Date: 2026-05-24

Scope:

- Time window: roughly 2025-05-24 to 2026-05-24.
- Priority venues: NeurIPS, ICLR, COLM, ACL, NAACL, EMNLP, TACL.
- Focus: Agentic RAG, multi-hop QA, search/retrieval RL, evidence utility, reranking, and process supervision.

## 1. High-Priority Papers

| ID | Paper | Venue | Category | Why It Matters |
| --- | --- | --- | --- | --- |
| P001 | Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning | NeurIPS 2025 | Process-supervised Agentic RAG | This is ReasonRAG, the main baseline and diagnostic entry point. It frames process supervision around query generation, evidence extraction, and answer generation. |
| P002 | Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning | COLM 2025 | Outcome-supervised search RL | Key outcome-RL baseline for search-augmented reasoning. It learns multi-turn search behavior but mainly uses final answer reward. |
| P003 | HiPRAG: Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation | ICLR 2026 | Hierarchical process reward | Directly addresses over-search and under-search with parsable steps and knowledge-grounded process rewards. |
| P004 | DecEx-RAG: Boosting Agentic Retrieval-Augmented Generation with Decision and Execution Optimization via Process Supervision | EMNLP Industry 2025 | Decision/execution process supervision | Separates decision and execution optimization, and explicitly attacks sparse global reward in Agentic RAG. |
| P005 | R3-RAG: Learning Step-by-Step Reasoning and Retrieval for LLMs via Reinforcement Learning | Findings EMNLP 2025 | Step-by-step reasoning and retrieval RL | Uses cold start plus RL, with outcome reward and relevance-based document verification as process reward. |
| P006 | InfoGain-RAG: Boosting Retrieval-Augmented Generation through Document Information Gain-based Reranking and Filtering | EMNLP 2025 | Evidence utility / reranking | Defines document information gain by comparing LLM confidence with and without a document, then trains a reranker. |
| P007 | Utility-Focused LLM Annotation for Retrieval and Retrieval-Augmented Generation | EMNLP 2025 | Utility annotation for retrieval | Uses LLM-generated utility annotations to improve retrieval and RAG, including experiments on HotpotQA. |
| P008 | REARANK: Reasoning Re-ranking Agent via Reinforcement Learning | EMNLP 2025 | Reasoning reranker / RL | Trains a reasoning listwise reranking agent with RL, useful for evidence ranking but not full Agentic RAG control. |
| P009 | GRADA: Graph-based Reranking against Adversarial Documents Attack | EMNLP 2025 | Noise robustness / adversarial reranking | Shows that semantically similar but misleading documents can attack RAG, matching the noisy retrieval badcases observed in ReasonRAG. |
| P010 | mt RAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems | TACL 2025 | Multi-turn RAG evaluation | Not a ReasonRAG-style training method, but useful for arguing that multi-step/multi-turn RAG evaluation is under-specified. |

## 2. Slightly Earlier But Still Necessary Context

These are just outside the strict one-year window or are older foundational works, but they are necessary for positioning:

| Paper | Venue/Year | Why Keep It |
| --- | --- | --- |
| DPA-RAG: Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation | WWW 2025 / arXiv 2024 | Important for retriever-LLM preference alignment and preference-data construction. |
| PA-RAG: RAG Alignment via Multi-Perspective Preference Optimization | NAACL 2025 | Important for generator-side RAG preference alignment. |
| Self-RAG | ICLR 2024 | Classic adaptive retrieval/self-reflection baseline. |
| ReAct | ICLR 2023 | Classic reasoning-action framework for tool-using agents. |
| IRCoT | ACL 2023 | Important iterative retrieval baseline for multi-hop QA. |

## 3. Do These Papers Share ReasonRAG-like Problems?

Yes, but they expose different slices of the same deeper issue.

### 3.1 Shared Problem A: Outcome Reward Is Too Sparse

ReasonRAG, HiPRAG, DecEx-RAG, R3-RAG, and Search-R1 all point to the same weakness:

- final answer reward is delayed;
- correct early steps can be penalized because of later mistakes;
- wrong intermediate retrieval or evidence extraction is hard to diagnose;
- RL exploration is inefficient.

This matches our badcases where the model takes several steps, retrieves partial evidence, but fails to turn it into a complete evidence chain.

### 3.2 Shared Problem B: Search Decisions Are Not Reliably Controlled

HiPRAG emphasizes over-search and under-search. DecEx-RAG separates decision and execution. R3-RAG trains step-by-step retrieval. These are all variants of the same issue observed in ReasonRAG:

- the model does not know when to search;
- subqueries can repeat or drift;
- retrieved evidence does not always advance the reasoning state;
- stopping is often premature or unsupported.

### 3.3 Shared Problem C: Retrieved Documents Are Not Equal in Utility

InfoGain-RAG, Utility-Focused Annotation, REARANK, DPA-RAG, and GRADA all show that relevance alone is not enough:

- documents can be topically relevant but useless;
- documents can be semantically similar but misleading;
- useful evidence may be ranked below noisy passages;
- generator preference and retriever score can be misaligned.

This matches our ReasonRAG badcases where the gold evidence exists in top-50/top-100 but does not enter the final evidence chain.

### 3.4 Shared Problem D: Current Methods Still Under-Model Reasoning State

This is the most important gap for our work.

Most evidence utility methods score:

```text
score(question, document)
```

or:

```text
score(question, document set)
```

But Agentic RAG needs:

```text
score(original question, previous subqueries, previous evidence, current subquery, candidate document)
```

The difference is crucial in multi-hop QA. A document can be irrelevant to the original question in isolation, but highly valuable after a bridge entity has been found. Conversely, a document can be relevant to the original question but redundant or misleading under the current reasoning state.

## 4. The Logic Gap We Can Claim

The strongest thesis-level gap is:

> Existing Agentic RAG work has recognized that multi-step retrieval reasoning needs process supervision, and existing evidence-utility work has recognized that retrieved documents differ in usefulness. However, these two lines remain weakly connected: current methods rarely model document utility, query quality, and stop decisions jointly under the current reasoning state. As a result, they still lack a unified state-aware mechanism for diagnosing and repairing trajectory-level errors in multi-hop QA.

In shorter paper language:

> The missing piece is state-aware process reward: a step-level reward mechanism that evaluates whether each query, retrieved document, and stop decision contributes new, supportive, and non-noisy information toward completing the evidence chain.

## 5. Why SAPR-RAG Is a Reasonable Direction

SAPR-RAG can be positioned as filling the gap between two existing research lines:

1. Process-supervised Agentic RAG
   - ReasonRAG, HiPRAG, DecEx-RAG, R3-RAG
   - Strength: optimize multi-step search/reasoning behavior
   - Weakness: process rewards are often action-level or relevance-level, not fully state-aware evidence-chain rewards

2. Evidence utility and reranking
   - DPA-RAG, InfoGain-RAG, Utility-Focused Annotation, REARANK, GRADA
   - Strength: distinguish useful vs. noisy evidence
   - Weakness: mostly static query-document or listwise ranking, not embedded into Agentic RAG trajectory control

SAPR-RAG should connect them:

```text
state-aware query reward
  + state-aware evidence reward
  + state-aware stop reward
  -> trajectory repair for multi-hop Agentic RAG
```

## 6. Immediate Reading Order

Read in this order:

1. ReasonRAG
2. HiPRAG
3. DecEx-RAG
4. R3-RAG
5. Search-R1
6. InfoGain-RAG
7. Utility-Focused Annotation
8. REARANK
9. DPA-RAG
10. PA-RAG

The first five define the Agentic RAG/process-reward line. The next five define the evidence utility/preference alignment line.

## 7. Primary Sources Checked

- ReasonRAG / Process vs. Outcome Reward: https://openreview.net/forum?id=h3LlJ6Bh4S
- Search-R1: https://openreview.net/pdf?id=Rwhi91ideu
- HiPRAG: https://openreview.net/pdf/eec0d2003efb6373a48484da49cd8f466110b7c3.pdf
- DecEx-RAG: https://aclanthology.org/2025.emnlp-industry.99/
- R3-RAG: https://aclanthology.org/2025.findings-emnlp.554/
- InfoGain-RAG: https://aclanthology.org/2025.emnlp-main.365/
- Utility-Focused LLM Annotation: https://aclanthology.org/2025.emnlp-main.88/
- REARANK: https://aclanthology.org/2025.emnlp-main.125/
- GRADA: https://aclanthology.org/2025.emnlp-main.1132/
- mt RAG: https://aclanthology.org/2025.tacl-1.36/
