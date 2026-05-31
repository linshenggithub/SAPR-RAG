# Review Summary (Round 1) — SAPR-RAG

**Input source**: `.aris/traces/research-review/20260528_run03/review_round1.md`
**Date**: 2026-05-28

## 1) Reviewer’s main verdict
- Direction is promising but **not yet publishable** if positioned as “prompt-judge scoring of query/evidence/stop.”
- Publishable framing should be **state-conditioned progress / action-value modeling** for Agentic RAG, with **compute-matched** evidence that modular critics improve trajectory quality beyond:
  - generic step-level PRMs,
  - static rerankers,
  - search-control heuristics.

## 2) What the reviewer found strong
- The failure taxonomy is real: query repetition/drift, missing bridge entities, retrieval noise, unsupported intermediate answers, premature stopping.
- “State-aware evidence utility” is the clearest novelty wedge: extend from `score(query, doc)` to `score(question, history, current step, doc)`.
- Modular decomposition (query/evidence/stop) improves diagnosis + ablation clarity.

## 3) Critical risks to address
- **Incremental novelty risk** vs nearby process-reward / MDP / PRM lines (ReasonRAG, DecEx-RAG, HiPRAG, ProRAG).
- Prompt judges are not a contribution; they are **costly/unstable/answer-leaky** and easily dismissed.
- Multi-dimension evidence checklists may be redundant; reviewers want a **progress/value** interpretation.
- Compute confounds must be eliminated: **no extra retrieval calls, candidates, or tokens** unless matched.

## 4) Concrete “minimum experiments” demanded
- State-aware evidence utility vs non-state reranker: final F1, supporting-fact recall, chain completeness.
- Query reward reduces drift/repetition under same budget; compare to candidate voting/self-consistency.
- Stop reward improves over/under-search under same/better F1; compare to HiPRAG-style heuristics.
- Modular heads > single scalar PRM under same data.
- Distill prompt reward into lightweight models.
- Generalization: train/dev on HotpotQA/2Wiki, test on MuSiQue/Bamboogle.

## 5) Success criteria implied
- Demonstrate improvements beyond ReasonRAG-LoRA under matched compute (target: **~+3 F1 avg** as an aspirational bar; smaller but consistent gains + trajectory-level improvements may still be acceptable).
- Provide trajectory-level evidence: fewer repeated queries, less entity drift, fewer premature stops, fewer unsupported intermediate claims.
