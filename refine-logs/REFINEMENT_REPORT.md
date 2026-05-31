# Refinement Report — Reframing SAPR-RAG into State-Conditioned Progress / Action-Value Modeling

**Date**: 2026-05-28

## A) Starting point (what we are moving away from)
The initial SAPR-RAG framing looked like:
- “Use an LLM judge to score **query / evidence / stop** with a checklist (relevance/novelty/supportiveness/etc.).”

Main issue: reviewers can (reasonably) dismiss this as a heuristic reranker + rule-based repair wrapper.

## B) Refined thesis (what we are moving toward)
We reframe SAPR-RAG as **state-conditioned progress / action-value modeling** for Agentic RAG.

### Key change
Instead of hand-defining multiple subjective dimensions, each module estimates:
- a **progress value** of a state, `V(s_t)` (expected probability of a correct, evidence-supported final answer under the remaining budget), and/or
- an **action-value** `Q(s_t, a)` (expected progress after taking candidate action `a`).

Then SAPR-RAG selects actions that maximize **expected progress**, not checklist agreement.

### What counts as “state”
`s_t` is explicitly trajectory-conditioned and includes:
- original question `q0`
- step index / remaining budget
- history of subqueries and retrieved/selected evidence
- intermediate answer / extracted entities (optional)
- (optional) a compact “what is missing” representation (“gap”) derived from history

This makes SAPR fundamentally different from static rerankers.

## C) What we keep vs reject
### Kept
- Modular action space: **query selection**, **evidence selection**, **stop decision**.
- “State-aware evidence utility” as the strongest, easiest-to-validate module.
- Prompt judges as **a bootstrapping device** for labels / distillation (V0), not as the end contribution.

### Explicitly rejected
- Large-scale GRPO / online RL / MCTS as a first step.
- Increasing retrieval calls, step counts, or token budgets to create confounded gains.
- Treating the method as only “top-3 from top-10 reranking” without a progress/value story.

## D) Novelty positioning (tightened)
We position SAPR-RAG as:
- **A modular, state-conditioned critic layer** that estimates progress/action-values for heterogeneous actions in Agentic RAG, enabling **online trajectory repair under fixed budgets**.

Differentiation sketch:
- **ReasonRAG**: has process rewards; SAPR emphasizes *state-conditioned value/advantage modeling* and *compute-matched trajectory repair* with modular Q heads and explicit state features.
- **DecEx-RAG**: MDP for decomposition/execution; SAPR targets *retrieval-reasoning control* and integrates as a critic wrapper.
- **HiPRAG**: search control; SAPR generalizes to query/evidence actions in the same value framework.
- **ProRAG**: learned PRM + search/RL; SAPR aims to show modular critics + state features yield gains without online RL.
- **DPA-RAG-style rerankers**: preference learning for doc ranking; SAPR extends to state-conditioned marginal utility and covers query/stop actions.

## E) Planning gate check
- **Final method thesis**: SAPR-RAG is a state-conditioned value/Q modeling layer that selects query/evidence/stop actions to maximize expected progress and repairs weak steps during inference.
- **Dominant contribution**: state-conditioned action-value modeling for multi-step retrieval reasoning actions.
- **Rejected complexity**: large-scale GRPO, MCTS, extra retrieval budget, prompt-judge-as-final.
- **Remaining concerns**: novelty vs existing PRM/process reward systems; compute confounds; whether state features truly matter.
- **Frontier primitive**: optional only (prompt judges for bootstrapping). The core claim must survive after distillation.
