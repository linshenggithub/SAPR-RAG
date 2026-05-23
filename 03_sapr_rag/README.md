# SAPR-RAG

SAPR-RAG stands for:

> State-Aware Process Reward for Agentic RAG

The goal is to repair multi-step retrieval reasoning trajectories by scoring intermediate actions conditioned on the current reasoning state.

## Core Modules

| Module | Role |
| --- | --- |
| Query Reward | Select subqueries that preserve entities, avoid repetition, and advance the next hop. |
| Evidence Reward | Select documents that are useful under the current question, history, and subquery state. |
| Stop Reward | Decide whether current evidence is sufficient to support the final answer. |
| Repair Mechanism | Trigger query rewrite, evidence reranking, or continued retrieval for weak steps. |

## V0 Pipeline

```text
Question
  -> Generate candidate subqueries
  -> State-aware Query Reward
  -> Retrieve top-k documents
  -> State-aware Evidence Reward
  -> Generate intermediate answer
  -> State-aware Stop Reward
  -> Continue or answer
```

