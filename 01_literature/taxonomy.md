# Literature Taxonomy

## 1. Traditional RAG and Complex QA

Focus:

- Open-domain QA
- Multi-hop QA
- Dense retrieval
- Reranking
- FiD-style generation

Purpose:

Use this category to build the background section for the midterm report and master thesis.

## 2. Agentic RAG and Retrieval-Augmented Reasoning

Focus:

- Active retrieval
- Adaptive retrieval
- Multi-round retrieval
- Query decomposition
- Reasoning-enhanced retrieval
- RL-based Agentic RAG
- Process-supervised RAG

Representative methods:

- ReAct
- FLARE
- Self-RAG
- Adaptive-RAG
- DeepRAG
- Search-R1
- ReasonRAG
- DecEx-RAG
- ProRAG

Key limitation to verify:

Existing Agentic RAG methods introduce multi-step retrieval and dynamic decisions, but often under-model whether each action truly advances the evidence chain.

## 3. Evidence Utility and Retriever-LLM Alignment

Focus:

- Document reranking
- Document refinement
- Document compression
- Evidence utility modeling
- Retriever-LLM preference alignment
- Noise robustness

Representative methods:

- DPA-RAG
- BIDER
- RECOMP
- Chain-of-Note
- Utility-based passage selectors

Key limitation to verify:

Most evidence utility methods score static question-document or question-document-set usefulness, rather than usefulness conditioned on the current reasoning state.

## 4. Process Reward and RL for Agentic RAG

Focus:

- Outcome reward
- Process reward
- Step-level reward
- Trajectory preference
- MCTS data construction
- DPO, GRPO, PPO
- LLM-as-Judge process scoring
- Reward model distillation

Purpose:

Support SAPR-RAG method design and the AAAI paper plan.

