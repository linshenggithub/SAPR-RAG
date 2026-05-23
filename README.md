# Agentic RAG Process Optimization

This repository is the research workspace for:

> 面向复杂问答的 Agentic RAG 多步检索推理过程优化研究

The project studies common failure modes in Agentic RAG methods for complex QA, using ReasonRAG as a representative baseline, diagnostic platform, and experimental testbed. The long-term goal is to develop state-aware evidence utility modeling and state-aware process rewards for repairing multi-step retrieval reasoning trajectories.

## Research Direction

Current title:

> State-Aware Process Rewards for Repairing Reasoning Trajectories in Agentic RAG

Core problem:

Agentic RAG pipelines can generate multiple retrieval and reasoning steps, but the intermediate process is often weakly supervised. Common errors include repeated or drifting subqueries, missing bridge entities, noisy retrieved documents, unsupported intermediate answers, premature stopping, and broken evidence chains.

## Main Threads

1. Baseline reproduction and diagnosis
   - Reproduce ReasonRAG on HotpotQA, 2Wiki, PopQA, MuSiQue, and Bamboogle.
   - Save trajectories, analyze badcases, and build a reusable error taxonomy.

2. State-aware evidence utility modeling
   - Score candidate documents using the original question, previous subqueries, previous evidence, current subquery, and candidate document.
   - Adapt ideas from DPA-RAG-style preference learning to multi-step Agentic RAG states.

3. SAPR-RAG
   - Build state-aware process rewards for query selection, evidence selection, and stop decision.
   - Repair weak reasoning trajectories during inference.

4. Literature and writing
   - Maintain a structured literature survey.
   - Reuse notes for the midterm report, AAAI submission, and master thesis.

## Repository Layout

```text
00_project_management/   Weekly plans, meeting notes, progress reports
01_literature/           Literature survey, taxonomy, paper notes
02_baseline_reasonrag/   ReasonRAG baseline references, configs, results, badcases
03_sapr_rag/             Proposed method, reward modules, ablations
04_experiments/          Unified run configs, logs, metrics, tables, figures
05_reports/              Midterm report, AAAI paper, master thesis materials
06_notes/                Daily notes, ideas, debug logs, writing notes
07_assets/               Figures, diagrams, slides
docs/                    Setup, experiment protocol, coding and writing standards
```

## External Codebases

This repository does not vendor full external baseline code by default. Local working copies are referenced instead.

- ReasonRAG local path: `/home/mayi/RAG/ReasonRAG`
- DPA-RAG local path: `/home/mayi/RAG/DPA-RAG`
- DecEx-RAG local path: `/home/mayi/RAG/DecEx-RAG`
- RoleRAG local path: `/home/mayi/RAG/RoleRAG`

## Current Priority

1. Fix the repository structure and research workflow.
2. Complete the first version of the literature survey.
3. Freeze ReasonRAG baseline configurations and trajectory format.
4. Build HotpotQA step-level evidence preference data.
5. Implement SAPR-RAG V0 on small samples.

