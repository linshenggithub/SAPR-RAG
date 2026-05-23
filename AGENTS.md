# AGENTS.md

This repository is a long-term research workspace for a master thesis on Agentic RAG process optimization.

## Research Goal

The project studies:

> 面向复杂问答的 Agentic RAG 多步检索推理过程优化研究

ReasonRAG is not the final research object. It is used as:

1. a representative Agentic RAG baseline;
2. a source of trajectories and badcases;
3. an experimental platform for validating process-level optimization methods.

## Current Research Route

Priority route:

1. Reproduce and stabilize ReasonRAG baselines.
2. Analyze badcases and build a taxonomy.
3. Construct step-level document preference data for HotpotQA.
4. Implement state-aware evidence utility modeling.
5. Implement SAPR-RAG with Query Reward, Evidence Reward, and Stop Reward.
6. Run small-scale experiments before expanding to full datasets.

Do not jump directly into large-scale GRPO unless the reward and data pipeline have been validated.

## External Repositories

Keep external codebases outside this repository unless there is a clear reason to vendor a small adapted module.

- ReasonRAG: `/home/mayi/RAG/ReasonRAG`
- DPA-RAG: `/home/mayi/RAG/DPA-RAG`
- DecEx-RAG: `/home/mayi/RAG/DecEx-RAG`
- RoleRAG: `/home/mayi/RAG/RoleRAG`

When changing external repositories, inspect their git status first and do not revert unrelated local changes.

## File Policy

Do not commit:

- model checkpoints;
- datasets;
- corpora;
- FAISS indexes;
- raw jsonl inference dumps;
- experiment output directories;
- logs and caches;
- private keys or credentials.

Use `.gitignore` for generated artifacts. Small tables, markdown summaries, configs, and figures are allowed when useful for research tracking.

## Experiment Policy

Every meaningful experiment should have:

1. a run config;
2. a result directory;
3. a metrics file or table entry;
4. a short experiment note;
5. a clear label distinguishing real results from expected or planned results.

Never present expected numbers as completed experimental results.

## Collaboration Style

Before code or file changes:

1. check the current git status;
2. keep edits scoped;
3. avoid destructive commands;
4. preserve the user's local work.

When asked to continue the project, first read this file, then inspect the relevant README or roadmap file.

