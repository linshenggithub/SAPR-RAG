## Project Context

默认使用中文和用户交流；技术术语、论文标题、模型名、数据集名、路径、命令和代码标识可保留英文。

This project is the long-term research workspace for:

> 面向复杂问答的 Agentic RAG 多步检索推理过程优化研究

When continuing work in this repository, first read `AGENTS.md`, then inspect `README.md`, `ROADMAP.md`, `MANIFEST.md`, `refine-logs/PIPELINE_SUMMARY.md`, and the task-relevant directory. Treat `AGENTS.md` as the full project operating guide and single source of truth for research positioning, paths, server rules, experiment hygiene, and writing standards.

Current execution setup:

- Executor: Claude Code connected to GLM, using project-local ARIS skills under `.claude/skills/`.
- External reviewer: Codex MCP (`mcp__codex__codex`) when a skill requires a senior reviewer or cross-model critique.
- Do not use Copilot CLI as the primary executor for this project unless the user explicitly asks.

Current research state:

- SAPR-RAG has been refined from prompt-judge heuristic scoring into state-conditioned progress / action-value modeling for Agentic RAG trajectory repair.
- Latest core artifacts:
  - `refine-logs/FINAL_PROPOSAL.md`
  - `refine-logs/EXPERIMENT_PLAN.md`
  - `refine-logs/EXPERIMENT_TRACKER.md`
  - `refine-logs/PIPELINE_SUMMARY.md`
  - `MANIFEST.md`
- First experiment priority for the current SAPR-E line: Evidence-only go/no-go on HotpotQA dev subset, matched compute, `debug_result` first.
- Do not interpret earlier "no large-scale GRPO" notes as a ban on SFT/RL. The project should explore SFT, DPO, PRM training, GRPO/online RL, or larger-compute routes when they are the fastest credible path for a novel and feasible idea. The rule is to avoid expensive RL before small-scale evidence and a stable data/reward pipeline, not to avoid RL altogether.

Server and path reminders:

- Local control server: current 4 x RTX 3090 machine.
- Remote experiment server: `rag-5090`, 3 x RTX 5090, user `mayi`.
- Remote ReasonRAG baseline repo: `/home/mayi/ReasonRAG`.
- Remote research repo: `/home/mayi/RAG/agentic-rag-process-optimization`.
- Do not overwrite/delete remote ReasonRAG `output/`, `corpus/`, `indexes/`, `dataset/`, or `training_dataset/`; previous reproduction results live there.

<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 78 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/home/mayi/aris_repo`).
Update with: `bash /home/mayi/aris_repo/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->
