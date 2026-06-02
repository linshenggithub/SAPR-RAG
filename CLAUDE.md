## Project Context

默认使用中文和用户交流；技术术语、论文标题、模型名、数据集名、路径、命令和代码标识可保留英文。

This project is the long-term research workspace for:

> 面向复杂问答的 Agentic RAG 多步检索推理过程优化研究

When continuing work in this repository, first read `AGENTS.md`, then inspect `README.md`, `ROADMAP.md`, `MANIFEST.md`, `docs/history.md`, `gate0/GATE0_STATUS.md`, and the task-relevant directory. Treat `AGENTS.md` as the full project operating guide and single source of truth for research positioning, paths, server rules, experiment hygiene, and writing standards.

Code-level规则（命名 / 路径 / debug / AI 执行行为约束）的详细版在 `docs/coding_standard.md`，AGENTS.md §11.5 有摘要。**跑实验遇到慢/卡/报错时必须先停下来报告，不允许默默降级**——这条规则违反过会让全部 debug 产物作废，详见 commit `cb867d1`。

Current execution setup:

- Executor: Claude Code connected to GLM, using project-local ARIS skills under `.claude/skills/`.
- External reviewer: Codex MCP (`mcp__codex__codex`) when a skill requires a senior reviewer or cross-model critique.
- Do not use Copilot CLI as the primary executor for this project unless the user explicitly asks.

Current research state:

- SAPR-RAG idea 演化记录：见 `docs/history.md`（v1 → v2 → v3 → v4）。
- 当前在做的事：Gate 0 验证（GPT-4o 重标 50 条 trajectory 看 typed eval 是否区分得出 scalar 区分不出的分支）。
- Gate 0 状态：`gate0/GATE0_STATUS.md`。
- 主要 docs：`docs/proposal.md` / `docs/experiment_plan.md` / `docs/experiment_tracker.md` / `docs/pipeline.md`。
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
