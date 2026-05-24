# AGENTS.md

This repository is the long-term research workspace for:

> 面向复杂问答的 Agentic RAG 多步检索推理过程优化研究

Use this file as the project operating guide. When continuing work in this repository, read this file first, then inspect `README.md`, `ROADMAP.md`, and the task-relevant directory.

## 1. Research Positioning

This project is not a single-method modification of ReasonRAG. ReasonRAG has three roles:

1. representative Agentic RAG baseline;
2. entry point for observing trajectory-level failures;
3. experimental platform for validating process-level optimization.

The broader research field includes:

1. retrieval-augmented generation;
2. complex QA and multi-hop QA;
3. Agentic RAG / retrieval-augmented reasoning;
4. process-supervised RAG;
5. evidence utility modeling and process reward optimization.

The project inherits two core problems from the proposal:

1. mismatch between retriever scores and LLM knowledge preference;
2. entity loss, path drift, noisy evidence, and missing supervision in complex QA reasoning processes.

## 2. Current Research Basis

Completed or already observed:

1. ReasonRAG official LoRA checkpoint has been reproduced.
2. Initial metrics exist for HotpotQA, 2Wiki, PopQA, MuSiQue, and Bamboogle.
3. Badcases include repeated subqueries, query drift, entity loss, retrieval noise, broken evidence chains, unsupported intermediate answers, and premature stopping.
4. Future work should abstract common Agentic RAG problems instead of only patching ReasonRAG.

## 3. Main Research Route

The planned route is:

```text
Fix research direction
  -> Survey Agentic RAG / Process-Supervised RAG / Evidence Utility literature
  -> Summarize common limitations by method taxonomy
  -> Validate those limitations with ReasonRAG reproduction and badcase analysis
  -> Propose two optimization directions
  -> Finish the midterm report
  -> Extract SAPR-RAG as the AAAI 2027 paper line
  -> Expand both directions into the master thesis
```

Do not jump directly into large-scale GRPO unless the data pipeline, reward design, and small-scale verification are stable.

## 4. Two Optimization Directions

### 4.1 State-Aware Evidence Utility Modeling

Existing evidence utility methods usually score:

```text
score(question, document)
```

This project extends it to:

```text
score(original question, history subqueries, history evidence, current subquery, candidate document)
```

Evaluation dimensions:

- Relevance: whether the document answers the current subquery.
- Novelty: whether it adds information not already present in history.
- Supportiveness: whether it supports intermediate reasoning.
- Chain Contribution: whether it advances the final evidence chain.
- Noise Risk: whether it introduces distracting entities or wrong paths.

Stage role:

- midterm report: Evidence Reward module of SAPR-RAG;
- AAAI paper: auxiliary module and ablation component;
- master thesis: independent chapter and first core contribution.

### 4.2 SAPR-RAG: State-Aware Process Reward for Agentic RAG

Problem:

Agentic RAG errors propagate across query, retrieval, evidence, intermediate answer, and stop decision. Existing process supervision often lacks sufficient granularity for these errors.

Goal:

Score candidate subqueries, candidate evidence, and stop decisions under the current reasoning state, then select better actions and reduce error propagation.

Core modules:

- Query Reward: select subqueries that preserve entities, avoid repetition, and advance the next retrieval hop.
- Evidence Reward: select the most useful top-3 documents from top-10 retrieval results under the current state.
- Stop Reward: decide whether current evidence is enough to support the final answer.
- Repair Mechanism: trigger query rewriting, evidence reranking, or continued retrieval for weak steps.

Stage role:

- midterm report: preliminary self-owned method;
- AAAI paper: main submission line;
- master thesis: independent chapter and second core contribution.

## 5. Repository Layout Rules

Use this repository as the unified knowledge base, experiment record, code workspace, and writing material store.

```text
00_project_management/   Weekly plans, meeting notes, progress reports, milestones
01_literature/           Literature survey, taxonomy, paper notes, related work drafts
02_baseline_reasonrag/   ReasonRAG configs, scripts, trajectories, results, badcases, analysis
03_sapr_rag/             SAPR-RAG configs, scripts, reward prompts, reward modules, results, ablations
04_experiments/          Unified run configs, logs, metrics, tables, figures
05_reports/              Midterm report, AAAI 2027 paper, master thesis
06_notes/                Daily notes, idea notes, debug notes, writing notes
07_assets/               Figures, diagrams, slides
docs/                    Setup, experiment protocol, coding standard, writing standard
```

External codebases should normally stay outside this repository:

- ReasonRAG: `/home/mayi/RAG/ReasonRAG`
- DPA-RAG: `/home/mayi/RAG/DPA-RAG`
- DecEx-RAG: `/home/mayi/RAG/DecEx-RAG`
- RoleRAG: `/home/mayi/RAG/RoleRAG`

Do not vendor a whole external repository unless explicitly requested. Prefer links, summaries, small adapted modules, and reproducible configs.

## 6. Literature Management Rules

Literature belongs under:

```text
01_literature/
```

Main survey file:

```text
01_literature/literature_survey.csv
```

Required columns:

```text
Paper ID
Title
Year
Venue
Category
Task
Dataset
Backbone
Retriever
Method Summary
Key Contribution
Limitation
Code URL
Relation to My Work
Read Status
Note Path
```

Fixed literature categories:

```text
1. Traditional RAG and Complex QA
2. Agentic RAG and Retrieval-Augmented Reasoning
3. Evidence Utility and Retriever-LLM Alignment
4. Process Reward and RL for Agentic RAG
```

Paper notes go under:

```text
01_literature/paper_notes/YYYY_PaperName.md
```

Paper reading notes must be written in Chinese by default, so they can be reused directly for weekly reports, the midterm report, the AAAI draft, and the master thesis. English technical terms may be kept when needed. Important figures or tables from the original paper should be included when available, but the note should cite the source paper and avoid excessive verbatim copying. Follow the detailed standard in `docs/literature_note_standard.md`.

Use the fixed note structure:

```markdown
# Paper Title

## 0. 阅读结论
## 1. 核心信息
## 2. 摘要与问题定义
## 3. 图表速读
## 4. 方法拆解
## 5. 实验设计与结果解读
## 6. 论文贡献、局限与证据强度
## 7. 与本课题的关系
## 8. 可复现与代码阅读线索
## 9. 可用于写作的中文表述
## 10. 后续行动
```

For each key claim, mark whether it is `[论文明确提出]`, `[基于方法/实验设置推断]`, or `[本课题 badcase 对齐]`. Store key paper artifacts under `01_literature/paper_notes/images/<note_stem>/key_figures/` with an `index.md` file. Prefer original figures extracted from arXiv source packages by parsing `figure` / `includegraphics`; use PDF caption crops only when source figures are unavailable or when extracting LaTeX tables.

When the user asks for latest or recent papers, browse and prefer primary sources such as ACL Anthology, OpenReview, arXiv, official proceedings, and official project pages.

## 7. Experiment Management Rules

ReasonRAG reproduction belongs under:

```text
02_baseline_reasonrag/
```

SAPR-RAG method work belongs under:

```text
03_sapr_rag/
```

Each meaningful experiment must have an independent config:

```text
04_experiments/run_configs/run_YYYYMMDD_xxx.yaml
```

Required config fields:

```yaml
run_name:
date:
server:
gpu:
model:
adapter:
dataset:
retriever:
corpus:
top_k:
max_steps:
temperature:
num_candidates:
method:
baseline:
output_path:
log_path:
metric_path:
note:
```

Each experiment should produce or link:

```text
1. run_config.yaml
2. raw_output.jsonl
3. trajectories.jsonl
4. metrics.json
5. badcases.jsonl
6. experiment_note.md
```

Experiment tables go under:

```text
04_experiments/tables/
```

Experiment figures go under:

```text
04_experiments/figures/
```

Always label results clearly:

- `real_result`: actually run and verified;
- `debug_result`: small-scale sanity check;
- `expected_result`: planned or estimated;
- `failed_run`: failed but diagnostically useful.

Never present expected numbers as completed experimental results.

## 8. Daily and Weekly Notes

Daily notes go under:

```text
06_notes/daily_notes/YYYY-MM-DD.md
```

Daily template:

```markdown
# YYYY-MM-DD Daily Research Note

## 1. Today's Goal
## 2. Completed Work
## 3. Experimental Runs
## 4. Problems Found
## 5. Decisions Made
## 6. Papers Read
## 7. Tomorrow's Plan
```

Weekly plans and summaries go under:

```text
00_project_management/weekly_plans/YYYY-WW.md
```

Weekly template:

```markdown
# Week YYYY-WW Plan and Summary

## 1. Weekly Goal
## 2. Completed Items
## 3. Key Experimental Results
## 4. Key Reading Progress
## 5. Risks
## 6. Next Week Plan
```

## 9. Branch and Commit Rules

Main branch:

```text
main
```

Development branch:

```text
dev
```

Commit format:

```text
[type] short description
```

Allowed types:

- `[lit]`: literature;
- `[exp]`: experiment;
- `[feat]`: method feature;
- `[fix]`: bug fix;
- `[analysis]`: analysis;
- `[writing]`: writing;
- `[docs]`: documentation;
- `[config]`: configuration.

Examples:

```text
[lit] add notes for Self-RAG and FLARE
[exp] run ReasonRAG-LoRA on HotpotQA
[feat] implement state-aware query reward
[analysis] add badcase taxonomy for 2Wiki
[writing] draft midterm report related work
```

## 10. Resource Usage Rules

Known resources:

- Server A: RTX 5090 machines, used for main experiments, SAPR-RAG main method, candidate trajectory generation, LLM-as-Judge scoring, and later reward model training.
- Server B: 4 x RTX 3090, used for baseline reruns, retrieval preprocessing, badcase classification, ablations, and metric recomputation.
- No cross-server distributed training by default.
- Use task-level parallelism across machines.

Every experiment should record server, GPU, model, retriever, corpus, output path, and log path.

## 11. File and Git Hygiene

Do not commit:

- model checkpoints;
- large datasets;
- corpora;
- FAISS indexes;
- raw inference dumps;
- large `.jsonl` trajectory files unless intentionally sampled and small;
- experiment output directories;
- logs and caches;
- private keys or credentials.

Commit small configs, markdown summaries, scripts, tables, and figures when they help research tracking.

Before modifying files:

1. inspect `git status`;
2. keep edits scoped;
3. do not revert unrelated local changes;
4. do not delete user work;
5. avoid destructive commands unless explicitly requested.

When changing external repositories, inspect their git status first.

## 12. Stage Timeline

Key milestones:

```text
2026-05-23 to 2026-05-31: direction convergence and baseline freeze
2026-06-01 to 2026-06-07: literature survey V1
2026-06-08 to 2026-06-15: SAPR-RAG V0
2026-06-16 to 2026-06-23: expanded experiments and ablations
2026-06-24 to 2026-06-30: midterm report finalization
2026-07-01 to 2026-07-05: AAAI method freeze
2026-07-06 to 2026-07-15: AAAI main and ablation experiments
2026-07-16 to 2026-07-28: AAAI paper writing and submission
2026-08 to 2026-12: master thesis experiment expansion
2027-01 to 2027-04: master thesis writing
2027-05 to 2027-06: revision and defense
```

AAAI 2027 target:

```text
Title: State-Aware Process Rewards for Repairing Reasoning Trajectories in Agentic RAG
```

Master thesis target:

```text
Title: 面向复杂问答的 Agentic RAG 多步检索推理过程优化研究
```

## 13. Current Execution Order

Follow this execution order unless the user explicitly changes priorities:

1. Maintain the GitHub research repository.
2. Build and update the repository directory structure.
3. Maintain `README.md`, `ROADMAP.md`, `TODO.md`, `CHANGELOG.md`, and this file.
4. Build literature survey and paper notes.
5. Write `research_direction.md`.
6. Freeze ReasonRAG-LoRA baseline.
7. Save and summarize complete trajectories.
8. Complete 30 core paper survey.
9. Summarize two common limitations.
10. Implement SAPR-RAG V0.
11. Run small experiments on HotpotQA, 2Wiki, MuSiQue, and Bamboogle.
12. Build main and ablation tables.
13. Write the midterm report.
14. Freeze the AAAI method version.
15. Run AAAI main experiments and ablations.
16. Prepare AAAI 2027 submission.
17. Expand work one and work two for the master thesis.

## 14. Core Project Statement

本课题面向复杂问答中的 Agentic RAG 多步检索推理过程优化问题，系统调研任务性能改进类 Agentic RAG、生成质量与证据质量改进类 RAG、证据效用建模和过程监督奖励建模等方向，归纳出现有方法在状态感知证据效用建模和细粒度过程奖励修复方面的共性不足。课题以 ReasonRAG 作为代表性 baseline 进行复现和 badcase 诊断，提出状态感知证据效用建模与状态感知过程奖励轨迹修复两项工作。本课题使用 GitHub 作为统一知识库和项目管理平台，对文献、代码、实验、笔记、报告和论文进行全过程管理，保证研究过程可追踪、实验结果可复现、写作材料可复用。
