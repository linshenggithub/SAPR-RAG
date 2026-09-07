# 文献调研入口

## 当前重点

- [GRPO 改进与长程 Agent 信用分配系统调研](related_work_drafts/grpo_long_horizon_credit_assignment_survey_20260906.md)：
  覆盖 GRPO 优化技巧、推理过程信用分配、多轮 Agent 信用分配和 Agentic RAG
  过程监督；包含 ICML/NeurIPS/ICLR/COLM 2024–2026 正式论文与预印本分层，
  并给出 SAPR-RAG 方向二的创新边界和实验建议。
- [OPD 用于 Agentic RAG 的顶会文献调研](related_work_drafts/opd_agentic_rag_survey.md)：
  覆盖外部 teacher OPD、自蒸馏和 Agentic RAG 的结合。
- [研究方向与问题分类](related_work_drafts/research_direction_and_problem_taxonomy.md)：
  汇总 Agentic RAG 的 Query、Evidence 和 Stop 控制问题。

## 结构化台账

- [literature_survey.csv](literature_survey.csv)：论文级结构化索引；
- [taxonomy.md](taxonomy.md)：Agentic RAG 方法谱系；
- [paper_notes/](paper_notes/)：单篇论文笔记。

## 使用规则

1. venue 与录用状态优先以官方 proceedings、OpenReview 或 ACL Anthology 为准；
2. arXiv、CoRR、withdrawn 和 rejected 工作可用于方法启发，但不得写成顶会证据；
3. 写论文前回到一手论文核对公式、实验设置和限制，不直接引用二手综述结论；
4. 新增论文时同步更新 `literature_survey.csv`，并记录与 SAPR-RAG 的具体关系。
