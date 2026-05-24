# ProRAG: Process-Supervised Reinforcement Learning for Retrieval-Augmented Generation

## 1. Basic Information

- Year: 2026
- Venue: arXiv
- Paper: https://arxiv.org/abs/2601.21912
- PDF: https://arxiv.org/pdf/2601.21912
- Code: https://github.com/lilinwz/ProRAG
- Task: 检索增强问答、过程监督强化学习
- Dataset: PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle
- Backbone: 待代码确认
- Retriever: 待代码确认

## 2. Motivation

- [论文明确提出] Outcome RL 存在 reward sparsity 和 credit assignment 问题。
- [论文明确提出] 长轨迹中最终答案错误无法定位具体出错步骤。
- [论文明确提出] RAG 过程中存在 process hallucination，需要 step-level feedback。

## 3. Method

- [论文明确提出] 四阶段流程：SFT warmup、MCTS-based PRM、PRM-guided refinement、process-supervised RL。
- [论文明确提出] 将 step-level process reward 与 outcome signal 结合。
- [基于方法/实验设置推断] 其 PRM 是强相关工作，但不一定细分 query/evidence/stop 三类错误来源。

## 4. Experiments

- [论文明确提出] 实验覆盖 PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle。
- [论文明确提出] 报告 EM/F1 等指标，并强调 process-supervised RL 的收益。
- [基于方法/实验设置推断] 数据集与 ReasonRAG 完全接近，是 SAPR-RAG 的直接竞争或对照方法之一。

## 5. Main Results

- [论文明确提出] Process reward 能缓解 outcome reward 的稀疏和 credit assignment 问题。
- [论文明确提出] PRM-guided refinement 可以提升后续 RL 训练质量。

## 6. Limitations

- [基于方法/实验设置推断] Step reward 可能仍是整体步骤质量判断，不一定能解释 query、evidence、stop 哪个模块失败。
- [基于方法/实验设置推断] Evidence utility 是否状态化需要代码或方法细节进一步确认。
- [基于方法/实验设置推断] 如果 PRM 依赖 MCTS 和外部 judge，成本和稳定性也需要评估。

## 7. Relation to My Research

- ProRAG 是最接近 SAPR-RAG 的 process-supervised RL 竞争方向之一。
- SAPR-RAG 需要清楚说明差异：不是泛化的 PRM，而是把过程奖励拆成 Query Reward、Evidence Reward、Stop Reward，并绑定当前 state。

## 8. Useful Sentences for Writing

- ProRAG 进一步证明，RAG 训练中的 credit assignment 不能只靠终答监督解决。
- 但要修复具体 Agentic RAG badcase，还需要将过程奖励拆解到 query、evidence 和 stop 的可解释动作层。

## 9. Follow-up Ideas

- C 档代码调研时重点看 ProRAG 的 PRM 数据格式、MCTS 轨迹和 reward target。
- 尝试用 SAPR-RAG 三头 reward 复现 ProRAG 中 step reward 的一部分功能。
