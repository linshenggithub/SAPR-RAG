# DecEx-RAG: Boosting Agentic Retrieval-Augmented Generation with Decision and Execution Optimization via Process Supervision

## 1. Basic Information

- Year: 2025
- Venue: EMNLP 2025 Industry
- Paper: https://aclanthology.org/2025.emnlp-industry.99/
- PDF: https://aclanthology.org/2025.emnlp-industry.99.pdf
- Code: https://github.com/sdsxdxl/DecEx-RAG
- Task: Agentic RAG QA
- Dataset: PopQA, NQ, AmbigQA, HotpotQA, 2WikiMultiHopQA, Bamboogle
- Backbone: 待代码确认
- Retriever: 待代码确认

## 2. Motivation

- [论文明确提出] Search-R1 类 outcome RL 存在 exploration inefficiency、sparse reward、ambiguous global feedback。
- [论文明确提出] Agentic RAG 的过程可以拆成 decision 和 execution，二者需要不同的优化信号。
- [基于方法/实验设置推断] 这与 ReasonRAG 的过程监督路线一致，但更强调执行过程中的动作选择。

## 3. Method

- [论文明确提出] 将 Agentic RAG 建模为包含 decision 与 execution 的 MDP。
- [论文明确提出] 使用 process-level policy optimization 替代只依赖 final answer 的全局反馈。
- [论文明确提出] 通过 pruning strategy 提升数据构建效率。

## 4. Experiments

- [论文明确提出] 实验覆盖六个 QA 数据集。
- [论文明确提出] 论文报告平均绝对提升约 6.2%，并且剪枝策略使数据构建效率接近 6 倍。
- [基于方法/实验设置推断] 实验数据集与 ReasonRAG 高度重合，适合作为同类 baseline 或 related work。

## 5. Main Results

- [论文明确提出] 过程级 decision/execution 优化能显著优于 outcome-only RL。
- [论文明确提出] 数据构建效率是该方法的重要卖点。

## 6. Limitations

- [基于方法/实验设置推断] Decision/execution 分解仍比较粗，未把证据效用单独建模为 `U(d | q, s_t)`。
- [基于方法/实验设置推断] Retrieval noise、entity loss、unsupported intermediate answers 更像由过程优化间接缓解，而不是显式奖励对象。
- [基于方法/实验设置推断] 需要代码确认其 process label、prompt 和 pruning 规则。

## 7. Relation to My Research

- SAPR-RAG 可借用 DecEx-RAG 的 MDP 表述，但把 state 定义得更具体：原问题、历史 query、历史 evidence、当前中间结论、未闭合槽位。
- 与 DecEx-RAG 的差异应写清楚：SAPR-RAG 不只优化 decision/execution，而是细化为 query/evidence/stop 三类状态感知奖励。

## 8. Useful Sentences for Writing

- DecEx-RAG 说明 Agentic RAG 的优化对象不应只是最终答案，而应包括决策与执行过程。
- 然而，决策/执行层面的过程监督仍不足以刻画当前状态下候选证据的边际效用。

## 9. Follow-up Ideas

- B 档代码调研时重点看其 MDP state/action/reward 的具体实现。
- 对比 DecEx-RAG 的 decision reward 与 SAPR-RAG 的 Query/Stop Reward 边界。
