# HiPRAG: Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation

## 1. Basic Information

- Year: 2026
- Venue: ICLR 2026
- Paper: https://openreview.net/forum?id=Gt4v9WBPzm
- PDF: https://openreview.net/pdf?id=Gt4v9WBPzm
- Code: https://github.com/qualidea1217/HiPRAG
- Task: Agentic RAG QA
- Dataset: NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle
- Backbone: Qwen2.5, Llama-3.2
- Retriever: 搜索/检索工具，细节待代码确认

## 2. Motivation

- [论文明确提出] Agentic RAG 存在 over-search 和 under-search：搜太多浪费成本且引入噪声，搜太少会导致信息不足。
- [论文明确提出] 只靠最终答案无法稳定训练模型何时搜索、何时不搜索。
- [基于方法/实验设置推断] HiPRAG 的重点是 search necessity，而不是具体证据 utility。

## 3. Method

- [论文明确提出] 将 reasoning trajectory 拆成可解析步骤。
- [论文明确提出] 使用 hierarchical process rewards 评估每一步 search / non-search 的必要性。
- [基于方法/实验设置推断] 该层级奖励为 Stop Reward 提供直接参考，但 Evidence Reward 仍需额外设计。

## 4. Experiments

- [论文明确提出] 实验覆盖七个 QA benchmarks。
- [论文明确提出] 报告 Qwen2.5 / Llama-3.2 平均准确率提升，并显著降低 over-search。
- [论文明确提出] 指标包括 accuracy、over-search rate、under-search rate 等效率/行为指标。

## 5. Main Results

- [论文明确提出] 层级过程奖励能改善模型的搜索决策，使模型更有效地决定是否检索。
- [论文明确提出] Over-search / under-search 可被显式量化和优化。

## 6. Limitations

- [基于方法/实验设置推断] HiPRAG 解决“该不该搜”，但没有完整回答“搜什么”和“检索到的哪个文档在当前状态最有用”。
- [基于方法/实验设置推断] Query 内容质量、entity preservation、document noise risk 不是其最核心的显式奖励对象。
- [基于方法/实验设置推断] 需要代码确认其 hierarchical reward 的具体解析和判分规则。

## 7. Relation to My Research

- Stop Reward 可以直接对标 HiPRAG。
- SAPR-RAG 的创新点应是从 search necessity 扩展到 state-aware query/evidence/stop utility。

## 8. Useful Sentences for Writing

- HiPRAG 将 Agentic RAG 的搜索控制问题从隐式行为转化为可优化的层级过程奖励。
- 但搜索必要性并不等同于证据效用；复杂问答还需要判断当前状态下哪条证据能推进证据链。

## 9. Follow-up Ideas

- 用 HiPRAG 的 over-search / under-search 指标评价 ReasonRAG badcase。
- 在 SAPR-RAG 中加入“未闭合槽位”作为 under-search 判据。
