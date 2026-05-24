# REX-RAG: Reasoning Exploration with Policy Correction in Retrieval-Augmented Generation

## 1. Basic Information

- Year: 2025
- Venue: arXiv
- Paper: https://arxiv.org/abs/2508.08149
- PDF: https://arxiv.org/pdf/2508.08149
- Code: https://github.com/MiliLab/REX-RAG
- Task: 开放域问答、多跳问答、RL-based RAG
- Dataset: NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle
- Backbone: Qwen2.5-3B, Qwen2.5-7B
- Retriever: 搜索/检索工具，细节待代码确认

## 2. Motivation

- [论文明确提出] RL-based RAG 容易进入 dead-end trajectory，模型可能过早沿着错误但自信的路径继续推理。
- [论文明确提出] 单纯从当前 policy 采样会限制探索空间，导致训练数据中缺少有价值的修复路径。
- [基于方法/实验设置推断] 这与 ReasonRAG badcase 中的“检索几轮仍 evidence=None，最终胡猜答案”相似。

## 3. Method

- [论文明确提出] Mixed Sampling Strategy 结合 probe sampling 和 exploratory prompts 扩大推理探索。
- [论文明确提出] Policy Correction Mechanism 用 importance sampling 修正探索策略和目标策略之间的分布偏移。
- [基于方法/实验设置推断] 该方法主要解决探索不足，不直接定义当前状态下的证据效用。

## 4. Experiments

- [论文明确提出] 实验覆盖七个 QA benchmarks。
- [论文明确提出] 报告 Qwen2.5-3B 平均提升约 5.1%，Qwen2.5-7B 平均提升约 3.6%。
- [基于方法/实验设置推断] 这些结果说明探索机制有用，但没有直接证明其能减少 query drift 或 unsupported intermediate answers。

## 5. Main Results

- [论文明确提出] 扩大探索并校正 policy 可以缓解 RL-RAG 训练中的 dead-end problem。
- [基于方法/实验设置推断] REX-RAG 的贡献更偏训练采样策略，而不是 reward 结构本身。

## 6. Limitations

- [基于方法/实验设置推断] Query drift 可能被探索缓解，但没有由 reward 显式惩罚。
- [基于方法/实验设置推断] Evidence 是否支撑中间结论没有成为独立判断。
- [基于方法/实验设置推断] 如果没有状态效用评分，探索可能产生更多候选路径，但仍难以选择真正推进证据链的路径。

## 7. Relation to My Research

- REX-RAG 可为 SAPR-RAG 的 Repair Mechanism 提供“逃离 dead end”的思想。
- SAPR-RAG 可以用 Query/Evidence/Stop 低分作为触发探索或重写的条件，而不是单纯依赖 exploratory prompt。

## 8. Useful Sentences for Writing

- REX-RAG 指出，RL-based Agentic RAG 的问题不仅是奖励稀疏，还包括训练轨迹容易陷入错误死路。
- 这说明轨迹修复机制需要和过程奖励结合，而不是只靠一次性策略优化。

## 9. Follow-up Ideas

- 将 ReasonRAG badcase 中连续 evidence=None 的样本标为 dead-end trajectory。
- 对低 Evidence Reward 的步骤触发 REX 风格探索，但用状态奖励选择修复后的轨迹。
