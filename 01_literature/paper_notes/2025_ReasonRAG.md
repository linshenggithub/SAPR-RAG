# Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning

## 1. Basic Information

- Year: 2025
- Venue: NeurIPS 2025
- Paper: https://openreview.net/forum?id=h3LlJ6Bh4S
- PDF: https://openreview.net/pdf?id=h3LlJ6Bh4S
- Code: https://github.com/Applied-Machine-Learning-Lab/ReasonRAG
- Task: Agentic RAG, multi-hop QA
- Dataset: HotpotQA, 2Wiki, PopQA, MuSiQue, Bamboogle
- Backbone: Qwen2.5-7B
- Retriever: BGE

## 2. Motivation

- [论文明确提出] Search-R1 类 outcome RL 存在 low exploration efficiency、gradient conflict、sparse reward。
- [论文明确提出] Agentic RAG 包含 query generation、evidence extraction、answer generation 等过程动作，适合过程级监督。
- [基于本课题实验观察] ReasonRAG 是本课题最重要的 baseline 和 badcase 来源。

## 3. Method

- [论文明确提出] 构造 RAG-ProGUIDE，为 query generation、evidence extraction、answer generation 提供过程监督数据。
- [论文明确提出] 使用 process reward 相比 outcome reward 提供更细粒度训练信号。
- [基于方法/实验设置推断] 其过程奖励仍没有完全覆盖 state-aware evidence utility、bridge entity preservation 和 stop decision。

## 4. Experiments

- [论文明确提出] 实验覆盖 HotpotQA、2Wiki、PopQA、MuSiQue、Bamboogle。
- [论文明确提出] 只用 5k training instances 即优于 Search-R1 和传统 RAG，而 Search-R1 使用更大规模训练数据。
- [基于本课题实验观察] 我们已复现得到各数据集初步指标，并观察到多类 badcase。

## 5. Main Results

- [论文明确提出] Process reward 在 Agentic RAG 中比单纯 outcome reward 更有效。
- [论文明确提出] 小规模高质量过程监督数据可以产生明显收益。

## 6. Limitations

- [基于本课题 badcase 推断] Query generation 仍会出现重复、复杂、合并多跳、偏移等问题。
- [基于本课题 badcase 推断] Evidence extraction 可能无法稳定推进完整证据链。
- [基于本课题 badcase 推断] Gold evidence rank 靠后、检索噪声、premature stop、unsupported answer 仍然存在。

## 7. Relation to My Research

- ReasonRAG 是本课题的代表性 baseline、问题观察入口和实验验证平台。
- SAPR-RAG 的定位应是解释“为什么 ReasonRAG 的过程奖励还不够细”，并补上 state-aware evidence utility 与 trajectory repair。

## 8. Useful Sentences for Writing

- ReasonRAG 证明了过程监督对于 Agentic RAG 的有效性，但其 badcase 也表明，过程奖励仍需进一步绑定当前推理状态和证据链进展。
- 本课题不是对 ReasonRAG 的局部修补，而是以 ReasonRAG 为窗口抽象 Agentic RAG 的共性过程控制问题。

## 9. Follow-up Ideas

- 固定 ReasonRAG baseline 配置，保存完整 trajectory。
- 按 Query/Evidence/Stop 三类错误重新标注 ReasonRAG badcase。
- 将 SAPR-RAG V0 插入 ReasonRAG pipeline，先做小样本验证。
