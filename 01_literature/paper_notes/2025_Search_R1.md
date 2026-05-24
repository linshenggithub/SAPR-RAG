# Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning

## 1. Basic Information

- Year: 2025
- Venue: COLM 2025
- Paper: https://openreview.net/forum?id=Rwhi91ideu
- PDF: https://openreview.net/pdf?id=Rwhi91ideu
- arXiv: https://arxiv.org/abs/2503.09516
- Code: https://github.com/PeterGriffinJin/Search-R1
- Task: RL-based search-augmented reasoning
- Dataset: NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle
- Backbone: Qwen2.5-3B, Qwen2.5-7B
- Retriever: 搜索引擎/检索工具

## 2. Motivation

- [论文明确提出] Prompting LLM 使用搜索并不可靠，模型需要被训练成能够有效与搜索引擎交互。
- [论文明确提出] 通过 RL 可以让模型在 step-by-step reasoning 中主动生成搜索查询。
- [基于方法/实验设置推断] 这是 ReasonRAG 和后续 process reward 工作的 outcome-RL 起点。

## 3. Method

- [论文明确提出] 训练 LLM 在推理过程中生成多轮 search queries。
- [论文明确提出] 使用 retrieved token masking 稳定训练。
- [论文明确提出] 奖励主要是 simple outcome-based reward function。

## 4. Experiments

- [论文明确提出] 实验覆盖七个 QA 数据集。
- [论文明确提出] Qwen2.5-7B 和 Qwen2.5-3B 相对 RAG baselines 有明显提升。
- [基于方法/实验设置推断] 它证明模型能学会 search action，但不保证每一步搜索和证据选择都是最优的。

## 5. Main Results

- [论文明确提出] Outcome-RL 可以显著提升搜索增强问答性能。
- [论文明确提出] 模型能在推理过程中主动调用搜索，而不是被动接收一次性检索结果。

## 6. Limitations

- [论文明确提出] 奖励是 outcome-based，训练信号稀疏。
- [基于方法/实验设置推断] 终答错误时难以判断 query、retrieval、evidence extraction、stop 哪一步出错。
- [基于方法/实验设置推断] Query drift、entity loss、unsupported intermediate answer 需要后续过程监督方法解决。

## 7. Relation to My Research

- Search-R1 是 SAPR-RAG 论文中最自然的 outcome-RL 对照。
- ReasonRAG 已经在 Search-R1 基础上证明 process reward 更好；SAPR-RAG 要进一步说明 state-aware process reward 为什么更细。

## 8. Useful Sentences for Writing

- Search-R1 证明 RL 可以训练 LLM 形成主动搜索行为，但 outcome reward 无法为中间检索推理步骤提供充分信用分配。
- 这为 ReasonRAG 和 SAPR-RAG 的过程监督路线提供了背景动机。

## 9. Follow-up Ideas

- 在 related work 中把 Search-R1 写为 outcome-supervised Agentic RAG 的代表。
- 用 Search-R1 -> ReasonRAG -> SAPR-RAG 的递进关系组织方法动机。
