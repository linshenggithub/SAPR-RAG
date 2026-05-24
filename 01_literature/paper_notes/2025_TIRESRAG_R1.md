# From Sufficiency to Reflection: Reinforcement-Guided Thinking Quality in Retrieval-Augmented Reasoning for LLMs

## 1. Basic Information

- Year: 2025
- Venue: arXiv
- Paper: https://arxiv.org/abs/2507.22716
- PDF: https://arxiv.org/pdf/2507.22716
- Code: https://github.com/probe2/TIRESRAG-R1
- Task: 多跳问答、检索增强推理
- Dataset: HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle
- Backbone: 待代码确认
- Retriever: 待代码确认

## 2. Motivation

- [论文明确提出] 仅依赖最终答案奖励会忽略中间推理质量，无法区分信息不足、推理错误和答案-推理不一致。
- [论文明确提出] 检索增强推理需要模型知道当前信息是否足够、推理是否可靠，以及是否需要反思修正。
- [基于方法/实验设置推断] 这篇论文与 ReasonRAG 的核心共同点是都认为过程信号比单纯 outcome reward 更可靠。

## 3. Method

- [论文明确提出] 使用 think-retrieve-reflect 流程，将推理、检索、反思组织成闭环。
- [论文明确提出] 引入 sufficiency reward、reasoning quality reward、reflection reward 等多维奖励。
- [基于方法/实验设置推断] Sufficiency reward 与 SAPR-RAG 的 Stop Reward 最接近，但该文没有把 evidence utility 显式写成状态条件函数。

## 4. Experiments

- [论文明确提出] 实验覆盖 HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle。
- [论文明确提出] 指标包括 EM 以及 LLM-as-Judge 类评价。
- [基于方法/实验设置推断] 这些数据集与 ReasonRAG baseline 高度重合，因此适合作为中期报告中的同谱系方法。

## 5. Main Results

- [论文明确提出] 多维过程奖励能改善检索增强推理质量。
- [论文明确提出] 论文把信息充分性与反思能力作为独立维度，说明 Agentic RAG 的失败不只是检索失败。

## 6. Limitations

- [基于方法/实验设置推断] Query redundancy / drift 没有被独立建模。
- [基于方法/实验设置推断] Bridge entity preservation 仍不是显式奖励项。
- [基于方法/实验设置推断] Sufficiency 判断关注“够不够回答”，但没有细分哪个候选证据能补齐当前缺口。

## 7. Relation to My Research

- 可将 sufficiency reward 扩展为 Stop Reward。
- 可将 reflection reward 用于 SAPR-RAG 的 Repair Mechanism：当 evidence 或 stop 低分时触发反思和重检索。

## 8. Useful Sentences for Writing

- TIRESRAG-R1 将检索增强推理的失败拆解为信息不足、推理错误和答案-推理不一致，为过程奖励设计提供了更细的错误视角。
- 但该类 reward 仍需要进一步与当前证据状态和下一跳证据效用绑定。

## 9. Follow-up Ideas

- 用 ReasonRAG trajectory 标注 sufficiency / reasoning / reflection 三类错误，验证其 taxonomy 是否覆盖我们的 badcase。
- 将 sufficiency reward 与未闭合实体槽位结合，形成更可计算的 Stop Reward。
