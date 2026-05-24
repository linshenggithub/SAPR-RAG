# MCTS-RAG: Enhancing Retrieval-Augmented Generation with Monte Carlo Tree Search

## 1. Basic Information

- Year: 2025
- Venue: Findings of EMNLP 2025
- Paper: https://aclanthology.org/2025.findings-emnlp.672/
- PDF: https://aclanthology.org/2025.findings-emnlp.672.pdf
- Code: https://github.com/yale-nlp/MCTS-RAG
- Task: 复杂问答、动态检索推理
- Dataset: ComplexWebQA, GPQA, FoolMeTwice
- Backbone: LLaMA 3.1-8B 等
- Retriever: 论文层面待代码确认

## 2. Motivation

- [论文明确提出] 标准 RAG 通常把检索和推理分开处理，模型先拿到一批文档，再基于这些文档生成答案，缺少推理过程中动态补充信息的能力。
- [论文明确提出] MCTS 可以增强 test-time reasoning，但如果没有外部知识支撑，搜索路径可能停留在模型内部知识和错误假设上。
- [基于方法/实验设置推断] 这篇论文试图把“树搜索”变成 Agentic RAG 的推理控制器，而不是只把检索当作一次性前处理。

## 3. Method

- [论文明确提出] 将检索与推理联合到 Monte Carlo Tree Search 中，每个节点表示候选 reasoning state 或 reasoning path。
- [论文明确提出] 在树搜索过程中动态调用检索，用检索结果扩展或修正推理路径。
- [基于方法/实验设置推断] MCTS-RAG 的核心优势是提高探索能力，但它没有把 query、document、stop action 分解成独立的可解释 reward head。

## 4. Experiments

- [论文明确提出] 实验覆盖 ComplexWebQA、GPQA、FoolMeTwice。
- [论文明确提出] 官方代码仓库说明该方法能让较小模型在复杂问答和事实验证场景中获得明显提升。
- [基于方法/实验设置推断] 该评测更偏重最终答案准确率和 test-time search 效果，而不是细粒度诊断 query drift、entity loss 或 unsupported intermediate answer。

## 5. Main Results

- [论文明确提出] 通过把 retrieval 纳入 MCTS，模型可以在推理过程中自适应补充外部知识。
- [基于方法/实验设置推断] 结果说明“扩大搜索空间”对复杂问答有效，但并不自动解决证据效用和轨迹忠实性问题。

## 6. Limitations

- [基于方法/实验设置推断] MCTS 提升了 trajectory exploration，但没有显式定义 `U(document | question, state)`。
- [基于方法/实验设置推断] Bridge entity preservation 没有作为独立奖励，因此仍可能搜索到看似合理但偏离实体链的路径。
- [基于方法/实验设置推断] 计算成本可能高于普通 Agentic RAG，需要额外考虑预算控制和剪枝策略。

## 7. Relation to My Research

- 这篇论文说明“多路径探索”很重要，但我的课题更关注“如何判断每一步证据是否真正推进当前证据链”。
- SAPR-RAG 可以借鉴其 trajectory exploration，但应把 value function 拆成 Query Reward、Evidence Reward 和 Stop Reward。

## 8. Useful Sentences for Writing

- MCTS-RAG 表明，复杂问答中的检索和推理不应被视为两个割裂阶段，而应在推理过程中动态交互。
- 然而，仅扩大搜索空间仍不足以保证证据链的正确推进；模型还需要判断当前状态下候选证据的真实边际效用。

## 9. Follow-up Ideas

- 将 MCTS 作为候选轨迹生成器，用 SAPR-RAG 的状态奖励对节点或边进行重打分。
- 在树搜索节点中显式保存 history evidence、bridge entity 和未闭合信息槽，作为 Evidence Reward 的输入。
