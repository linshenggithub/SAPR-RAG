# 面向 ReasonRAG 方向的 Agentic RAG 文献谱系

本文件基于 `deep-research-report.md` 与 Franklin A 档调研结果重建，目标是服务“面向复杂问答的 Agentic RAG 多步检索推理过程优化研究”。本轮只做论文层面的 A 档调研，不做代码深读。

## 1. 方向基线：从 Outcome Reward 到 Process Reward

### Search-R1

- 类别：Outcome-RL Agentic RAG。
- 贡献：证明 LLM 可以通过强化学习学会在推理过程中主动生成搜索查询，并利用检索结果回答问题。
- 局限：主要使用最终答案奖励，训练信号稀疏，难以定位错误来自 query、retrieval、evidence extraction 还是 stop decision。
- 对本课题作用：作为“只看终答奖励不够”的对照基线。

### ReasonRAG

- 类别：Process-Supervised Agentic RAG。
- 贡献：提出 RAG-ProGUIDE，为 query generation、evidence extraction、answer generation 构造过程级奖励。
- 局限：过程监督已经比 outcome reward 细，但从现有 badcase 看仍存在 query drift、重复子查询、证据链断裂、检索噪声、提前停止等问题。
- 对本课题作用：代表性 baseline、问题观察入口、实验验证平台。

## 2. 过程奖励强化类 Agentic RAG

代表论文：

- TIRESRAG-R1 / From Sufficiency to Reflection
- DecEx-RAG
- HiPRAG
- ProRAG
- Search-P1
- VERITAS / Beyond Correctness

共同趋势：

1. 从 final answer reward 转向 step-level、path-level、faithfulness-level reward。
2. 试图缓解 sparse reward、credit assignment、process hallucination、over-search / under-search。
3. 开始把“推理过程是否合理”作为训练对象，而不只是把最终答案作为训练目标。

共同不足：

1. 很多 reward 仍然是动作级或路径级，没有把“当前证据状态下某个文档是否真正有用”定义为核心对象。
2. Query 质量、Evidence 质量、Stop 决策经常混在一个整体过程分数里，错误来源不够可解释。
3. Bridge entity 保真、未支撑中间答案、检索噪声风险仍缺少独立且可计算的奖励项。

对 SAPR-RAG 的启发：

- Query Reward 要判断 query 是否针对当前未解决缺口。
- Evidence Reward 要判断文档是否在当前状态下提供新信息、支持中间推理、推进证据链。
- Stop Reward 要判断当前证据是否足以支撑最终答案，而不仅是减少搜索次数。

## 3. 探索与轨迹结构改造类

代表论文：

- MCTS-RAG
- REX-RAG
- RAGShaper

共同趋势：

1. 不再只做单条贪心推理路径，而是通过 MCTS、mixed sampling、teacher trajectory synthesis 扩展搜索空间。
2. 强调 Agentic RAG 容易陷入 dead end，需要更强的探索和纠错轨迹。
3. 把噪声、干扰和错误路径显式放入训练或推理环境。

共同不足：

1. 轨迹探索可以找到更多候选路径，但不自动等价于知道“当前状态下哪个证据最有价值”。
2. MCTS 或数据合成成本较高，且容易依赖教师模型或启发式 value。
3. 对 query drift、entity loss、premature stop 的约束大多是间接的。

对 SAPR-RAG 的启发：

- Repair Mechanism 可以借鉴 REX-RAG 的 dead-end 逃逸和 RAGShaper 的噪声纠错轨迹。
- MCTS-RAG 可为候选轨迹生成提供探索机制，但 SAPR-RAG 的核心应是可解释的状态奖励，而不是单纯扩大搜索。

## 4. Evidence Utility / Retriever-LLM 对齐类

代表论文：

- Utility-Focused LLM Annotation for Retrieval and RAG
- LLM-Specific Utility
- Utility-Aligned Embeddings / UAE

共同趋势：

1. 明确指出 retrieval relevance 不等于 generative utility。
2. 从“文档是否相关”转向“文档是否能帮助某个 LLM 生成更好答案”。
3. 尝试用 LLM annotation、model-specific benchmark、distillation 等方式对齐 retriever 与 LLM 的证据偏好。

共同不足：

1. Utility 多数仍定义为静态 query-document 或 LLM-document 关系。
2. 没有纳入 Agentic RAG 中的 history evidence、current subquery、intermediate answer、remaining gap。
3. 难以解释多跳推理中为什么某个文档在第 t 步有用，而在第 t+1 步冗余或有害。

对 SAPR-RAG 的启发：

- 本课题的关键创新可以写成：从 `U(d | q)` / `U(d | q, LLM)` 推进到 `U(d | q, s_t)`。
- Evidence Reward 应显式包含 relevance、novelty、supportiveness、chain contribution、noise risk。

## 5. 传统或非完整 Agent 回路的证据优化类

代表论文：

- Evidence Tree Search

共同趋势：

1. 关注检索后证据集合选择，而不是完整的 Agentic RAG 搜索-推理-停止闭环。
2. 强调多句证据组合、冗余过滤、证据集合质量评估。

共同不足：

1. 不直接建模 query rewriting / search action / stop policy。
2. 更像 evidence selector 或 evidence compressor，而不是完整 trajectory controller。

对 SAPR-RAG 的启发：

- 可作为 Evidence Reward 的局部证据集合评估器。
- 但 SAPR-RAG 需要进一步把证据评估嵌入 query、evidence、stop 三类动作的过程控制中。

## 6. 本课题的逻辑缺口

现有研究已经分别证明：

1. Agentic RAG 需要 process reward，而不能只依赖 final answer reward。
2. 检索文档的 relevance 不等于 LLM 真正需要的 utility。
3. 多步推理轨迹中的 dead end、noise、unfaithfulness、over-search / under-search 都是真实问题。

但仍缺少统一回答：

> 在第 t 步，给定原问题、历史子查询、历史证据、当前中间结论和剩余推理缺口，哪个 query / document / stop action 最能推进完整证据链？

这就是 SAPR-RAG 的定位：

```text
State-Aware Evidence Utility
  + Query Reward
  + Evidence Reward
  + Stop Reward
  + Repair Mechanism
  -> trajectory-level repair for Agentic RAG
```
