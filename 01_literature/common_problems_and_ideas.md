# 核心方法共性问题与优化 Idea

本文件只基于当前保留的 6 篇核心同类方法进行归纳：Search-R1、ReasonRAG、DecEx-RAG、HiPRAG、ProRAG、Search-P1。它们共同构成 ReasonRAG 方向的主线：从 outcome reward 到 process reward，再到层级奖励和路径奖励塑形。

## 1. 共性问题矩阵

| 共性问题 | 主要出处 | 证据性质 | 对应 ReasonRAG badcase | 可转化优化 idea |
| --- | --- | --- | --- | --- |
| Final-answer reward 稀疏，credit assignment 不清 | Search-R1; ReasonRAG; DecEx-RAG; ProRAG; Search-P1 | 论文明确提出或方法动机直接对应 | 终答错误后难以判断是 query、retrieval、evidence extraction 还是 stop 错 | 将轨迹反馈拆成 Query Reward、Evidence Reward、Stop Reward |
| Query 生成不稳定 | ReasonRAG; DecEx-RAG; ProRAG; Search-P1 | 论文过程优化动机 + 本课题 badcase 归纳 | Corliss Archer 例子中模型把两跳合并成一个复杂 query，只检到第一跳信息 | Query Reward：惩罚重复、过宽、合并多跳的 query，奖励对准当前缺口的 query |
| Evidence extraction / intermediate claim 缺少充分支撑 | ReasonRAG; ProRAG; Search-P1 | 论文明确关注 process hallucination / path quality，本课题进一步细化 | Evidence=None 或证据不足时仍生成答案或中间结论 | Evidence Reward + Step Entailment：要求中间结论被当前 evidence set 支撑 |
| Stop decision 不可靠 | HiPRAG; ReasonRAG; DecEx-RAG | HiPRAG 直接讨论 over-search / under-search，其他方法从过程决策角度支撑 | 只查到 Shirley Temple 后停止，没有继续查询 Chief of Protocol | Stop Reward：判断证据是否覆盖全部 hop，是否仍有未闭合实体/关系 |
| Path-level / process-level reward 仍偏粗 | ReasonRAG; ProRAG; Search-P1; DecEx-RAG | 基于方法边界推断 | 路径得分无法解释具体错在 query、evidence 还是 stop | 三头奖励模型：query/evidence/stop 分开打分，再组合成 trajectory score |
| 状态感知证据效用缺失 | ReasonRAG; HiPRAG; ProRAG; Search-P1; DecEx-RAG | 基于方法边界 + badcase 推断 | 正确证据可能被召回但 rank 靠后或未被模型使用 | 定义 `U(d | q, s_t)`，评价文档在当前推理状态下的边际贡献 |

## 2. 两类核心问题

### 2.1 多步推理过程控制能力不足

这一类问题关注 Agent 自身如何生成查询、组织推理、校验证据和决定停止。

具体表现包括：

- query drift / query redundancy；
- 多跳问题被合并成过复杂的单个 query；
- bridge entity 没有稳定传递到下一跳；
- 中间结论没有被证据支撑；
- premature stop、over-search、under-search；
- 过程 reward 无法定位具体错误来源。

这类问题说明：现有 Agentic RAG 虽然已经能够主动搜索，但仍没有稳定掌握“当前应该问什么、是否应该继续、当前结论是否可靠”。

### 2.2 状态感知证据利用不足

这一类问题关注检索到的候选文档如何被排序、选择和利用。

具体表现包括：

- 检索结果包含噪声，模型容易被同名实体或相似事件干扰；
- 正确证据即使被召回，也未必被放到当前推理需要的位置；
- evidence extraction reward 往往没有显式判断文档是否提供新信息、是否支撑中间结论、是否推进完整证据链；
- 现有过程奖励多关注步骤或路径质量，但没有独立建模当前状态下的证据边际效用。

这类问题说明：现有方法还没有充分回答“在第 `t` 步，给定当前已有证据和未解决缺口，哪篇文档最有用”。

## 3. 对本课题的优化 Idea

### Query Reward

目标：判断 query 是否对准当前未解决的信息缺口。

可计算维度：

- 是否重复历史 query；
- 是否过宽或合并多个 hop；
- 是否保留 bridge entity；
- 是否覆盖当前未闭合实体、关系、时间或地点槽位；
- 是否能够引导检索到下一跳证据。

### Evidence Reward

目标：判断候选文档在当前状态下是否真正有用。

可计算维度：

- Relevance：是否回答当前子查询；
- Novelty：是否提供历史证据没有的新信息；
- Supportiveness：是否支撑当前中间结论；
- Chain Contribution：是否推进最终证据链；
- Noise Risk：是否引入同名实体、错误关系或错误时间线。

核心形式：

```text
U(d | q, s_t)
```

其中 `s_t` 包含原始问题、历史子查询、已有证据、中间结论、未闭合信息槽和剩余检索预算。

### Stop Reward

目标：判断当前是否真的应该停止检索并输出答案。

可计算维度：

- 当前 evidence set 是否覆盖全部 hop；
- final answer 是否被 evidence set 蕴含；
- 是否仍有未闭合 bridge entity 或关系槽；
- 继续检索的边际收益是否可能大于成本。

### Repair Mechanism

当某一类 reward 低时，不直接沿着错误轨迹继续生成，而是触发修复动作：

- query 低分：重写 query 或拆分 query；
- evidence 低分：扩大 top-k、重排文档、过滤噪声；
- stop 低分：继续检索或要求补充证据；
- entailment 低分：拒绝 unsupported intermediate answer。

## 4. 可写入论文的中心表述

现有 Agentic RAG 过程优化方法已经证明了 process reward、层级奖励和路径奖励对于复杂问答是必要的，但它们仍主要在步骤或轨迹层面提供反馈，缺少对当前推理状态下 query、evidence、stop 三类决策的可解释区分。尤其是在 ReasonRAG 类方法中，证据是否对当前状态真正有用、是否能支撑中间结论、是否推进完整证据链，仍没有被建模为独立的状态感知效用。因此，本课题拟提出面向多跳问答的状态感知过程优化机制，将 Query Reward、Evidence Reward、Stop Reward 与 Repair Mechanism 结合起来，缓解 query drift、检索噪声、unsupported intermediate answer 和 premature stop 等问题。
