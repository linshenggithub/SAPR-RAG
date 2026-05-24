# 研究方向、任务定义与核心方法问题归纳

## 1. 目前具体研究方向

本课题的具体研究方向是：**面向复杂多跳问答的 Agentic RAG 多步检索推理过程优化**。

更具体地说，本课题关注 ReasonRAG、Search-R1、DecEx-RAG、HiPRAG、ProRAG、Search-P1 这条方法线：大语言模型在推理过程中主动生成查询、调用检索器、读取证据、更新中间结论，并决定是否继续检索或输出答案。

与传统 RAG 相比，本课题研究的核心不是一次性检索增强生成，而是多步 agent 回路中的过程控制问题：

```text
当前应该查询什么？
当前检索到的文档是否真正有用？
当前证据是否足以支撑中间结论或最终答案？
如果当前步骤错误，应该重写查询、重排证据，还是继续检索？
什么时候应该停止检索并输出答案？
```

因此，本课题的研究重点可以进一步表述为：**ReasonRAG 类 Agentic RAG 方法在多跳问答中的状态感知过程控制与证据利用优化**。

## 2. 具体任务定义

本课题面向 HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle 等复杂多跳问答任务。这类任务通常需要模型跨越多个实体、关系或文档构建证据链，才能得到最终答案。

给定一个复杂问题 `Q`，Agentic RAG 系统在第 `t` 步可表示为：

```text
s_t = {
  original question,
  historical subqueries,
  retrieved documents,
  selected evidence,
  intermediate claims,
  remaining reasoning gaps,
  search budget
}
```

系统需要在该状态下做出三类关键决策：

1. **Query decision**：下一步应该生成什么子查询；
2. **Evidence decision**：候选文档中哪些证据真正有助于推进当前证据链；
3. **Stop decision**：当前证据是否足以支持最终答案，是否应该停止检索。

因此，本课题可以形式化为：

> 在多步检索推理过程中，给定当前推理状态 `s_t`，学习或设计能够评价 query、evidence 与 stop action 的状态感知过程反馈机制，从而减少 query drift、检索噪声、证据链断裂、未支撑中间结论和过早停止等失败。

## 3. 为解决该任务已有的核心方法类别

### 3.1 Outcome-RL Agentic RAG

代表方法：Search-R1。

Search-R1 使用最终答案正确性作为强化学习奖励，训练 LLM 在推理过程中主动调用搜索工具。它的重要意义在于证明了搜索行为可以通过 RL 学到，LLM 可以从固定检索输入转向主动搜索。

但这类方法的奖励主要来自最终答案。当模型答错时，很难判断错误来自 query generation、document retrieval、evidence extraction、intermediate reasoning 还是 stop decision。

### 3.2 Process-Supervised Agentic RAG

代表方法：ReasonRAG。

ReasonRAG 将 Agentic RAG 从 outcome reward 推进到 process reward，通过 RAG-ProGUIDE 对 query generation、evidence extraction 和 answer generation 提供过程监督。它说明复杂问答中的 Agentic RAG 不能只靠最终答案反馈，过程监督本身是有效的。

但 ReasonRAG 仍暴露出一些可观察的过程失败：query 可能过宽或合并多个 hop，第一跳证据没有稳定推进为第二跳查询，证据不足时仍可能生成答案，检索停止也可能过早。

### 3.3 Decision / Execution Process Optimization

代表方法：DecEx-RAG。

DecEx-RAG 将 Agentic RAG 建模为 decision / execution 优化问题，强调不同阶段的过程监督。这类方法比单一 outcome reward 更细，有助于缓解全局反馈模糊的问题。

但它仍没有充分回答：在当前状态下，某个 query 是否真正对准未解决缺口，某篇文档是否真正推进证据链，以及当前是否应该停止。

### 3.4 Hierarchical Process Reward

代表方法：HiPRAG。

HiPRAG 关注 Agentic RAG 中的 over-search 与 under-search，并用层级过程奖励判断是否需要继续搜索。它直接对应多步检索推理中的搜索必要性和停止控制问题。

但 HiPRAG 更强调“是否搜索”，而本课题进一步关心“搜什么”“候选证据是否有用”“当前证据是否足以支撑答案”。

### 3.5 Process-Supervised RL / PRM

代表方法：ProRAG。

ProRAG 通过 SFT warmup、MCTS-based PRM、PRM-guided refinement 和 process-supervised RL 优化 RAG 过程，目标是缓解 process hallucination 并提升训练稳定性。

但 PRM 通常更像对步骤或轨迹整体质量打分，未必能解释具体错误来自 query、evidence 还是 stop decision。

### 3.6 Path-Centric Reward Shaping

代表方法：Search-P1。

Search-P1 关注路径级奖励塑形，从失败样本中提取训练信号，改善 Agentic RAG 训练中的 credit assignment 与稳定性。

但路径级奖励仍需要进一步拆解：一条路径不好，到底是 query 没对准、证据没支撑、停止过早，还是中间推理走偏？这正是本课题试图细化的问题。

## 4. 现有方法存在的两类核心问题

基于当前保留的 6 篇核心方法和 ReasonRAG badcase 观察，现有方法的问题可以归纳为两大类。

### 4.1 第一类问题：多步推理过程控制能力不足

这类问题发生在 Agent 自身的推理控制过程中，表现为模型不能稳定决定“下一步该问什么、当前结论是否可靠、什么时候应该继续检索或停止”。

具体包括：

1. **查询生成不稳定**
   模型可能生成重复、过宽、过复杂或合并多个 hop 的查询。例如 Corliss Archer badcase 中，问题本应拆成“谁饰演 Corliss Archer”和“该演员担任过什么政府职位”两个子问题，但模型将两步合成一个复杂 query，导致检索只命中第一跳信息。

2. **Bridge entity 传递不稳定**
   多跳问答依赖第一跳得到的关键实体作为下一跳检索约束。但 ReasonRAG badcase 中，模型即使找到了 Shirley Temple，也没有稳定将该实体推进到下一跳政府职位查询中。

3. **中间结论缺少证据校验**
   模型可能在 evidence 不足甚至 evidence=None 时生成中间答案或最终答案。ProRAG 等过程监督方法关注 process hallucination，也说明中间过程可靠性是 Agentic RAG 的关键问题。

4. **停止决策不可靠**
   模型可能过早停止，也可能过度搜索。HiPRAG 明确关注 over-search 与 under-search，说明搜索必要性和停止控制本身就是核心问题。

5. **过程奖励仍然偏粗**
   ReasonRAG、DecEx-RAG、ProRAG、Search-P1 都尝试从不同角度细化过程反馈，但很多反馈仍停留在步骤或路径整体质量层面，难以明确区分错误来自 query、evidence 还是 stop decision。

这类问题可以概括为：

> 现有 Agentic RAG 方法虽然已经具备主动检索和过程监督能力，但仍缺少对多步推理过程的精细控制，尤其缺少对查询生成、证据支撑和停止决策的可解释反馈。

### 4.2 第二类问题：状态感知证据利用不足

这类问题发生在检索文档的排序、选择和利用过程中，表现为模型不能判断“当前状态下哪篇文档真正有用”。

具体包括：

1. **检索结果容易受到噪声干扰**
   多跳问答中常见同名实体、相似标题、相似事件或热门无关文档。ReasonRAG 的 Evolution / Nicolas Cage badcase 中，检索结果被不同版本的 Evolution 信息污染，导致正确证据 rank 靠后。

2. **正确证据即使被召回，也未必被模型使用**
   在复杂问答中，正确证据可能已经位于较大的候选集合中，但模型缺少机制判断哪个文档最能推进当前证据链。

3. **Evidence extraction reward 不等于状态感知证据效用**
   ReasonRAG 已经对 evidence extraction 提供过程监督，但该监督更接近“当前抽取是否合理”，并没有充分建模“这篇文档相对于当前状态的边际贡献”。

4. **路径奖励容易吸收证据错误，缺少证据级解释**
   ProRAG 和 Search-P1 等方法强调过程或路径奖励，但如果路径最终不好，仍需要进一步判断是证据 rank 错、证据噪声大、证据没有支撑中间结论，还是 query 本身错误。

5. **缺少 `U(d | q, s_t)` 建模**
   现有核心方法大多没有显式建模文档在当前推理状态 `s_t` 下的效用。这里的状态应包括历史子查询、已有证据、中间结论、未闭合实体/关系和剩余检索预算。

这类问题可以概括为：

> 现有方法更多在步骤或路径层面优化 Agentic RAG，但没有充分判断文档在当前多步推理状态下是否能够推进证据链。

## 5. 本课题的切入点

针对上述两类问题，本课题可以将核心创新定位为：**状态感知的 Agentic RAG 过程优化**。

具体来说，本课题希望从已有过程奖励进一步推进到：

```text
U(d | q, s_t)
```

其中 `s_t` 表示当前多步推理状态，包括原始问题、历史子查询、已有证据、中间结论、未闭合实体/关系和剩余检索预算。

在此基础上，可以设计三类过程反馈：

1. **Query Reward**
   判断当前 query 是否对准未解决的信息缺口，是否保留 bridge entity，是否避免重复、过宽或合并多跳。

2. **Evidence Reward**
   判断候选文档是否在当前状态下提供新信息、支撑中间结论、推进证据链，并控制噪声风险。

3. **Stop Reward**
   判断当前证据是否已经足以支撑最终答案，是否仍有未闭合实体或关系，是否应该继续检索。

进一步地，当 Query Reward、Evidence Reward 或 Stop Reward 显示当前步骤质量较低时，系统可以触发 repair mechanism，例如重写 query、重排文档、扩大 top-k、拒绝无支撑中间结论或继续检索。

因此，本课题不是单纯提出一个新的 reranker，也不是只改进 query rewriting，而是试图将 **多步推理过程控制** 与 **状态感知证据利用** 统一起来，修复 ReasonRAG 类 Agentic RAG 方法在复杂多跳问答中的关键失败模式。
