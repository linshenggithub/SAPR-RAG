# 共性问题、证据出处与优化 Idea

本文件汇总 Franklin A 档文献调研、网页端 Deep Research 证据核查与 ReasonRAG badcase 观察。表中“论文明确提出”表示论文中直接讨论了该问题；“基于方法/实验设置推断”表示该缺口来自对论文方法边界与本课题 badcase 的归纳。

网页端证据核查补充了一个重要约束：**不要把所有问题都写成论文直接提出**。目前一手证据最硬的是 coarse reward、faithfulness、搜索行为失衡、retriever-LLM utility mismatch 与 benchmark 粒度不足；`entity loss / bridge entity 保真` 更像 ReasonRAG badcase 引出的合理研究假设，还需要额外补查实体跟踪或 query decomposition 相关文献。

## 1. 共性问题矩阵

| 共性问题 | 出处论文 | 出处证据 | 对应 ReasonRAG badcase | 可转化优化 idea | 信息性质 |
| --- | --- | --- | --- | --- | --- |
| Query drift / redundancy | MCTS-RAG; ProRAG; HiPRAG; RAGShaper; ReasonRAG | MCTS-RAG 直接讨论 vague queries 与重复检索；ProRAG 讨论冗余检索和过程幻觉；HiPRAG 讨论 over-search / under-search；RAGShaper 构造干扰环境训练纠错；ReasonRAG badcase 中存在重复、复杂、合并多跳的 query | Corliss Archer 例子中模型把“找演员”和“找政府职位”合并成一个复杂 query，导致只检到第一跳信息 | Query Reward：惩罚重复 query，奖励命中当前未闭合信息槽、保留 bridge entity、分解粒度合理的 query | 部分论文明确提出 + 本课题归纳 |
| Entity loss / bridge entity 保真不足 | Evidence Tree Search; ReasonRAG; 待补查 entity tracking / query decomposition 文献 | 当前 16 篇 Agentic RAG / utility 论文很少直接使用 bridge entity preservation 作为显式术语；ETS 可支持多句证据依赖，但不能单独证明 entity loss 是领域高频术语 | 多跳问题中找到了 Shirley Temple 但没有稳定推进到其政府职位；桥接实体没有成为下一跳约束 | Entity Preservation Reward：跟踪 bridge entity、关系词、时间/地点槽位，惩罚丢失关键实体的 query 或 answer | 主要来自本课题 badcase 推断；证据强度需后续补强 |
| Retrieval noise 敏感 | RAGShaper; Evidence Tree Search; Utility-Focused Annotation; ReasonRAG | RAGShaper 构造 perception/cognition distractors；ETS 指出长文档含冗余和无关内容；Utility-Focused 说明 relevance 与 utility 不等价 | Evolution / Nicolas Cage 例子被同名电影和相似实体污染，正确证据 rank 靠后 | Evidence Reward：同时评分 relevance、novelty、supportiveness、chain contribution、noise risk | 论文明确提出 + 本课题归纳 |
| Unsupported intermediate answers | VERITAS; TIRESRAG-R1; ProRAG; ReasonRAG | VERITAS 关注 faithfulness；TIRESRAG-R1 讨论 faulty reasoning 与 answer-reasoning inconsistency；ProRAG 关注 process hallucination | Evidence=None 或证据不足时仍生成答案；中间结论看似合理但没有被检索证据支撑 | Step Entailment Reward：每步中间结论必须被当前 evidence set 支撑，否则触发 repair | 论文明确提出 + 本课题归纳 |
| Premature stop / 停搜价值未建模 | HiPRAG; TIRESRAG-R1; DecEx-RAG; REX-RAG; AgenticRAGTracer; ReasonRAG | HiPRAG 显式讨论 under-search；TIRESRAG-R1 讨论推理过早中断；DecEx-RAG 将终止决策单独建模；REX-RAG 讨论 dead ends；AgenticRAGTracer 诊断 premature collapse 与 over-extension | 只查到 Shirley Temple 后停止，没有继续查询 Chief of Protocol | Stop Reward：检查 evidence sufficiency、未闭合实体/关系、反事实继续检索收益 | 论文明确提出 + 本课题归纳 |
| Coarse-grained reward / credit assignment 不清 | Search-R1; ReasonRAG; DecEx-RAG; ProRAG; Search-P1 | Search-R1 使用 outcome-based reward；后续 process/path reward 论文都在缓解 sparse reward 和 credit assignment | 终答错了无法定位到底是 query、retrieval、evidence extraction 还是 stop 错 | 三头奖励：Query Reward、Evidence Reward、Stop Reward，再外接 trajectory-level path score | 论文明确提出 + 本课题归纳 |
| State-aware evidence utility 缺失 | Utility-Focused Annotation; LLM-Specific Utility; UAE; HiPRAG; ReasonRAG | Utility 工作证明 relevance 不等于 utility，但多为静态 query-document、LLM-document 或单步 utility；HiPRAG 强在是否搜索而非当前状态下哪个证据有用 | Gold evidence 可能在 top-50/top-100，但模型不会判断当前状态下哪个文档最能推进证据链 | 定义 `U(d | q, s_t)`，其中 `s_t` 包含历史 evidence、当前 subquery、中间结论、未闭合信息槽和预算 | 基于方法/实验设置推断 |
| Retriever-LLM alignment 静态化 | Utility-Focused Annotation; LLM-Specific Utility; UAE | 这些工作都在对齐 retriever 与 LLM utility，但未进入多步 agent state | 检索器按相似度返回文档，LLM 需要的是能推进当前证据链的文档 | State-conditioned reranker / retriever distillation：输入 `[question; state summary; current subquery; document]` | 论文明确提出 utility 问题；动态化为本课题归纳 |
| Failure-labeled / hop-level benchmark 不足 | AgenticRAGTracer; RAGShaper; ReasonRAG | AgenticRAGTracer 指出现有 benchmark 多只有 final QA，缺少 hop-level 中间问题；RAGShaper 指出高质量纠错轨迹人工标注不可扩展；ReasonRAG badcase 需要人工重构失败步骤 | 当前 badcase 多是自然语言分析，缺少统一的 hop_id、failure_type、repair_action 标签 | Failure Bank：按 hop 标注 query drift、retrieval noise、unsupported claim、premature stop、over-extension，并记录可执行 repair action | 论文明确提出 + 本课题工程归纳 |

## 2. 核心缺口表述

单篇论文的 limitation 并不等于本课题的研究缺口。本课题要抓的是多个方向反复暴露的共同盲点：

> 直接证据层：现有 Agentic RAG 文献已经明确指出，仅依赖 final-answer outcome reward 会造成 reward sparsity、credit assignment 不足和局部错误难定位；多步检索推理中还会出现 process hallucination、faithfulness 问题、over-search / under-search、premature stop；retrieval relevance 与 LLM utility 之间存在系统性错位；传统 final QA benchmark 难以定位 hop-level 失败。
>
> 综合推断层：现有 process reward 与 evidence utility 两条线仍未充分结合。前者主要回答“如何给轨迹或步骤打分”，后者主要回答“哪篇文档对某个 query 或某个 LLM 有用”，而很少有工作显式回答：在第 t 步，给定当前轨迹状态，下一条证据的边际效用是什么。这正是 state-aware evidence utility 的选题空间。

更具体地说，现有方法大多没有回答：

```text
在第 t 步，给定当前已检索证据、当前中间结论、当前子查询、未闭合实体/关系和剩余预算，
哪个文档真正有用？
哪个查询真正对准缺口？
现在是否应该停止？
如果当前步骤低质量，应该重写、重排还是继续检索？
```

## 3. 对 SAPR-RAG 的模块化启发

### Query Reward

目标不是判断 query 是否流畅，而是判断 query 是否对准当前缺口。

可计算维度：

- bridge entity 是否保留；
- query 是否重复历史 query；
- query 是否只针对一个合理 hop；
- query 是否覆盖未闭合实体、关系、时间、地点槽位；
- query 是否避免把多个 hop 合并成不可检索的大 query。

### Evidence Reward

目标不是判断 document 是否相关，而是判断 document 在当前状态下是否有用。

可计算维度：

- Relevance：是否回答当前子查询；
- Novelty：是否提供历史证据没有的新信息；
- Supportiveness：是否支撑当前中间结论；
- Chain Contribution：是否推进最终证据链；
- Noise Risk：是否引入同名实体、错误时间线、错误关系。

### Stop Reward

目标不是简单减少搜索，而是判断“现在停是否有充分证据”。

可计算维度：

- 当前 evidence set 是否覆盖全部 hop；
- final answer 是否被 evidence set 蕴含；
- 是否仍有未闭合 bridge entity 或关系槽；
- 继续搜索的边际收益是否可能大于成本。

### Repair Mechanism

当 Query/Evidence/Stop 任一 reward 低时，不直接输出答案，而是触发修复动作：

- query 低分：重写 query 或拆分 query；
- evidence 低分：扩大 top-k、重排文档、过滤噪声；
- stop 低分：继续检索或要求补充证据；
- entailment 低分：拒绝 unsupported intermediate answer。

### Failure Bank and Hop-Level Evaluation

目标不是只积累 badcase 文本，而是形成可训练、可评测、可复查的失败数据。

可记录字段：

- hop_id：当前是第几跳；
- subquery：模型生成的当前查询；
- retrieved_docs：检索候选及 rank；
- selected_evidence：模型抽取或使用的证据；
- intermediate_claim：当前中间结论；
- failure_type：query drift、retrieval noise、unsupported claim、premature stop、over-extension；
- repair_action：rewrite query、rerank evidence、continue search、reject claim、stop with support。

## 4. 可写入论文的中心表述

面向复杂问答的 Agentic RAG，多步检索推理的核心瓶颈不是“不会检索”，而是“不会根据当前轨迹状态判断下一条证据的真实效用，也不会把这种效用反馈为细粒度过程奖励”。因此，本课题提出的 SAPR-RAG 应将状态感知证据效用建模与 query/evidence/stop 三类过程奖励结合起来，用于修复 ReasonRAG 类方法中的 query drift、retrieval noise、unsupported intermediate answers、premature stop 以及本课题 badcase 中观察到的 bridge entity loss。
