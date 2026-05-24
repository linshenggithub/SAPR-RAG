# 网页端 Deep Research 证据核查补充

来源：用户提供的 ChatGPT Deep Research 共享页  
链接：https://chatgpt.com/s/t_6a12bd81c49c8191a5c8d610cc587f47  
整理日期：2026-05-24

## 1. 本次补充的价值

网页端 Deep Research 的主要价值不是重新列论文，而是把现有 A 档调研中的判断做了证据分层：

1. 原有 15 篇表中缺少一个数据与诊断方向的关键条目：AgenticRAGTracer。
2. 对每篇论文的共性问题标签区分了“论文直接提出”和“基于方法边界推断”。
3. 修正了两个容易写过头的判断：`entity loss / bridge entity 保真` 在当前 16 篇中不是高频显式术语；HiPRAG 不适合被写成 judge 校准缺失的负例。
4. 强化了 SAPR-RAG 的核心表述：process reward 与 evidence utility 两条线都在推进，但仍缺少 trajectory-state 条件下的 evidence utility。

## 2. 新增关键条目

### AgenticRAGTracer

- Paper: https://arxiv.org/abs/2602.19127
- PDF: https://arxiv.org/pdf/2602.19127
- Code: https://github.com/YqjMartin/AgenticRAGTracer
- Dataset: https://huggingface.co/datasets/YqjMartin/AgenticRAGTracer

它是诊断与 benchmark 论文，不是优化方法论文。arXiv 摘要指出现有 benchmark 往往只提供最终问答，缺少中间 hop-level questions，因此无法定位 agent 在哪一步失败。Hugging Face 数据集卡显示其数据按 2-hop、3-hop、4-hop 以及 inference / comparison 组织，适合为本课题的 failure bank 与 hop-level evaluation 提供格式参考。

对 SAPR-RAG 的直接启发：

- badcase 不应只记录 final answer，应记录 hop_id、subquery、retrieved_docs、selected_evidence、intermediate_claim、stop/continue 和 failure_type；
- 实验评价应增加 collapse rate、over-extension rate、step sufficiency、trajectory faithfulness；
- 该论文可支撑“final QA benchmark 不足以诊断 Agentic RAG 多步失败”的论述。

## 3. 证据强度修正

### 直接证据更强的共性问题

- Coarse-grained outcome reward：Search-R1、ReasonRAG、TIRESRAG-R1、DecEx-RAG、ProRAG、Search-P1 都能支撑。
- Faithfulness / unsupported intermediate reasoning：VERITAS、TIRESRAG-R1、ProRAG、REX-RAG 支撑更强。
- Search imbalance：HiPRAG、DecEx-RAG、AgenticRAGTracer 对 over-search、under-search、premature stop、over-extension 的证据更直接。
- Retriever-LLM utility mismatch：Utility-Focused LLM Annotation、LLM-Specific Utility、UAE 是最核心证据。
- Benchmark granularity insufficiency：AgenticRAGTracer 是最直接证据，RAGShaper 可支撑高质量纠错轨迹数据不足。

### 更适合作为综合推断的缺口

- State-aware evidence utility：不是某篇论文直接承认“我们没做”，而是由 utility 线仍停留在 query-document / LLM-document 层、process reward 线仍以 step/path/action 打分为主共同推出。
- Entity loss / bridge entity preservation：这是 ReasonRAG badcase 中非常真实的问题，但当前 16 篇里显式术语证据不足，后续应补查 entity tracking、query decomposition、bridge entity carryover 相关文献。

## 4. 可写入论文的更稳表述

直接证据层：

> 现有 Agentic RAG 文献已经明确指出，仅依赖最终答案的 outcome reward 会造成训练信号稀疏、credit assignment 不足和局部错误难定位；多步检索推理中还会出现过程幻觉、faithfulness 不足、过度检索、检索不足、过早停止和轨迹过度延伸；同时，retrieval relevance 与 LLM generation utility 之间存在系统性错位。

综合推断层：

> 尽管过程奖励和 evidence utility 两条研究线都在快速发展，前者主要关注如何评价推理轨迹或动作，后者主要关注文档对 query 或特定 LLM 的生成效用。现有工作仍较少显式回答：在第 t 步，给定当前已检索证据、中间结论、子查询和未闭合信息槽，下一条候选证据的边际效用是什么。

## 5. 已合并到仓库的改动

- `literature_survey.csv` 新增 P016 AgenticRAGTracer。
- `paper_notes/2026_AgenticRAGTracer.md` 新增中文阅读笔记。
- `taxonomy.md` 新增“诊断与 Hop-Level Benchmark 类”。
- `common_problems_and_ideas.md` 增加证据分层、修正 entity loss 证据强度，并新增 failure-labeled / hop-level benchmark 不足。
