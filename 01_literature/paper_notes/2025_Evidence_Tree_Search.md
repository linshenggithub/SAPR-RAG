# Enhancing Retrieval-Augmented Generation via Evidence Tree Search

## 1. Basic Information

- Year: 2025
- Venue: ACL 2025 Long
- Paper: https://aclanthology.org/2025.acl-long.1175/
- PDF: https://aclanthology.org/2025.acl-long.1175.pdf
- Code: 未找到官方代码；论文称将发布
- Task: Evidence selection, long-context QA, multi-hop QA
- Dataset: LongBench 中 2WikiMultiHopQA, HotpotQA, MuSiQue, MultiFieldQA, Qasper
- Backbone: 多种 reader，细节待确认
- Retriever: 检索后 evidence selection

## 2. Motivation

- [论文明确提出] Retriever 返回的长文档通常包含冗余和无关内容。
- [论文明确提出] 多句证据之间存在组合依赖，不能只单独评价每一句证据。
- [论文明确提出] 多证据选择面临监督稀缺和搜索空间爆炸。

## 3. Method

- [论文明确提出] 将 evidence retrieval 建模为 evidence tree expansion。
- [论文明确提出] 每条路径表示一个候选 evidence set。
- [论文明确提出] 用 MCTS 评估 evidence set quality，并用 Early-Terminating Beam Search 降低推理成本。

## 4. Experiments

- [论文明确提出] 实验覆盖 LongBench 中多个 QA 数据集。
- [论文明确提出] 指标包括 EM/F1 等任务指标。
- [论文明确提出] 在不同 reader 上相对已有方法取得提升。

## 5. Main Results

- [论文明确提出] Evidence tree 可以更好建模多句证据组合。
- [论文明确提出] 检索后证据选择对 RAG 性能有显著影响。

## 6. Limitations

- [基于方法/实验设置推断] ETS 不是完整 Agentic RAG loop，不处理 query generation、query rewrite、stop policy。
- [基于方法/实验设置推断] 更像局部 evidence selector，而不是 trajectory controller。
- [基于方法/实验设置推断] 没有把当前子查询和历史检索轨迹纳入完整过程奖励。

## 7. Relation to My Research

- ETS 可作为 Evidence Reward 的局部模块参考。
- SAPR-RAG 需要比 ETS 多出 query/evidence/stop 的全流程状态控制。

## 8. Useful Sentences for Writing

- Evidence Tree Search 说明，证据质量不仅取决于单个句子的相关性，还取决于多个证据之间的组合关系。
- 但多步 Agentic RAG 还需要进一步控制何时生成查询、何时选择证据、何时停止检索。

## 9. Follow-up Ideas

- 在 Evidence Reward 中加入 evidence set-level contribution，而不是只给单文档打分。
- 用 ETS 思路搜索 top-k 文档中的最优证据组合，再反馈给 SAPR-RAG 的中间答案生成。
