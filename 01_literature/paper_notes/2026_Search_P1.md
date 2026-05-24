# Search-P1: Path-Centric Reward Shaping for Stable and Efficient Agentic RAG Training

## 1. Basic Information

- Year: 2026
- Venue: arXiv / OpenReview PDF，正式 venue 待确认
- Paper: https://arxiv.org/abs/2602.22576
- PDF: https://arxiv.org/pdf/2602.22576
- Code: 未找到官方代码
- Task: Agentic RAG RL 训练
- Dataset: NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle, AD-QA
- Backbone: 待确认
- Retriever: 搜索/检索工具

## 2. Motivation

- [论文明确提出] Agentic RAG 训练受 sparse outcome reward 影响，失败样本通常无法提供有效训练信号。
- [论文明确提出] Step-level reward 仍可能忽视路径整体质量和顺序不敏感的 coverage。
- [基于方法/实验设置推断] 该文与 SAPR-RAG 都关注 trajectory-level credit assignment。

## 3. Method

- [论文明确提出] 提出 path-centric reward，将奖励从单步扩展到路径级。
- [论文明确提出] 使用 order-agnostic step coverage 和 soft scoring 从失败样本中提取训练信号。
- [论文明确提出] Dual-Track Path Scoring 结合 self-consistency 与 reference alignment。

## 4. Experiments

- [论文明确提出] 实验覆盖多个 QA benchmarks，包括 HotpotQA、2Wiki、MuSiQue、Bamboogle。
- [论文明确提出] 报告相对 Search-R1 等基线的平均 accuracy 提升。
- [基于方法/实验设置推断] 论文重点是训练稳定性和路径级 reward，而非静态/动态 evidence utility 对比。

## 5. Main Results

- [论文明确提出] Path-centric reward 可以利用失败样本，改善 RL 训练稳定性和效率。
- [论文明确提出] 路径级 credit assignment 比只看最终答案更细。

## 6. Limitations

- [基于方法/实验设置推断] Path score 告诉我们路径好坏，但未必解释路径中 query、document、stop 的具体错误来源。
- [基于方法/实验设置推断] 如果 reference planner 依赖外部强模型，构造成本会较高。
- [基于方法/实验设置推断] Unsupported intermediate answers 仍需要更强的证据蕴含检查。

## 7. Relation to My Research

- SAPR-RAG 可以把 Search-P1 的 path score 拆成实体覆盖、证据支撑、停止价值三个可解释分数。
- Search-P1 是 trajectory-level reward 的重要相关工作，但 SAPR-RAG 更强调状态感知 evidence utility。

## 8. Useful Sentences for Writing

- Search-P1 表明，失败轨迹并非无用数据；通过路径级 reward shaping，错误样本也能提供稳定训练信号。
- 本课题进一步关注路径中每个检索动作和证据选择动作的状态条件效用。

## 9. Follow-up Ideas

- 用 ReasonRAG 失败轨迹构造 path-level labels，再拆解成 Query/Evidence/Stop labels。
- 将 path reward 作为外层目标，三头状态 reward 作为内层诊断。
