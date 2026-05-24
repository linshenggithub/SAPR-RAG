# RAGShaper: Eliciting Sophisticated Agentic RAG Skills via Automated Data Synthesis

## 1. Basic Information

- Year: 2026
- Venue: ACL ARR 2026 Jan Submission / arXiv
- Paper: https://openreview.net/forum?id=VnppICw50X
- PDF: https://openreview.net/attachment?id=VnppICw50X&name=pdf
- Code: 未找到官方代码
- Task: Agentic RAG 数据合成、抗噪轨迹生成
- Dataset: NQ, PopQA, AmbigQA, Bamboogle
- Backbone: Teacher agent 待确认
- Retriever: 带干扰的检索环境

## 2. Motivation

- [论文明确提出] 训练鲁棒 Agentic RAG 需要高质量轨迹数据，但人工标注成本高且难覆盖真实噪声。
- [论文明确提出] 检索环境中存在 perception/cognition 级 adversarial distractors，模型需要学会识别和纠错。
- [基于方法/实验设置推断] 这与 ReasonRAG 中 retrieval noise 和错误路径污染高度相关。

## 3. Method

- [论文明确提出] 使用 InfoCurator 自动构建 dense information trees。
- [论文明确提出] 构造 perception 和 cognition 两类 adversarial distractors。
- [论文明确提出] 通过 constrained navigation 迫使 teacher agent 在干扰条件下生成纠错轨迹。

## 4. Experiments

- [论文明确提出] 实验覆盖 NQ、PopQA、AmbigQA、Bamboogle 等数据集。
- [论文明确提出] 评价关注 EM/F1 等任务指标，以及训练出的 agent 是否具备复杂检索技能。
- [基于方法/实验设置推断] 它更像数据合成工作，而不是完整的 online reward learning 框架。

## 5. Main Results

- [论文明确提出] 高质量自动合成轨迹可以诱导更复杂的 Agentic RAG 技能。
- [论文明确提出] 带噪检索环境有助于训练模型面对干扰证据时进行纠错。

## 6. Limitations

- [基于方法/实验设置推断] 主要依赖 SFT/数据合成，不是状态感知过程奖励。
- [基于方法/实验设置推断] Query drift、premature stop 和 evidence utility 仍缺少在线 reward 闭环。
- [基于方法/实验设置推断] 代码未找到，复现成本和数据构建细节需要后续确认。

## 7. Relation to My Research

- RAGShaper 可为 SAPR-RAG 的 failure-labeled data 和 noisy preference data 构造提供思路。
- Evidence Reward 可以显式学习区分真正证据和 perception/cognition distractors。

## 8. Useful Sentences for Writing

- RAGShaper 表明，真实检索环境中的干扰文档不是边缘问题，而是训练 Agentic RAG 纠错能力的关键条件。
- 这支持本课题将 noise risk 纳入状态感知证据效用。

## 9. Follow-up Ideas

- 在 HotpotQA 中自动构造同名实体、时间线冲突、关系相似的 hard negatives。
- 用 ReasonRAG 失败样本生成“纠错轨迹”，训练 Repair Mechanism。
