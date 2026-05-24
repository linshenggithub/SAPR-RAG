# Beyond Correctness: Rewarding Faithful Reasoning in Retrieval-Augmented Generation

## 1. Basic Information

- Year: 2025
- Venue: arXiv
- Paper: https://arxiv.org/abs/2510.13272
- PDF: https://arxiv.org/pdf/2510.13272
- Code: 未找到官方代码
- Task: RAG faithful reasoning, RL search agent evaluation
- Dataset: NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle
- Backbone: 待确认
- Retriever: 搜索/检索工具

## 2. Motivation

- [论文明确提出] 只奖励最终答案正确可能导致 chain-of-thought unfaithfulness。
- [论文明确提出] 检索增强推理中，模型可能给出正确答案，但推理过程与检索信息并不忠实对应。
- [基于方法/实验设置推断] 这直接对应 ReasonRAG 中 unsupported intermediate answers 和 evidence=None 后仍回答的问题。

## 3. Method

- [论文明确提出] 提出 VERITAS，评估 information-think、think-search、think-answer 三类 faithfulness。
- [论文明确提出] 将忠实性维度作为 reward 或评价对象，补充最终 correctness。
- [基于方法/实验设置推断] VERITAS 强在过程忠实性评价，但没有直接将 retriever 的 state-aware utility 作为训练目标。

## 4. Experiments

- [论文明确提出] 评估 SearchR1、ReSearch 等 RLVR search agents。
- [论文明确提出] 使用多种 QA 数据集并关注 faithfulness 与 task performance。
- [基于方法/实验设置推断] 该文适合支持本课题在实验中加入 faithfulness / supportiveness 指标。

## 5. Main Results

- [论文明确提出] 仅优化 correctness 的搜索 agent 仍可能存在明显的不忠实推理。
- [论文明确提出] 忠实性应作为 RAG 过程优化的重要目标。

## 6. Limitations

- [基于方法/实验设置推断] Faithfulness reward 没有直接转化为“当前文档是否有用”的 state-aware evidence utility。
- [基于方法/实验设置推断] Query drift 和 premature stop 不是该文最核心的单独建模对象。
- [基于方法/实验设置推断] 代码未找到，复现其 judge 细节和 reward 实现需要后续确认。

## 7. Relation to My Research

- VERITAS 为 SAPR-RAG 的 Supportiveness / Step Entailment Reward 提供强文献支撑。
- Evidence Reward 不应只看文档相关性，还要判断该文档是否支撑当前中间结论。

## 8. Useful Sentences for Writing

- VERITAS 指出，RAG 系统的最终答案正确并不保证推理过程忠实于检索证据。
- 因此，Agentic RAG 的过程奖励需要覆盖中间结论与证据之间的支撑关系。

## 9. Follow-up Ideas

- 在 ReasonRAG trajectory 中标注 information-think、think-search、think-answer 三类忠实性错误。
- 用 NLI 或 LLM-as-Judge 判断每步 intermediate answer 是否被 evidence set 蕴含。
