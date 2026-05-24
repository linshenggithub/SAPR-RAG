# Utility-Focused LLM Annotation for Retrieval and Retrieval-Augmented Generation

## 1. Basic Information

- Year: 2025
- Venue: EMNLP 2025 Main
- Paper: https://aclanthology.org/2025.emnlp-main.88/
- PDF: https://aclanthology.org/2025.emnlp-main.88.pdf
- Code: https://github.com/Trustworthy-Information-Access/Utility-Focused-LLM-Annotation
- Task: Retrieval, RAG
- Dataset: MS MARCO, BEIR, MS MARCO QA, NQ, HotpotQA
- Backbone: Qwen2.5-32B 用于标注
- Retriever: 检索模型细节待代码确认

## 2. Motivation

- [论文明确提出] Retrieval relevance 与 generative utility 并不等价。
- [论文明确提出] 人工标注文档 utility 成本高，难以扩展。
- [基于方法/实验设置推断] 这为本课题的 Evidence Reward 提供直接理论起点：不能只看相似度或相关性。

## 3. Method

- [论文明确提出] 使用 LLM 自动标注文档 utility。
- [论文明确提出] 提出 summed marginal likelihood 来利用多正例。
- [论文明确提出] 用 utility-focused annotation 改善 retrieval 和 RAG 性能。

## 4. Experiments

- [论文明确提出] Retrieval 实验覆盖 MS MARCO 和 BEIR。
- [论文明确提出] RAG 实验覆盖 MS MARCO QA、NQ、HotpotQA。
- [论文明确提出] 结果显示 LLM utility annotation 可以提升 OOD retrieval 和 RAG。

## 5. Main Results

- [论文明确提出] LLM 生成 utility 标注可以显著减少人工标注需求。
- [论文明确提出] 少量人工标签加 LLM 标签可接近全人工标注效果。

## 6. Limitations

- [基于方法/实验设置推断] Utility 仍是静态 query-document 关系，不包含 history evidence 或 current reasoning state。
- [基于方法/实验设置推断] 无法直接处理 Agentic RAG 的 query drift、premature stop 和中间结论支撑问题。
- [基于方法/实验设置推断] 标注质量依赖 LLM judge，需要校准和一致性检查。

## 7. Relation to My Research

- 这是 Evidence Reward 的直接起点。
- SAPR-RAG 需要把 `utility(q, d)` 扩展成 `utility(q, s_t, d)`。

## 8. Useful Sentences for Writing

- Utility-Focused LLM Annotation 证明，检索相关性并不能代表文档对生成答案的真实效用。
- 但在 Agentic RAG 中，文档效用还应取决于当前推理状态，而不仅是原始 query。

## 9. Follow-up Ideas

- 用该文标注思路为 HotpotQA step-level candidate documents 构造 utility labels。
- 在标注 prompt 中加入 previous evidence、current subquery 和 missing slots。
