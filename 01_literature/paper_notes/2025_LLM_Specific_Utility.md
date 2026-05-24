# LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation

## 1. Basic Information

- Year: 2025
- Venue: arXiv
- Paper: https://arxiv.org/abs/2510.11358
- PDF: https://arxiv.org/pdf/2510.11358
- Code: 未找到官方代码
- Task: RAG utility modeling
- Dataset: NQ, TriviaQA, MS MARCO-FQA
- Backbone: Qwen3-8B, Qwen3-14B, Qwen3-32B, Llama3.1-8B
- Retriever: 待确认

## 2. Motivation

- [论文明确提出] RAG 不应只优化 topical relevance，更应优化 passage 是否能帮助特定 LLM 生成正确完整答案。
- [论文明确提出] 同一篇文档对不同 LLM 的 utility 可能不同。
- [基于方法/实验设置推断] 这把 utility 从“通用相关性”推进到“模型特定有用性”。

## 3. Method

- [论文明确提出] 构建 LLM-specific gold utilitarian passages benchmark。
- [论文明确提出] 在多个 LLM 上分析文档 utility 与最终 RAG performance 的关系。
- [基于方法/实验设置推断] 方法仍主要是 single-shot RAG 或静态检索视角。

## 4. Experiments

- [论文明确提出] 数据集包括 NQ、TriviaQA、MS MARCO-FQA。
- [论文明确提出] 模型包括 Qwen3 系列和 Llama3.1-8B。
- [基于方法/实验设置推断] 不覆盖完整 Agentic RAG trajectory，也不直接评估多步 query/retrieval/stop。

## 5. Main Results

- [论文明确提出] Passage utility 具有 LLM-specific 性质。
- [论文明确提出] 对某个 LLM 有帮助的文档，不一定对另一个 LLM 同样有帮助。

## 6. Limitations

- [基于方法/实验设置推断] Utility 虽然变成 model-specific，但仍不是 state-specific。
- [基于方法/实验设置推断] 未建模当前已检证据、当前中间结论和下一跳缺口。
- [基于方法/实验设置推断] 对 Agentic RAG 的 query drift / stop decision 没有直接处理。

## 7. Relation to My Research

- 这篇论文给 SAPR-RAG 的核心定位提供一条清晰递进：relevance -> LLM-specific utility -> state-specific utility。
- 本课题应把 utility 条件从 LLM 身份进一步扩展到 reasoning state。

## 8. Useful Sentences for Writing

- LLM-Specific Utility 表明，证据有用性不是一个与生成模型无关的静态属性。
- 在多步 Agentic RAG 中，证据有用性还应进一步依赖当前轨迹状态。

## 9. Follow-up Ideas

- 构造对 Qwen2.5-ReasonRAG 特定的 state-aware evidence utility labels。
- 比较同一候选文档在不同 history evidence 下的 utility 变化。
