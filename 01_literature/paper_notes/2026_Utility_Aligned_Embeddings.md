# Aligning Dense Retrievers with LLM Utility via Distillation

## 1. Basic Information

- Year: 2026
- Venue: arXiv
- Paper: https://arxiv.org/abs/2604.22722
- PDF: https://arxiv.org/pdf/2604.22722
- Code: 未找到官方代码
- Task: Dense retrieval, RAG
- Dataset: QASPER
- Backbone: LLM reranker / reward model，细节待确认
- Retriever: Bi-encoder dense retriever

## 2. Motivation

- [论文明确提出] LLM reranking 可以更好反映 utility，但推理成本高。
- [论文明确提出] Dense retriever 速度快，但与 LLM utility 的对齐不足。
- [基于方法/实验设置推断] 这是 retriever-LLM alignment 的工程化路线。

## 3. Method

- [论文明确提出] 提出 Utility-Aligned Embeddings。
- [论文明确提出] 使用 perplexity reduction 得到 utility distribution，并将其蒸馏进 bi-encoder。
- [论文明确提出] 使用 Utility-Modulated InfoNCE，推理时不再需要 test-time LLM reranking。

## 4. Experiments

- [论文明确提出] 实验使用 QASPER。
- [论文明确提出] 报告 Recall@1、MAP、Token F1 以及速度提升。
- [论文明确提出] 相比 LLM reranking，推理速度显著提高。

## 5. Main Results

- [论文明确提出] Utility-aligned dense retriever 可以在保持高效率的同时提高 utility-oriented retrieval performance。
- [论文明确提出] 该路线说明 LLM utility 可以被蒸馏进轻量检索器。

## 6. Limitations

- [基于方法/实验设置推断] 方法仍是单步检索视角，不处理 Agentic RAG 的多步状态。
- [基于方法/实验设置推断] 没有建模 current subquery、history evidence、intermediate answer。
- [基于方法/实验设置推断] 数据集和任务与 HotpotQA 类多跳 Agentic RAG 仍有差距。

## 7. Relation to My Research

- 可作为后续 state-conditioned retriever distillation 的工程路线。
- SAPR-RAG 第一阶段可先做 reranker / reward model，后续再将 state-aware utility 蒸馏进 retriever。

## 8. Useful Sentences for Writing

- UAE 表明，LLM 偏好的证据效用可以通过蒸馏方式迁移到高效 dense retriever。
- 本课题进一步关注多步推理状态下的 utility 对齐，而不是单步查询下的 utility 对齐。

## 9. Follow-up Ideas

- 用 `state summary + current subquery` 作为 retriever 输入，训练 state-conditioned bi-encoder。
- 将 Evidence Reward 的标签蒸馏到 dense retriever，用于降低在线 rerank 成本。
