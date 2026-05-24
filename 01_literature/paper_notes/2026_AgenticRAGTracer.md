# AgenticRAGTracer: A Hop-Aware Benchmark for Diagnosing Multi-Step Retrieval Reasoning in Agentic RAG

## 1. Basic Information

- Year: 2026
- Venue: arXiv；项目 README 标注为 ACL 2026 Findings
- Paper: https://arxiv.org/abs/2602.19127
- PDF: https://arxiv.org/pdf/2602.19127
- Code: https://github.com/YqjMartin/AgenticRAGTracer
- Dataset: https://huggingface.co/datasets/YqjMartin/AgenticRAGTracer
- Task: Agentic RAG 多步检索推理诊断
- Dataset: 2-hop、3-hop、4-hop；inference 与 comparison 两类任务
- Backbone: 多种闭源/开源 LLM，具体模型需在 B 档阅读正文表格后补齐
- Retriever: 项目代码提供 e5 检索服务配置

## 2. Motivation

- [论文明确提出] 现有多跳问答 benchmark 通常只提供最终问题和最终答案，缺少逐 hop 的中间问题，因此很难定位 agent 到底在哪一步失败。
- [论文明确提出] 多跳推理需要模型进行有计划的思考和多步交互，正好适合作为评估 Agentic RAG 能力的测试场。
- [论文明确提出] 人工构造多跳 benchmark 成本高、扩展性弱，因此作者希望用 LLM 自动构造并进行 step-by-step validation。
- [基于方法/实验设置推断] 这篇论文可以补强本课题的数据与评测动机：如果只看 final EM/F1，ReasonRAG 的 query drift、premature stop、over-extension 很难被系统归因。

## 3. Method

- [论文明确提出] AgenticRAGTracer 是一个 hop-aware benchmark，核心不是提出新的 Agentic RAG 优化算法，而是构造可以逐步诊断多步检索推理失败的数据集。
- [论文明确提出] 数据覆盖多个领域，包含 1305 个样本，并设计为避免与主流 benchmark 重叠。
- [论文明确提出] 数据集包含中间 hop-level questions，用于连接 atomic questions 与最终 multi-hop query。
- [论文明确提出] Hugging Face 数据集卡显示数据按 2-hop、3-hop、4-hop，以及 inference、comparison 任务类型组织为 jsonl。
- [基于方法/实验设置推断] 它提供的是诊断维度：某一步是否过早塌缩、是否过度延伸、是否没有按逻辑结构分配推理步数，而不是一个直接可替代 ReasonRAG 的训练框架。

## 4. Experiments

- [论文明确提出] 作者在多个大语言模型上评估 Agentic RAG 多步检索推理能力。
- [论文明确提出] 摘要报告，最难子集上即便强模型也只有较低 EM 表现，说明现有 LLM 在 hop-aware 多步检索推理上仍然明显不足。
- [论文明确提出] 论文做了 hop-aware diagnosis，用于区分推理链是过早停止、步数不足，还是向无关方向过度延伸。
- [基于方法/实验设置推断] 该实验形式非常适合作为 SAPR-RAG 的诊断评测补充：不仅报告最终答案，还报告每一步 query、evidence、intermediate answer 和 stop decision 是否合理。

## 5. Main Results

- [论文明确提出] 传统只含 final QA 的 benchmark 缺少诊断能力，而 AgenticRAGTracer 可以分析 agent 的失败步骤。
- [论文明确提出] 失败主要表现为两类轨迹结构问题：过早塌缩和过度延伸。
- [论文明确提出] 这些失败说明模型不能稳定按任务逻辑结构分配推理步骤。
- [基于方法/实验设置推断] 这为 SAPR-RAG 的 Stop Reward 和 trajectory repair 提供了评测证据：模型不仅要答对，还要在正确 hop 数和正确信息缺口上推进。

## 6. Limitations

- [基于方法/实验设置推断] 该工作主要是 benchmark 和诊断协议，不直接给出新的训练 reward 或 reranker。
- [基于方法/实验设置推断] 数据主要由 LLM 自动构造，虽然适合扩展，但后续用于训练时仍需要抽样人工核验，避免把构造偏差引入 reward model。
- [基于方法/实验设置推断] 该数据集与 ReasonRAG 的 trace schema 不一定完全一致，需要额外做格式转换，才能接入本课题的 failure bank。
- [基于方法/实验设置推断] 论文强调 hop-level questions，但 state-aware evidence utility 仍需要本课题自行定义，例如 `U(d | q, s_t)`。

## 7. Relation to My Research

- AgenticRAGTracer 可以作为 SAPR-RAG 的“诊断侧基线”：它回答的是如何发现第几 hop 出错，而 SAPR-RAG 进一步回答如何用状态感知 reward 修复错误。
- 它可以指导 ReasonRAG badcase 标注格式：每条样本不仅记录 final answer，还记录 hop_id、subquery、retrieved_docs、selected_evidence、intermediate_answer、stop/continue 标签。
- 它特别适合支撑本课题中“benchmark 粒度不足”的论证，因为它直接指出 final QA 无法定位 agent failure step。
- 它也可用于设计新的评价指标：hop accuracy、step sufficiency、collapse rate、over-extension rate、trajectory faithfulness。

## 8. Useful Sentences for Writing

- 现有多跳问答评测往往只关注最终问答对，难以解释 Agentic RAG 在多步检索推理过程中究竟在哪个 hop 出错。
- AgenticRAGTracer 说明，复杂问答失败不只是答案错误，还包括推理链过早塌缩和过度延伸这两类轨迹结构失衡。
- 因此，本课题需要在 ReasonRAG badcase 分析中引入 hop-level trace 标注，把最终答案错误分解到 query、evidence、intermediate answer 和 stop decision 等局部错误。
- AgenticRAGTracer 是诊断工具，而 SAPR-RAG 的目标是在诊断基础上进一步学习可执行的状态感知过程奖励。

## 9. Follow-up Ideas

- 抽取 ReasonRAG 的 HotpotQA badcase，按 AgenticRAGTracer 风格补充 hop-level 标签。
- 设计一个 `failure_bank.jsonl`，字段包含 `question`、`gold_answer`、`hop_id`、`subquery`、`retrieved_docs`、`selected_evidence`、`intermediate_claim`、`failure_type`、`repair_action`。
- 在实验中报告 collapse rate 与 over-extension rate，看 SAPR-RAG 是否真的减少提前停止和无效延伸。
- B 档代码调研时重点查看 `evaluation.py`、`multihop_pipeline.py` 和数据 jsonl 格式，判断能否复用其评测脚本或数据结构。
