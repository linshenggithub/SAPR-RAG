# SAPR-E v0 Pipeline

**Date**: 2026-06-04
**Status**: 中期答辩冲刺主线，已收窄

## 1. v0 的唯一改动

SAPR-E v0 只改 ReasonRAG 的检索文档选择，不改其他 pipeline 行为。

```text
ReasonRAG baseline:
query -> retrieve top-3 -> document_analysis_prompt

SAPR-E v0:
query -> retrieve top-10 -> SAPR-E v0 score -> select top-3 -> document_analysis_prompt
```

也就是说，模型最终看到的文档数仍然是 3 篇；区别只是这 3 篇是从 top-10 候选里重新排序选出来的。

## 2. 禁止混入 v0 的改动

正式 v0 实验中不要改：

- ReasonRAG prompt；
- ReasonRAG state machine；
- query 生成逻辑；
- query 解析逻辑；
- answer 解析逻辑；
- stopping behavior；
- generator 模型；
- dataset；
- corpus / index；
- `max_iter`、`max_tokens` 等 baseline 配置。

如果要做 query-fix、offline reretrieve、inferred_subquery 诊断，必须单独标成 ablation 或历史诊断，不能叫 v0 主实验。

## 3. 当前可信入口

主线入口应优先使用：

```text
03_sapr_rag/scripts/run_sapr_e_v0_minimal_rerank_ablation.py
```

原因：它的目标是保留 ReasonRAG 原始 batch 状态机、prompt routing、日志和停止逻辑，只替换检索边界的文档选择。

历史脚本：

```text
03_sapr_rag/scripts/run_sapr_e_v0_e2e_eval.py
```

该脚本重写了较多 `run_batch()` 细节，容易把 v0 和 query fallback / 解析逻辑变化混在一起。除非重新审计，否则不要作为 v0 主线证据。

## 4. 实验对齐要求

baseline 与 SAPR-E v0 必须完全共享：

- 同一个可靠 generator；
- 同一个 HotpotQA dev 切片；
- 同一个 BGE encoder；
- 同一个 FAISS index；
- 同一个 corpus；
- 同一个 `max_iter`；
- 同一个 `max_tokens=256`；
- 同一个 `retrieval_topk` 输出给模型：3 篇。

SAPR-E v0 额外允许：

- candidate top-k = 10；
- 用 v0 打分策略在 top-10 中选 3 篇。

## 5. 模型注意事项

正式 v0 generator 对齐用户 baseline：

```text
/home/mayi/LLaMA-Factory/examples/merge_lora/output/qwen2.5-7B-lora-dpo-RAG-ProGuide
```

这是 LoRA 合并后的完整模型。不要把下面这个 adapter 目录当作 vLLM 直接使用的完整模型：

```text
/home/mayi/models/Qwen2.5-7B-Instruct-ReasonRAG-Lora
```

正式实验必须在 metrics / run log 中记录 generator 路径。

## 6. 现有 retrieval 指标的定位

`04_experiments/overnight_summary.md` 中的 hit@3 信号可以作为弱证据参考，但不是最终结论，原因是其中的 decision points / inferred_subquery 来自历史诊断流程，且并非收窄版在线 e2e。

正式结论必须来自收窄后的在线 e2e：

```text
baseline top-3 vs SAPR-E top-10 rerank top-3
```

并报告 EM/F1、文档是否进入 `document_analysis_prompt`、以及 case study。
