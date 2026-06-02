# SAPR-RAG Overnight Pipeline Summary

**Date**: 2026-05-30
**Status**: ⚠️ Evidence-only signal confirmed (200-sample), but **end-to-end EM/F1 无提升**
**Session**: New Claude session (previous crashed on api.z.ai tool result 500)

---

## 1. Pipeline Stages

| # | Stage | Status | Output |
|---|-------|--------|--------|
| 1 | ReasonRAG baseline trajectory (v2) | ✅ Done | `metrics/20260529_evidence_debug_30samples_v2/` |
| 2 | Evidence decision point export (v2, query-fixed) | ✅ Done | `logs/20260530_evidence_decision_top10_queryfix/` |
| 3 | inferred_subquery reretrieval (固化) | ✅ Done | `logs/20260530_evidence_decision_top10_queryfix_reretrieved/` |
| 4 | 4-way comparison + no-history ablation | ✅ Done | `metrics/20260530_sapr_evidence_v0_reretrieved/` |

**Original empty-query comparison** (Stage 3/4 from first session) results are retained but invalid — see §4.

---

## 2. Baseline Trajectory Results (Stage 1)

**Run**: `20260529_evidence_debug_30samples_v2`
**Config**: Qwen2.5-7B + ReasonRAG LoRA, bge-base retriever, topk=3, max_iter=8, RTX 5090

| Metric | 30-sample | Full dev reference |
|--------|----------:|-------------------:|
| EM | **0.2333** (7/30) | 0.3495 |
| F1 | **0.3283** | 0.4560 |
| Answer rate | 96.7% (29/30) | — |
| Max-out rate | 3.3% (1/30) | — |
| Avg time | 29.1 s/example | — |

---

## 3. Evidence Decision Point Export (Stage 2)

**Run**: `20260530_evidence_decision_top10_queryfix`
**Script**: `03_sapr_rag/scripts/export_evidence_decision_points.py`

| Check | Value |
|-------|------:|
| Items (summaries) | 30 |
| Retrieval steps | 113 |
| Total decision points | 143 |
| inferred_subquery non-empty | 113/113 (100%) |
| pipeline_query empty | 113/113 (100%) |

### Inferred Subquery Quality

- dev_0: "Ed Wood's nationality" ← good
- dev_1: "the woman who portrayed Corliss Archer in the film" ← good
- dev_4: "the" ← too generic
- Overall: ~85% meaningful subqueries, ~15% overly short/generic

---

## 4. Empty-Query Bug (Original Data, Fixed)

The original `evidence_decision_points.jsonl` had all 113 retrieval steps using empty queries (ReasonRAG batch mode never generates `"So the next query is ..."`), producing identical retrieval results. **Original comparison all-zeros — invalid.**

**Fix**: Post-hoc reretrieval using `inferred_subquery` with BGE query instruction prefix `"Represent this sentence for searching relevant passages: "`.

---

## 5. ✅ Reretrieved Evidence Selector Comparison (Main Result)

**Data**: `20260530_evidence_decision_top10_queryfix_reretrieved`
**Method**: BGE-base + FAISS flat, inferred_subquery + query instruction prefix
**Gold recall (top-10)**: 72/226 per-step (31.9%), 26/60 unique gold docs (43.3%)

### 4-Way Comparison + No-History Ablation

| Selector | Hit@3 | Recall@3 | Noise Rate | Item Hit% |
|----------|------:|---------:|-----------:|----------:|
| retriever_top3 | 0.4690 | 0.2389 | 0.8407 | 56.7% |
| static_qd | 0.5221 | 0.2832 | 0.8112 | 56.7% |
| **sapr_e_v0** | **0.5398** | **0.2699** | **0.8201** | **63.3%** |
| sapr_e_no_hist | 0.5133 | 0.2566 | 0.8289 | 60.0% |

### Deltas

| Comparison | Hit@3 Δ | Recall@3 Δ |
|------------|--------:|-----------:|
| SAPR-E vs retriever_top3 | **+0.0708** | **+0.0310** |
| SAPR-E vs static_qd | +0.0177 | −0.0133 |
| SAPR-E vs no-history | **+0.0265** | **+0.0133** |

### Verdict: ✅ DIRECTIONAL SIGNAL DETECTED

1. **SAPR-E v0 > retriever_top3**: +7.1pp hit@3, +3.1pp recall@3, +6.6pp item hit%
2. **History matters**: SAPR-E with history > without (+2.7pp hit@3, +1.3pp recall@3)
3. **Static QD also beats retriever**: keyword overlap is a strong baseline for re-ranking
4. **SAPR-E best overall on hit@3 and item-level hit rate**

### Caveats

- 30-sample subset, not statistically significant alone
- Noise rates remain high (~82–84%) — SAPR-E doesn't reduce noise vs baselines
- SAPR-E recall@3 slightly below static_qd (−1.3pp) — state-awareness helps precision more than recall
- Per-step gold recall limited by subquery being single-entity focused

### 6-Way Ablation (30-sample)

| Selector | Hit@3 | Recall@3 | NoiseRate | ItemHit% | vs retriever Δ |
|----------|------:|---------:|----------:|---------:|:--------------:|
| retriever_top3 | 0.4690 | 0.2389 | 0.8407 | 56.7% | — |
| static_qd | 0.5221 | 0.2832 | 0.8112 | 56.7% | +5.3pp |
| **sapr_e_v0** | **0.5398** | **0.2699** | **0.8201** | **63.3%** | **+7.1pp** |
| sapr_e_no_hist | 0.5133 | 0.2566 | 0.8289 | 60.0% | +4.4pp |
| sapr_e_no_title | 0.4602 | 0.2301 | 0.8466 | 53.3% | −0.9pp |
| sapr_e_no_subquery | 0.5133 | 0.2566 | 0.8289 | 60.0% | +4.4pp |

**Ablation deltas vs sapr_e_v0 (full):**

| 去掉维度 | Hit@3 Δ | Recall@3 Δ | ItemHit Δ | 解读 |
|----------|--------:|-----------:|----------:|------|
| −title实体匹配 (Dim5) | **−8.0pp** | **−4.0pp** | **−10.0pp** | 最大贡献；去掉后低于 retriever baseline |
| −history新颖性 (Dim4) | −2.7pp | −1.3pp | −3.3pp | 有贡献但非主要 |
| −subquery匹配 (Dim1) | −2.7pp | −1.3pp | −3.3pp | 与 history 贡献相当 |

**维度重要性排序**: title_entity(×1.0) >> subquery(×2.0) ≈ history(×0.5) > question(×1.0) ≈ entity(×1.5)

**Item hit counts**: retriever 17/30, static 17/30, v0 **19/30**, no_hist 18/30, no_title 16/30, no_subq 18/30

---

## 6. Original Empty-Query Comparison (Invalid, Retained for Record)

### 3-Way (`20260530_sapr_evidence_v0`) — all zeros

| Selector | Hit@3 | Recall@3 | Noise Rate | Item Hit% |
|----------|------:|---------:|-----------:|----------:|
| retriever_top3 | 0.0000 | 0.0000 | 1.0000 | 0.0% |
| static_qd | 0.0000 | 0.0000 | 1.0000 | 0.0% |
| sapr_e_v0 | 0.0000 | 0.0000 | 1.0000 | 0.0% |

### 4-Way + No-History (`20260530_sapr_evidence_v0_no_hist`) — all zeros

Same as above. All selectors 0% due to identical placeholder retrieval docs.

---

## 7. Reretrieval Script

**Script**: `03_sapr_rag/scripts/reretrieve_evidence_with_inferred_subquery.py`
**Server**: rag-5090 GPU 1 (RTX 5090)
**Key details**:
- BGE query prefix: `"Represent this sentence for searching relevant passages: "` (matches FlashRAG)
- CLS pooling + L2 normalization (matches index construction)
- On-demand corpus line fetch (527 unique docs from 22M corpus) instead of full load
- Timing: encode 0.3s, FAISS load 213s, search 689s, corpus fetch 93s

---

## 8. Next Steps

1. **Scale up**: Run on larger HotpotQA dev subset (200–500 examples) for statistical significance
2. **SAPR-E v1 improvements**:
   - LLM-based evidence utility scorer (replace heuristic with small LM)
   - History-weighted novelty (current Dim 4 is weak +0.5 weight)
   - Better subquery extraction (fix ~15% generic subqueries)
3. **End-to-end test**: Plug SAPR-E selector into actual ReasonRAG pipeline, measure EM/F1 impact
4. **Noise reduction**: Current ~82% noise rate needs improvement for practical use

---

## 10. 200-Sample Evidence Validation (Confirmed Signal)

**Data**: `20260530_evidence_decision_top10_queryfix_200_reretrieved`
**Gold recall (top-10)**: 348/1468 per-step (23.7%)

### 6-Way Ablation (200-sample)

| Selector | Hit@3 | Recall@3 | NoiseRate | ItemHit% | vs retriever |
|----------|------:|---------:|----------:|---------:|:------------:|
| retriever_top3 | 0.3256 | 0.1744 | 0.8837 | 47.7% | — |
| static_qd | 0.3311 | 0.1853 | 0.8765 | 49.2% | +1.5pp |
| **sapr_e_v0** | **0.3420** | **0.1792** | **0.8806** | **51.8%** | **+4.1pp** |
| sapr_e_no_hist | 0.3501 | 0.1832 | 0.8778 | 51.8% | +4.1pp |
| sapr_e_no_title | 0.2943 | 0.1580 | 0.8946 | 46.2% | −1.5pp |
| sapr_e_no_subquery | 0.3283 | 0.1717 | 0.8856 | 50.3% | +2.6pp |

**Ablation vs sapr_e_v0**: −title −4.8pp, −subquery −1.4pp, −history **+0.8pp** (history slightly negative on 200-sample)

**Item hits**: retriever 95/199, SAPR-E 103/199 (+8 items)

**结论**: Evidence-only 方向性信号在 200 条上稳定（+4.1pp item hit）。但 no_hist 略优于 v0，history 维度贡献为负。

---

## 11. ❌→⚠️ End-to-End Test (配置错误导致无效，待重做)

**原始判定已推翻。** 端到端 EM/F1 无提升的根因是 `max_tokens=32` 配置错误，非 SAPR-E 方法缺陷。详见 [诊断报告](logs/20260531_sapr_e_e2e_diagnosis.md)。

| 实验 | EM | F1 | 检索方式 | 文档选择 |
|------|----|----|---------|---------|
| Baseline v1 (旧) | 0.2333 | 0.3283 | 空 query, top-3 | retriever top-3 |
| SAPR-E v1 | 0.2333 | 0.3171 | 空 query, top-10 | SAPR-E rerank top-3 |
| SAPR-E v2 (修复) | 0.2333 | 0.3171 | inferred_subquery, top-10 | SAPR-E rerank top-3 |

### 根因（2026-05-31 诊断）

实验配置将 `max_tokens` 从 ReasonRAG 原始的 `256` 改为 `32`，导致：

1. 模型每步只能生成 ~30 token，来不及输出 `"So the next query is..."` 结论标记
2. `get_flags()` 永远返回 `"None"`（30/30 items, 143/143 steps）
3. Batch pipeline 路由永远走 `answer_generation_prompt`（无文档 slot），`document_analysis_prompt`（有文档）从未触发
4. 模型 30/30 条的完整 thought 序列在 baseline 和 SAPR-E 之间 **byte-level 完全一致**
5. SAPR-E 重排的文档从未被模型看到

### 待做

- [ ] 修正 `max_tokens=256` 后重跑 baseline 30-sample
- [ ] 修正后重跑 SAPR-E e2e 30-sample
- [ ] 确认文档被注入、轨迹有差异后，重新评估 EM/F1

---

## 9. File Index

| Artifact | Path |
|----------|------|
| Baseline metrics | `04_experiments/metrics/20260529_evidence_debug_30samples_v2/metrics.json` |
| Baseline config | `04_experiments/run_configs/run_20260529_evidence_debug.yaml` |
| Baseline note | `04_experiments/logs/20260529_evidence_debug_30samples_v2/experiment_note.md` |
| Decision points (v2, original) | `04_experiments/logs/20260530_evidence_decision_top10_queryfix/evidence_decision_points.jsonl` |
| Decision points (reretrieved) | `04_experiments/logs/20260530_evidence_decision_top10_queryfix_reretrieved/evidence_decision_points.jsonl` |
| Validation report | `04_experiments/logs/20260530_evidence_decision_top10_queryfix/validation_report.md` |
| **4-way comparison metrics** | `04_experiments/metrics/20260530_sapr_evidence_v0_reretrieved/metrics.json` |
| **200-sample 6-way metrics** | `04_experiments/metrics/20260530_sapr_evidence_v0_200_reretrieved/metrics.json` |
| 200-sample decision points | `04_experiments/logs/20260530_evidence_decision_top10_queryfix_200/evidence_decision_points.jsonl` |
| 200-sample reretrieved | `04_experiments/logs/20260530_evidence_decision_top10_queryfix_200_reretrieved/evidence_decision_points.jsonl` |
| SAPR-E e2e script | `03_sapr_rag/scripts/run_sapr_e_v0_e2e_eval.py` |
| 6-way ablation script | `03_sapr_rag/scripts/compare_6way_selector_ablation.py` |
| Old 3-way metrics (invalid) | `04_experiments/metrics/20260530_sapr_evidence_v0/metrics.json` |
| Old 4-way ablation (invalid) | `04_experiments/metrics/20260530_sapr_evidence_v0_no_hist/metrics.json` |
| Comparison script | `03_sapr_rag/scripts/compare_3way_evidence_selectors.py` |
| Reretrieval script | `03_sapr_rag/scripts/reretrieve_evidence_with_inferred_subquery.py` |
| Exporter script | `03_sapr_rag/scripts/export_evidence_decision_points.py` |
| **E2E 诊断报告** | `04_experiments/logs/20260531_sapr_e_e2e_diagnosis.md` |
| **This summary** | `04_experiments/overnight_summary.md` |
