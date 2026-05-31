# Experiment Plan — SAPR-RAG (Minimum Go/No-Go Under Matched Compute)

**Date**: 2026-05-28
**Backbone**: ReasonRAG-LoRA (frozen)
**Datasets**: HotpotQA / 2Wiki / MuSiQue / Bamboogle

## 1) Compute matching rules (non-negotiable)
Match across all variants:
- `max_steps`
- retrieval calls
- `top_k`
- `num_candidates`
- token budget (as close as possible)

If SAPR uses *C* candidates, baselines must also use *C* candidates (random pick / majority vote / self-consistency).

## 2) Minimum go/no-go suite (do these first)
### (A) Evidence-only: state-conditioned evidence utility
**Variants**:
- Baseline: ReasonRAG-LoRA retrieval order
- Non-state reranker: query-doc reranker (DPA-RAG-style)
- SAPR-E: state-aware evidence Q (uses history)
- Ablation: SAPR-E (history removed)

**Metrics**:
- Final: EM/F1
- Evidence: Recall@5/10, MRR (HotpotQA/2Wiki)
- Trajectory: noise@k / wrong-entity distractor rate

**Go criteria** (minimum):
- consistent improvement on HotpotQA+2Wiki under matched compute, OR
- neutral F1 but strong evidence-metric + trajectory-metric improvements with no budget increase.

### (B) Stop-only: premature-stop vs over-search
**Variants**:
- Baseline stop heuristic
- SAPR-S: stop Q
- Ablation: stop Q without history

**Metrics**:
- avg steps
- premature-stop proxy (missing supporting facts at final)
- final EM/F1

### (C) Query-only: drift/repetition reduction
**Variants**:
- Candidate random pick (control)
- Candidate voting/self-consistency (control)
- SAPR-Q: query Q
- Ablation: query Q without history

**Metrics**:
- repetition rate
- entity preservation proxy (NER overlap with history)
- final EM/F1

## 3) Next: modular vs scalar PRM (novelty isolation)
Compare:
- Single scalar step PRM trained on same data
- Modular Q heads (`Q_q/Q_e/Q_s`) trained on same data

## 4) Generalization
Train/dev on HotpotQA+2Wiki; test on MuSiQue+Bamboogle.

## 5) Run order (decision gated)
1) Evidence-only on HotpotQA dev subset
2) Evidence-only on HotpotQA full dev + 2Wiki dev
3) Add stop-only, then query-only
4) Modular vs scalar PRM
5) Generalization

## 6) Config template
Use `04_experiments/run_configs/TEMPLATE.yaml` with fixed constants (edit to match baseline):
- `top_k: 10`
- `max_steps: 3`
- `num_candidates: 5`
