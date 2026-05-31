# Experiment Tracker v2 — ClosureRAG

**Date**: 2026-05-30
**Status**: Planning phase, awaiting Gate 0

---

## Run Log

| Run ID | Gate | Date | Dataset | N | System | EM | F1 | Premature Stop Rate | Unsupported Claim Rate | Bridge Entity Recall | Avg Steps | Status |
|--------|------|------|---------|---|--------|-----|-----|---------------------|----------------------|---------------------|-----------|--------|
| - | G0 | - | HotpotQA | 50 | ReasonRAG baseline | - | - | - | - | - | - | pending |
| - | G0 | - | HotpotQA | 50 | ClosureRAG-prompt (Board+Stop) | - | - | - | - | - | - | pending |
| - | G1 | - | HotpotQA | 80 | +Claim Gate | - | - | - | - | - | - | pending |
| - | G2 | - | HotpotQA | 200 | Full prompt + heuristic | - | - | - | - | - | - | pending |
| - | G2 | - | HotpotQA | 200 | ReasonRAG + more steps | - | - | - | - | - | - | pending |
| - | G2 | - | HotpotQA | 200 | Post-hoc verification | - | - | - | - | - | - | pending |
| - | G3 | - | HotpotQA | 500 | ClosureRAG-trained | - | - | - | - | - | - | pending |
| - | G3 | - | 2Wiki | 500 | ClosureRAG-trained | - | - | - | - | - | - | pending |
| - | G4 | - | All 4 | full | Full system + baselines | - | - | - | - | - | - | pending |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-30 | SAPR-RAG v1 → ClosureRAG v2 | Phase 3-4 review: v1 四模块同等贡献 novelty 不硬，收缩为 Board + Claim Gate + Stop Closure |
| 2026-05-30 | 砍掉 branch rollback | 审稿人建议：复杂度高，会被要求和 MCTS 对比 |
| 2026-05-30 | Evidence selection 降级为 heuristic | 审稿人建议：LLM judge 打分太慢太贵，不是核心贡献 |
| 2026-05-30 | Board schema 从 6 字段砍到 4 字段 | 审稿人建议：最小充分状态，减少出错面 |

---

## Data Artifacts

| Artifact | Description | Status |
|----------|-------------|--------|
| Board annotations (500 trajectories) | Phase 1 数据积累 | pending |
| Claim-evidence pairs (~600) | Phase 1 数据积累 | pending |
| Closure labels (slot/claim/chain) | Phase 1 数据积累 | pending |
| Human validation (100-200) | 人工校验 Board + claim 标注 | pending |

---

## Key Files

- Proposal: `refine-logs/FINAL_PROPOSAL_v2.md`
- Experiment Plan: `refine-logs/EXPERIMENT_PLAN_v2.md`
- Phase 3 Novelty: `idea-stage/PHASE3_NOVELTY_VERIFICATION.md`
- Phase 4 Review: `idea-stage/PHASE4_CRITICAL_REVIEW.md`
- Literature Landscape: `idea-stage/LITERATURE_LANDSCAPE.md`
- ReasonRAG Improvement Ideas: `idea-stage/REASONRAG_IMPROVEMENT_IDEAS.md`
