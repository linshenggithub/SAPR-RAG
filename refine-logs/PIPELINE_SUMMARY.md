# Pipeline Summary

**Problem**: Agentic RAG fails due to trajectory-level control errors because action utility is state-dependent but commonly scored statically.
**Final Method Thesis**: SAPR-RAG is a modular, state-conditioned progress / action-value modeling layer that selects query/evidence/stop actions to maximize expected progress and repairs weak steps under fixed compute.
**Final Verdict**: READY
**Date**: 2026-05-28

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- Dominant contribution: state-conditioned action-value modeling for query/evidence/stop actions (compute-matched trajectory repair).
- Explicitly rejected complexity: large-scale GRPO/online RL/MCTS; extra retrieval/tokens; prompt judges as final system.

## Must-Prove Claims
- State/history features causally matter (history-removal ablations reduce gains).
- Beats static rerankers and scalar PRMs under matched compute.

## First Runs to Launch
1. Evidence-only: HotpotQA dev subset (baseline vs non-state vs SAPR-E vs no-history).
2. Evidence-only: HotpotQA + 2Wiki full dev.
3. Stop-only and query-only modules.

## Next Action
- Proceed to `/run-experiment`.
