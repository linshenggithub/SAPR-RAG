# Pipeline Summary

**Problem**: Agentic RAG fails due to trajectory-level control errors because action utility is state-dependent but commonly scored statically.
**Final Method Thesis**: SAPR-RAG is a modular, state-conditioned progress / action-value modeling layer that selects query/evidence/stop actions to maximize expected progress and repairs weak steps under fixed compute.
**Final Verdict**: READY
**Date**: 2026-05-28

## Final Deliverables
- Proposal: `docs/proposal.md` (latest, was `refine-logs/FINAL_PROPOSAL.md`，已并入 docs/)
- History 演化记录: `docs/history.md`
- Experiment plan: `docs/experiment_plan.md`
- Experiment tracker: `docs/experiment_tracker.md`

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
