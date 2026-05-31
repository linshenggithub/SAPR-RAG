# Experiment Tracker — SAPR-RAG (Go/No-Go)

## Evidence-Only Validation (2026-05-30)

| ID | Goal | Dataset | Variant | Compute matched? | Status | Result Summary |
|---|---|---|---|---|---|---|
| E2-30 | Evidence-only main (30-sample) | HotpotQA dev 30 | 6-way comparison on reretrieved data | yes | ✅ **done** | SAPR-E v0 hit@3=0.5398 > retriever 0.4690 (+7.1pp), item_hit 63.3% > 56.7% |
| E3-30 | Evidence ablation (30-sample) | HotpotQA dev 30 | no_hist, no_title, no_subquery | yes | ✅ **done** | no_title largest drop (−8.0pp hit@3); no_hist and no_subquery each −2.7pp |
| E2-200 | Evidence-only main (200-sample) | HotpotQA dev 200 | 6-way comparison on reretrieved data | yes | 🔄 running | Rag-5090 GPU 1, reretrieval + ablation in progress |
| E2-full | Evidence-only main (full dev) | HotpotQA dev 7405 | 6-way comparison | yes | not_started | Blocked on E2-200 results |

## Planned Experiments (from EXPERIMENT_PLAN.md)

| ID | Goal | Dataset | Variant | Compute matched? | Status |
|---|---|---|---|---|---|
| S2 | Stop-only main | HotpotQA/2Wiki | stop Q | yes | not_started |
| Q3 | Query-only main | HotpotQA/2Wiki | query Q | yes | not_started |
| M2 | Modular vs scalar | HotpotQA/2Wiki | Q heads vs scalar PRM | yes | not_started |
| G1 | Generalization | MuSiQue/Bamboogle | best SAPR | yes | not_started |

## Key Decisions

- **2026-05-30**: E2-30 shows directional signal ✅. Scaling to 200-sample (E2-200) for statistical significance.
- **2026-05-30**: Dim5 (title entity match) is the strongest single dimension in v0 heuristic scorer.
- **2026-05-30**: Empty-query bug fixed via inferred_subquery reretrieval with BGE query prefix.
