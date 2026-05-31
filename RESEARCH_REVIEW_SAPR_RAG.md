# Research Review: Is SAPR-RAG a publishable direction?

**Scope.** Critically review SAPR-RAG (State-Aware Process Repair for Agentic RAG) as a research direction: query/evidence/stop rewards + repair mechanism for ReasonRAG-style multi-step trajectories on complex QA (HotpotQA/2Wiki/MuSiQue/Bamboogle).

**Primary repo context used.**
- `AGENTS.md` (project positioning + SAPR-RAG definition)
- `06_notes/idea_notes/sapr_rag_idea_draft.md` (method sketch + experiments)
- `01_literature/taxonomy.md`, `01_literature/common_problems_and_ideas.md`, `01_literature/related_work_drafts/research_direction_and_problem_taxonomy.md`
- `02_baseline_reasonrag/README.md` (baseline metrics snapshot)

**External review status.** Codex MCP reviewer timed out; see `.aris/traces/research-review/20260527_run01/`.

---

## 1) Executive assessment

SAPR-RAG is **plausibly publishable** *if* you make the contribution **crisp and falsifiable** and avoid it being perceived as “a bundle of heuristics + extra compute.” The most defensible novelty is **state-aware evidence utility** and **decision-decomposed process evaluation** (query / evidence / stop) with **repair-triggered control**, plus **diagnostic metrics** that demonstrate *why* the improvements happen.

However, the default framing (“we add rewards for query/evidence/stop and repair”) is at high risk of being judged as:
- incremental relative to process-supervised / PRM / stop-control work;
- an engineering wrapper rather than a scientific contribution;
- confounded by extra inference budget and LLM-judge leakage.

**Recommendation:** Commit only after a small “go/no-go package” shows (i) consistent evidence-level gains *and* (ii) at least modest end-task gains under compute-matched settings, with ablations that isolate which head fixes which failure.

---

## 2) Where the novelty really is (and isn’t)

### Likely novelty (if executed well)
1. **State-aware evidence utility**: formalize and validate `U(d | q, s_t)` where `s_t` includes history evidence/queries/intermediate claims/gap, not just `score(q, d)`.
2. **Decision-decomposed evaluation**: explicit separation of *query quality*, *evidence utility*, and *stop sufficiency* as distinct, interpretable signals rather than a single step/path score.
3. **Repair as controlled intervention**: showing that *triggered* rewrites/reranks/continue decisions correct *specific* failure types (bridge loss, noise confusion, premature stop), with measurable causal links.
4. **A diagnostic protocol** that makes agentic RAG failures measurable (e.g., bridge preservation rate, unsupported intermediate claim rate, premature stop rate) and ties them to module activations.

### Weak novelty / “obvious engineering” risk
- “Add a reranker” alone is not novel.
- “Add a verifier to prevent hallucination / require evidence” is common.
- “Add a stop criterion” is an established knob.
- “Use an LLM-as-judge to score steps” is now standard; reviewers demand careful leakage/control.

**What makes it scientific:** the *state definition*, the *utility decomposition*, and *evidence that these signals better explain + improve trajectory quality than existing process/path rewards*.

---

## 3) Closest prior work (and how reviewers may map you onto it)

Based on your taxonomy (`01_literature/taxonomy.md`) the most relevant nearby clusters are:
- **ReasonRAG**: already introduces process supervision for query generation / evidence extraction / answer generation. Reviewers may ask: “Aren’t your rewards just a rephrasing of ProGUIDE signals?”
- **ProRAG / PRM-guided methods**: process reward models that score steps; reviewers may ask: “Isn’t your query/evidence/stop decomposition just a multi-head PRM?”
- **HiPRAG**: explicit over-search / under-search control; reviewers may ask: “Is your Stop Reward substantively different or just re-implementing search-necessity?”
- **DecEx-RAG**: decision/execution MDP framing; reviewers may ask: “Do you add anything beyond splitting the action space and scoring it?”
- **Search-P1 / path reward shaping**: trajectory-level credit assignment; reviewers may ask: “Is your repair controller just another form of reward shaping/credit assignment?”
- **Search-R1 / outcome-RL**: mostly background, but sets the “why process signals” motivation.

**Required differentiation sentence (you must be able to defend):**
> Prior work provides step/path feedback, but does not explicitly model *state-conditioned* marginal utility of evidence and does not separate query/evidence/stop errors into distinct, actionable signals with compute-matched causal interventions.

If you can’t back that with experiments/metrics, the novelty collapses.

---

## 4) Feasibility & what will be accepted as “enough”

### V0 (prompt-based judges only): feasible, but risky as a paper core
V0 can be compelling **as a diagnosis + protocol paper** (failure bank + diagnostics + strong ablations) or as a “system” paper, but top-tier reviewers often require:
- either *training* (a learned reward model / reranker) **or**
- extremely careful controls showing improvements are not due to larger inference compute / stronger LLM / judge bias.

If V0 is purely prompt heuristics, expect pushback: “This is a pipeline with hand-designed prompts.”

### Minimal path to publishability
A credible minimum path is:
1. **Evidence utility result that is hard to dismiss**: state-aware reranking improves gold-evidence rank/recall and reduces noise@k across datasets.
2. **Compute-matched end-task gains**: EM/F1 improvements with the *same or lower* retrieval/LLM calls, or show a Pareto curve (quality vs cost).
3. **Ablations with causal alignment**: each head fixes the failure types it claims to fix.
4. **At least one learned component** (optional but strong): train a small reranker or RM on your constructed preference pairs and show it generalizes beyond HotpotQA.

---

## 5) Fatal weaknesses / likely reviewer attacks (plan defenses now)

1. **Compute confound / “more tries wins”**
   - If SAPR-RAG does extra steps, higher top-k, or more LLM calls, reviewers will say improvements are budget-based.
   - **Defense:** report cost, match budgets, provide Pareto curves.

2. **LLM-as-judge leakage / circularity**
   - If the same model family judges and is being improved, you can get self-fulfilling gains.
   - **Defense:** use a separate judge model; use annotation subsets; rely on dataset supervision when available (Hotpot supporting facts) for evidence metrics.

3. **Overfitting to one dataset / annotation artifact**
   - HotpotQA supporting facts make evidence scoring easier; may not transfer to MuSiQue/2Wiki.
   - **Defense:** show at least 2 datasets; design dataset-agnostic metrics (unsupported claim rate, stop accuracy).

4. **Unclear definition of “remaining gap” / state**
   - If the state includes a magic “remaining_gap” generated by an LLM, reviewers may say you injected extra reasoning.
   - **Defense:** ablate with/without gap; show robustness; constrain gap to extractable slots.

5. **Contribution dilution (too many modules)**
   - Bundled improvements are hard to attribute.
   - **Defense:** anchor around *one* flagship contribution (evidence utility), make query/stop/repair supporting modules.

6. **Evaluation mismatch**
   - Evidence rank improving but final EM not improving may look like “proxy not aligned.”
   - **Defense:** add intermediate metrics (second-hop success, entailment rate) and explain pipeline bottlenecks.

---

## 6) Minimum experiments before committing (go/no-go)

This is the smallest set that (a) de-risks feasibility and (b) best predicts whether this can become a solid paper.

### A. Baseline + trajectories (must-have)
- Freeze a ReasonRAG baseline config and trajectory schema.
- For HotpotQA dev: store per-step subquery, retrieved list, selected evidence, intermediate claim, stop, cost.

### B. Failure Bank V0 (must-have)
- Build **100 badcases** step-level Failure Bank with failure types.
- Report: distribution + “rerankable/repairable” share.
- **Sanity:** small human audit of labels (even 30–50 steps) for credibility.

### C. Evidence utility flagship (must-have)
- Compare rerankers on **per-step** doc ranking quality:
  - original retrieval order vs query-doc reranker vs state-aware reranker
  - include an oracle upper bound
- Report: Recall@k / MRR of gold evidence, noise@k.
- Then show downstream effect: EM/F1 change (compute-matched), plus repair-rate on the Failure Bank.

### D. One more head (pick one) (high ROI)
Pick **Stop verifier** *or* **Query repair** for V0 (not both), to avoid story dilution.
- Stop verifier is often higher leverage (directly targets unsupported/premature stop).

### E. Compute-matched ablation table (must-have)
- At minimum: baseline vs +E vs (baseline + extra steps/top-k without rerank) to isolate the effect.

### F. Dataset generalization (must-have for strong venues)
- Replicate A–C on **2Wiki or MuSiQue** (even small subset) to show non-Hotpot dependence.

If A–C fail to produce a strong evidence-utility story, do not commit to full SAPR-RAG yet.

---

## 7) Results-to-claims matrix (avoid overclaiming)

### Scenario (i): evidence ranking improves, EM doesn’t
Allowed claims:
- “State-aware utility improves evidence ranking metrics (MRR/Recall@k) and reduces retrieval noise under multi-hop states.”
- “Evidence ranking is a measurable bottleneck, but downstream generator remains limiting.”
Not allowed:
- “Improves QA performance” or “repairs trajectories” (unless you show repair-rate / entailment improvements clearly).

### Scenario (ii): EM improves, evidence metrics don’t
Allowed claims:
- “Repair controller improves end-task performance via query/stop control even without improving gold evidence rank.”
- “The failure mode is not evidence ranking but stopping/claim verification.”
Not allowed:
- “We improved evidence utility modeling.”
Action: shift novelty anchor to query/stop control + entailment.

### Scenario (iii): improves only on HotpotQA, not on MuSiQue/2Wiki
Allowed claims:
- “Effective in settings with strong supporting-fact supervision / Wikipedia entity structure.”
- “Diagnosis reveals dataset-specific bottlenecks.”
Not allowed:
- general claims about agentic RAG broadly.
Action: refine scope or add robustness/transfer experiment.

---

## 8) Reframing suggestion (to maximize acceptance)

Make the paper’s “one-sentence contribution” something like:
> We introduce *state-aware evidence utility* for multi-step RAG and show it yields interpretable, compute-controlled improvements in evidence quality and downstream QA, with a failure-bank-driven diagnosis protocol.

Then present query/stop rewards as extensions/ablations, not the core novelty.

---

## 9) Immediate next actions (concrete)

1. Define and freeze the **trajectory schema** and cost accounting.
2. Build Failure Bank V0 (100 badcases) + 30-step human audit.
3. Implement + evaluate state-aware evidence reranking (LLM-judge V0), with compute-matched baselines.
4. If (3) is promising, decide between: train a lightweight reranker OR add stop verifier next.

---

## External Reviewer Round 1 (gpt-5.5, xhigh)

- Trace: `.aris/traces/research-review/20260528_run03/review_round1.md`
- ThreadId: `019e6a5b-5ff0-7ed1-9044-5c21f04eb9ef`

Key takeaways:
- **Novelty risk is the #1 threat**: must show what ReasonRAG/DecEx/HiPRAG/ProRAG cannot already do.
- Prompt-judge V0 is fine for debugging/data, but **not a contribution** for top venues.
- Strongest anchor: **state-aware evidence utility** (history-conditioned), but must beat strong non-state rerankers under **compute-matched** settings.
- Required: modular heads > single scalar PRM; repair > scoring-only; generalize beyond HotpotQA.
