# Research Idea Report

**Direction**: 面向复杂问答的 Agentic RAG 多步检索推理过程优化
**Generated**: 2026-05-31
**Ideas evaluated**: 10 generated → 6 survived filtering → 0 piloted → 3 recommended
**Sources**: arXiv 30-paper landscape + Codex brainstorming (gpt-5.5)

---

## Landscape Summary

本课题研究 Agentic RAG 在多跳复杂问答中的轨迹控制问题。通过 ReasonRAG baseline 复现和 badcase 分析，确认了 8 种失败类型（query drift、entity loss、retrieval noise、premature stop 等）。文献调研覆盖 30 篇论文，发现最直接的竞争者是 FAIR-RAG（gap analysis + iterative refinement, HotpotQA +8.3pp）和 PRISM（三模块 agentic retrieval）。现有方法虽然将历史轨迹放入 prompt，但缺少可分解、可干预、可修复的结构化过程状态。我们修正了"Step PRM 不用历史"的错误判断——ReasonRAG 的 generation 和 evaluation 都包含 Previous Thoughts，但历史是作为扁平文本拼入，无法做维度级消融。

---

## Recommended Ideas (ranked)

### 🏆 Idea 1: Structured State Ablation for Agentic RAG

- **Hypothesis**: 不是"有没有 history"决定效果，而是不同状态组件对不同决策有不同因果作用——query generation 更依赖 entity chain，evidence selection 更依赖 history documents，stop decision 更依赖 accumulated support
- **Minimum experiment**: 抽 200-500 条 ReasonRAG trajectories，构造 6 种 prompt ablation（full flat / remove docs / remove queries / remove thoughts / entity chain only / evidence summary only），比较 action accuracy、gold evidence recall、stop quality
- **Expected outcome**: 若成立 → 证明 flat history 掩盖了关键状态因素，为结构化 state design 提供实证基础；若不成立 → flat history 已经足够，不需要结构化
- **Novelty**: 9/10 — 没有人做过 Agentic RAG 的状态组件因果拆解
- **Feasibility**: 仅需 prompt 改写 + 重跑推理，无需训练，1-2 周，<100 GPU-hours
- **Risk**: LOW
- **Contribution type**: diagnostic + empirical finding
- **Closest work**: ReasonRAG (flat history, no ablation), AgenticRAGTracer (diagnosis, no state ablation)
- **Reviewer's likely objection**: "这只是 ablation study，不是新方法" → 回应：ablation 揭示的因果结构直接指导后续方法设计，是 process-level understanding 的基础贡献
- **Why we should do this**: 最便宜、最稳、结果正负都有价值，为后续所有方法设计提供实证基础

### 🥈 Idea 2: Failure-Type Conditional Repair Policy

- **Hypothesis**: Agentic RAG 的错误不是同质的，按 failure type 触发 targeted repair（entity-preserving rewrite / anti-repeat / noise-aware rerank / continue-stop correction）比 generic refinement 更有效
- **Minimum experiment**: 用 8 类 badcase taxonomy，先做 LLM+规则 failure classifier，每类设计 repair prompt，对 HotpotQA badcases 做 replay，比较 repaired trajectory 的 EM/F1
- **Expected outcome**: 若成立 → failure-type-specific repair 优于 generic refinement，证明 RAG 错误异质性；若不成立 → generic repair 已经足够，或 failure classifier 不够准
- **Novelty**: 8/10 — Error-Aware PRM 做了 math error typing，但没人做 RAG-specific failure taxonomy → repair mapping
- **Feasibility**: LLM judge + rule-based classifier + repair prompts，2-3 周，无需训练
- **Risk**: MEDIUM（failure classifier 准确性是关键风险）
- **Contribution type**: new method + diagnostic
- **Closest work**: FAIR-RAG (gap-based repair, not from taxonomy), Error-Aware PRM (math, not RAG)
- **Reviewer's likely objection**: "LLM judge 做分类不够可靠" → 回应：先用 LLM judge 验证方向，后续可蒸馏为小模型
- **Why we should do this**: 直接利用已有的 badcase taxonomy，把诊断转化为可执行的修复

### 🥉 Idea 3: Compute-Matched Query/Evidence/Stop Repair

- **Hypothesis**: 不同错误来源的收益不均衡——query repair 解决 missing_bridge_entity/repeated_query，evidence repair 解决 noise/gold_rank_low，stop repair 解决 premature/over_search。固定预算下存在最优模块组合
- **Minimum experiment**: 用 LLM-as-judge 实现三个独立 scorer，固定检索预算，做 7 组消融（Q-only / E-only / S-only / Q+E / Q+S / E+S / Full），比较 EM/F1 和 per-failure-type repair rate
- **Expected outcome**: 若成立 → 证明模块化设计优于通用方法，且不同模块解决不同失败类型；若不成立 → 说明错误类型高度耦合，无法独立修复
- **Novelty**: 7/10 — PRISM 有三模块但耦合，无人做独立 ablation + fixed-budget
- **Feasibility**: 需要实现三个 scorer + repair pipeline，3-4 周
- **Risk**: MEDIUM（实现复杂度较高，4 周内完成有压力）
- **Contribution type**: new method + empirical finding
- **Closest work**: PRISM (3 agents, coupled), SAPR-RAG proposal (refine-logs/)
- **Reviewer's likely objection**: "LLM judge 作为 reward 不够新颖" → 回应：核心贡献是模块化消融框架，不是 reward model 本身
- **Why we should do this**: 最完整的实验框架，消融清楚，是论文的 backbone 实验

---

## Survived but Not Top-3

| # | Idea | Rank | Why not top-3 |
|---|------|------|---------------|
| 4 | Retrieval Marginal Benefit Estimation | 4 | Idea 2/3 的子集，可合并到 stop reward |
| 5 | Evidence Chain Continuity Verifier | 5 | 有价值但窄，可作为 Idea 2 的一个 repair dimension |
| 7 | Counterfactual Trajectory Editing | 6 | 机制性强但实现成本高，可作为 Idea 1 的验证手段 |

## Eliminated Ideas

| Idea | Reason eliminated |
|------|-------------------|
| 6. Negative Result: Evidence-Only Insufficient | 有价值但需要先修复 config bug 重跑，时间成本不确定；且 negative result 单独发 AAAI 不够，更适合作为论文中的一个 section |
| 8. Process Reward Decomposition Benchmark | 3-4 周建 benchmark 太重，AAAI deadline 前不现实；且与 AgenticRAGTracer 定位重叠 |
| 9. Entity-Preserving Query Reward | 是 Idea 2 的子集（failure type = missing_bridge_entity 时的 repair），不需要单独作为 idea |
| 10. State Compression vs Full History | 是 Idea 1 的子集（ablation 的一个维度），不需要单独作为 idea |

---

## Suggested Paper Story

Codex 建议的论文主线（也是我认为最合理的）：

> **论文骨架**：Idea 1 (State Ablation, §4) → Idea 2 (Failure-Type Repair, §5) → Idea 3 (Compute-Matched Full Framework, §6)
>
> §1-2 Introduction + Related Work
> §3 Badcase Taxonomy + Failure Bank（已有的分析）
> §4 Structured State Ablation（证明不同状态组件的因果作用）
> §5 Failure-Type Conditional Repair（基于 failure taxonomy 的定向修复）
> §6 Compute-Matched Query/Evidence/Stop Full Framework（完整消融）
> §7 Experiments (HotpotQA / 2Wiki / MuSiQue / Bamboogle)
>
> 核心贡献：(1) Agentic RAG state component 因果分析 (2) RAG-specific failure-type → repair mapping (3) 三模块解耦消融

---

## Next Steps

- [ ] 用户选择 top idea(s) 或调整方向
- [ ] 对 top idea 做 `/novelty-check` 深度验证
- [ ] 对 top idea 做 `/research-review` 外部评审
- [ ] 进入实现阶段
