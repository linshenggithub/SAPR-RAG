# Divergent Idea Novelty Cross-check

**日期**: 2026-05-30
**方法**: Codex (gpt-5.5) 文献检索 + 逐条竞品分析
**对照文献库**: 01_literature 6 篇核心论文 + idea-stage/LITERATURE_LANDSCAPE.md 30 篇扩展论文
**评估视角**: 每个 idea 能否作为 ReasonRAG 改进方向（不是完全替代）

---

## 总览

| Idea | Novelty | 核心竞品 | ReasonRAG 适配建议 |
|------|---------|---------|-------------------|
| 1. Question-to-World Model | ⚠️ PARTIAL | GraphRAG, HippoRAG, TRACE | 降级为 state representation |
| 2. ASP RAG | ⚠️ PARTIAL | LLM+ASP 已有，RAG+ASP 做 QA 尚可 | Stop Reward verifier |
| 3. Hypothesis-First | ❌/⚠️ | HCQR (2602.14735) 已很接近 | Repair mechanism |
| 4. Experiment Design | ⚠️ PARTIAL | InfoGainRAG, iG2RAG | **Query Reward** |
| 5. Entity Ledger | ❌ NOT NOVEL | LedgerRAG, SEAL-RAG 名字都撞 | 内部机制，不可主贡献 |
| 6. Debate-RAG | ❌/⚠️ | MAIN-RAG, AC-RAG | Verifier，不适合主方法 |
| 7. Program-of-Retrieval | ❌ NOT NOVEL | PyRAG (2605.12975) | 改动太大，不适合 ReasonRAG |
| 8. Counterfactual RAG | ❌ NOT NOVEL | RAGONITE, ContextCite | **离线标注器** |
| 9. Memory Palace | ⚠️ PARTIAL | MemPalace 命名已占 | State slot 建模 |
| 10. One-Shot Massive | ❌/⚠️ | ECoRAG, LongRAG, submodular | **强 baseline** |

---

## 逐条详细分析

### Idea 1: Question-to-World Model — ⚠️ PARTIAL

**核心竞品**:

1. **GraphRAG** (2404.16130, Microsoft, 2024): 从文本构建 entity graph + community summaries → 回答问题。已经覆盖"先构建结构化知识再回答"。
2. **HippoRAG** (2405.14831, NeurIPS 2024): 用 KG/Personalized PageRank 做跨文档联想和多跳检索。
3. **TRACE** (2509.18988, 2025): KG Generator + reasoning chain constructor 生成可追踪知识推理链。

**差异化空间**: 不是全局 KG，而是 ReasonRAG 每一步的 state-local world model，显式服务于 query/evidence/stop reward。

**ReasonRAG 适配**: 作为 evidence-state memory。每步把 top-k evidence 抽成局部 entity-relation graph，辅助判断 entity loss、path drift、unsupported intermediate answer。

---

### Idea 2: ASP RAG — ⚠️ PARTIAL

**核心竞品**:

1. **LLM as Logical Solvers** (2311.06158, 2023): NL → symbolic formulation → solver，路线已验证。
2. **LLM-Based ASP Frameworks** (2412.18589, 2024/2025): LLM 生成/修复 ASP 程序。
3. **ASP for KG QA** (2209.01598, 2022): 用 ASP 做 KG 问答中的约束满足。

**差异化空间**: 面向 open-domain complex QA，检索填充候选 facts，ASP 不替代 LLM 而是校验多跳约束。

**ReasonRAG 适配**: 作为 Stop Reward / Final Verification module。ReasonRAG 生成 trajectory → LLM 把 question + evidence 编译为 ASP facts/rules → solver 检查约束满足。

---

### Idea 3: Hypothesis-First Retrieval — ❌/⚠️

**核心竞品**:

1. **HyDE** (ACL 2023): 先生成 hypothetical document 再检索，"先假设再检索"的基本范式已覆盖。
2. **GenGround** (ACL 2024 Findings): 先生成中间内容/分解，再 iterative grounding，非常接近"假设→检索验证"。
3. **HCQR** (2602.14735, 2026): 先生成多个假设、聚类并据此构造查询，已经覆盖 hypothesis-first retrieval。

**差异化空间**: 明确做 prove/refute/discriminate 三类检索动作；假设竞争转成 process reward；专门修复 ReasonRAG 的 premature stopping。

**ReasonRAG 适配**: 当 evidence 不足时，先生成 3-5 个候选 answer hypotheses → 为每个假设生成支持/反驳检索 query → verifier 选择继续方向。

---

### Idea 4: Retrieval as Experiment Design — ⚠️ PARTIAL

**核心竞品**:

1. **iG2RAG** (2510.08865, 2025): 显式用 information gain 指导图/检索增强生成。
2. **InfoGainRAG** (2502.14844, 2025): 文档加入前后答案分布变化衡量信息价值。
3. **Active retrieval** 系列: FLARE 等自适应检索决策。

**差异化空间**: 不只是 document utility，而是 Agentic RAG 的 action selection——每一步选择最能区分候选答案或减少 stop uncertainty 的 observation。与 ReasonRAG 的 query/evidence/stop 三类决策绑定。

**ReasonRAG 适配**: **最适合做 Query Reward**。对候选 subquery 估计 expected uncertainty reduction：
```
reward(q_t) = expected reduction in answer entropy
            + bridge entity coverage
            - repetition penalty
```

---

### Idea 5: Entity Ledger RAG — ❌ NOT NOVEL

**核心竞品**:

1. **LedgerRAG** (Electronics, 2026): 名字和核心概念都撞了——用 evidence ledger 追踪和验证证据。
2. **SEAL-RAG** (ICLR 2026 submission): 直接使用 Live Entity Ledger，面向 traceable/verifiable RAG。

**Novelty 判定**: 名字和主干已被覆盖。不可作为主创新点。

**ReasonRAG 适配**: 降级为内部机制 Entity-state tracker / candidate evidence accounting，用于检测 entity loss、bridge entity drift。

---

### Idea 6: Debate-RAG — ❌/⚠️

**核心竞品**:

1. **MAIN-RAG** (ACL 2025): 多 agent 过滤和验证检索内容。
2. **AC-RAG** (2503.18191, 2025): adversarial/collaborative RAG 场景。
3. 多个 Advocate/Skeptic/Judge 框架 (2505.15760 等): 多角色辩论批判已成常见变体。

**差异化空间**: Skeptic 不只 critique 而是生成 counter-evidence retrieval query；Judge 检查 minimal sufficient evidence set；与 ReasonRAG trajectory repair 结合。

**ReasonRAG 适配**: 适合做离线 badcase analyzer 或 inference-time verifier，不建议作为主方法名。

---

### Idea 7: Program-of-Retrieval — ❌ NOT NOVEL

**核心竞品**:

1. **PyRAG** (2605.12975, 2026): LLM 生成 Python workflow，把 retrieval/reasoning/computation 编成可执行程序。非常接近。
2. **PlanRAG** (2406.12430, 2024): 先生成 plan，再执行检索和推理。
3. **ReAct / DSPy**: 可编程检索/工具调用框架。

**Novelty 判定**: "LLM 生成可执行检索程序"已非常拥挤。除非 DSL 有独特语义（entity binding, evidence join, contradiction check），不建议主创新。

**ReasonRAG 适配**: 把 ReasonRAG 自由文本 subquery 改成结构化 retrieval plan，但改动太大，适配成本高。

---

### Idea 8: Counterfactual RAG — ❌ NOT NOVEL

**核心竞品**:

1. **RAGONITE** (2407.14599, 2024): 明确使用 counterfactual data augmentation / evidence perturbation 增强 RAG。
2. **InfoGainRAG** (2502.14844): 加入/移除文档前后答案变化衡量，本质上接近 counterfactual utility。
3. **ContextCite** (2409.00729, 2024): 上下文扰动/移除做归因。

**Novelty 判定**: "移除/替换证据看答案是否改变"作为 evidence attribution 已被覆盖。不建议单独立题。

**ReasonRAG 适配**: **非常适合做离线标注器**——对每步 evidence 做 leave-one-out，标为 high utility / redundant / noise。与 state-aware evidence utility 方向很契合。

---

### Idea 9: Memory Palace RAG — ⚠️ PARTIAL

**核心竞品**:

1. **MemPalace** (2025/2026): "Memory Palace" 命名已被使用。
2. **HippoRAG / GraphRAG / Mem0**: 结构化记忆 RAG 已是常见路线。

**差异化空间**: rooms 不是长期记忆，而是 per-question reasoning state；每个 room 对应错误类型（entity/timeline/claim/conflict）。

**ReasonRAG 适配**: 作为 state representation——历史 evidence 解析进 typed memory slots → Query Reward 检查是否填补空 slot → Stop Reward 检查 answer 所需 slots 是否齐全。

---

### Idea 10: One-Shot Massive Retrieval + Compression — ❌/⚠️

**核心竞品**:

1. **ECoRAG** (Findings ACL 2025): 面向 RAG 的 evidence compression。
2. **LongRAG** (2406.15319, 2024): 长上下文减少多轮检索需求。
3. **Submodular optimization for RAG** (2407.08962): 从大量候选中选覆盖性/多样性最优 evidence subset。

**差异化空间**: 适合做对照实验——回答"ReasonRAG 多步检索到底比 one-shot high-recall compression 强在哪里"。

**ReasonRAG 适配**: 作为强 baseline + fallback repair。ReasonRAG 失败时用 one-shot 作为 backup；用 one-shot evidence set 反向标注 ReasonRAG 哪些步骤是 redundant/drift。

---

## 综合结论

### 1. 没有单独的范式级 idea 可以直接立题

10 个发散性 idea 都有已发表的相近工作。最危险的（Idea 5/7/8）连名字和核心概念都被覆盖。这不是 Codex 的 idea 质量问题，而是 Agentic RAG 方向竞争极其激烈。

### 2. 但有 4 个 idea 的降级版本对 ReasonRAG 改进有价值

按适配优先级排序：

| 优先级 | 来源 Idea | ReasonRAG 适配角色 | 具体用法 |
|--------|----------|-------------------|---------|
| 🥇 1 | Idea 4 (Experiment Design) | **Query Reward 信号** | 用 expected information gain 替代/增强 heuristic query reward |
| 🥈 2 | Idea 8 (Counterfactual) | **离线 evidence 标注器** | leave-one-out 标注 evidence utility/necessity，替代 heuristic scorer |
| 🥉 3 | Idea 10 (One-Shot Massive) | **强 baseline + repair fallback** | 证明多步 vs 单步的差异，失败时 fallback |
| 4 | Idea 1/9 (World Model / Memory Palace) | **State representation** | 把 flat history 改为 typed entity/relation/claim slots |

### 3. 回到 SAPR-RAG 框架的可能性

这些降级适配本质上都在 SAPR-RAG 的 Query/Evidence/Stop 三模块框架内：
- **Query Reward** ← Idea 4 (information gain)
- **Evidence Reward** ← Idea 8 (counterfactual 标注)
- **Stop Reward** ← Idea 2 (ASP constraint check) / Idea 10 (one-shot fallback)
- **State Representation** ← Idea 1/9 (typed slots)

### 4. 下一步建议

1. **不要追求全新范式**——这个方向竞争太激烈，单独范式创新风险极高
2. **回到 ReasonRAG 改进主线**——但用新获得的文献知识改进 SAPR-RAG 的具体模块设计
3. **最有价值的新实验**：先做 Idea 10 (One-Shot vs Multi-Step 对比)，回答"ReasonRAG 的多步检索是否真的必要"，这本身就是一个有价值的研究发现
4. **Query Reward 优先尝试 Idea 4 的 information gain 思路**，比 heuristic rule 更有理论根基

---

## 新发现的竞品论文（需要补充到文献库）

| 论文 | 年份 | 与本课题关系 |
|------|------|------------|
| GraphRAG (2404.16130) | 2024 | 结构化知识 + RAG |
| HippoRAG (2405.14831) | NeurIPS 2024 | KG 联想记忆 + 多跳检索 |
| GenGround (ACL 2024 Findings) | 2024 | 先生成再 grounding |
| HCQR (2602.14735) | 2026 | 假设聚类 + query reasoning |
| iG2RAG (2510.08865) | 2025 | 信息增益指导 RAG |
| LedgerRAG (Electronics 2026) | 2026 | Evidence ledger（名字撞了） |
| SEAL-RAG (ICLR 2026 sub) | 2026 | Live Entity Ledger |
| PyRAG (2605.12975) | 2026 | Python 可执行 RAG workflow |
| ECoRAG (Findings ACL 2025) | 2025 | RAG evidence compression |
| LongRAG (2406.15319) | 2024 | 长上下文 RAG |
| RAGONITE (2407.14599) | 2024 | 反事实增强 RAG |
| ContextCite (2409.00729) | 2024 | 上下文归因 |
| MAIN-RAG (ACL 2025) | 2025 | 多 agent RAG 验证 |
| PlanRAG (2406.12430) | 2024 | 先 plan 再检索 |
| TRACE (2509.18988) | 2025 | 知识推理链追踪 |
