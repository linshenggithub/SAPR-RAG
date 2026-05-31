# Phase 3: Deep Novelty Verification

**日期**: 2026-05-30
**方法**: Codex (gpt-5.5) 多源检索 + 竞品分析
**验证对象**: 主线 4 个 idea（State Board / Marginal Utility / Claim Gate / Stop-by-Closure）

---

## 总览

| Idea | 竞品密度 | Novelty | 最危险竞品 |
|------|---------|---------|-----------|
| A. Evidence State Board | 中高 | ⚠️ PARTIAL | S2G-RAG (2604.23783) |
| B. State-Conditioned Evidence Marginal Utility | 高 | ⚠️ PARTIAL | IG-Search (2604.15148), SIGHT (2602.11551), IGP (2601.17532) |
| C. Claim-Support Gate | 高 | ⚠️ PARTIAL | RAGChecker (2408.08067) — 但多数是 post-hoc，非 transition gate |
| D. Stop-by-Closure | 很高 | ⚠️ 接近 ❌ | S2G-RAG, SURE-RAG (2605.03534), PAR²-RAG (2603.29085) |

**结论**: 四个 idea 单独都不够 novel，但合并为统一框架后 novelty 显著增强。

---

## Idea A: Evidence State Board — ⚠️ PARTIAL

### 最近竞品

1. **S2G-RAG** (2604.23783, ACL ARR 2026): 维护 sentence-level Evidence Context，每轮判断 evidence memory 是否足够回答；不足时输出 structured gap items，映射成下一轮 query。非常接近"开放信息槽 / 下一跳需求 / sufficiency state"。
2. **PyRAG** (2026): 把 multi-hop RAG 表示成 executable Python program，用变量暴露 intermediate states。不是 evidence board，但"显式状态而非自由文本 CoT"方向很近。
3. **Baleen** (2101.00436, NeurIPS 2021): 每 hop 将检索结果 condensed 成 compact context，但不是 slot/entity/claim board。
4. **IRCoT** (2212.10509, ACL 2023): CoT sentence 作为 evolving retrieval query，仍是自然语言 CoT，不是结构化 board。

### 差异化

- S2G-RAG 主要是 sufficiency + gap items + evidence context，**没有完整建模 entity graph、claim-support relation、conflict ledger**
- PyRAG 的 state 是程序变量，不是面向 evidence closure 的语义状态板
- IRCoT / Baleen 是隐式状态压缩，不是可审计的结构化 board

### 差异化建议

- 不叫"structured history"，写成 **Evidence Closure Board for MCTS-based Agentic RAG**
- Board schema 固定且可评估：entities, slots, claims, support_edges, conflicts, missing_links, next_hop_requirements
- 明确集成到 ReasonRAG 的 MCTS node state

---

## Idea B: State-Conditioned Evidence Marginal Utility — ⚠️ PARTIAL

### 最近竞品

1. **IG-Search** (2604.15148, 2026): 对每个 search step 计算 retrieved documents 对 gold answer confidence 的 information gain，作为 step-level RL reward。非常接近"检索证据的边际信息贡献"。
2. **SIGHT** (2602.11551, 2026): Self-Evidence Support + Information-Gain Driven Diverse Branching，识别能最大降低不确定性的关键状态。
3. **IGP** (2601.17532, 2026): Information Gain Pruning，用 generator-aligned utility signal 选择/剪枝 evidence。
4. **S2G-RAG** (2604.23783): 用 structured gaps 指导下一步检索，本质在问"当前缺什么证据"。

### 差异化

- IG-Search 的 IG 是相对 gold answer confidence 的 step reward，**不知道"补了哪个 slot、引入哪个 bridge entity"**
- SIGHT 的 IG 用于 branching intervention，不是 top-k candidate 在结构化 state board 下的 U(d|s_t)
- IGP 更偏 final-context selection，不是 multi-hop board-conditioned marginal contribution

### 差异化建议

- 不叫 information gain，否则被 IG-Search/SIGHT/IGP 压住
- 形式化为：`U(d | B_t, q_t) = slot_gain + bridge_gain + support_gain + conflict_resolution - redundancy - noise_risk`
- 重点评估 **utility attribution**：每个文档为什么被选中，是否真的补了 board 中的 open slot

---

## Idea C: Claim-Support Gate — ⚠️ PARTIAL

### 最近竞品

1. **RAGChecker** (2408.08067, 2024): 将回答拆成 atomic claims，检查 response claims vs retrieved context / ground truth，用于 faithfulness 诊断。
2. **MedRAGChecker** (2601.06519, 2026): biomedical RAG 的 atomic claim decomposition + evidence-grounded NLI。
3. **RT4CHART** (2603.27752, 2026): local-to-global hierarchical verification，标注 entailed / contradicted / baseless。
4. **SimulRAG** (2509.25459, 2025): claim-level generation + uncertainty estimation。

### 差异化

- 上述多数是 **post-hoc evaluation / diagnostic / answer verification**
- 我们把 claim-evidence entailment 放进 Agentic RAG 的**状态转移门控**：每步 intermediate thought 产生后抽 claim，若 unsupported 则不能进入 answer/stop，必须 repair
- 如果集成到 ReasonRAG MCTS，claim gate 可成为 expansion/pruning 条件

### 差异化建议

- 不主张"claim-level verification"是新的（已很拥挤）
- 主张：**transition-time claim-support gating for multi-step Agentic RAG**
- Gate 输出不只是 binary：supported / contradicted / insufficient / off-state，触发不同 repair

---

## Idea D: Stop-by-Closure — ⚠️ 接近 ❌

### 最近竞品

1. **S2G-RAG** (2604.23783, 2026): 每轮判断 evidence memory 是否支持回答，不支持则输出 structured missing gap items 并继续检索。**最危险的竞品**。
2. **SURE-RAG** (2605.03534, 2026): evidence sufficiency verification：coverage、relation strength、disagreement、conflict、retrieval uncertainty → selective answering/abstention。
3. **PAR²-RAG** (2603.29085, 2026): breadth-first anchoring + depth-first refinement + evidence sufficiency control。
4. **Don't Stop Early** (2604.24978, 2026): evidence-based completion criteria 防止 premature stopping。

### 差异化

- S2G-RAG 已覆盖"structured gap + sufficiency + continue retrieval/stop"
- SURE-RAG 已覆盖"evidence sufficiency + coverage/conflict/uncertainty + abstain"
- PAR²-RAG 已覆盖 multi-hop QA 中的"evidence sufficiency controller"
- **唯一强差异**：停止条件基于 Evidence State Board 的 slot-level evidence-backed claim closure，嵌入 ReasonRAG 的 MCTS + step PRM

### 差异化建议

- **D 不单独作为主创新**，降级为框架的一个模块（Closure Reward / Stop Reward）
- 和 S2G-RAG 对比时突出：S2G 输出 gap items；我们维护完整 board，对 slot → claim → evidence 进行闭合验证
- 做 hard negative：answer confidence 高但 evidence chain 未闭合

---

## 新发现竞品论文（需补充到文献库）

| 论文 | 年份 | arXiv ID | 核心机制 | 与我们的关系 |
|------|------|----------|---------|------------|
| **S2G-RAG** | 2026 | 2604.23783 | structured gap + sufficiency judging + iterative retrieval | ⭐⭐⭐ 最直接竞品 |
| **IG-Search** | 2026 | 2604.15148 | step-level information gain reward | ⭐⭐ B 的竞品 |
| **SIGHT** | 2026 | 2602.11551 | self-evidence + IG-driven diverse branching | ⭐⭐ B 的竞品 |
| **IGP** | 2026 | 2601.17532 | generator-aligned evidence pruning | ⭐⭐ B 的竞品 |
| **SURE-RAG** | 2026 | 2605.03534 | sufficiency + uncertainty verification + selective answering | ⭐⭐⭐ D 的竞品 |
| **PAR²-RAG** | 2026 | 2603.29085 | breadth anchoring + depth refinement + sufficiency control | ⭐⭐⭐ D 的竞品 |
| **Don't Stop Early** | 2026 | 2604.24978 | evidence-based completion criteria | ⭐ D 的背景竞品 |
| RAGChecker | 2024 | 2408.08067 | claim-level faithfulness diagnosis | C 的背景竞品 |
| MedRAGChecker | 2026 | 2601.06519 | biomedical claim verification | C 的背景竞品 |
| RT4CHART | 2026 | 2603.27752 | hierarchical hallucination detection | C 的背景竞品 |

---

## Phase 3 结论

### 单独 novelty 不够，合并为统一框架后显著增强

**四个 idea 合并为**：

> **State-Aware Evidence Closure for Process-Reward Guided Agentic RAG**
>
> 把结构化 Evidence Closure Board 注入 ReasonRAG 的 MCTS + step PRM 全流程

| 模块 | 角色 | 对应 ReasonRAG 改动 |
|------|------|-------------------|
| Evidence Closure Board (A) | 状态表示 | 替代 flat history |
| State-Conditioned Evidence Marginal Utility (B) | evidence action reward | 替代 top-k 全注入 |
| Claim-Support Gate (C) | reasoning transition constraint | 替代单一 step PRM 分 |
| Stop-by-Closure (D) | termination reward | 替代 flag-based stop |

### 与 S2G-RAG（最危险竞品）的差异化

| 维度 | S2G-RAG | 我们 |
|------|---------|------|
| 状态表示 | evidence memory + gap items | 完整 Board（entities/slots/claims/support edges/conflicts） |
| 集成方式 | 独立 iterative pipeline | 嵌入 ReasonRAG MCTS node state |
| Claim 验证 | 不做 | transition-time claim-support gate |
| Evidence 选择 | gap → query 映射 | state-conditioned marginal utility with attribution |
| Stop | sufficiency judge | slot-level evidence-backed claim closure |

### ReasonRAG 不足全覆盖

| ReasonRAG 不足 | 解法模块 |
|---------------|---------|
| Flat history 无维度消融 | Board (A) |
| 无 state-aware evidence utility | Marginal Utility (B) |
| 停搜决策不可解释 | Stop-by-Closure (D) |
| Query drift / 重复 / 合并多跳 | Board-guided Query (A) |
| Bridge entity 丢失 | Board entity tracking (A) + Evidence Utility (B) |
| 检索噪声 | Evidence Utility noise_risk (B) |
| Gold document rank 低 | Marginal Utility slot_gain (B) |
| Unsupported claim | Claim-Support Gate (C) |
| Premature stop | Stop-by-Closure (D) |
| Over search | Stop-by-Closure (D) + Board 闭合检测 |
| Step PRM 不能区分错误来源 | 四决策点独立信号 |
