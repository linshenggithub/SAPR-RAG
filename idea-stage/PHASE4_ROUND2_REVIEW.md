# Phase 4 Round 2: Re-review after Revision

**日期**: 2026-05-30
**审稿人**: Codex (gpt-5.5)
**审稿对象**: ClosureRAG 修改版（Board 3 字段 + 砍掉 Evidence Selection + 明确训练方案）

---

## Overall Score: 6/10（上次 4/10）

Borderline workshop / weak conference paper proposal。比上版显著改善，但距离 AAAI 稳投还需要补硬核实验和定位。

---

## 上次批评的回应情况

| 上次批评 | 回应情况 |
|---------|---------|
| Board schema 太大 | ✅ 基本回应。3 字段比原来可控。 |
| 拼装感太强 | ⚠️ 部分回应。模块从 4 变 3，但 Extract/Validate/Merge/Closure/Repair 仍多。 |
| Evidence Selection 太贵 | ✅ 回应了。砍掉 Evidence Selection 是正确降风险。 |
| Process-reward guided 名不副实 | ✅ 基本回应。改名 ClosureRAG。 |
| Board 更新不可靠 | ⚠️ 部分回应。Extract-Validate-Merge 是对的方向，但 Extract/Validate 仍是瓶颈。 |
| 核心贡献不清 | ✅ 明显改善。"结构化 closure 替代标量 step PRM"比之前清楚。 |
| 和 S2G-RAG 区别不够硬 | ❌ 仍然是最大问题。 |

---

## 剩余四大问题

### 问题 1: Closure Reward 的分母不清楚

`total_required_gaps` 和 `all_claims` 在真实 inference 中通常**不可知**。

- 如果来自 gold reasoning chain → 只能用于训练/离线评估，不能做 inference-time reward
- 如果来自模型自己抽取 → reward 会被 extractor 的漏检操控：**少抽 gap 就高分，少生成 claim 就高 support ratio**

**建议改成 transition reward**:
```
R_gap_delta = resolved_gaps - new_unresolved_gaps - repeated_gaps
R_claim = supported_claims / generated_claims
R_stop = evidence_sufficient && no_critical_open_gap && no_unsupported_core_claim
```

### 问题 2: open_gaps 为空 ≠ 可以停止

LLM extractor 可能漏掉缺口。多跳 QA 里最危险的是 **"unknown unknowns"**：模型不知道自己少了哪一跳。

必须引入 sufficiency stress test：让 validator 判断 final answer 是否由 claims + evidence entail，而不是只看 gap list 为空。

### 问题 3: Claim 没有 provenance

`supported_claims` 如果只是文本列表，不知道被哪个 evidence sentence 支撑，审稿人会质疑"可归因"是否成立。

建议：Board 公开 3 字段保持简单，但**实现里**给每个 claim 存 provenance（claim → evidence_ids/sentence_ids）。

### 问题 4: Repair 机制还很虚

"unsupported → 补检索；gap 未闭合 → 重写 query" 具体策略、触发条件、是否改变 ReasonRAG query generation 都不清楚。如果不是核心贡献，不要过度强调。

---

## Board 3 字段是否足够

作为最小状态够用。但缺：

| 缺失项 | 为什么需要 | 建议 |
|--------|----------|------|
| evidence provenance | claim→evidence 映射，"可归因"才站得住 | 实现里加，不放 Board schema |
| relation/constraint | 多跳是关系链，不是实体集合 | 2 个月内可不加 |
| conflict/uncertainty | 检索结果可能矛盾 | 2 个月内可不加 |
| gap priority | 哪些 gap 是关键缺口 | 2 个月内可不加 |

---

## "只改两个点"是否足够作为论文贡献

**可以，但要换一种说法**。

❌ 不要说："我们只改了状态表示和停止判断"
✅ 应该说："We show that explicit closure-state tracking is a minimal intervention that substantially improves stopping reliability, repair decisions, and attribution under identical retrieval/injection settings."

包装成 **minimal intervention with causal diagnosis**，用 ablation 支撑。

前提是实验必须证明三件事：
1. ReasonRAG 的主要错误确实来自 premature stop / unsupported claim / unresolved gap
2. Closure Board 能显著降低这些错误
3. 改进不是来自更多 LLM 调用或更强 judge，而是来自 closure state 本身

---

## 训练方案是否可行

Phase 1 + Phase 2 在 2 个月内基本可行。Phase 3 (RL) 不建议作为 AAAI 主线承诺。

**更稳的路线**:
1. AAAI 主实验以 **prompt ClosureRAG** 为主
2. SFT extractor/validator 作为效率压缩或附加实验
3. RL 放 future work，最多做小规模 proof-of-concept

---

## Minimum Viable Paper

能投，但还需要补三类东西：

**必须补的实验**:
- ReasonRAG vs ClosureRAG: 四数据集, same top-k, same max steps
- Stopping accuracy: premature stop rate, over-retrieval rate
- Board quality: entity precision/recall, gap recall, claim support precision
- Ablation: 只用 gaps / gaps+entities / gaps+claims / no validator
- Cost report: 额外 latency, LLM calls, tokens

**必须补的定位**:
- 和 S2G-RAG 的区别钉死：claim-support transition gate + closure reward + ReasonRAG trajectory repair + attribution evaluation
- 修正 R_gap 和 R_claim 的分母问题
- 明确 reward 只用于训练还是 inference control

---

## 最终判断

> 这版比上次强很多，已经从"方向发散的 proposal"变成了"可执行的最小论文方案"。但距离 AAAI 级别还差一个硬核点：必须证明 Closure Board 不只是 S2G-style gap judging 的变体，而是能通过 **claim-level support tracking** 和 **closure-based stopping** 修复 ReasonRAG 的具体轨迹错误。

**当前分数: 6/10**。补齐 S2G 对比、reward 定义、Board provenance 评测后可到 **7/10**。
