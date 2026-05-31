# ReasonRAG 改进 Idea 报告（基于深度文献分析）

**日期**: 2026-05-30
**方法**: Codex (gpt-5.5) 基于用户 01_literature/ 6 篇详细 paper notes + badcase 分析 + 8 种 failure type
**核心叙事**: 从 trajectory-level process reward 走向 **state-closure-aware process control**
**关键差异化**: 不只是多头打分，而是状态闭合驱动的过程控制

---

## 方法论说明

前两轮 Codex brainstorming 的局限：
- 第一轮：给 gap list + 30 篇摘要 → 产出 SAPR-RAG 增量改进，和已有 proposal 重叠
- 第二轮：要求范式发散 → 产出 10 个新范式 idea，但 novelty cross-check 发现多数已被覆盖
- **第三轮**：喂入用户在 01_literature/ 下的 6 篇深度 paper notes（每篇含方法拆解、局限、接口差异、可转化模块、写作表述），以及 common_problems_and_ideas.md 和 taxonomy.md → 产出直接针对 ReasonRAG 具体组件的改进方案

---

## Idea 总览

| # | Idea | 改动组件 | 解决 badcase | 论文角色 |
|---|------|---------|-------------|---------|
| 1 | Evidence State Board | state representation + prompt | query drift, entity loss, unsupported claim, premature stop, over search | 主线：状态表示 |
| 2 | Evidence Marginal Utility | retrieval selection | retrieval noise, gold_rank_low, entity loss, over search | 主线：证据选择 |
| 3 | Claim-Support Gate | reward signal + routing | unsupported claim, premature stop, query drift | 主线：推理约束 |
| 5 | Stop-by-Closure | stop reward + routing | premature stop, over search, unsupported claim | 主线：停止决策 |
| 4 | Bridge Entity Conservation | query prompt + repair | query drift, repeated query, entity loss, noise | 增强：query 质量 |
| 6 | Distractor-Aware Portfolio | retrieval selection | retrieval noise, gold_rank_low, over search, drift | 增强：反噪声 |
| 7 | Tree-Mode + Router Fix | routing logic | premature stop, over search, unsupported claim, repeated query | 工程可靠性 |
| 8 | Failure-Attribution PRM | evaluate_thoughts 扩展 | 覆盖 8 类 | 分析工具 |

---

## 主线 Idea 详细分析

### Idea 1: Evidence State Board — 结构化状态替代 flat history

**核心思想**: 把 ReasonRAG 的 `" ".join(thoughts)` 改成结构化 state board，显式记录已确认实体、开放信息槽、证据支持关系、冲突点、下一跳需求。

**改动组件**: `state representation` + prompt。Tree mode 中每步更新 StateBoard，作为 Previous Thoughts 的替代或补充。

**解决 badcase**: query drift, missing_bridge_entity, unsupported_claim, premature_stop, over_search

**与 6 篇竞品的差异化**:
- ReasonRAG/ProRAG/Search-P1 仍主要对 step/path 打分
- DecEx-RAG 有 MDP state 但没有证据闭合状态
- HiPRAG 判断是否搜，但不表示"缺哪个槽"
- **无人维护结构化的 evidence closure state**

**最小验证**: HotpotQA 50 条；对比原 ReasonRAG tree mode vs StateBoard prompt。
- 指标: EM/F1、平均步数、bridge entity recall、premature_stop rate、unsupported claim rate
- 预期: 1-2 天

**最大风险**: LLM 生成的 board 本身出错，错误状态持续传播。

**Scale 路径**: 先用 prompt 版 state board；信号确认后蒸馏小型 state updater/verifier。

---

### Idea 2: State-Conditioned Evidence Marginal Utility — 证据边际效用

**核心思想**: 不是单独 evidence reranking，而是判断候选文档对当前 state board 的"边际贡献"：补了哪个 open slot、是否引入新桥接实体、是否降低歧义。

**改动组件**: retrieval 后新增 evidence utility scorer，在 top-10 中选 top-3 注入 prompt。

**解决 badcase**: retrieval_noise, gold_rank_low, missing_bridge_entity, over_search

**与 6 篇竞品的差异化**:
- InfoGainRAG/iG2RAG 看信息增益，但不绑定 ReasonRAG 的 reasoning state
- HiPRAG 管 search necessity，不管 top-k 哪篇最有用
- ReasonRAG step PRM 没有独立建模 U(d|s_t)
- **无人把 evidence utility 绑定到结构化推理状态**

**最小验证**: HotpotQA 100 条，保留 query 不变，只替换文档选择。
- 指标: gold doc hit@3、supporting fact recall、noise rate、F1
- 预期: 1-2 天

**最大风险**: LLM judge 评分慢且不稳定；需要 deterministic rubric。

**Scale 路径**: 用 LLM judge 生成 1k-5k 偏好对，训练 BGE-style cross-encoder 或 LoRA reward head。

---

### Idea 3: Claim-Support Gate — 中间结论必须绑定证据

**核心思想**: ReasonRAG 每步生成 intermediate thought 后，要求抽取 claim，检查 claim 是否被当前 evidence 支持；不支持则禁止进入 answer 或触发 repair。

**改动组件**: reward signal + routing logic。在 evaluate_thoughts 外加 step entailment gate。

**解决 badcase**: unsupported_claim, premature_stop, query_drift

**与 6 篇竞品的差异化**:
- ProRAG 提到 process hallucination，但仍是整体 PRM
- Search-P1 是 path reward，不做 step-level claim verification
- **无人把"claim 是否由证据支撑"变成 ReasonRAG 状态转移的硬约束/软门控**

**最小验证**: 30-50 条 badcase 先跑。
- 指标: unsupported claim rate、错误停止率、answer faithfulness、EM/F1
- 预期: 半天到 1 天

**最大风险**: 过严会阻止合理隐式推理，导致步数增加。

**Scale 路径**: 从 rule/LLM judge gate → 小型 NLI verifier，或只在低置信步骤触发。

---

### Idea 5: Stop-by-Closure — 停止 = 证据链闭合

**核心思想**: 停止前检查 state board 中所有必要 information slots 是否有 evidence-backed claim；未闭合则继续检索或重写 query。

**改动组件**: stop reward + routing logic。替代 batch mode 中简单 flag "So the answer is" 的路由。

**解决 badcase**: premature_stop, over_search, unsupported_claim

**与 6 篇竞品的差异化**:
- HiPRAG 关注 over/under-search，但没有细到"哪个槽未闭合"
- ReasonRAG 的 stop 由生成文本/PRM 间接决定，不做 evidence closure
- **无人把 stop 决策绑定到结构化的 evidence closure state**

**最小验证**: HotpotQA 100 条。
- 指标: premature_stop rate、over_search rate、平均步数、answer support rate、F1
- 预期: 1 天

**最大风险**: closure judge 偏保守导致 over_search。

**Scale 路径**: 把 closure 类型细分为 entity slot、relation slot、comparison slot、temporal slot，形成可解释 stop reward。

---

## 增强 Idea

### Idea 4: Bridge Entity Conservation Query Reward

生成子查询后检查是否保留原问题关键实体和上一跳 bridge entity；丢失则重写。
- 改动: query generation prompt + query reward/repair
- 解决: query_drift, repeated_query, missing_bridge_entity, noise
- 验证: HotpotQA/2Wiki 各 50 条，1 天
- 风险: 实体抽取错误

### Idea 6: Distractor-Aware Retrieval Portfolio

top-3 不只选高分文档，而选 portfolio：1 篇主相关 + 1 篇桥接补全 + 1 篇歧义消解/反干扰。
- 改动: retrieval selection + document injection prompt
- 解决: retrieval_noise, gold_rank_low, over_search, drift
- 验证: HotpotQA 100 条（含同名实体/相似事件问题），1-2 天
- 风险: portfolio 策略牺牲直接相关性

---

## 工程/分析 Idea

### Idea 7: Tree-Mode + Batch Router Verifier

把 batch mode 的 "query"/"answer"/"evidence"/None 文本检测替换成 state-machine verifier。
- 学术 novelty 单独不够，但作为所有方法的默认执行框架很有价值
- 我们已发现 batch mode 的 flag 路由是 max_tokens bug 的根因
- 适合放入 implementation reliability ablation

### Idea 8: Failure-Attribution PRM

每步输出 failure attribution（query_error / evidence_error / claim_error / stop_error / no_error），用它指导 repair。
- 比 SAPR-RAG 三头更具体：不是泛泛 Q/E/S reward，而是可执行的错误归因
- 适合先做 LLM judge 标注，积累数据后训练轻量 classifier

---

## 论文骨架建议

```
§1 Introduction
§2 Related Work
   - Search-R1 → ReasonRAG → HiPRAG/ProRAG/Search-P1 → DecEx-RAG 谱系
   - 指出共同缺口：无人维护 state-closure-aware process control
§3 Method: State-Closure-Aware Process Control for Agentic RAG
   §3.1 Evidence State Board（Idea 1）
   §3.2 Claim-Support Gate（Idea 3）
   §3.3 State-Conditioned Evidence Marginal Utility（Idea 2）
   §3.4 Stop-by-Closure（Idea 5）
§4 Experiments
   §4.1 Main Results（HotpotQA / 2Wiki / MuSiQue / Bamboogle）
   §4.2 Ablation: 4 个组件逐一去掉
   §4.3 Analysis: failure attribution accuracy, bridge entity recall, noise rate
§5 Discussion
   - Bridge Entity Conservation（Idea 4）作为 query enhancement
   - Distractor-Aware Portfolio（Idea 6）作为 retrieval enhancement
```

---

## 与前两轮 Idea 的关系

| 前两轮 Idea | 第三轮对应 | 关系 |
|------------|----------|------|
| State Ablation (第一轮 #1) | Idea 1 State Board | 消融 → 结构化替代 |
| Evidence Reward (第一轮 SAPR) | Idea 2 Marginal Utility | 启发式 → state-conditioned |
| Failure-Type Repair (第一轮 #2) | Idea 8 Attribution PRM | 三头 → 可执行错误归因 |
| Compute-Matched (第一轮 #3) | 整体框架 | budget-aware 的过程控制 |
| Hypothesis-First (第二轮 #3) | Idea 5 Stop-by-Closure | 假设竞争 → 闭合检查 |
| Information Gain (第二轮 #4) | Idea 2 Marginal Utility | 同源，适配 ReasonRAG |
| Counterfactual (第二轮 #8) | Idea 2 标注器 | 反事实 → 边际效用估计 |
| One-Shot Massive (第二轮 #10) | baseline 对照 | 强 baseline |

---

## 下一步

- [ ] 用户选择验证优先级
- [ ] Idea 1 (State Board) 最便宜最稳，建议先做
- [ ] Idea 3 (Claim-Support Gate) 最直接解决 unsupported claim
- [ ] Idea 7 (Router Fix) 是工程必须，可和 Idea 1 一起做
- [ ] 信号确认后 scale 到完整实验
