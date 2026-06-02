# SAPR-RAG Idea 演化历史

> **写给未来的我和未来的 AI**：这份文档总结 2026-05-28 → 2026-05-30 期间，SAPR-RAG idea 经历的 4 次重写。
> 当前活跃版本是 v4（见 [docs/proposal.md](./proposal.md)），本文只记录前三版"做过什么、结果如何、为什么淘汰"，避免后来者翻 20 多份带时间戳的快照重新踩坑。

---

## 时间线总览

```
v1 SAPR-RAG (05-28)
   ↓ Reviewer Round 1 反馈：novelty 不足、prompt-judge 易被 dismiss
v2 ClosureRAG (05-30)
   ↓ 自评 4/10：claim 无 provenance、reward 全局比例不可学
v3 TrajectoryClosure (05-30)
   ↓ 仍未回答"标量 PRM 真的盲吗"这一根本问题
v4 当前版（见 docs/proposal.md）
```

每次迭代都没有跑新的实验数据来对比，只是 idea / 命名 / 状态 schema 的重构。这本身是经验教训之一（见末节）。

---

## v1 SAPR-RAG（State-Aware Process Rewards，2026-05-28）

**已弃用**。原文见 git 历史中的 `refine-logs/FINAL_PROPOSAL.md`。

### 做了什么
- 把 ReasonRAG 重新定位为 **state-conditioned progress / action-value 建模**。
- 模块化为 3 个 critic 头：
  - **query selection**：从候选 subquery 中选最有信息增量的
  - **evidence selection**：state-aware utility `score(question, history, current step, doc)`
  - **stop decision**：避免过早停 / 过度搜
- Baseline 固定为 ReasonRAG-LoRA，强调 **compute-matched**（不增加 retrieval calls / candidates / tokens）。
- 训练用 prompt-judge 蒸馏出小型 critic（V0 → distilled）。

### 跑了什么
- **E2-30**：HotpotQA dev 30 条，6-way evidence-only 对比
  - SAPR-E v0：hit@3 = 0.5398
  - retriever baseline：hit@3 = 0.4690
  - 增益 +7.1pp，item_hit 63.3% vs 56.7%
- **E3-30**：evidence ablation
  - no_title 影响最大（−8.0pp hit@3）
  - no_hist 和 no_subquery 各 −2.7pp

### 结论与淘汰原因
- **方向有信号**，但 Reviewer Round 1 给出明确修改要求：
  - 容易被 dismiss 为 "prompt-judge reranker + rule-based repair wrapper"
  - novelty 相对 ReasonRAG / DecEx-RAG / HiPRAG / ProRAG 不够清晰
  - prompt judge 自身是 costly / unstable / answer-leaky，不能作为最终 contribution
- 必须升级为更结构化、更可学习的状态建模。

---

## v2 ClosureRAG（Evidence Closure Board，2026-05-30）

**已弃用**。原文见 git 历史中的 `refine-logs/FINAL_PROPOSAL_v2.md`。

### 做了什么
- 用 **显式 Evidence Closure Board** 替代 v1 的隐式 `s_t`：
  ```json
  {
    "entities": ["..."],
    "open_gaps": ["..."],
    "supported_claims": ["..."]
  }
  ```
- 用 **结构化 closure reward** 替代标量 step PRM，分三维：slot closure / claim support / entity chain。
- 大幅收缩 v1 的设计：砍掉 rollback、砍掉全 LLM judge evidence scoring、Evidence selection 降级为 heuristic。

### 跑了什么
- **没有跑新实验**，只是 idea / schema 重写。

### 结论与淘汰原因
- **自评 4/10**（按顶会审稿人视角）：
  - `supported_claims` 是文本列表，**没有 provenance**，无法回到具体 evidence 句，不可审计也难训练
  - closure reward 是"全局比例"（如"已 close 的 gap / 总 gap"），但**分母不可知**——总 gap 数本身需要 oracle
  - 与 S2G-RAG（gap-checking）的区别表述模糊，被认为是其变体
- 引出 v3 的关键修正：claim 必须带 evidence_ref；reward 必须是 per-step transition 而非全局比例。

---

## v3 TrajectoryClosure（Provenance-Aware Transition，2026-05-30）

**已弃用**。原文见 git 历史中的 `refine-logs/FINAL_PROPOSAL_v3.md`。

### 做了什么
- 在 v2 基础上给 claim 加 **provenance**：
  ```json
  {
    "entities": ["..."],
    "open_gaps": ["..."],
    "claims": [
      {
        "text": "...",
        "evidence_ref": "doc_3::sent_5",
        "status": "supported"
      }
    ]
  }
  ```
- reward 改为 **per-step transition reward**（不依赖未知分母）。
- 定位收敛到："**MCTS 节点评估的结构化升级**"，不是线性 pipeline 的 sufficiency check。

### 跑了什么
- **没有跑新实验**，只是 idea 重写。

### 结论与淘汰原因
- 方向对了，但仍**未回答最根本的问题**：
  > ReasonRAG 的标量 PRM 真的"对分支选择是盲的"吗？兄弟节点 Q 值分布到底有多接近？如果其实分布很分散，typed eval 的收益空间就小。
- 缺一个数据驱动的 **Gate 0 验证**来支撑全部上层方法，而不是基于直觉持续重写命名。
- 这成了 v4 的核心动作：先做 Gate 0 实验 A（GPT-4o 重打 Q 值无偏验证），再决定 Go / Pivot / Stop。

---

## v4 当前版（FailureAttributedMCTS，进行中）

详见 [docs/proposal.md](./proposal.md)。核心改动：

- 在 v3 基础上聚焦 **failure-attributed branch selection in MCTS**
- typed transition evaluation：**φ_q（query 是否对准 gap）/ φ_c（claim 是否被支撑）/ φ_s（stop 是否合理）**
- directed expansion：query_fail → rewrite，claim_fail → supplement，stop_fail → force continuation
- Typed-UCB / Bottleneck-UCB
- 当前阶段：等 Gate 0 实验 A 结果，决定 Go / Pivot / Stop

---

## 关键经验教训（写给以后的我和 AI）

1. **不要在没有 Gate 0 数据支持前迭代 idea 命名**。前 3 版主要是 AI 主导的语言重构，没有产生新的实证差异；产生的 ~20 份带时间戳的 EXPERIMENT_PLAN / TRACKER / FINAL_PROPOSAL 反向掩埋了真正的 idea。
2. **数据来源的诚实性比方法学的精致更重要**。本仓库 `reward_data` 是用 Llama-70B-int4 复现的，**不是论文配置的 GPT-4o**。所有基于 reward_data 的统计（如"分支 Q 值高度同质化"）都必须先用 `gate0/relabel_q_with_gpt4o.py` 做 GPT-4o 无偏对照，再下结论。
3. **命名规范**：
   - 不要用 `_v1 / _v2 / _v3` 文件后缀。最新版直接覆盖；演化记录写到 `docs/history.md`。
   - 不要给同一份文档加时间戳副本。git 已经记录历史，时间戳副本只制造噪声。
   - 文件名小写，单词间用 `_`。
4. **Reviewer 视角自评要尽早**。v2 自评 4/10 暴露了 provenance / 分母不可知问题；类似的自评应该在 v1 投入实验前就做一次。
5. **保留可执行的最新版 + 一份阶段性总结，删掉所有中间快照**。这就是本次清理（cleanup.sh）做的事情。

---

## 索引：原始历史快照在 git 中的对应文件

如需考古，下列文件可在 cleanup.sh 执行前的 git 历史中找到：

| 阶段 | 文件 |
|---|---|
| v1 | `refine-logs/FINAL_PROPOSAL.md`, `refine-logs/REVIEW_SUMMARY.md`, `refine-logs/REFINEMENT_REPORT.md` |
| v2 | `refine-logs/FINAL_PROPOSAL_v2.md` |
| v3 | `refine-logs/FINAL_PROPOSAL_v3.md` |
| v1→v3 中间快照 | `refine-logs/*_20260528_200150.md`, `refine-logs/*_20260528_200527.md` |

均已通过 `git rm` 删除并随 cleanup.sh 提交，可在 git log 中追溯。
