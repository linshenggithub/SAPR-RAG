# Phase 4: External Critical Review

**日期**: 2026-05-30
**审稿人**: Codex (gpt-5.5) 扮演 NeurIPS/ICML/AAAI 级资深审稿人
**审稿对象**: State-Aware Evidence Closure for Process-Reward Guided Agentic RAG

---

## Overall Score: 4 / 10

Weak Reject 到 Borderline Reject。若实验证据很强、方法定义收缩得很干净，最多能冲到 6。

---

## Summary

本文试图将 Agentic RAG 的 flat trajectory 改为结构化 Evidence Closure Board，并在 query、evidence、claim、stop 四个节点加入状态感知控制与修复。核心目标是减少多跳问答中的 query drift、证据链断裂、unsupported claims 和 premature stopping。想法合理，但当前方法过于模块堆叠，缺少一个足够 sharp 的 algorithmic contribution，也没有证明这些复杂状态能被稳定维护。

---

## Strengths

1. **问题是真问题**。多跳 RAG 中 flat history、证据链不可解释、提前停止、intermediate hallucination 都是 ReasonRAG 类系统的真实失败模式。
2. **状态感知 evidence utility** 比普通 score(q, d) 更贴近 Agentic RAG 的过程需求。
3. **Stop-by-Closure 是最有潜力的部分**。相比让模型输出 stop flag，用 evidence-backed slots/claims/conflicts 控制停止更容易形成可解释实验和 error attribution。
4. **Claim-Support Gate 能直接攻击 process hallucination**。
5. 方向适合作为 ReasonRAG 的 failure diagnosis + process control extension。

---

## Weaknesses（关键！）

### 4a. 方法设计弱点

1. **Board 更新是最大单点故障**。Board 由 7B 模型或 LLM prompt 更新，很可能出错：错误实体被 confirmed、open_slot 被错误关闭、missing_link 被漏检、conflict 被幻觉生成。一旦 board 错，后续所有模块跟着错。把 flat history 的噪声变成了 structured state 的噪声，但没有证明后者更可控。

2. **Board schema 太大，缺少最小充分状态定义**。6 类字段，审稿人会问：哪些字段真正必要？为什么不是只用 open_gaps + supported_claims？没有理论或系统性消融，这像 prompt schema engineering。

3. **Claim extraction 可靠性被低估**。7B 模型对复杂多跳 QA 的 intermediate thought 做 atomic claim decomposition 很容易出错。

4. **Claim-Support Gate 本质上可能只是 noisy NLI**。没有说明 evidence granularity，没有说明 contradiction vs insufficiency 的判别标准。复杂 QA 里很多 evidence 是组合支持。

5. **U(d|board) 的 marginal utility 定义不够可计算**。slot_gain、bridge_gain 等没有明确 scoring function。LLM judge 对 top-10 逐个打分成本和 latency 很高。

6. **"Marginal utility" 没有真正体现 marginal**。要证明 U(d|board) 是边际效用，至少要建模相对已有 evidence 的新增贡献。

7. **四个模块强耦合，错误归因困难**。Board 影响 evidence utility，evidence utility 影响 claims，claims 更新 board，board 影响 stop。最终提升或下降时，很难说是哪一部分导致。

8. **Repair mechanism 太泛**。什么时候 rewrite？什么时候 retrieve？什么时候 rollback？没有具体 policy。

9. **"Process-reward guided" 名不副实**。如果不训练 PRM、不做 RL，只用 LLM judge / heuristic 控制，那不该叫 process-reward guided。应该叫 closure-guided process control。

### 4b. Novelty 质疑

1. **和 S2G-RAG 的区别目前不够硬**。更复杂不等于更 novel。
2. **整体像拼装系统**：structured state + evidence reranking + NLI verification + sufficiency stopping。
3. **Board vs gap items 本质区别需要可操作定义**。
4. **Claim-Support Gate vs post-hoc verification 区别不清**。
5. **Stop-by-Closure vs sufficiency check 边界不清**。

### 4c. 实验设计质疑

1. **7B 模型可能撑不起这个复杂 pipeline**。模块越多，error surface 越大。
2. **不做 RL 却想赢 RL baseline，论证压力很大**。
3. **消融组合负担很重**（4 模块 15 组合）。
4. **Baseline 不足会直接导致拒稿**。至少需要 ReasonRAG, vanilla iterative RAG, S2G-RAG-like, claim-verification baseline, reranker baseline。
5. **指标不能只看 EM/F1**。必须报告过程指标。
6. **Badcase-only 实验有 selection bias**。
7. **需要 same-budget 实验**，否则 reviewer 质疑 unfair compute。

### 4d. 可行性风险

1. **1 周内做 4 个最小验证实验过于乐观**。
2. **推理成本可能爆炸**。每步 top-10 scoring + claim extraction + support gate + stop closure。
3. **AAAI 2027 时间线很危险**。2 个月要完成方法、代码、四数据集实验、消融、写作。

### 4e. 写作和定位风险

1. **核心贡献不清**：method、diagnosis 还是 system？
2. **Prompt engineering 风险**。
3. **标题中 "Reward" 会引来和训练型 PRM/RL 方法的直接比较**。

---

## Questions for Authors（审稿人提问）

1. Board 的状态转移函数是什么？规则、LLM prompt、训练模型？
2. Board 错误如何检测和恢复？
3. 如何定义 "closure"？形式化判定条件？
4. Atomic claims 抽取准确率如何评估？
5. Claim-support 判断的 evidence granularity？
6. U(d|board) 的 scoring function 具体是什么？
7. Repair policy 如何选择 rewrite / retrieve / rollback / stop？
8. 和 S2G-RAG 在相同 backbone、相同 retriever、相同 max turns 下是否对比？
9. 如果 full system 提升，如何证明不是来自更长检索步数或更多 token budget？
10. LLM judge 是否参与了 test-time decision？evaluation leakage？

---

## Minimum Viable Improvements（审稿人建议）

### 必须做

1. **把核心贡献收缩为一个：Stop-by-Closure + minimal Board**。不要四模块同等贡献。
2. **定义最小 Board schema**：entities, open_slots, supported_claims, missing_links, conflicts。砍掉 next_hop_requirements。
3. **给出 closure 的可执行定义**：
   - 每个 required slot 至少有一个 supported claim
   - final answer entity 被 evidence-backed reasoning path 支持
   - no active contradiction involving answer-critical claims
4. **做过程指标**：unsupported claim rate, premature stop rate, query repetition, evidence sufficiency, latency。
5. **做强 baseline**：ReasonRAG, ReasonRAG + more steps, S2G-like controller。
6. **做局部人工标注**：100-200 条 trajectory step annotations。

### 应该砍掉

1. 砍掉完整四模块同等贡献叙事
2. 暂时砍掉 branch rollback
3. 砍掉 "process-reward guided" 的强说法，改成 "closure-guided process control"
4. 砍掉 top-10 全 LLM marginal scoring 的默认方案

### 应该强调

1. "Closure as an interpretable stopping criterion"
2. "Transition-time verification prevents error propagation"
3. "Failure attribution"——能归因到具体错误类型
4. "Cost-controlled test-time controller"——无需训练、可插拔

---

## Constructive Suggestions

1. **重命名**：ECO-RAG (Evidence Closure-guided Process Control) 或 ClosureRAG
2. **给出明确 algorithm box**
3. **Board 更新拆成 "extract" + "validate"**，不要让 LLM 一步生成完整 board
4. **Evidence Utility 先用轻量版本**：embedding similarity + novelty + entity_gain，不要全 LLM judge
5. **Claim Gate 只做 answer-critical claims**，不要对每个 thought 的所有原子 claim 都 gate
6. **Stop Closure 做 deterministic + judge fallback**
7. **做 same-budget 实验**
8. **增加 failure transition matrix**
9. **做 oracle diagnostic**（oracle board, oracle claim support, oracle stop, oracle evidence）
10. **明确 S2G-RAG 对比点**：我们维护 claim-evidence graph + transition-time 阻断 unsupported claim + closure graph 控制 stop

---

## 最终判断

> 研究方向值得继续，但当前版本会被 top-venue reviewer 认为"概念正确、系统臃肿、novelty 不硬、实验风险高"。两个月内最务实的路线是砍成一篇更 sharp 的论文：
>
> **Evidence Closure Board + Stop/Claim Gate for preventing unsupported and premature reasoning in Agentic RAG**
>
> 先证明结构化 closure 能显著减少 premature stop 和 unsupported claims，再谈 evidence marginal utility 和 query repair。
