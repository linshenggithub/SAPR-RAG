# Phase 4 Round 3: Objective Re-review

**日期**: 2026-05-30
**审稿人**: Codex (gpt-5.5)
**Prompt 类型**: 完全客观版（无上次分数、无回应标记、主动暴露弱点）
**结果**: **4/10**

---

## Overall Score: 4/10

去掉引导性表述后，审稿人给了 **4/10**——和第一轮相同。

核心判断：**ClosureRAG 很容易被审稿人认为是 S2G-RAG 的 ReasonRAG 适配版，而不是独立新方法。**

---

## 审稿人的核心观点

### 和 S2G-RAG 的对应关系几乎一一映射

| ClosureRAG | S2G-RAG |
|-----------|---------|
| Evidence Closure Board | Evidence Memory / Evidence Context |
| open_gaps | Gap Items |
| Closure check | Sufficiency judging |
| open_gap → rewrite query | Gap-to-query mapping |
| supported_claims | sentence-level evidence support |

### Board 3 字段不够

适合做诊断日志，但不足以做可靠控制器。缺少：
- evidence provenance（claim→evidence 句子级映射）
- relation/path state（推理链走到哪一跳）
- conflict/contradiction 状态
- query history 和 gap-query 对齐

### Closure Reward 定义有严重问题

- 分母不可知
- claim 粒度可操纵（多生成容易支持的 claim 就高分）
- 不区分关键 claim 和无关 claim
- 不惩罚错误/冗余 evidence

### "只改两个点"不够作为论文贡献

审稿人认为："如果 generator、retriever、thought 都不动，性能提升到底来自一个 heuristic stop controller，还是来自额外 LLM judge 的外部算力？"

---

## 审稿人建议的升级方向

> **Provenance-aware trajectory closure modeling for Agentic RAG process control**
>
> 不只是记录 gap，而是显式建模 query、evidence、claim、gap、stop decision 之间的**状态转移和可验证支撑关系**。

具体来说：
1. 每个 claim 必须绑定 evidence sentence/span（provenance）
2. Closure Reward 从全局比例改为 action-level process reward
3. 定义可执行的 repair 算法
4. 覆盖 query-evidence-claim-stop 的状态演化，不只是 gap 列表
