# Experiment Plan v2 — ClosureRAG

**Date**: 2026-05-30
**Backbone**: ReasonRAG-LoRA (Qwen2.5-7B) on vLLM
**Retriever**: BGE-base + FAISS
**Datasets**: HotpotQA / 2Wiki / MuSiQue / Bamboogle
**Compute**: 3×RTX 5090 (rag-5090) + 4×RTX 3090 (local)
**Supersedes**: `refine-logs/EXPERIMENT_PLAN.md` (v1)

---

## 1) Same-Budget 规则（所有实验必须遵守）

所有方法固定以下参数（基于 ReasonRAG 原始配置 my_config.yaml）：

```yaml
max_steps: 3          # 最大检索步数
retrieval_topk: 5     # 每步检索候选数（ReasonRAG 默认 5）
inject_top: 5         # 注入 prompt 的文档数（ReasonRAG 默认全注入）
max_tokens: 256       # ReasonRAG 原始配置（不用 32!）
use_reranker: False   # 与 ReasonRAG 相同
```

**如果 ClosureRAG 需要更多步数才能 closure**，单独报告 same-budget 和 adaptive-budget 两组实验。

---

## 2) Gate 0: Board + Stop Closure 最小验证

**目的**: 用最快速度判断 Board + Stop Closure 是否有方向性信号。

**配置**:
- 数据: HotpotQA dev 50 条（随机采样，含 2-hop 和 3-hop）
- 模式: ReasonRAG tree mode（已验证比 batch mode 可靠）
- Board: Prompt 版（LLM judge 生成）
- Stop: Deterministic closure rule
- Evidence selection: 不改，用原始 top-3
- Claim Gate: 不加（Gate 1 才加）

**对比**:
| 系统 | Board | Stop | Evidence |
|------|-------|------|----------|
| ReasonRAG (baseline) | flat history | flag-based | top-3 by rank |
| ClosureRAG-Gate0 | Closure Board | closure rule | top-3 by rank |

**指标**:
- EM / F1
- bridge entity recall（关键实体是否出现在 trajectory 中）
- premature_stop rate（supporting facts 未全覆盖就停的比例）
- avg steps
- latency

**Go/No-Go 判断**:
- ✅ Go: EM 或 F1 有正向趋势 + premature_stop rate 下降 ≥ 5pp
- ⚠️ Weak: 过程指标改善但 EM/F1 不变 → 继续加模块
- ❌ No-Go: 过程指标也不改善 → 方向性错误，止损

**时间**: 2-3 天
**算力**: 1×5090

---

## 3) Gate 1: + Claim Gate

**目的**: 验证 transition-time claim-support gate 是否减少 unsupported claims。

**配置**:
- 数据: Gate 0 的 50 条 + 30 条已知 badcase（unsupported_claim / premature_stop 为主）
- Board: Prompt 版
- Claim Gate: LLM judge 标注 supported/unsupported/insufficient
- Repair: unsupported → 继续检索; insufficient → 重写 query

**对比**:
| 系统 | Board | Claim Gate | Stop |
|------|-------|-----------|------|
| ReasonRAG | flat | 无 | flag |
| +Gate0 | Board | 无 | closure |
| +Gate1 | Board | **transition gate** | closure |

**指标**:
- unsupported_claim rate（LLM judge 标注）
- repair success rate（repair 后 trajectory 是否变好）
- EM / F1
- avg steps（Gate 是否导致 over_search）

**Go/No-Go**:
- ✅ unsupported_claim rate 下降 ≥ 10pp
- ❌ 无改善或误杀率过高

**时间**: 2-3 天
**算力**: 1×5090

---

## 4) Gate 2: 全部 Prompt 版 + 200 条

**目的**: 在足够大的样本上验证完整 prompt 版系统的信号。

**配置**:
- 数据: HotpotQA dev 200 条
- 全部模块: Board + Claim Gate + Stop Closure + 轻量 Evidence Heuristic
- 对比:

| 系统 | Board | Claim | Stop | Evidence |
|------|-------|-------|------|----------|
| ReasonRAG | flat | 无 | flag | top-3 |
| +MoreSteps | flat | 无 | flag | top-3, max_steps=5 |
| PostHoc | flat | **post-hoc** verify | flag | top-3 |
| ClosureRAG-prompt | Board | **transition** gate | closure | heuristic |

**指标**:
- EM / F1
- 全部过程指标（unsupported claim rate, premature stop rate, bridge entity recall, evidence chain completeness, query repetition, avg steps, latency）
- failure transition matrix（每步错误归因到 query/evidence/claim/stop）

**关键判断**:
- ClosureRAG vs ReasonRAG: 是否提升？
- ClosureRAG vs +MoreSteps: 提升是否来自"搜得更多"？如果不是 → 核心贡献成立
- ClosureRAG vs PostHoc: transition gate 是否比 post-hoc 好？如果是 → 核心差异化成立

**数据积累**: 同步收集 Board 标注、claim 标注、closure 标签，用于 Phase 2 训练。

**时间**: 1 周
**算力**: 2×5090

---

## 5) Gate 3: SFT 训练版

**前提**: Gate 2 信号正向。

**训练内容**:

### 5a. Board Updater (LoRA SFT)

- 训练数据: Gate 2 积累的 200 × ~3步 = ~600 条 Board 标注
- 目标: 输入 (thought, evidence, current_board) → 输出 updated_board
- 模型: Qwen2.5-7B + LoRA
- 预期: Board 更新准确率 > LLM prompt 版

### 5b. Claim Verifier

- 训练数据: Gate 2 积累的 ~600 条 claim-evidence 对
- 目标: claim + evidence → supported/unsupported/insufficient
- 模型: DeBERTa-large 或 Qwen2.5-7B + LoRA
- 预期: F1 > LLM judge prompt 版

### 5c. Closure Reward Model (LoRA SFT)

- 训练数据: Gate 2 积累的 ~600 条 Board → (R_slot, R_claim, R_chain) 标签
- 目标: Board state → 三维 closure score
- 模型: Qwen2.5-7B + LoRA

**评估**:
- Trained 版 vs Prompt 版: 各模块准确率
- 端到端: Trained ClosureRAG vs Prompt ClosureRAG vs ReasonRAG
- 数据: 500 条 HotpotQA + 500 条 2Wiki

**时间**: 2 周
**算力**: 3×5090

---

## 6) Gate 4: 全量实验

**前提**: Gate 3 完成。

### 6a. 主实验

| 数据集 | 样本量 | 指标 |
|--------|--------|------|
| HotpotQA | full dev | EM, F1, 过程指标 |
| 2WikiMultiHopQA | full dev | EM, F1, 过程指标 |
| MuSiQue | full dev | EM, F1, 过程指标 |
| Bamboogle | full test | EM, F1, 过程指标 |

### 6b. Baseline 对比

| 方法 | 来源 | 训练方式 |
|------|------|---------|
| ReasonRAG | NeurIPS 2025 | Process-supervised RL |
| ReasonRAG + more steps | 我们的变体 | 同上, max_steps=5 |
| Vanilla iterative RAG | 自己实现 | 无 |
| Post-hoc claim verification | RAGChecker-style | NLI model |
| S2G-RAG-like sufficiency controller | 复现或 reimplementation | Rule-based |
| ClosureRAG-prompt | 我们 | 无训练 |
| ClosureRAG-trained | 我们 | SFT |

### 6c. 消融实验

| 实验 | 验证 |
|------|------|
| Full vs -Board (flat history) | Board 的贡献 |
| Full vs -ClaimGate | Claim Gate 的贡献 |
| Full vs -StopClosure | Stop Closure 的贡献 |
| Full vs -EvidenceHeuristic | Evidence heuristic 的贡献 |
| Prompt vs Trained (各模块) | 训练的价值 |
| Oracle Board | Board 完美时的系统上限 |
| Oracle Claim | Claim verification 完美时的上限 |
| Oracle Stop | Stop closure 完美时的上限 |
| Oracle All | 全部完美时的上限 |

### 6d. Error Attribution Analysis

- 对每条 trajectory 标注每步错误类型: query_error / evidence_error / claim_error / stop_error / no_error
- 对比 ClosureRAG vs ReasonRAG 的错误分布
- 证明 ClosureRAG 的三维 reward 比 step PRM 有更好的归因能力

### 6e. Same-Budget vs Adaptive-Budget

- Same-budget: 固定 max_steps=3, top_k=10, inject_top=3
- Adaptive-budget: 允许 ClosureRAG 根据 closure 状态调整步数（但报告总 cost）

**时间**: 2 周
**算力**: 3×5090

---

## 7) Run Order Summary

```
Week 1 (06/01-06/07): Gate 0 + Gate 1
  → 最小验证，决定是否继续

Week 2 (06/08-06/14): Gate 2
  → 200 条验证 + 数据积累
  → 如果信号弱，在此止损

Week 3-4 (06/15-06/28): Gate 3
  → SFT 训练 + 500 条评估

Week 5-6 (06/29-07/12): Gate 4
  → 全量实验 + 消融 + baseline

Week 7-8 (07/13-07/27): 写作 + 打磨
  → Related Work, Method, Experiments, Analysis

07/28: AAAI 2027 submission
```

---

## 8) 成功标准

**Minimum viable paper**:
- ClosureRAG 在 HotpotQA + 2Wiki 上 EM 或 F1 显著优于 ReasonRAG (same budget)
- 过程指标（unsupported claim rate, premature stop rate）显著改善
- 消融证明 Board + Claim Gate + Stop Closure 各自贡献
- Error attribution 对比证明三维 reward 比标量 step PRM 归因更准确

**Stretch goal**:
- MuSiQue + Bamboogle 上也有效
- Oracle 分析揭示瓶颈在 Board 更新准确率（指导未来工作）
- Process RL (closure-guided GRPO) 进一步提升
