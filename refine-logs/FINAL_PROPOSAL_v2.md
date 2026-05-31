# FINAL PROPOSAL v2 — Evidence Closure-guided Process Control for Agentic RAG

**Working title**: *ClosureRAG: Structured Evidence Closure Rewards for Process-Controlled Agentic RAG*
**Date**: 2026-05-30
**Status**: Phase 4.5 draft, pending user review
**Supersedes**: `refine-logs/FINAL_PROPOSAL.md` (v1 SAPR-RAG)

---

## 0) Problem Anchor

**问题**: 多跳复杂问答中的 Agentic RAG 系统（如 ReasonRAG）在多步检索推理过程中存在严重的轨迹控制问题。ReasonRAG 的 step PRM 给出标量分数（0-1），无法归因错误来自 query、evidence 还是 stop。这导致 8 类 badcase（query drift、entity loss、retrieval noise、unsupported claim、premature stop 等）无法被精准修复。

**根因**: 传统 process reward 是标量信号——"这一步好不好"——但不知道"哪里不好、为什么不好、怎么修"。

**我们的核心主张**: 用 **结构化 Evidence Closure 状态** 替代标量 step score，从中衍生出可归因的 closure reward，用于训练和推理。

**一句话贡献**:
> 我们提出 Evidence Closure Board 作为 Agentic RAG 的结构化过程状态，从中衍生出 slot closure / claim support / entity chain 三维 closure reward，替代传统标量 step PRM，实现可归因的过程控制和训练。

---

## 1) 与 v1 (SAPR-RAG) 的关键变化

| 维度 | v1 SAPR-RAG | v2 ClosureRAG |
|------|------------|---------------|
| 状态表示 | 隐式 `s_t`（未具体定义 schema） | **显式 Evidence Closure Board**（3 字段） |
| Reward 形式 | Q(s_t, a_t) 标量 | **结构化 closure reward**（slot/claim/entity 三维） |
| 训练信号 | pairwise preference over actions | **Board state → closure labels** |
| 与 S2G-RAG 区别 | 不明确 | **transition-time claim gate + closure graph stop + 嵌入 MCTS** |
| 模块数量 | 3 个（Q/E/S reward heads） | 3 个（Board + Claim Gate + Stop Closure），Evidence 选择降级为 heuristic |
| 审稿人反馈 | 未处理 | **大幅收缩**，砍掉 rollback、砍掉全 LLM judge evidence scoring |

---

## 2) Method: Evidence Closure Board + Closure Reward

### 2.1 Evidence Closure Board 定义

Board 是一个结构化状态，在 ReasonRAG 每步推理后更新：

```json
{
  "entities": ["Shirley Temple", "Chief of Protocol", "1976"],
  "open_gaps": ["1976年的美国总统"],
  "supported_claims": [
    "Shirley Temple 1976年被任命为Chief of Protocol"
  ]
}
```

**3 个字段，不更多**：
- `entities`: 已确认实体（文本列表）
- `open_gaps`: 未解决的信息缺口（文本列表，合并了原 open_slots 和 missing_links）
- `supported_claims`: 有证据句支撑的中间结论（文本列表）

**为什么只有 3 个**：
- `open_slots`（信息需求）和 `missing_links`（证据链断裂）说的事高度重叠，合并为 `open_gaps`
- `conflicts` 可从 supported_claims 之间的矛盾推断
- `next_hop_requirements` 可从 open_gaps 自动派生
- `support_edges` 不需要单独标注，claim 文本本身隐含来源
- 没有嵌套结构 → 7B 模型更新时出错概率最低
- bridge entity 追踪由 `entities` 字段覆盖：如果 gap 描述里提到了 entities 中的实体，说明 bridge 没丢

**示例推理过程**（问题: "Shirley Temple 被任命为 Chief of Protocol 时，当时的美国总统是谁？"）：

```
Step 0 (初始化):
  entities: [Shirley Temple, Chief of Protocol]
  open_gaps: [任命时间, 那个时间的美国总统]
  supported_claims: []

Step 1 (检索 "Shirley Temple Chief of Protocol"，找到 1976 年任命):
  entities: [Shirley Temple, Chief of Protocol, 1976]
  open_gaps: [1976年的美国总统]
  supported_claims: [Shirley Temple 1976年被任命为Chief of Protocol]

Step 2 (检索 "1976 US President"，找到 Gerald Ford):
  entities: [Shirley Temple, Chief of Protocol, 1976, Gerald Ford]
  open_gaps: []
  supported_claims: [..., 1976年美国总统是Gerald Ford]

  → open_gaps 为空 → closure = true → 停止，答案 Gerald Ford
```

### 2.2 Board 更新机制

**两阶段更新**（审稿人建议）：

```
Step 1: Extract（抽取）
  输入: 当前 thought + 新检索文档
  输出: candidate entities, candidate claims, candidate slots
  实现: Prompt (验证期) → LoRA SFT (正式期)

Step 2: Validate（验证）
  输入: candidate claims + 当前 evidence
  输出: supported / unsupported / insufficient
  实现: NLI model 或 LLM judge (验证期) → Trained classifier (正式期)

Step 3: Merge（确定性合并）
  规则: supported claims → 加入 board; unsupported → 触发 repair
  实现: Deterministic Python 代码，不依赖 LLM
```

**为什么是 Extract-Validate-Merge 而不是一步生成**：
- 避免 LLM 一步生成完整 Board 导致 hallucination
- Extract 和 Validate 可以分别评估和训练
- Merge 是确定性的，不会引入额外错误

### 2.3 Closure Reward 定义

从 Board 状态衍生两个维度的 reward：

```
R_closure(s_t) = R_gap + R_claim

R_gap: open_gaps 的闭合比例
  = 1 - |open_gaps| / |total_required_gaps|
  衡量：信息缺口是否全部解决（合并了原 slot closure + chain completeness）

R_claim: supported_claims 的比例
  = |supported_claims| / |all_extracted_claims|
  衡量：中间结论是否有证据支撑
```

**与标量 step PRM 的本质区别**：

| 维度 | Step PRM (ReasonRAG) | Closure Reward (我们) |
|------|---------------------|---------------------|
| 信号形式 | 标量 0-1 | 二维结构化 (gap, claim) |
| 可归因性 | "这一步 0.3 分" → 不知道为什么 | "gap 闭合 60%, claim 支撑 40%" → 知道缺什么 |
| 训练目标 | 学会"整体做好" | 学会"闭合每个 gap, 支撑每个 claim" |
| Repair 指导 | 低分 → 不知道修什么 | gap 未闭合 → 补检索; claim unsupported → 补证据 |

### 2.4 Inference-time Process Control

```
输入: question, retriever (BGE), generator (Qwen2.5-7B + LoRA)

初始化:
  Board = {
    entities: 从问题抽取的实体,
    open_slots: 从问题分解的信息需求,
    supported_claims: [],
    missing_links: []
  }

循环 (max_steps = N):
  1. Query Generation
     - 从 Board.open_slots 生成子查询
     - 检查: 是否对准某个 open_slot? 是否保留 bridge entity?
     - 不满足则重写

  2. Retrieval
     - 用子查询检索 top-10

  3. Evidence Selection (轻量 heuristic)
     - slot_gain: 和 open_slot 的 embedding 相似度
     - novelty: 和已有 evidence 的最大相似度取负
     - entity_gain: 新增实体是否连接已有 entities
     - noise_risk: 候选文档实体和问题实体的冲突程度
     - 选 top-3 注入 prompt

  4. Thought Generation
     - ReasonRAG generator 生成 intermediate thought

  5. Board Update (Extract → Validate → Merge)
     - Extract: 从 thought + evidence 抽取 candidate claims/entities
     - Validate: 检查 claims 是否被 evidence 支持
     - Merge: 确定性规则更新 Board

  6. Closure Check (deterministic + judge fallback)
     - 规则: 每个 open_slot 有 supported_claim?
            final answer entity 被 chain 连接?
            无 missing_link?
     - 全部满足 → 停止, 生成答案
     - 未满足 → 继续
     - 边界情况 → LLM judge fallback

  7. Repair (if needed)
     - unsupported claim → 补检索
     - open_slot 未闭合 → 重写 query

输出: final answer + Board (可审计)
```

---

## 3) Training Plan

### 3.1 Phase 1: 数据积累 (Prompt/Rule 原型)

**目的**: 验证 Board + Claim Gate + Stop Closure 是否有信号；积累训练数据。

**方法**:
1. 在 ReasonRAG tree mode 上跑 500 条 HotpotQA trajectories
2. 用 LLM judge 在每步生成 Board 标注（entities, open_slots, claims, missing_links）
3. 用 LLM judge 标注每步 claim support status
4. 用 HotpotQA supporting facts 做自动 closure 评估
5. 人工校验 100-200 条 Board 标注质量

**产出**:
- 500 条 trajectory × 每步 Board annotations
- 500 条 closure labels（slot/claim/chain 三维）
- 100-200 条人工校验数据

**预期时间**: 1 周
**算力**: 3×5090 (ReasonRAG 推理 + LLM judge)

### 3.2 Phase 2: 模块训练 (SFT)

**3.2a Board Updater (LoRA SFT)**

- 输入: thought + evidence + 当前 Board
- 输出: 更新后的 Board (JSON)
- 数据: Phase 1 积累的 Board 标注
- 训练: Qwen2.5-7B + LoRA, supervised fine-tuning
- 算力: 3×5090, 几小时

**3.2b Claim Verifier (Classifier)**

- 输入: claim + evidence snippet
- 输出: supported / unsupported / insufficient
- 数据: Phase 1 积累的 claim-evidence 标注
- 训练: 小模型（DeBERTa-large 或 LoRA on 7B）
- 算力: 很小

**3.2c Closure Reward Model (LoRA SFT)**

- 输入: Board state
- 输出: R_slot, R_claim, R_chain 三维分数
- 数据: Phase 1 积累的 closure labels
- 训练: Qwen2.5-7B + LoRA
- 算力: 几小时

**预期时间**: 1-2 周
**算力**: 3×5090

### 3.3 Phase 3: Process RL (可选，依赖信号强度)

如果 Phase 2 的 trained 版本比 prompt 版有显著提升，可以进一步做：

- **训练信号**: Closure Reward (R_slot + R_claim + R_chain)
- **方法**: 类似 ReasonRAG 的 GRPO，但用 closure reward 替代 step PRM
- **目标**: 让模型学会主动生成能闭合 Board 的 thought/query
- **算力**: 如果需要，可以申请更多 GPU

---

## 4) Experiment Plan

### 4.1 验证优先级（决策门控）

```
Gate 0 (3天): Prompt 版 Board + Stop Closure
  → 50 条 HotpotQA, 对比原 ReasonRAG
  → 看: bridge entity recall, premature_stop rate, EM/F1
  → 决策: 信号是否正向？负向则方向性错误，止损

Gate 1 (3天): + Claim Gate
  → 50 条 badcase, 看 unsupported_claim rate
  → 决策: Claim Gate 是否减少 unsupported claims？

Gate 2 (1周): + 全部 prompt 版 + 200 条
  → 200 条 HotpotQA, 完整 prompt 版 vs ReasonRAG
  → 看: EM/F1, 过程指标, latency
  → 决策: 整体信号是否足够强，值得进 Phase 2 训练？

Gate 3 (2周): Trained 版
  → Phase 2 SFT 训练各模块
  → 500 条 HotpotQA + 500 条 2Wiki
  → 看: trained vs prompt vs ReasonRAG

Gate 4 (2周): 全量实验
  → 四数据集 (HotpotQA/2Wiki/MuSiQue/Bamboogle)
  → 多 baseline, 消融, 过程指标
```

### 4.2 必须的 Baseline

| Baseline | 为什么必须 |
|---------|----------|
| ReasonRAG (原始) | 核心 baseline |
| ReasonRAG + more steps | 证明提升不是"搜得更多" |
| Post-hoc claim verification | 证明 transition gate 比 post-hoc 好 |
| S2G-RAG-like sufficiency controller | 最危险竞品 |
| Vanilla iterative RAG | 下界 |

### 4.3 必须的消融

| 消融 | 验证什么 |
|------|---------|
| Full vs 去掉 Board（用回 flat history） | Board 是否必要 |
| Full vs 去掉 Claim Gate | Claim Gate 的贡献 |
| Full vs 去掉 Stop Closure | Stop Closure 的贡献 |
| Prompt 版 vs Trained 版 | 训练的价值 |
| Oracle Board / Oracle Claim / Oracle Stop | 系统上限分析 |

### 4.4 必须的指标

**最终答案**:
- EM, F1

**过程质量（核心贡献必须用这些验证）**:
- unsupported claim rate
- premature stop rate
- bridge entity recall
- evidence chain completeness (supporting fact recall)
- query repetition rate
- average retrieval steps
- latency / token cost

**Error Attribution（核心差异化）**:
- failure transition matrix: 每步错误归因到 query_error / evidence_error / claim_error / stop_error
- 对比 ReasonRAG step PRM 的归因能力

### 4.5 Same-Budget 规则

所有方法固定:
- max_steps
- retrieved docs per question
- generator calls
- token budget

---

## 5) Novelty Positioning

### 与 S2G-RAG（最危险竞品）的区别

| 维度 | S2G-RAG | ClosureRAG |
|------|---------|-----------|
| 状态表示 | gap items（缺什么） | Evidence Closure Board（已确认什么 + 缺什么 + 哪些有支撑） |
| 停止判断 | sufficiency judge | **三维修 closure reward** (slot + claim + chain) |
| 中间验证 | 不做 | **transition-time claim-support gate** 阻断 unsupported claim |
| 训练信号 | 无（rule-based） | **Closure reward → SFT → 可选 RL** |
| 集成方式 | 独立 iterative pipeline | **嵌入 ReasonRAG MCTS** |

### 核心差异化（一句话）

> S2G-RAG 判断"证据够不够回答"；我们维护结构化 Board 并在**中间步骤**就阻断 unsupported claim，用**三维修 closure reward** 替代标量 sufficiency judging。

### 与 6 篇核心竞品的定位

```
Search-R1 → outcome reward
ReasonRAG → step PRM (标量)
HiPRAG    → search necessity (标量)
ProRAG    → process RL (标量)
Search-P1 → path reward (标量)
DecEx-RAG → MDP decomposition (标量)

ClosureRAG → structured closure reward (三维: slot/claim/chain)
             + transition-time claim gate
             + evidence closure board
```

---

## 6) Risk Assessment

| 风险 | 级别 | 缓解措施 |
|------|------|---------|
| Board 更新不准确 | 高 | Extract-Validate-Merge 分离；Oracle Board 实验测上限 |
| 7B 模型能力不足以维护 Board | 中 | Phase 1 先验证；如果 prompt 版就有信号说明可行 |
| S2G-RAG 竞品抢先 | 中 | S2G 不做 transition-time claim gate，不做三维 closure reward |
| 不做 RL 打不过 trained baseline | 中 | Phase 2 SFT 先上；Phase 3 RL 可选 |
| AAAI deadline 时间紧 | 中 | Gate 0-2 共 2 周出信号；如果弱则及时止损 |
| Evidence heuristic 太弱 | 低 | 降级为辅助模块，不是核心贡献 |

---

## 7) Explicit Non-goals

- 不在未经 Gate 0-2 验证的情况下直接做大规模实验
- 不在核心贡献中强调 evidence selection（降级为 heuristic）
- 不做 branch rollback（审稿人建议砍掉）
- 不叫 "process-reward guided"（除非真的训练 reward model 做 RL）
- 不和 MCTS/tree search 方法比搜索效率（我们的贡献是过程控制，不是搜索策略）

---

## 8) Timeline

```
2026-06-01 ~ 06-07: Gate 0-1 (Prompt 版 Board + Claim Gate + Stop Closure)
2026-06-08 ~ 06-14: Gate 2 (200 条验证 + 数据积累)
2026-06-15 ~ 06-28: Gate 3 (SFT 训练 + 500 条实验)
2026-06-29 ~ 07-12: Gate 4 (全量实验 + 消融 + baseline)
2026-07-13 ~ 07-20: 写作 (Related Work + Method + Experiments)
2026-07-21 ~ 07-27: 打磨 + 内部 review
2026-07-28: AAAI 2027 submission
```
