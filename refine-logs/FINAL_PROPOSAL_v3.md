# FINAL PROPOSAL v3 — Trajectory Closure Modeling for Agentic RAG

**Working title**: *TrajectoryClosure: Provenance-Aware State Transition Evaluation for Agentic RAG*
**Date**: 2026-05-30
**Status**: Phase 4.5 revised after objective review (4/10 → 修改中)
**Supersedes**: `refine-logs/FINAL_PROPOSAL_v2.md` (ClosureRAG)

---

## 0) 核心变化：v2 → v3

| 维度 | v2 ClosureRAG | v3 TrajectoryClosure |
|------|--------------|---------------------|
| Board | 3 个文本列表 | **每条 claim 带 provenance** |
| Reward | 全局比例（分母不可知） | **per-step transition reward**（不依赖分母） |
| 定位 | "S2G-RAG 的 ReasonRAG 适配" | **MCTS 节点评估的结构化升级** |
| 和 S2G 区别 | gap checking 的变体 | **tree search 中的 transition evaluation，不是线性 pipeline 的 sufficiency check** |
| 审稿人分数 | 4/10 (objective) | 待评估 |

---

## 1) Problem Anchor

**问题**: ReasonRAG 的 MCTS + step PRM 给每个节点一个 0-1 标量分。这个分数不能回答：
- 当前节点的 query 是否对准了未解决的 gap？
- 当前节点的 evidence 是否支撑了新的 claim？
- 当前分支应该扩展、剪枝还是终止？

**根因**: 标量 PRM 把多种决策质量压缩成一个数，丢失了可归因性。

**和 v2 的区别**: v2 说的是"flat history 不好"→ 现在说的是"MCTS 节点评估太粗"→ 更具体的切入点。

---

## 2) Method: Trajectory State + Transition Reward

### 2.1 Trajectory State（替代 Board）

每个 MCTS 节点维护一个 **Trajectory State**，包含 3 个字段，其中 claims 带 provenance：

```json
{
  "entities": ["Shirley Temple", "Chief of Protocol", "1976"],
  "open_gaps": ["1976年的美国总统"],
  "claims": [
    {
      "text": "Shirley Temple 1976年被任命为Chief of Protocol",
      "evidence_ref": "doc_3::sent_5",
      "status": "supported"
    }
  ]
}
```

**和 v2 的关键区别**:
- `claims` 不再是文本列表，每条 claim 都绑定了具体 evidence 句子（doc_id::sent_id）
- `evidence_ref` 使得 claim validation 可审计、可训练、可归因

**为什么还是 3 个顶层字段**:
- `entities` + `open_gaps` 保持简单文本列表（NER + gap 抽取足够可靠）
- `claims` 是唯一有嵌套结构的字段（因为 provenance 是审稿人要求的核心差异化）

### 2.2 State Transition（核心概念）

在 MCTS 中，从一个节点到下一个节点是一次 **transition**：

```
Transition_t = (State_t, Action_t, State_{t+1})

其中 Action_t 包括:
  - query_t: 这一步生成了什么子查询
  - evidence_t: 检索到了什么文档
  - thought_t: 模型生成了什么推理
```

我们评估每次 transition 的**质量**，而不是评估单个节点的标量分。

### 2.3 Transition Reward（替代 Closure Reward）

**不依赖分母**，只看当前 step 的增量：

```
R_transition(t) = R_query(t) + R_claim(t) + R_stop(t)

R_query(t): 查询质量
  = gap_targeting(query_t, open_gaps_t)    // query 是否对准某个 open_gap
  + entity_preservation(query_t, entities_t) // query 是否保留 bridge entity
  - repetition(query_t, query_history)       // 是否重复历史 query

R_claim(t): claim 质量（需要 provenance）
  = Σ supported_claims_with_provenance / Σ new_claims  // 有 evidence_ref 的 claim 比例
  - Σ unsupported_claims / Σ new_claims               // 无 evidence 支撑的 claim 比例

R_stop(t): 停止质量
  = 1 if open_gaps is empty AND all_answer_claims_have_provenance
  = -1 if premature (open_gaps not empty but model wants to stop)
  = 0 otherwise (continue)
```

**和 v2 Closure Reward 的关键区别**:

| 维度 | v2 Closure Reward | v3 Transition Reward |
|------|------------------|---------------------|
| 分母 | `total_required_gaps`（不可知） | **只用当前步的增量**（可知） |
| 可操纵性 | 少抽 gap 就高分 | **只看 query→gap 对应关系，不看 gap 总数** |
| 粒度 | 全局比例 | **per-step per-action** |
| 类型 | 1 个比例 | **3 类 action reward** |
| provenance | 无 | **claim 必须有 evidence_ref 才计入 supported** |

### 2.4 Transition Evaluation 在 MCTS 中的位置

```
ReasonRAG 原始 MCTS:
  select node → expand (generate query → retrieve → generate thought)
  → evaluate with step PRM (标量 0-1) → backpropagate

TrajectoryClosure:
  select node → expand (generate query → retrieve → generate thought)
  → update Trajectory State (Extract-Validate-Merge with provenance)
  → evaluate with Transition Reward (R_query + R_claim + R_stop)
  → backpropagate structured reward
```

**和 S2G-RAG 的根本区别**:

S2G-RAG 是**线性 iterative pipeline**：
```
check sufficiency → output gap → map to query → retrieve → repeat
```

TrajectoryClosure 是 **tree search 中的节点评估**：
```
expand node → compute transition reward → 用 reward 指导 MCTS 选枝/剪枝/终止
```

| 维度 | S2G-RAG | TrajectoryClosure |
|------|---------|-------------------|
| 架构 | 线性 iterative pipeline | **嵌入 MCTS tree search** |
| 评估对象 | 每轮 sufficiency | **每次 state transition** |
| 评估粒度 | sufficiency yes/no | **R_query + R_claim + R_stop 三维** |
| Claim 追踪 | 无 | **provenance-aware (claim → evidence_ref)** |
| 决策影响 | 决定是否继续检索 | **影响 MCTS 选枝、剪枝、终止** |
| 训练信号 | 无（rule-based） | **transition reward → SFT → 可选 RL** |

### 2.5 State Update（Extract-Validate-Merge with Provenance）

```
Step 1: Extract（1 次 LLM 调用）
  输入: thought_t + retrieved docs + 当前 State_t
  输出: candidate entities, candidate gaps, candidate claims
  每个 candidate claim 附带 evidence_ref（来自哪篇文档的哪句话）

Step 2: Validate（NLI model / LLM judge）
  输入: (claim, evidence_ref 指向的原文句子)
  输出: supported / unsupported
  关键: 验证对象是具体的 claim-sentence 对，不是笼统的"有没有支撑"

Step 3: Merge（确定性 Python 代码）
  - supported claims + provenance → 加入 State.claims
  - unsupported claims → 标记，可能触发 repair
  - new entities → 加入 State.entities（去重）
  - resolved gaps → 从 State.open_gaps 移除
  - new gaps → 加入 State.open_gaps
```

### 2.6 Repair Policy（具体化）

```
if R_query(t) < threshold:
  → query 没对准 gap 或丢失 bridge entity
  → 用 open_gap 中的具体描述重写 query
  → 要求新 query 包含 entities 中的关键实体

if R_claim(t) < threshold:
  → 有 unsupported claims
  → 检查 unsupported claim 对应的 gap
  → 如果 gap 仍然 open → 为这个 gap 生成新 query → 补检索

if R_stop(t) == -1:
  → 模型想停但 open_gaps 不为空
  → 强制继续，用 open_gaps 中优先级最高的 gap 生成 query
```

**和 v2 的区别**: 每个 repair 动作都绑定到具体的 reward 维度和具体的 state 字段，不是笼统的"重写 query"。

---

## 3) Training Plan

### Phase 1: Prompt 版验证 + 数据积累（1-2 周）

- Trajectory State update: Prompt 版（LLM 输出带 provenance 的 JSON）
- Transition Reward: Heuristic 计算（基于 prompt 输出的结构化字段）
- 在 500 条 HotpotQA trajectories 上积累标注数据
- 每条标注包含: State_t, Action_t, State_{t+1}, R_query, R_claim, R_stop

### Phase 2: SFT 训练（1-2 周）

**State Updater (LoRA SFT)**:
- 输入: thought + evidence + 当前 State
- 输出: 更新后的 State JSON（包含 claim provenance）
- 数据: Phase 1 积累的 ~1000 条 State 标注
- 训练: Qwen2.5-7B + LoRA

**Claim Validator**:
- 输入: (claim_text, evidence_sentence)
- 输出: supported / unsupported
- 数据: Phase 1 积累的 claim-evidence 对
- 训练: DeBERTa 或 LoRA

**Transition Reward Model (可选)**:
- 输入: (State_t, Action_t, State_{t+1})
- 输出: (R_query, R_claim, R_stop)
- 只在需要做 RL 时训练

### Phase 3: Process RL（可选）

- 用 Transition Reward 替代 step PRM
- 方法: 类似 ReasonRAG 的 GRPO
- 只在 Phase 1-2 信号确认后考虑

---

## 4) 实验设计

### Same-Budget 规则

```yaml
# 基于 ReasonRAG 实际配置
retrieval_topk: 5
inject_top: 5
max_steps: 3
max_tokens: 256
use_reranker: False
```

额外成本需要诚实报告：每步多 1 次 Extract + N 次 Validate + 1 次 Merge。

### Baseline

| Baseline | 为什么必须 |
|---------|----------|
| ReasonRAG | 核心 baseline |
| ReasonRAG + extra LLM judge stop | 证明提升不是来自额外 LLM 调用 |
| S2G-like sufficiency controller | 最直接竞品 |
| Post-hoc claim verification | 证明 transition-time 比 post-hoc 好 |

### 消融

| 实验 | 验证 |
|------|------|
| Full vs 去掉 provenance（claim 只存文本） | provenance 的价值 |
| Full vs 去掉 R_query | query reward 的贡献 |
| Full vs 去掉 R_claim | claim reward 的贡献 |
| Full vs 去掉 R_stop | stop reward 的贡献 |
| Full vs 标量 step PRM（用同一数据训练） | 结构化 reward vs 标量 reward |
| Prompt 版 vs Trained 版 | 训练的价值 |
| Oracle State / Oracle Reward | 上限分析 |

### 指标

**最终答案**: EM, F1

**过程质量**:
- premature stop rate
- unsupported claim rate
- query repetition rate
- bridge entity preservation rate
- claim provenance accuracy（claim → evidence_ref 是否正确）
- average retrieval steps
- latency / extra LLM calls / token cost

**Transition 评估**:
- transition productive rate（每次 transition 是否关闭了 ≥1 gap、新增了 ≥1 supported claim）
- error attribution accuracy（能归因到 query/claim/stop 的比例）

---

## 5) Novelty Positioning

### 核心贡献（一句话）

> 我们提出 Trajectory State 和 per-step Transition Reward，将 ReasonRAG 的 MCTS 节点评估从标量 step PRM 升级为 provenance-aware 的结构化 transition evaluation。

### 和 S2G-RAG 的区别（升级版）

| 维度 | S2G-RAG | TrajectoryClosure |
|------|---------|-------------------|
| 架构 | 线性 iterative pipeline | **嵌入 MCTS tree search** |
| 评估对象 | 每轮 sufficiency (yes/no) | **每次 state transition (三维 reward)** |
| Claim | 不追踪 | **provenance-aware (claim → evidence_ref)** |
| Reward | 无 | **R_query + R_claim + R_stop** |
| 训练信号 | 无 | **transition reward → SFT → 可选 RL** |
| 错误归因 | 无 | **归因到 query/claim/stop 三类决策** |
| 决策影响 | 决定是否继续 | **影响 MCTS 选枝、剪枝、终止** |

### 论文 Story

```
§1 Introduction
   ReasonRAG 的 MCTS + step PRM 无法归因错误来源

§2 Related Work
   Process reward 系列 (ReasonRAG/ProRAG/HiPRAG/Search-P1)
   Sufficiency control 系列 (S2G-RAG/SURE-RAG/PAR²-RAG)
   Claim verification 系列 (RAGChecker/MedRAGChecker)
   指出：无人把 provenance-aware state transition 引入 MCTS 节点评估

§3 Method
   §3.1 Trajectory State 定义（带 provenance）
   §3.2 State Transition 概念
   §3.3 Transition Reward (R_query + R_claim + R_stop)
   §3.4 State Update (Extract-Validate-Merge with provenance)
   §3.5 Repair Policy

§4 Experiments
   §4.1 Main Results (HotpotQA/2Wiki/MuSiQue/Bamboogle)
   §4.2 Transition Reward vs Scalar Step PRM
   §4.3 Ablation (provenance, R_query, R_claim, R_stop)
   §4.4 Error Attribution Analysis
   §4.5 Cost Analysis

§5 Discussion
   oracle analysis, limitation, future work (process RL)
```

---

## 6) Risk Assessment

| 风险 | 级别 | 缓解 |
|------|------|------|
| 7B 模型无法可靠输出带 provenance 的 JSON | 高 | Phase 1 先验证；如果 prompt 版就不可靠，用更简单的 provenance 格式 |
| S2G-RAG 抢先发表 | 中 | 我们的 MCTS 集成 + transition reward + provenance 是 S2G 没做的 |
| Transition Reward 定义仍然是 heuristic | 中 | 先用 heuristic 验证信号；如果有效再训练 reward model |
| 额外 LLM 调用导致 unfair comparison | 中 | 诚实报告 cost；做 "same total LLM budget" 对照实验 |
| AAAI deadline 时间紧 | 中 | Gate 0-2 用 prompt 版验证信号，2 周出结果 |

---

## 7) Timeline

```
2026-06-01 ~ 06-07: Gate 0-1 (Prompt 版 Trajectory State + Transition Reward)
2026-06-08 ~ 06-14: Gate 2 (200 条验证 + 数据积累)
2026-06-15 ~ 06-28: Gate 3 (SFT 训练 + 500 条实验)
2026-06-29 ~ 07-12: Gate 4 (全量实验 + 消融 + baseline 对比)
2026-07-13 ~ 07-27: 写作 + 打磨
2026-07-28: AAAI 2027 submission
```
