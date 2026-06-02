# FINAL PROPOSAL v4 — Typed Transition Evaluation for Directed MCTS in Agentic RAG

**Working title**: *TrajectoryClosure: Failure-Attributed Branch Selection for MCTS-based Agentic RAG*
**Date**: 2026-05-31
**Status**: Phase 4.5 revised after objective review (5/10 → 修改中)
**Supersedes**: `refine-logs/FINAL_PROPOSAL_v3.md` (TrajectoryClosure v3)

---

## 0) 核心变化：v3 → v4

| 维度 | v3 TrajectoryClosure | v4 TrajectoryClosure |
|------|---------------------|---------------------|
| 核心贡献 | "结构化 state + transition reward" | **failure-attributed branch selection → directed expansion in MCTS** |
| Reward 定义 | 3 个 heuristic scalar 描述 | **形式化定义，每个组件可计算** |
| MCTS 集成 | "嵌入 MCTS"但没说怎么用 reward | **typed expansion strategy：failure type 决定扩展策略** |
| Provenance | 单句 evidence_ref | **multi-evidence provenance（set of refs）** |
| Anti-gaming | 无 | **R_claim 使用 F1-style + minimum claim floor** |
| Compute fairness | 只控制 retrieval_topk | **same-inference-budget（total LLM tokens）** |
| State quality | 假设正确 | **oracle analysis + sensitivity study** |
| 数据现实 | 未分析数据集实际格式 | **新增 §1.5：明确 gold 仅到文档标题级，过程标签全部自构造** |
| 训练路线 | LLM judge 银标准（循环监督） | **SOTA 教师蒸馏 + SFT + RL（消除循环监督）** |
| 审稿人分数 | 5/10 | 待评估 |

### v4 的核心论点转移

v3 说："我们加了 provenance 和 transition reward" → 审稿人："这是 S2G + RAGChecker 的 MCTS 适配"

v4 说：**"我们用 typed transition evaluation 实现 MCTS 中的 failure-attributed branch selection，使得不同失败类型的分支获得不同的扩展策略。这在 S2G 的线性 pipeline 中不可能做到，因为线性 pipeline 没有分支。"**

---

## 1) Problem Anchor

**问题**: ReasonRAG 的 MCTS 用标量 step PRM (0-1) 评估节点。标量分数无法区分失败类型，导致 MCTS 的 branch selection 是"盲"的——两个分支可能总分相同，但失败原因完全不同。

**具体来说**：

```
问题: "Shirley Temple 被任命为 Chief of Protocol 时，当时的美国总统是谁？"

Branch A (step PRM = 0.4):
  query: "Shirley Temple 任命时间"
  → 检索到正确文档，但 thought 遗漏了关键信息
  → 失败原因: claim_fail（没从 evidence 中正确抽取 claim）

Branch B (step PRM = 0.4):
  query: "美国总统名单"  ← 丢失了 bridge entity "1976"
  → 检索到无关文档
  → 失败原因: query_fail（query 没有对准 open gap，丢失 bridge entity）
```

**标量 PRM 无法区分 Branch A 和 Branch B**——都是 0.4 分。但最优扩展策略完全不同：
- Branch A：query 好但 claim 差 → 保持 query，重新生成 thought
- Branch B：query 差 → 重写 query，保留 bridge entity

**这就是 MCTS 中 typed transition evaluation 的价值**：不只是一个更好的分数，而是改变 tree search 本身的行为。

**为什么 S2G-RAG 不能做**：S2G 是线性 pipeline，每步只有一条路径，没有分支比较。在单路径上判断"够不够"（sufficiency）和在多分支间比较"哪个更有前途"（branch selection）是本质上不同的问题。

---

## 1.5) Data Reality：数据集提供什么、我们的方法需要什么

### 数据集实际提供了什么

**ReasonRAG 训练数据实际格式**（`/home/mayi/RAG/ReasonRAG/training_data/hotpotqa.jsonl`，共 90447 条）：

```json
{
  "id": "train_0",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "golden_answers": ["Arthur's Magazine"],
  "metadata": {
    "type": "comparison",           // 或 "bridge"
    "level": "medium",              // easy / medium / hard
    "supporting_facts": {
      "title": ["Arthur's Magazine", "First for Women"],
      "sent_id": [0, 0]
    },
    "context": {
      "title": ["Radio City", "History of Albanian football", ...共10篇],
      "sentences": [
        ["Radio City is India's first private FM...", ...],
        ["Football in Albania existed before...", ...],
        ...
      ]
    }
  }
}
```

**关键结构特征**：

1. **顶层字段**：`id` + `question` + `golden_answers` + `metadata`（FlashRAG 标准格式）
2. **supporting_facts 在 metadata 内部**，且是 dict 格式（`{title: [...], sent_id: [...]}`），不是原始 HotpotQA 的 list 格式
3. **context 也在 metadata 内部**，也是 dict 格式（`{title: [...], sentences: [[...], [...]]}`），共 10 篇文档
4. **supporting_facts 的 title 和 sent_id 是一一对应的**：`title[i]` 文档的第 `sent_id[i]` 句是金标准支撑句子

**实际 bridge 类型例子**：

```
问题: "The Oberoi family is part of a hotel company that has a head office in what city?"
答案: Delhi

supporting_facts:
  Oberoi family[0]:    "The Oberoi family is an Indian family famous for involvement
                        in hotels, namely through The Oberoi Group."
  The Oberoi Group[0]: "The Oberoi Group is a hotel company with its head office in Delhi."

context: 10篇文档（含上述2篇金标准 + 8篇干扰文档）
```

**数据分布**：bridge 72991 条（80.7%）、comparison 17456 条（19.3%）

**2WikiMultiHopQA**：类似结构，也有 `supporting_facts`，部分版本还包含更结构化的 `evidences`、`evidences_id`、`answer_id`。

### 数据集有 vs 方法需要的

| 数据集有（final-level gold） | 方法需要但数据集没有（trajectory-level） |
|---------------------------|--------------------------------------|
| 最终答案（answer） | 每步应该生成什么子查询 |
| 最终 supporting_facts（哪些文档/句子支撑答案） | 每步检索回来的 top-5 里哪些该被关注 |
| 问题类型（bridge / comparison） | 每步的缺口列表应该是什么 |
| 难度等级（easy / medium / hard） | 什么时候应该停止检索 |
| | 每步的中间结论是否被证据支撑 |
| | 失败归因到查询/断言/停止哪个维度 |

**核心鸿沟**：数据集提供**最终答案级别的金标准**，但 Agentic RAG 需要**过程级别的轨迹监督**。这个鸿沟正是我们的研究空间。

### supporting_facts 的实际可用范围

supporting_facts 是人工标注的金标准，但在 Agentic RAG 训练流程中**不能直接当句子级标签用**，原因：

1. **文档来源不同**：supporting_facts 引用的是数据集预给的 10 篇文档中的句子。ReasonRAG 检索回来的文档来自自己的语料库（BGE + FAISS），文档内容、句子切分可能不一样。
2. **索引无法对齐**：supporting_facts 的 `[title, sent_id]` 中的 `sent_id` 是数据集预给文档中的句子编号，不能直接映射到检索回来的文档。

**实际能用的范围**：

| 用法 | 可行性 | 说明 |
|------|--------|------|
| 文档标题级覆盖检查 | **可行** | 检查检索结果是否包含了 supporting_facts 中的文档标题 |
| 训练数据质量粗过滤 | **可行** | 如果检索结果连金标准文档都没覆盖 → 标注不可靠 |
| Oracle 检索实验 | **可行** | 确保检索结果包含所有金标准文档，模拟"检索完美" |
| 句子级 provenance 验证 | **不可行** | 检索文档和金标准文档的句子切分不一致 |
| RL 句子级覆盖奖励 | **不可行** | 无法精确对齐到检索文档的句子 |
| 过程级轨迹标签 | **不可行** | 数据集不提供每步的 gap/claim/query 标注 |

### 对方法各组件的影响

| 方法组件 | 对 gold 标注的依赖 | 实际数据来源 |
|---------|-------------------|------------|
| $\phi_q$（查询质量） | **不依赖** | 纯计算：NER + 集合运算 + Jaccard，不需要任何标注 |
| $\phi_c$（断言质量） | **不依赖** | NLI 验证（claim vs evidence sentence），不需要 gold 标注 |
| $\phi_s$（停止质量） | **弱依赖** | 依赖 open_gaps 是否为空 → 缺口由 LLM 抽取，非 gold |
| 缺口抽取 | **无 gold** | LLM judge 标注 + 自一致性过滤 + 文档标题级 gold 校验 |
| 断言 provenance | **无 gold** | LLM judge 标注 + NLI 验证 |
| 失败归因 | **无 gold** | 由 $\phi_q, \phi_c, \phi_s$ 自动计算得出 |

**关键设计**：$\phi_q$ 完全不依赖任何标注或 LLM 输出（只依赖 NER），这是方案最可靠的部分。$\phi_c$ 依赖 NLI 而非 gold。$\phi_s$ 依赖缺口抽取的准确性，是最脆弱的环节。

---

## 2) Method

### 2.1 Trajectory State（带 multi-evidence provenance）

每个 MCTS 节点维护：

```json
{
  "entities": ["Shirley Temple", "Chief of Protocol", "1976"],
  "open_gaps": ["1976年的美国总统"],
  "claims": [
    {
      "text": "Shirley Temple 于1976年被任命为 Chief of Protocol",
      "evidence_refs": ["doc_3::sent_5", "doc_3::sent_6"],
      "status": "supported"
    }
  ]
}
```

**和 v3 的区别**：
- `evidence_refs`（复数）：一条 claim 可以被多个 evidence sentence 联合支撑。多跳 QA 的关键结论经常需要多句联合支持，单句 provenance 是过度简化。
- 类型：`evidence_refs: Set[doc_id :: sent_id]`

**三个字段的角色**：

| 字段 | 类型 | 用于 | 为什么不需要更多字段 |
|------|------|------|-------------------|
| `entities` | `List[str]` | bridge entity tracking, query preservation check | NER + 去重足够可靠 |
| `open_gaps` | `List[str]` | gap targeting, stop condition | 自然语言描述，LLM 抽取 + gold SF 校验 |
| `claims` | `List[{text, evidence_refs, status}]` | claim provenance, repair trigger | 唯一嵌套字段，provenance 是核心差异化 |

---

### 2.2 Typed Transition Evaluation（形式化定义）

**定义**：给定 transition $\tau_t = (S_t, a_t, S_{t+1})$，我们计算 typed evaluation $\Phi(\tau_t) = (\phi_q, \phi_c, \phi_s)$。

#### $\phi_q$：Query Quality（formal）

$$\phi_q(\tau_t) = \underbrace{\text{Overlap}(\mathcal{E}(q_t), \mathcal{E}(G_t))}_{\text{gap targeting}} \cdot \underbrace{\text{Overlap}(\mathcal{E}(q_t), B_t \cap \mathcal{E}(G_t))}_{\text{bridge preservation}} \cdot \underbrace{(1 - \max_{q' \in Q_{<t}} \text{Sim}(q_t, q'))}_{\text{non-redundancy}}$$

其中：
- $\mathcal{E}(x)$：从文本 $x$ 中抽取的 named entity 集合（NER，确定性）
- $G_t$：$S_t$ 中的 open_gaps（自然语言列表）
- $B_t$：$S_t$ 中的 entities（bridge entity 集合）
- $Q_{<t}$：历史 query 集合
- $\text{Overlap}(A, B) = |A \cap B| / |A|$ if $|A| > 0$ else $1$（query 中有多少实体对准了目标）
- $\text{Sim}(q_1, q_2)$：token-level Jaccard similarity

**性质**：
- $\phi_q \in [0, 1]$
- 每个因子可独立计算、独立评估
- 不依赖 LLM judge，只依赖 NER + set operation + Jaccard
- 失败模式可精确诊断：哪个因子低 → gap targeting 差 / bridge 丢失 / query 重复

#### $\phi_c$：Claim Quality（formal, anti-gaming）

$$\phi_c(\tau_t) = \frac{|\{c \in \Delta C_t : \text{Supp}(c)\}| + \epsilon}{\max(|\Delta C_t|, \kappa) + \epsilon}$$

其中：
- $\Delta C_t = C_{t+1} \setminus C_t$：本步新增 claims
- $\text{Supp}(c)$：claim $c$ 至少有一条 evidence_refs 中的句子通过 NLI 验证
- $\kappa = \lceil |G_t| / 2 \rceil$：minimum expected claims floor（每 2 个 open gap 至少产生 1 个 claim）
- $\epsilon = 0.5$：Laplace smoothing，避免除零

**Anti-gaming 设计**：
- 如果模型不生成 claim（$|\Delta C_t| = 0$），分母是 $\kappa$（期望 claim 数），不是 0 → 得分趋近 0
- 如果模型生成 trivial claim，NLI 验证会标记为 unsupported → 分子不增加
- 如果模型少报 gap 来降低 $\kappa$ → gap 数量由 Extract 阶段独立决定，不由 generator 控制

**Multi-evidence provenance**：
- $\text{Supp}(c)$ 的判定：claim $c$ 的 `evidence_refs` 中至少有一条句子通过 NLI → supported
- 但 high-quality claim 应该有更完整的 provenance：一个 claim 的所有关键陈述都应该有 evidence 支撑
- 正式版可进一步拆分为 claim-level precision (supported facts / claimed facts) 和 recall (supported facts / required facts)

#### $\phi_s$：Stop Quality（formal）

$$\phi_s(\tau_t) = \begin{cases} +1 & \text{if } G_{t+1} = \emptyset \land \forall c \in \text{AnsClaims}(q, C_{t+1}): \text{Supp}(c) \\ -1 & \text{if } G_{t+1} \neq \emptyset \land \text{model stops at } t \\ 0 & \text{otherwise (continue)} \end{cases}$$

其中：
- $G_{t+1}$：transition 后的 open_gaps
- $\text{AnsClaims}(q, C)$：与问题 $q$ 直接相关的 claims（由 NLI 判定 claim 是否能直接回答问题）
- $\text{Supp}(c)$：claim 至少有一条 evidence 通过 NLI

**性质**：
- $\phi_s \in \{-1, 0, +1\}$，三值而非连续
- 不依赖分母
- 正停止条件：gaps 全部关闭 且 答案 claim 全部有 provenance
- 负停止条件：gaps 未关闭但模型想停（premature stop）

---

### 2.3 Failure Attribution（MCTS 中的核心差异化）

给定 typed evaluation $\Phi(\tau_t) = (\phi_q, \phi_c, \phi_s)$，我们定义 failure type：

$$f(\tau_t) = \begin{cases} \text{success} & \text{if } \phi_q \geq \theta_q \land \phi_c \geq \theta_c \land \phi_s \geq 0 \\ \text{query\_fail} & \text{if } \phi_q < \theta_q \land \phi_q = \min(\phi_q, \phi_c, \phi_s) \\ \text{claim\_fail} & \text{if } \phi_c < \theta_c \land \phi_c = \min(\phi_q, \phi_c, \phi_s) \\ \text{stop\_fail} & \text{if } \phi_s = -1 \land \phi_q \geq \theta_q \land \phi_c \geq \theta_c \\ \text{mixed} & \text{otherwise (multiple failures)} \end{cases}$$

**关键**：$f(\tau_t)$ 将每次 transition 的失败归因到具体的 failure type。这不仅是诊断信息——它**直接决定 MCTS 的扩展策略**。

**为什么这在 S2G-RAG 中不可能**：

S2G-RAG 是线性 pipeline：
```
step 1 → sufficiency? → no → gap → query → step 2 → sufficiency? → ...
```
每步只有一条路径。S2G 可以判断"这一步够不够"，但无法在多个候选路径之间比较"哪条路更有前途"。

MCTS 中有多条分支：
```
                    root
                   / | \
              node_A node_B node_C
              / \       |
          node_A1 node_A2  node_B1
```
每条分支可能失败的原因不同。Typed transition evaluation 使得：
- node_A：query_fail → 扩展时优先重写 query
- node_B：claim_fail → 扩展时优先补充检索
- node_C：stop_fail → 强制继续

**这种基于 failure type 的差异化扩展策略在标量 PRM 中不可能**，因为标量 PRM 无法归因失败类型。两个分支都得了 0.4 分，MCTS 不会知道哪个该优先扩展。

---

### 2.4 Directed Expansion Strategy（MCTS 专用）

标准 MCTS 扩展：所有子节点用同一策略生成（通常是模型自回归采样）。

**我们的 typed expansion**：

```
在节点 n 选择扩展策略时:

if f(n) = query_fail:
  → Directed Expansion: 重写 query
  - 保留 entities 中的 bridge entity
  - 用 open_gaps 中优先级最高的 gap 作为 query 核心内容
  - 避免重复历史 query
  - 检索后正常 thought generation
  → 生成 1 个新子节点

if f(n) = claim_fail:
  → Supplementary Retrieval Expansion
  - 不改 query，保持当前 query
  - 为 unsupported claim 对应的 open_gap 补充检索
  - 用新 evidence 重新生成 thought
  → 生成 1 个新子节点

if f(n) = stop_fail:
  → Forced Continuation
  - 不允许生成最终答案
  - 用 open_gaps 中优先级最高的 gap 生成下一条 query
  → 生成 1 个新子节点

if f(n) = success:
  → 正常扩展（如果还有未尝试的 action）或标记为 terminal
```

**和 ReasonRAG 原始 MCTS 的区别**：

| 维度 | ReasonRAG MCTS | TrajectoryClosure MCTS |
|------|---------------|----------------------|
| 节点评估 | step PRM scalar (0-1) | typed evaluation $(\phi_q, \phi_c, \phi_s)$ + failure attribution $f$ |
| 分支选择 | UCB1 on scalar reward | **UCB1 on typed reward**（见 2.5）|
| 扩展策略 | 统一策略（模型自回归） | **failure-attributed directed expansion** |
| 剪枝 | 基于 scalar score | **基于 failure type**：mixed failure → 低优先级 |
| 终止条件 | flag-based / max depth | **gap closure + claim provenance** |
| 诊断能力 | 无 | **每步 failure type 可追溯** |

---

### 2.5 Typed Backpropagation in MCTS

标准 UCB1：

$$\text{UCB1}(n) = \bar{Q}(n) + c \sqrt{\frac{\ln N(\text{parent})}{N(n)}}$$

我们的 **Typed-UCB**：

$$\text{UCB}_{\text{typed}}(n) = w_q \cdot \bar{\phi}_q(n) + w_c \cdot \bar{\phi}_c(n) + w_s \cdot \bar{\phi}_s^+(n) + c \sqrt{\frac{\ln N(\text{parent})}{N(n)}}$$

其中：
- $\bar{\phi}_q(n)$：从根到节点 $n$ 的所有 transition 的平均 $\phi_q$
- $\bar{\phi}_c(n)$：同上，$\phi_c$
- $\bar{\phi}_s^+(n) = \max(0, \bar{\phi}_s(n))$：$\phi_s$ 的正部分（stop reward 只在有停止决策时生效）
- $w_q, w_c, w_s$：权重，初始化为 $(1/3, 1/3, 1/3)$，可通过验证集调参或学习
- $c$：标准 exploration constant

**替代方案（Bottleneck-UCB）**：

不使用加权求和，而是使用**瓶颈维度**：

$$\text{UCB}_{\text{bottleneck}}(n) = \min_d \left[\bar{\phi}_d(n) + c \sqrt{\frac{\ln N(\text{parent})}{N(n)}}\right]$$

直觉：选择"最弱维度最高"的分支。如果一个分支的 $\phi_q$ 很高但 $\phi_c$ 很低，说明 query 好但 claim 差 → 这个分支值得扩展（好 query，需要更好的 claim generation）。

**两种 UCB 的实验对比**：作为消融实验的一部分。

---

### 2.6 State Update（Extract-Validate-Merge）

```
Extract (1 次 LLM 调用):
  输入: thought_t + retrieved_docs_t + current_state S_t
  输出: {
    new_entities: [str],
    new_gaps: [str],
    new_claims: [{text, evidence_refs: Set[str]}],
    resolved_gaps: [str]
  }

Validate (N 次 NLI 调用，N = |new_claims|):
  对每个 new_claim:
    输入: (claim.text, evidence_refs 中每个句子的原文)
    输出: supported / unsupported
  注: NLI 用小模型（DeBERTa-large），不用 7B LLM

Merge (确定性 Python):
  supported_claims + provenance → add to S.claims
  unsupported_claims → mark, trigger repair
  new_entities → add to S.entities (去重)
  resolved_gaps → remove from S.open_gaps
  new_gaps → add to S.open_gaps
```

**State quality 保障**：
- Extract 阶段的输出可用 HotpotQA gold supporting facts 做**文档标题级**校验
  - 具体方法：检查检索结果是否覆盖了 supporting_facts 中提到的文档标题
  - 如果文档标题覆盖率 < 50% → 该 trajectory 的 state annotation 标记为 low-confidence
  - 注意：只做文档标题级匹配，不做句子级对齐（数据集预给文档和检索文档的句子切分不一致）
- NLI 验证提供句子级的断言支撑判断（不依赖 gold 标注）
- Oracle 实验：用 gold supporting facts 的文档标题构造"检索完美"条件，测系统上限
- Sensitivity 实验：对 state 加入可控噪声，测 $\phi_q, \phi_c, \phi_s$ 的 robustness

---

### 2.7 完整算法

```
Algorithm: TrajectoryClosure-Guided MCTS

Input: question q, retriever R, generator G, max_depth D
Output: answer a, trajectory diagnostics

1. root ← MCTSNode(state = InitState(q))
2. for depth = 1 to D:
3.   for each active node n in current frontier:
4.     // Standard MCTS expansion with typed strategy
5.     if n.failure_type = query_fail:
6.       child ← DirectedExpand(n, strategy=QUERY_REWRITE)
7.     elif n.failure_type = claim_fail:
8.       child ← DirectedExpand(n, strategy=SUPPLEMENTARY_RETRIEVAL)
9.     elif n.failure_type = stop_fail:
10.      child ← DirectedExpand(n, strategy=FORCED_CONTINUATION)
11.    else:
12.      child ← StandardExpand(n)  // 模型自回归采样
13.
14.    // Evaluate transition
15.    τ ← (n.state, child.action, child.state)
16.    Φ(τ) ← (φ_q, φ_c, φ_s)  // typed evaluation (Section 2.2)
17.    f(τ) ← FailureAttribution(Φ(τ))  // Section 2.3
18.
19.    // Backpropagate typed reward
20.    BackpropTyped(child, Φ(τ))
21.
22.  // Select nodes for next frontier using Typed-UCB
23.  frontier ← SelectByTypedUCB(all_children, budget=max_children)
24.
25. // Terminal: select best path
26. best_path ← argmax_n UCB_typed(n) among terminal nodes
27. a ← GenerateAnswer(best_path.final_state)
28. return a, {failure_types, Φ values per step}
```

---

## 3) 和 S2G-RAG 的根本区别（重写）

审稿人质疑："S2G 适配到 MCTS" 不等于方法创新。我们的回应：

**S2G-RAG 解决的问题**：在单条路径上判断"证据是否足够" → 线性 sufficiency check
**我们解决的问题**：在多条分支间比较"哪条路更有前途、应该怎么扩展" → tree search 中的 failure-attributed branch selection

| 维度 | S2G-RAG | TrajectoryClosure | 为什么不只是"适配" |
|------|---------|-------------------|-------------------|
| 搜索结构 | 单路径（线性） | **多分支（tree）** | 有/无分支比较是质的差异 |
| 评估目标 | "当前证据够不够" | **"这次 transition 的 query/claim/stop 哪个维度失败"** | sufficiency vs. failure attribution |
| 评估结果 | binary (continue/stop) | **typed failure label** | 二值 vs. 四类标签 |
| 决策影响 | 决定是否继续检索 | **决定扩展策略（query rewrite / supplementary retrieval / forced continuation）** | "是否继续" vs. "怎么继续" |
| Branch selection | 无（只有一条路） | **Typed-UCB** | S2G 不存在这个问题 |
| Credit assignment | 无 | **branch-level, typed** | S2G 无法做 |

**一句话**：S2G 判断"够不够"，我们判断"哪条路有前途、该怎么走"。前者是 sufficiency check，后者是 **failure-attributed search control**。

---

## 4) Training Plan：SOTA 蒸馏 + SFT + RL

### 整体思路

训练数据不是让 7B 模型级别的 LLM judge 自己标自己（循环监督），而是用 **SOTA 模型（GPT-4 级别）作为教师**生成轨迹级标签，在 7B 学生模型上做 SFT（模仿学习），再用 RL（自我探索）让学生超越教师的示范。

```
教师（GPT-4 级别）                           学生（Qwen2.5-7B）
    │                                              │
    ▼                                              │
  看到轨迹 + gold supporting_facts                  │
  标注每步的: 缺口 / 断言+provenance / 失败类型      │
    │                                              │
    ▼                                              │
  过滤低质量标注                                     │
    │                                              │
    └──── 强银标准轨迹标签 ────→ SFT（模仿学习） ──→ │
                                                   │
                                            RL（自我探索）
                                            奖励: EM/F1 + 文档覆盖率 + NLI
                                                   │
                                                   ▼
                                            部署到 ReasonRAG MCTS
```

**为什么这比"7B judge 标注"好**：

| 维度 | 7B LLM judge 标注 | SOTA 教师标注 |
|------|-----------------|-------------|
| 标注者 vs 学习者 | 同水平（循环监督） | **教师远强于学生** |
| 标注质量上限 | ≈ 7B 自身能力 | **≈ GPT-4 能力** |
| 能否超越标注 | 不能 | **能（RL 探索超越 SFT 数据）** |
| 审稿人接受度 | 低（"银标准不可靠"） | **高（蒸馏+RL 是成熟范式）** |

---

### Phase 1: Prompt 版验证 + SOTA 教师标注（1-2 周）

#### 1a. 收集原始轨迹

在 ReasonRAG 上跑 HotpotQA（tree mode），收集 500 条轨迹。每条轨迹包含：

```
问题: "The Oberoi family is part of a hotel company that has its head office in what city?"
Gold supporting_facts: {title: ["Oberoi family", "The Oberoi Group"], sent_id: [0, 0]}
Gold answer: ["Delhi"]

Step 0: query="Oberoi family hotel" → 检索到5篇文档 → thought="..."
Step 1: query="The Oberoi Group head office" → 检索到5篇文档 → thought="..."
Step 2: 停止 → 答案 "Delhi"
```

#### 1b. 用 SOTA 模型标注每一步

对轨迹中的每一步，给 GPT-4 级别的模型提供：

```
输入：
  - 原始问题
  - 当前步的 query、检索到的文档（含标题和句子）、模型生成的 thought
  - 前几步的状态（如果有的话）
  - Gold supporting_facts（从 metadata['supporting_facts'] 读取）
  - Gold answer

要求 SOTA 模型输出：
  1. 轨迹状态标注:
     - 当前步应该有哪些实体
     - 当前步应该有哪些未解决的缺口
     - 当前步应该抽取哪些断言，每个断言对应的证据句子（doc_id::sent_id）
     - 哪些缺口在这一步被解决了
  2. 质量评判:
     - 这一步的 query 质量如何（好/中/差 + 原因）
     - 这一步的 claim 质量如何（断言是否被证据支撑）
     - 这一步是否应该停止
  3. 失败归因:
     - 失败类型: success / query_fail / claim_fail / stop_fail / mixed
     - 如果有失败，正确的下一步应该怎么做
```

**SOTA 教师的关键优势**：可以看到 gold supporting_facts，所以它能知道"这条轨迹最终需要哪些文档和句子"，从而更准确地判断每一步的缺口、断言和失败原因。7B judge 没有这个信息。

#### 1c. 标注数据过滤

**第 1 层：规则过滤（自动）**
```python
# provenance 一致性: evidence_refs 指向的句子必须存在于检索文档
# claim-evidence 对齐: supported claim 和其 evidence_ref 的 token overlap > 阈值
# gap 进展合理性: |open_gaps| 不应在后期暴涨
# gold supporting facts 文档标题覆盖率:
#   sf = item.metadata['supporting_facts']
#   gold_titles = set(sf['title'])
#   retrieved_titles = {doc.title for doc in retrieved_docs}
#   coverage = len(gold_titles & retrieved_titles) / len(gold_titles)
#   coverage < 0.5 → 标记为 low-confidence
```

**第 2 层：教师自一致性过滤**
- 对关键样本用 SOTA 模型标注两次（不同 temperature）
- 两次标注的缺口/断言一致性低 → 标记为 uncertain

**第 3 层：人工抽检**
- 规则过滤标红的: 全部看（~5-10%）
- 教师自一致性低的: 全部看（~10%）
- 随机抽 5%: 估算教师标注准确率（SOTA 模型预期 > 85%）

#### 1d. 同步做 Prompt 版验证

在教师标注的同时，用 prompt 版跑 50 条验证信号：
- 实验 A（typed vs scalar branch selection）→ 核心贡献的 Go/No-Go
- 如果 typed 和 scalar 选的分支一样 → 停止，不继续训练

**数据产出**：
- 500 条轨迹 × ~3 步 = ~1500 个 step annotations（强银标准）
- 50 条 prompt 版验证结果
- Gate 0 Go/No-Go 判断

**成本估算**：
- SOTA API 调用: 1500 次 × ~$0.04/次 ≈ **$60**
- ReasonRAG 推理: 500 条 × ~10 秒/条 ≈ 1.5 小时（1×5090）
- 人工抽检: ~100 条 × 2-3 分钟/条 ≈ **3-5 小时**

---

### Phase 2: SFT 模仿学习（1-2 周）

用 Phase 1 积累的教师标注数据，在 7B 模型上做 LoRA SFT。

#### 2a. State Updater (LoRA SFT)

- **目的**: 让 7B 模型学会像 GPT-4 一样做轨迹状态更新
- **输入**: thought + 检索到的文档 + 当前状态
- **输出**: 更新后的状态 JSON（缺口、断言 + provenance、实体）
- **数据**: Phase 1 过滤后的教师标注（~1200 条 step annotations）
- **模型**: Qwen2.5-7B + LoRA
- **训练量**: ~1200 条，LoRA 微调，1×5090，2-4 小时

#### 2b. Claim Validator

- **目的**: 判断断言是否被证据句子支撑（替代 LLM judge，降延迟）
- **输入**: (claim_text, evidence_sentence_text)
- **输出**: supported / unsupported
- **数据**: Phase 1 教师标注中提取的 claim-evidence pairs（~3000 对）
- **模型**: DeBERTa-large 或 Qwen2.5-7B + LoRA
- **训练量**: 很小，1 小时内

#### 2c. Failure Classifier (轻量)

- **目的**: 根据轨迹特征判断失败类型
- **输入**: $(\phi_q, \phi_c, \phi_s)$ + 轨迹特征
- **输出**: failure type ∈ {success, query_fail, claim_fail, stop_fail, mixed}
- **数据**: Phase 1 教师标注的失败类型标签
- **模型**: 轻量分类器（不需要 7B，可以是 MLP 或小型 transformer）

#### 2d. 评估 SFT 效果

- 在 200 条 held-out 轨迹上比较: SFT 模型 vs prompt 版 vs 教师标注
- 指标: 缺口抽取 F1、断言抽取 F1、provenance 准确率、失败归因准确率
- 如果 SFT 模型接近教师标注（> 80% 一致性）→ Phase 3 有意义
- 如果差距大 → 继续积累数据或调整 prompt

---

### Phase 3: RL 自我探索（1-2 周，可选）

SFT 只能学到教师做过的。RL 让模型自己探索，可能发现教师没教过的更好策略。

#### 3a. 奖励信号

| 奖励 | 来源 | 粒度 | 可靠性 |
|------|------|------|--------|
| 最终答案 EM/F1 | 数据集金标准 | 问题级 | **最高** |
| 文档标题覆盖率 | gold supporting_facts | 轨迹级 | 高 |
| NLI 验证通过率 | DeBERTa | 步级 | 中 |
| 步数效率 | 自动计算 | 轨迹级 | 高（少步 = 好） |
| 失败归因一致性 | 与教师标注对比 | 步级 | 中 |

**关键**: 没有句子级 gold 奖励（检索文档和金标准文档的句子切分不一致），但文档级覆盖 + 最终 EM/F1 的组合已经足够做 RL。

#### 3b. RL 方法

- 方法: GRPO（和 ReasonRAG 用的一样）
- **关键区别**: reward 是 typed 的（三维），不是标量
- 效果: policy gradient 可以分别引导 query generation、claim extraction、stop decision
- 模型: Phase 2 的 SFT 模型作为初始化
- 算力: 3×RTX 5090，1-2 天

#### 3c. RL 的价值

RL 要验证的核心假设: **SFT 模型通过自我探索，能否在某些情况下超越教师标注?**

具体来说:
- 教师标注的是"正确的轨迹状态"，但教师可能对某些问题类型的缺口分解做得不好
- RL 让模型在这些情况下尝试不同的策略，如果最终答案 EM/F1 提升了 → 说明探索有效
- 如果 RL 后效果和 SFT 一样 → RL 价值有限，论文中作为 ablation 报告即可

---

### Phase 4: 全量实验（2 周）

- 四数据集: HotpotQA / 2Wiki / MuSiQue / Bamboogle
- 对比: Prompt 版 vs SFT 版 vs SFT+RL 版 vs ReasonRAG vs S2G
- 消融 + 过程指标 + 时延报告
- 详见 §5 实验设计

---

### 时间线（更新版）

```
2026-06-01 ~ 06-07: Phase 1
  → 跑 ReasonRAG 收集 500 条轨迹
  → GPT-4 标注轨迹级标签（~$60）
  → Prompt 版 Gate 0 验证（50 条）
  → Go/No-Go 判断: typed vs scalar branch selection 是否不同

2026-06-08 ~ 06-14: Phase 2
  → SFT 训练 State Updater + Claim Validator + Failure Classifier
  → 200 条 held-out 评估: SFT 模型 vs 教师标注一致性
  → 如果 < 80% 一致性 → 补数据或调 prompt

2026-06-15 ~ 06-28: Phase 3
  → RL 训练（如果 Phase 2 信号正向）
  → 500 条 HotpotQA + 500 条 2Wiki 评估

2026-06-29 ~ 07-12: Phase 4
  → 全量实验 + 消融 + S2G baseline + 时延分析

2026-07-13 ~ 07-27: 写作 + 打磨

2026-07-28: AAAI 2027 submission
```

---

## 5) 实验设计

### 5.1 Same-Inference-Budget 规则

所有实验必须报告以下三种 budget 设置之一：

| Budget 类型 | 规则 | 目的 |
|------------|------|------|
| **Strict same-budget** | 总 LLM tokens ≤ ReasonRAG 的 1.0x | 证明提升不是来自更多计算 |
| **Relaxed same-budget** | 总 LLM tokens ≤ ReasonRAG 的 1.5x | 允许少量额外开销 |
| **Retrieval same-budget** | retrieval_topk=5, max_steps=3 (与 ReasonRAG 相同) | 只控制检索预算 |

固定参数（所有实验）：
```yaml
retrieval_topk: 5
max_steps: 3
max_tokens: 256
use_reranker: False
generator: Qwen2.5-7B-Instruct
retriever: BGE-base + FAISS
```

### 5.2 必须的 Baseline

| Baseline | 为什么必须 | 对比意义 |
|---------|----------|---------|
| ReasonRAG (原始) | 核心 baseline | 整体提升 |
| ReasonRAG + extra LLM call (同等 token budget) | 控制计算量 | 证明提升不是来自更多 LLM 调用 |
| S2G-RAG faithful reimplementation | 最直接竞品 | 和 S2G 在 HotpotQA 上正面比较 |
| Post-hoc claim verification | 证明 transition-time > post-hoc | 我们的 claim gate 是在线的 |
| Vanilla iterative RAG | 下界 | 确认 ReasonRAG baseline 有效 |
| ReasonRAG + structured state only (no typed reward) | 消解 state 本身的贡献 | 是 state 带来的，还是 typed evaluation 带来的？ |

**关于 S2G-RAG baseline**：审稿人明确要求 faithful reimplementation，不能用"S2G-like controller"模糊带过。S2G-RAG 的核心模块（Evidence Memory, Gap Items, Sufficiency Judging, Gap-to-Query Mapping）需要在 ReasonRAG 框架内忠实复现。

### 5.3 消融实验

| 实验 | 验证什么 | 预期 |
|------|---------|------|
| Full vs 去掉 typed evaluation（用回 scalar UCB1） | typed evaluation 的贡献 | **核心消融** |
| Full vs 去掉 directed expansion（扩展策略不区分 failure type） | directed expansion 的贡献 | **核心消融** |
| Full vs 去掉 provenance（claim 不带 evidence_refs） | provenance 的价值 | |
| Full vs 去掉 $\phi_q$ | query reward 的贡献 | |
| Full vs 去掉 $\phi_c$ | claim reward 的贡献 | |
| Full vs 去掉 $\phi_s$ | stop reward 的贡献 | |
| Full vs 去掉 multi-evidence provenance（退化为单句） | multi-evidence 的价值 | |
| Typed-UCB vs Bottleneck-UCB vs 加权 UCB | UCB 策略对比 | |
| Prompt 版 vs Trained 版 | 训练的价值 | |
| Oracle State（用 gold SF 文档标题构造"检索完美"条件） | 系统上限 | |
| Oracle Reward（用文档级 gold 覆盖率 + 最终 EM/F1 计算） | reward 质量上限 | |
| Noisy State（对 state 加入可控噪声） | state quality sensitivity | |

### 5.4 必须的指标

**最终答案**: EM, F1

**过程质量**:
- premature stop rate
- unsupported claim rate
- query repetition rate
- bridge entity preservation rate
- claim provenance accuracy（claim → evidence_refs 是否正确，需人工标 or gold SF 校验）
- average retrieval steps
- total LLM tokens / latency

**Typed Evaluation 专属（核心差异化指标）**:
- failure attribution accuracy：系统判断的 failure type 是否和人工标注一致
- typed vs scalar branch selection agreement：typed evaluation 和 scalar PRM 选的分支有多不同
- directed expansion success rate：按 failure type 扩展后，trajectory 是否改善
- error attribution coverage：能被 typed evaluation 归因的错误占总错误的比例

### 5.5 关键对比实验

**实验 A：Typed vs Scalar Branch Selection**

```
目的：证明 typed evaluation 在 MCTS 中的 branch selection 优于 scalar

方法：
1. 在同一 MCTS 树上，分别用 typed evaluation 和 scalar PRM 选择 top-1 分支
2. 比较两种选择下的最终答案 EM/F1
3. 统计"typed 选了 scalar 没选的分支"的比例

关键问题：typed evaluation 选的分支是否比 scalar 选的更好？
如果 typed 和 scalar 总是选同一条分支 → typed evaluation 没有实际价值
如果 typed 选的分支确实更好 → 核心贡献成立
```

**实验 B：Directed vs Undirected Expansion**

```
目的：证明 failure-attributed directed expansion 优于 uniform expansion

方法：
1. Fixed total expansion budget（如 max_children=3）
2. 对比：directed expansion（按 failure type 扩展）vs uniform expansion（随机扩展）
3. 比较 EM/F1 和过程指标

关键问题：知道"为什么失败"是否帮助"怎么扩展"？
```

**实验 C：Cost-Quality Pareto**

```
目的：证明提升不是来自更多计算

方法：
1. x 轴：总 LLM tokens per trajectory
2. y 轴：EM / F1
3. 画三条曲线：ReasonRAG, TrajectoryClosure, ReasonRAG + extra LLM calls
4. 如果 TrajectoryClosure 的曲线在 ReasonRAG 上方 → 同等计算下更高效
```

---

## 6) Novelty Positioning

### 核心贡献（一句话）

> 我们提出 typed transition evaluation，将 MCTS 中的节点评估从标量 reward 升级为 failure-attributed branch selection，实现基于 failure type 的差异化扩展策略。整个方法不依赖轨迹级金标准标注——$\phi_q$ 纯计算、$\phi_c$ 依赖 NLI、$\phi_s$ 依赖结构化状态——从数据集已有的 final-level gold 中提取过程级控制信号。

### 三个明确贡献（不贪多）

1. **Typed transition evaluation**: 将 process reward 从标量分解为 query/claim/stop 三个可独立计算、可归因的维度，每个维度有形式化定义
2. **Failure-attributed branch selection**: 在 MCTS tree search 中，利用 typed evaluation 实现基于 failure type 的分支选择和差异化扩展（query_fail → rewrite, claim_fail → supplement, stop_fail → force continue）
3. **Evidence-grounded repair policy**: 每个 repair 动作绑定到具体的 failure type 和具体的 state 字段（gap description, unsupported claim, bridge entity），可审计、可训练

### 和 S2G-RAG 的区别（最终版）

**S2G-RAG**: 线性 sufficiency controller（单路径，判断"够不够"）
**TrajectoryClosure**: MCTS failure-attributed search controller（多分支，判断"哪个有前途、怎么走"）

这不是"S2G 适配到 MCTS"，而是在 MCTS 中解决一个 S2G 不存在的问题：**多分支间的 typed credit assignment**。

### 论文 Story

```
§1 Introduction
   Agentic RAG 的数据困境：数据集只提供 final-level gold（答案 + supporting facts），
   但过程控制需要 trajectory-level supervision（每步的 query/gap/claim/stop 质量）
   → 现有方法（ReasonRAG 等）用标量 step PRM，无法归因失败类型
   → MCTS 中两个失败原因完全不同的分支得到相同分数
   → MCTS 无法做出 informed branch selection
   → 引出 typed transition evaluation 的动机：
      从 final-level gold 提取过程级控制信号，不依赖轨迹级金标准

§2 Related Work
   §2.1 Process reward for Agentic RAG (ReasonRAG/ProRAG/HiPRAG/Search-P1)
       → 都是 scalar，无法归因
   §2.2 Sufficiency and gap tracking (S2G-RAG/SURE-RAG/PAR²-RAG)
       → 线性 pipeline，无 branch comparison
   §2.3 Claim verification (RAGChecker/MedRAGChecker)
       → post-hoc 诊断，非 transition-time gate
   Gap: 无人把 failure-attributed evaluation 引入 MCTS branch selection，
        也无人从 final-level gold 中提取不依赖轨迹标注的过程信号

§3 Method
   §3.1 Trajectory State with multi-evidence provenance
   §3.2 Typed Transition Evaluation (φ_q, φ_c, φ_s) — formal definitions
   §3.3 Failure Attribution and Directed Expansion Strategy
   §3.4 Typed-UCB for MCTS Backpropagation
   §3.5 State Update: Extract-Validate-Merge

§4 Experiments
   §4.1 Main Results (HotpotQA/2Wiki/MuSiQue/Bamboogle)
   §4.2 Typed vs Scalar Branch Selection (Experiment A)
   §4.3 Directed vs Undirected Expansion (Experiment B)
   §4.4 Cost-Quality Pareto Analysis (Experiment C)
   §4.5 Ablation (provenance, φ_q, φ_c, φ_s, UCB variants)
   §4.6 Error Attribution Analysis
   §4.7 Oracle and Sensitivity Analysis

§5 Discussion
   oracle analysis, limitation, future work
```

---

## 7) Risk Assessment

| 风险 | 级别 | 缓解 |
|------|------|------|
| 7B 模型无法可靠输出 multi-evidence JSON | 高 | Phase 1 先验证；降级为单句 provenance 仍有 typed evaluation 价值 |
| Typed evaluation 和 scalar PRM 选的分支一样 | 高 | 实验 A 专门测这个；如果一样 → typed evaluation 没有实际价值，核心贡献崩塌 |
| S2G-RAG 抢先发表 | 中 | 我们的贡献是 MCTS branch selection，S2G 不做 |
| Extract 阶段 gap 漏检导致 $\phi_s$ 误判 | 中 | Oracle State 实验测上限；用 gold SF 文档标题做自动 sanity check；$\phi_s$ 的 -1 判断不依赖缺口（只要 claim 无 provenance 就为负） |
| 额外 LLM 调用导致 unfair comparison | 中 | 实验 C 的 cost-quality Pareto + strict same-budget |
| $\kappa$（minimum claim floor）设不好 | 低 | 作为超参数在验证集上调 |
| AAAI deadline 时间紧 | 中 | Gate 0-2 共 2 周出信号；核心实验 A（typed vs scalar）3 天可做 |

---

## 8) Timeline

```
2026-06-01 ~ 06-07: Gate 0-1
  → Prompt 版 Trajectory State + Typed Evaluation
  → 50 条 HotpotQA, 重点看 Experiment A (typed vs scalar branch selection)
  → 这是核心 Go/No-Go: 如果 typed 和 scalar 选的分支一样 → 止损

2026-06-08 ~ 06-14: Gate 2
  → 200 条验证 + 数据积累
  → Experiment B (directed vs undirected expansion)
  → Experiment C (cost-quality Pareto)

2026-06-15 ~ 06-28: Gate 3
  → SFT 训练 + 500 条实验

2026-06-29 ~ 07-12: Gate 4
  → 全量实验 + 消融 + S2G-RAG baseline

2026-07-13 ~ 07-27: 写作 + 打磨

2026-07-28: AAAI 2027 submission
```

---

## 9) v4 修改清单（对应审稿人六大致命问题）

| 审稿人问题 | v4 如何回应 |
|-----------|-----------|
| Incremental over S2G-RAG | 核心贡献重新定位为 **failure-attributed branch selection**，S2G 在线性 pipeline 中不存在 branch comparison 问题 |
| Reward heuristic | $\phi_q, \phi_c, \phi_s$ 都有形式化数学定义，$\phi_q$ 只依赖 NER + set op + Jaccard，不依赖 LLM judge |
| State correctness assumed | 新增 Oracle State / Oracle Reward / Noisy State 三组实验，用 gold SF 自动校验 state quality |
| Compute unfairness | 新增 same-inference-budget 规则（total LLM tokens），新增 Experiment C (cost-quality Pareto) |
| Provenance 太浅 | evidence_refs 从单句改为 **Set[doc_id::sent_id]**，支持多句联合支持 |
| R_claim gaming | 新增 $\kappa$ minimum claim floor + $\epsilon$ Laplace smoothing，不生成 claim 时得分趋近 0 |
| 贡献不清晰 | 收窄为三个明确贡献（typed evaluation / directed expansion / evidence-grounded repair） |
| 缺 S2G faithful reimplementation | baseline 中明确要求 S2G-RAG faithful reimplementation |
| 未分析数据集实际格式 | 新增 §1.5 明确：gold 仅到文档标题级，$\phi_q$ 不依赖任何标注，过程标签全部自构造，训练数据过滤用文档标题覆盖率而非句子级对齐 |
| 训练路线不清晰 / 循环监督 | **训练路线重构为 SOTA 蒸馏 + SFT + RL**：用 GPT-4 级教师生成轨迹级标签，7B 学生做 SFT 模仿学习，RL 自我探索超越教师示范。消除循环监督，符合成熟范式 |
