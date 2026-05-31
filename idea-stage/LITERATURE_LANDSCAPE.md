# Literature Landscape: Agentic RAG 多步检索推理过程优化

**日期**: 2026-05-31
**搜索范围**: 8 轮 arXiv API + 项目已有 6 篇核心论文 + Semantic Scholar（rate limited）
**覆盖子方向**: process reward, trajectory optimization, evidence gap, adaptive retrieval, PRM, MCTS, self-reflective RAG, graph-based reasoning
**总计**: 30 篇高相关论文

---

## 一、按方向的文献全景

### 方向 A: Agentic RAG 多步检索推理（核心方向）

| # | 论文 | 年份 | 来源 | 核心贡献 | 与本课题关系 |
|---|------|------|------|---------|------------|
| A1 | **FAIR-RAG** (2510.22344) | 2025-10 | arXiv | SEA 模块：gap checklist → 迭代补检索，HotpotQA F1=0.453 (+8.3pp) | ⭐⭐⭐ 最直接竞争者，gap analysis ≈ Stop Reward |
| A2 | **PRISM** (2510.14278) | 2025-10 | arXiv | 3-agent 架构（Question Analyzer + Retrieval Agent + Answer Verifier） | ⭐⭐⭐ 三模块分工类似 SAPR 三头 |
| A3 | **AgenticRAGTracer** (2602.19127) | 2026-02 | arXiv | 首个 hop-aware 诊断 benchmark，GPT-5 仅 22.6% EM | ⭐⭐⭐ 验证我们的 badcase observation |
| A4 | **ReaLM-Retrieve** (2604.26649) | 2026-04 | arXiv | 推理过程中何时检索的自适应决策 | ⭐⭐ 检索时机 ≈ Stop/Query Reward |
| A5 | **Adaptive Retrieval for Reasoning** (2601.04618) | 2026-01 | arXiv | Bridge document 检索，确保推理链覆盖 | ⭐⭐ 证据链连续性 |
| A6 | **Vendi-RAG** (2502.11228) | 2025-02 | arXiv | Diversity-quality tradeoff，HotpotQA +4.2% | ⭐⭐ 多样性 = evidence selection 维度 |
| A7 | **CARROT** (2411.00744) | 2024-11 | arXiv | 成本约束检索优化，MCTS + chunk utility 非单调 | ⭐⭐ compute-matched 思路相似 |
| A8 | **ToR** (2404.14464) | 2024-04 | arXiv | 树结构动态检索，独立推理路径降低噪声 | ⭐ tree 结构 vs chain 结构 |
| A9 | **IGMiRAG** (2602.07525) | 2026-02 | arXiv | 直觉引导 + 自适应深度挖掘 | ⭐ depth control |
| A10 | **CogPlanner** (2501.15470) | 2025-01 | arXiv | 认知规划驱动的 Agentic RAG | ⭐ planning for RAG |

### 方向 B: Process Reward Models (PRM)

| # | 论文 | 年份 | 核心贡献 | 与本课题关系 |
|---|------|------|---------|------------|
| B1 | **R-PRM** (2503.21295) | 2025-03 | 推理驱动的 PRM，不只打分还诊断错误类型 | ⭐⭐ 错误诊断 ≈ Failure Bank |
| B2 | **GroundedPRM** (2510.14942) | 2025-10 | Tree-guided + fidelity-aware PRM，可扩展标注 | ⭐⭐ tree-guided reward |
| B3 | **ReasonFlux-PRM** (2506.18896) | 2025-06 | 轨迹感知 PRM，针对长 CoT 的中间步骤评估 | ⭐⭐⭐ trajectory-aware ≈ state-conditioned |
| B4 | **Error-Aware Hierarchical PRM** (2505.19706) | 2025-05 | 错误类型分层的层级监督 | ⭐⭐ 错误分层 ≈ failure type taxonomy |
| B5 | **Hierarchical Multi-Step Reward** (2503.13551) | 2025-03 | 多步层级奖励，解决 reward hacking | ⭐⭐ 层级奖励设计 |
| B6 | **Step-DPO** (2406.12845) | 2024-06 | 步骤级偏好优化，长链推理的 DPO 变体 | ⭐⭐ step-level preference |

### 方向 C: Tree Search + Agent Planning

| # | 论文 | 年份 | 核心贡献 | 与本课题关系 |
|---|------|------|---------|------------|
| C1 | **LATS** (2310.04406) | 2023-10 | 统一 reasoning/acting/planning 的 tree search 框架 | ⭐⭐ ReasonRAG 的上游灵感 |
| C2 | **Tree Search for LM Agents** (2407.01476) | 2024-07 | LM agent 的树搜索，web automation 应用 | ⭐ 搜索策略 |
| C3 | **RAG-Star** (2412.12881) | 2024-12 | MCTS + query/answer-aware verification reward | ⭐⭐ MCTS + verification ≈ ReasonRAG 增强 |

### 方向 D: Self-Reflective / Adaptive RAG

| # | 论文 | 年份 | 核心贡献 | 与本课题关系 |
|---|------|------|---------|------------|
| D1 | **Self-RAG** (2310.11511) | 2023-10 | 学习检索/生成/批判的自反思机制 | ⭐⭐⭐ 基础工作，retrieval/critique token |
| D2 | **Self-MedRAG** (2601.04531) | 2026-01 | 自反思医学 RAG | ⭐ 领域应用 |
| D3 | **ReFeed** (2603.01417) | 2026-03 | 检索反馈引导的查询改写 | ⭐ query rewriting |

### 方向 E: Graph-based Retrieval Reasoning

| # | 论文 | 年份 | 核心贡献 | 与本课题关系 |
|---|------|------|---------|------------|
| E1 | **GRAIL** (2508.05498) | 2025-08 | 知识图谱上的检索增强推理 | ⭐ 结构化知识路径 |
| E2 | **UniKGQA** (2212.00959) | 2022-12 | 统一检索推理的 KG QA | ⭐ KG + multi-hop |

### 方向 F: Benchmark / Evaluation

| # | 论文 | 年份 | 核心贡献 | 与本课题关系 |
|---|------|------|---------|------------|
| F1 | **AgenticRAGTracer** (2602.19127) | 2026-02 | hop-aware 诊断 benchmark | ⭐⭐⭐ |
| F2 | **Evaluating Multi-Hop Reasoning** (2604.18234) | 2026-04 | RAG 多跳推理评估 | ⭐ 评估方法 |
| F3 | **Hard2Verify** (2510.13744) | 2025-10 | 开放式数学的步骤级验证 benchmark | ⭐ verification benchmark |

### 方向 G: Survey

| # | 论文 | 年份 | 核心贡献 |
|---|------|------|---------|
| G1 | **Agentic Reasoning Survey** (2601.12538) | 2026-01 | 全面综述：foundational → self-evolving → multi-agent |

---

## 二、项目已有 6 篇核心方法定位

| 方法 | 核心机制 | 过程监督粒度 | 状态感知 | 与 SAPR-RAG 差异 |
|------|---------|------------|---------|----------------|
| ReasonRAG | MCTS + step PRM | 步骤级 | 弱 | SAPR 用 state-conditioned Q 替代 step PRM |
| Search-R1 (2503.09516) | Outcome RL → Process RL | 轨迹→步骤 | 无 | SAPR 不做 online RL，offline critic |
| DecEx-RAG | MDP 分解/执行 | 动作级 | 部分 | SAPR 是 plug-in critic |
| HiPRAG | Over/under-search 控制 | 停止级 | 部分 | SAPR 统一 query/evidence/stop |
| ProRAG (2601.21912) | Learned PRM + search | 步骤级 | 无 | SAPR 强调 state features ablation |
| Search-P1 | 路径奖励塑形 | 路径级 | 无 | SAPR 用模块化三头 + repair |

---

## 三、核心 Gap 分析（6 个方向）

### Gap 1: 无人在 explicit trajectory state 上做 action-value modeling

- 现有方法：step PRM (B1-B6), 启发式控制 (A1, A7), tree search (C1-C3)
- 缺失：显式定义 s_t（history queries, evidence, gap, budget）→ Q(s_t, a) → 排序候选动作 → state ablation 验证
- ReasonFlux-PRM (B3) 最接近（trajectory-aware），但针对 math CoT，不针对 retrieval reasoning

### Gap 2: 诊断驱动的方法设计未形成闭环

- AgenticRAGTracer (F1) 做了 hop-level 诊断，但没有基于诊断设计修复
- FAIR-RAG (A1) 做了 gap-based 修复，但没有系统化的诊断→修复 pipeline
- PRISM (A2) 做了三模块分工，但没有 failure bank 驱动的模块设计
- **没有 "badcase taxonomy → failure bank → module design → repair" 的完整闭环**

### Gap 3: 三模块解耦 + compute-matched repair

- PRISM (A2) 有三模块但耦合在一起
- FAIR-RAG (A1) 的 gap analysis 把 query + stop 合在一起
- 没有人做 Query/Evidence/Stop 独立建模 + 独立消融 + 固定计算预算修复

### Gap 4: Error-type-aware reward for RAG

- Error-Aware PRM (B4) 做了错误分层，但针对 math reasoning
- R-PRM (B1) 做了错误诊断，但也是 math
- **没有人把 error-type-aware reward 应用到 Agentic RAG 的 query/evidence/stop 失败类型上**

### Gap 5: Retrieval timing decision during reasoning

- ReaLM-Retrieve (A4) 做了推理过程中的检索时机决策，但针对 long reasoning models (DeepSeek-R1)
- FAIR-RAG (A1) 用 gap detection 隐式控制检索时机
- **没有人显式建模 "当前状态下的检索边际收益" 作为可训练的 stop/query reward**

### Gap 6: Evidence chain continuity verification

- Adaptive Retrieval for Reasoning (A5) 关注 bridge documents
- CARROT (A7) 关注 chunk utility 的非单调性
- **没有人显式验证多跳推理中 evidence chain 的连续性（bridge entity 是否闭合）**

---

## 四、竞争风险评估

| 风险来源 | 论文 | 抢先程度 | 我们的区别 |
|---------|------|---------|-----------|
| Gap analysis + iterative | FAIR-RAG | ⚠️ 高 | 我们做 state Q + 三模块解耦 + offline critic + state ablation |
| 三模块分工 | PRISM | ⚠️ 中 | PRISM 是三 agent 级联，不是 critic + repair |
| Process reward | ReasonFlux-PRM | ⚠️ 中 | trajectory-aware but for math, not retrieval |
| MCTS + verification | RAG-Star | ⚠️ 低 | 我们不做 MCTS，做 lightweight critic |
| Self-reflective retrieval | Self-RAG | 基础工作 | Self-RAG 是 tokenize-level，我们是 trajectory-level |
| Error-type reward | Error-Aware PRM | ⚠️ 中 | for math, not RAG |

**时间窗口**：FAIR-RAG (2025-10), PRISM (2025-10), AgenticRAGTracer (2026-02), ReaLM-Retrieve (2026-04) — 方向非常活跃，但我们的切入角度（state-conditioned Q + 诊断驱动 + 三模块解耦）仍有独特空间。

---

## 五、最有潜力的发展方向（初步排序）

1. **诊断驱动的 Trajectory Repair** — badcase taxonomy → failure bank → 模块化 repair
2. **State-conditioned Action-value for Retrieval Reasoning** — 显式 Q(s_t, a) 替代 step PRM
3. **Evidence Chain Verification** — 多跳推理中的 bridge entity 闭合性检查
4. **Retrieval Timing as a Learnable Decision** — 检索边际收益的可训练 stop/query reward
5. **Error-type-aware RAG Reward** — 借鉴 Error-Aware PRM 的思路到 retrieval reasoning

---

## 六、搜索日志

| 轮次 | 搜索关键词 | 命中数 | 去重后新增 |
|------|-----------|-------|----------|
| R1 | agentic RAG process reward multi-hop reasoning | 10 | 6 |
| R2 | retrieval augmented reasoning trajectory optimization state-aware | 10 | 2 |
| R3 | evidence gap detection stop decision query decomposition | 10 | 3 |
| R4 | process reward model RL retrieval search | 10 | 3 |
| R5 | adaptive retrieval planning decompose query reformulation | 15 | 4 |
| R6 | PRM math reasoning step-level verification | 15 | 6 |
| R7 | MCTS retrieval augmented generation document selection | 15 | 1 |
| R8 | agent tool use planning reward shaping RL LLM | 15 | 1 |
| R9 | multi-hop CoT retrieval reasoning hallucination | 15 | 4 |
| R10 | search agent RL tool use web search information seeking | 15 | 1 |
| R11 | iterative retrieval generation interleaved | 15 | 1 |
| R12 | Self-RAG FLARE adaptive retrieval critique | 15 | 3 |
| R13 | DPO reward model preference learning reranking | 10 | 1 |
| R14 | graph-based retrieval reasoning knowledge graph multi-hop | 10 | 1 |
| R15 | LATS language agent tree search | 3 | 1 |
| R16 | GRAIL knowledge graph retrieval reasoning | 3 | 1 |
| R17 | error-aware reward hierarchical process model | 5 | 2 |

**Sources contributed**: arxiv (17 searches), local (6 paper notes), web (API errors, 0 contribution)
