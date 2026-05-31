# Research Brief: 面向复杂问答的 Agentic RAG 多步检索推理过程优化

## 1. Problem Statement

多跳复杂问答（HotpotQA, 2Wiki, MuSiQue, Bamboogle）中的 Agentic RAG 系统在多步检索推理过程中存在严重的轨迹控制问题，导致最终答案质量远低于预期。通过 ReasonRAG baseline 复现和 badcase 分析，已确认以下核心问题：

- **Query drift / redundancy**: 子查询漂移、重复、过宽或合并多个 hop
- **Entity loss**: bridge entity 在跨跳推理中丢失
- **Noisy evidence**: 检索结果被同名实体或相似事件污染
- **Missing evidence supervision**: 缺少对"当前状态下哪篇文档最有用"的建模
- **Premature stopping**: 证据不足时过早停止检索
- **Process control instability**: 步骤级错误传播，难以定位具体失败来源

## 2. Constraints

- **Baseline**: ReasonRAG-LoRA (Qwen2.5-7B + LoRA, vLLM)
- **Retriever**: BGE-base, FAISS flat index
- **Datasets**: HotpotQA / 2Wiki / MuSiQue / Bamboogle
- **Compute**: 3×RTX 5090 (rag-5090) + 4×RTX 3090 (local)
- **Timeline**: 中期报告 2026-06-30, AAAI 2027 submission 2026-07-28
- **Staged compute policy**: do not jump directly into large-scale GRPO/online RL before small-scale verification, but SFT/RL/PRM/DPO/GRPO are allowed and should be considered when they are the fastest credible route to a novel and feasible method.

## 3. What We Already Tried

### 3.1 SAPR-E v0 Evidence-only (Under diagnosis)

- 实现 heuristic 5-dim scorer (relevance, novelty, supportiveness, chain_contribution, noise_risk)
- Evidence-only 信号在 200-sample 上稳定：+4.1pp item hit vs retriever baseline
- 早期端到端结果显示 EM 不变、F1 降 1.1pp，但该结论已被诊断为配置错误导致的无效实验
- 根因定位：实验配置 `max_tokens=32`（应为 256），模型输出被截断，文档从未注入 prompt
- **当前状态**：不要把 SAPR-E 端到端无效当作定论；应在修正 `max_tokens=256`、确认文档注入后再判断是否继续、降级或转向

### 3.2 现有 SAPR-RAG Proposal (refine-logs/FINAL_PROPOSAL.md)

- State-conditioned progress / action-value modeling
- 三模块：Query Reward, Evidence Reward, Stop Reward + Repair Controller
- 已经过 research-refine-pipeline 多轮打磨
- 但尚未实施端到端实验验证

## 4. Literature Already Surveyed

已有 literature survey (01_literature/literature_survey.csv)，核心同类方法：

1. **Search-R1**: outcome reward → process reward 转型代表
2. **ReasonRAG**: MCTS + process reward，我们的 baseline
3. **DecEx-RAG**: MDP 分解/执行
4. **HiPRAG**: over/under-search 控制
5. **ProRAG**: learned PRM + search/RL
6. **Search-P1**: 路径奖励塑形

## 5. Domain Knowledge & Non-goals

- 本课题不是单一修改 ReasonRAG，而是从 Agentic RAG 共性问题出发提出优化
- Non-goals: 不把 prompt judge 作为最终系统；不靠不公平的额外检索/tokens 获得增益；不在缺少小规模验证和稳定数据/奖励管线时盲目投入大规模 GRPO/online RL
- Clarification: 这不是禁止 SFT/RL。若 idea 足够新颖、可行，并且早期实验显示值得投入，SFT、DPO、PRM 训练、GRPO/online RL 和更大算力都可以作为主线方案。
- ReasonRAG batch mode 存在 pipeline routing bug（max_tokens 配置 + flag 路由），tree mode 设计更合理
- 7B 模型对检索文档的利用能力有限，需要考虑模型本身的能力边界

## 6. Key Question for Idea Discovery

**用户刚刚搁置了 evidence-only 方向，希望重新探索 Agentic RAG 多步检索推理过程优化的新方向。**

需要回答的核心问题：
1. 除了 evidence re-ranking，还有哪些高潜力的优化切入点？
2. Query reward / Stop reward 单独做是否有足够的信号？
3. 是否存在更根本的范式创新（不只是给现有 pipeline 打补丁）？
4. 在当前 7B 模型 + 3×5090/4×3090 起步资源下，哪些方向最快能出验证信号？如果 idea 足够新颖和可行，哪些方案值得申请/调度更多算力继续推进？
