# 面向 ReasonRAG 方向的核心 Agentic RAG 文献谱系

本文件只保留与本课题任务最直接相关的核心方法：Search-R1、ReasonRAG、DecEx-RAG、HiPRAG、ProRAG、Search-P1。它们都围绕 Agentic RAG / retrieval-augmented reasoning 中的搜索行为、过程奖励、路径奖励或搜索停止决策展开，并且实验任务包含多跳问答或复杂开放域问答。

## 1. 方向主线

当前核心主线可以概括为：

```text
Outcome reward
  -> process reward
  -> hierarchical / path-centric reward
  -> state-aware query/evidence/stop process optimization
```

其中前三步是现有方法已经覆盖的部分，最后一步是本课题拟进一步推进的方向。

## 2. Outcome-RL 基线

### Search-R1

- 类别：Outcome-RL Agentic RAG。
- 贡献：证明 LLM 可以通过强化学习学会在推理过程中主动生成搜索查询，并利用检索结果回答问题。
- 局限：主要使用最终答案奖励，训练信号稀疏，难以定位错误来自 query、retrieval、evidence extraction 还是 stop decision。
- 对本课题作用：作为“只看终答奖励不够”的对照基线。

## 3. Process-Supervised Agentic RAG 基线

### ReasonRAG

- 类别：Process-Supervised Agentic RAG。
- 贡献：提出 RAG-ProGUIDE，为 query generation、evidence extraction、answer generation 构造过程级奖励。
- 局限：过程监督已经比 outcome reward 更细，但从现有 badcase 看仍存在 query drift、重复或复杂子查询、证据链断裂、检索噪声、提前停止等问题。
- 对本课题作用：代表性 baseline、问题观察入口、实验验证平台。

## 4. 决策/执行过程优化

### DecEx-RAG

- 类别：Decision and execution optimization via process supervision。
- 贡献：将 Agentic RAG 建模为 decision / execution MDP，通过过程监督优化不同阶段的行为。
- 局限：它证明了拆分决策和执行有价值，但仍没有把 query、evidence、stop 的错误类型细化为可解释的状态感知反馈。
- 对本课题作用：提供 MDP 视角，可作为本课题定义状态 `s_t` 和动作反馈的理论参照。

## 5. 搜索必要性与停止控制

### HiPRAG

- 类别：Hierarchical process rewards。
- 贡献：显式关注 Agentic RAG 中的 over-search 与 under-search，用层级过程奖励判断是否需要继续搜索。
- 局限：强在“是否搜索”的过程控制，但弱在“当前状态下搜什么”和“候选证据是否真正推进证据链”。
- 对本课题作用：直接支撑 Stop Reward 的必要性，同时暴露 Evidence Reward 与 Query Reward 的缺口。

## 6. 过程监督强化学习

### ProRAG

- 类别：Process-supervised reinforcement learning。
- 贡献：结合 SFT warmup、MCTS-based PRM、PRM-guided refinement 和 process-supervised RL 优化 RAG 过程，缓解 process hallucination。
- 局限：PRM 仍可能给出整体步骤分数，未必区分 query 错、evidence 错、stop 错或 unsupported claim。
- 对本课题作用：可作为过程奖励模型的强相关对照；本课题需要进一步拆分奖励头并引入状态感知证据效用。

## 7. 路径级奖励塑形

### Search-P1

- 类别：Path-centric reward shaping。
- 贡献：从失败样本中提取训练信号，通过路径级奖励改善 Agentic RAG 训练的稳定性和 credit assignment。
- 局限：路径级 reward 能缓解稀疏反馈，但仍需要进一步回答每一步 query 是否对准缺口、每篇证据是否支撑当前推理、当前是否应该停止。
- 对本课题作用：作为 trajectory-level reward 对照；本课题可将 path reward 拆解为 Query Reward、Evidence Reward、Stop Reward。

## 8. On-Policy Distillation 与 Agentic RAG

### GKD

- 类别：通用 On-Policy Distillation 基础。
- 贡献：让 student 在自身生成序列上学习 teacher 的 token distribution，缓解离线 KD 的 train-inference mismatch。
- 局限：ICLR 2024 原论文不研究 Agentic RAG；直接迁移到弱搜索 Agent 时，低质量 student trajectory 仍会污染训练上下文。
- 对本课题作用：提供 OPD 的标准定义、on-policy 数据来源和 divergence 选择依据。

### DGPO

- 类别：Selective On-Policy Distillation for Agentic RAG。
- 贡献：先以正确教师轨迹做 KD 冷启动，再让 student 在线检索；正确轨迹使用 RL reward，错误轨迹才接受外部 teacher KL。
- 结果：ACL 2026 主会长文；Qwen2.5-3B→0.5B 设置下 HotpotQA EM 0.342，七个 QA 数据集平均 EM 0.329，优于 PPO 0.238、GKD 0.240 和普通 KD 0.298。
- 局限：只报告 EM，主要验证 0.5B/1B student，teacher KL gate 是整轨迹级。
- 对本课题作用：从当前 privileged self-distillation 转向外部强 teacher OPD 时的首要复现基线。

### EOPD

- 类别：Entropy-Aware On-Policy Distillation。
- 贡献：在 teacher 高熵 token 上加入 top-k forward KL，缓解纯 reverse KL 的 mode collapse 和训练不稳定。
- 局限：ICML 2026 实验集中于数学推理，没有 Agentic RAG 证据。
- 对本课题作用：可用于 `<query>/<answer>` 行为分叉位置的 teacher entropy 审计和 KL 目标改进。

### SCoRe

- 类别：Student-Centered Agent Distillation。
- 贡献：teacher 只修 student 轨迹的最早错误，再从 verified prefix 做短程 RL，缩短长轨迹 credit assignment。
- 局限：不是严格 token-level OPD，teacher 标注和修正成本较高。
- 对本课题作用：可把 DGPO 的“失败轨迹选择性纠偏”进一步细化为首错 query/stop 后的局部纠偏。

完整证据边界与复现建议见：

```text
01_literature/related_work_drafts/opd_agentic_rag_survey.md
01_literature/paper_notes/2026_DGPO.md
```

## 9. 当前归纳出的两类问题

### 9.1 多步推理过程控制能力不足

核心表现：

- query 生成不稳定，可能重复、过宽、过复杂或合并多个 hop；
- bridge entity 在多跳转移中容易丢失；
- 中间结论缺少证据校验，存在 unsupported intermediate answer；
- 搜索停止决策不可靠，存在 premature stop、over-search、under-search；
- 过程奖励仍偏粗，错误来源难以明确归因到 query / evidence / stop。

对应文献支撑：

- Search-R1 体现 outcome reward 的稀疏性；
- ReasonRAG 说明 process reward 有效，但 badcase 暴露其过程控制仍不稳定；
- DecEx-RAG、ProRAG、Search-P1 都从不同角度试图缓解过程反馈和 credit assignment 问题；
- HiPRAG 明确关注 over-search / under-search。

### 9.2 状态感知证据利用不足

核心表现：

- 现有方法大多关注轨迹、步骤或搜索必要性，但没有充分判断“当前状态下哪篇证据真正有用”；
- 检索结果可能包含噪声，正确证据即使被召回也未必被使用；
- 文档是否提供新信息、是否支撑中间结论、是否推进证据链、是否引入噪声风险，尚未成为独立的状态感知奖励对象；
- 现有 process reward 往往没有显式建模 `U(d | q, s_t)`。

对应文献支撑：

- ReasonRAG 的 evidence extraction reward 是重要起点，但仍没有显式建模状态条件下的证据边际效用；
- HiPRAG 强在是否搜索，而不是搜索得到的文档如何按当前状态排序；
- ProRAG 和 Search-P1 强调过程/路径奖励，但 evidence utility 仍容易被整体 reward 吸收；
- DecEx-RAG 提供 MDP 视角，但没有直接展开 state-aware evidence utility。

## 10. 本课题定位

本课题拟在 ReasonRAG 类 process-supervised Agentic RAG 基础上，进一步提出状态感知过程优化机制：

```text
State s_t
  -> Query Reward: 当前 query 是否对准未解决缺口
  -> Evidence Reward: 当前文档是否提供新信息、支撑中间结论、推进证据链
  -> Stop Reward: 当前证据是否足够支撑最终答案
  -> Repair Mechanism: 低质量步骤触发重写、重排、继续检索或拒绝无支撑结论
```

其目标不是单纯提出一个新的 reranker，也不是只改 query rewriting，而是将多步推理过程控制和状态感知证据利用统一起来，修复 ReasonRAG 类方法在复杂多跳问答中的关键失败模式。
