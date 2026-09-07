# GRPO 改进与长程 Agent 信用分配系统调研

**调研日期**：2026-09-06
**目标**：为 SAPR-RAG 的 ICASSP 方向二判断创新空间，区分 GRPO 通用优化技巧、
长推理信用分配、长程 Agent 信用分配和 Agentic RAG 专用过程监督。
**检索范围**：ICML、NeurIPS、ICLR、COLM 官方 proceedings/OpenReview，
辅以 arXiv 与官方代码仓库。
**证据规则**：正式录用论文与预印本分开；论文方法和 venue 以一手页面为准。

---

## 0. 结论先行

1. **“GRPO 给整条轨迹所有 token 广播同一个 advantage，长程任务需要
   turn/step-level credit assignment”已经不是新问题。**
   ICML 2024 的 ArCHer、ICML 2025 的 VinePPO、NeurIPS 2025 的 GiGPO，
   以及 ICLR/ICML 2026 的 Tree-GRPO、iStar、ΔBelief-RL、HPO、InfoPO、
   AREW、BEACON 已从不同角度解决该问题。

2. **“用中间进展差值作为 turn reward”也不能单独构成创新。**
   ΔBelief-RL 用目标答案置信度变化，InfoPO 用遮蔽反馈后的反事实信息增益，
   BEACON 用 milestone 进展，HiPRAG 用搜索必要性过程奖励。我们原计划的
   `新增证据覆盖率 = Φ(s_{t+1}) - Φ(s_t)` 与这条方法线高度相邻。

3. **GRPO 通用改进大致分为四类**：
   - advantage/归一化与长度偏置：Dr.GRPO、DAPO；
   - importance ratio、clipping 与聚合稳定性：GSPO、GMPO、RIPO；
   - 隐式 step value 或树状估计：VinePPO、GN-IVO、Tree-GRPO；
   - 长程 Agent 的 turn/step 信用：GiGPO、iStar、ΔBelief-RL、HPO、
     InfoPO、AREW、BEACON。

4. **对 SAPR-RAG 最直接的强基线不是 DAPO/GSPO，而是 GiGPO、Tree-GRPO、
   HiPRAG、ΔBelief-RL 和 HPO。**
   DAPO/Dr.GRPO/GMPO/RIPO 主要解决训练稳定性、长度偏置、clipping 或探索
   坍塌，不直接回答“哪一轮 Query 应得多少信用”。

5. **原先的 Action-Causal GRPO 方案需要收缩创新声明。**
   单纯把 Answer F1 分给 Answer token、把新增 evidence 分给 Query token，
   在当前文献环境中更像合理的 RAG 专用实例化，而不是足够独立的新算法。

6. **仍可能成立的创新空隙是“检索—证据吸收—答案利用”的可验证因果分解。**
   现有工作大多以 turn、树节点、belief change 或整体 search necessity 为单位，
   尚未系统拆开：

   ```text
   Query 是否取得新证据
       -> Evidence 是否被正确吸收进状态
       -> Answer 是否真正利用该证据
   ```

   该方向必须与 AREW 的 Action Selection/Belief Tracking、InfoPO 的反事实
   信息增益、ΔBelief-RL 的 belief potential 做严格区别。

---

## 1. 为什么 GRPO 的信用分配在长程任务中失效

标准 GRPO 对同一 prompt 采样 \(G\) 条完整轨迹，用终局 reward 构造：

\[
A_i = \frac{R_i-\mu_R}{\sigma_R+\epsilon}
\]

同一轨迹中每个 token 通常共享 \(A_i\)。这带来四个问题：

1. **时间归因不清**：早期正确动作会因后续失败被惩罚；
2. **动作类型混淆**：Query、工具调用、证据抽取和 Answer 接收相同信用；
3. **零方差组**：同题 rollout 全对或全错时，组内 advantage 为 0；
4. **环境反馈被压扁**：每轮 observation 的信息增益在最终标量中消失。

SAPR-RAG 的 B 实验直接观察到：

- `frac_reward_zero_std` 约为 0.20–0.28；
- F1 reward 上升，但 relevance 长期震荡；
- 当前累计 relevance 和旧 marginal relevance 最终都被压成 sequence scalar；
- reward-v3 虽逐轮计算新增 evidence，仍没有把该值保留到对应 Query token。

因此，项目问题不是“缺一个新的 reward 项”，而是：

> 中间信号在进入 GRPO 前被聚合成轨迹标量，因果位置丢失。

---

## 2. GRPO 通用改进：哪些是技巧，哪些是信用分配

### 2.1 基础与长度/归一化偏置

| 工作 | Venue | 核心改动 | 解决问题 | 是否直接解决长程信用 |
|---|---|---|---|---|
| [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) | arXiv 2024 | 用同题采样组替代 critic 构造相对 advantage | 降低 PPO critic 成本 | 否，轨迹内仍共享 advantage |
| [Dr.GRPO](https://openreview.net/forum?id=5PAF7PAY2Y) | COLM 2025 | 去掉 reward std 与长度归一化造成的偏置 | 错误长答案膨胀、问题难度加权偏差 | 否 |
| [DAPO](https://openreview.net/forum?id=2a36EMSSTp) | NeurIPS 2025 | Clip-Higher、动态采样、token-level PG loss、超长 reward shaping | 熵坍塌、无效 prompt、长度稳定性 | 部分；动态采样缓解零方差，但不定位动作 |

**对项目的意义**：

- Dr.GRPO 值得作为长度偏置控制，尤其本项目历史上出现过长输出退化；
- DAPO 的 dynamic sampling 可减少全同分组，但它是“过滤无信号问题”，不是
  “给同一失败轨迹内的不同 Query 分配不同信用”；
- 二者适合作为优化器消融，不应被包装成 Agentic RAG 方法创新。

### 2.2 Ratio、clipping 与聚合稳定性

| 工作 | Venue | 核心改动 | 主要目标 | 与本项目关系 |
|---|---|---|---|---|
| [GSPO](https://openreview.net/forum?id=8hdHzwGLWY) | CoRR 2025，未确认顶会录用 | sequence likelihood ratio 与 sequence clipping | MoE/大规模 RL 稳定性 | 可作工程参考，不是信用分配基线 |
| [GMPO](https://openreview.net/forum?id=nCEs0tSwc2) | ICLR 2026 Poster | 用几何均值抑制 token reward/ratio 异常值 | 更新稳定性 | 可与新 advantage 正交组合 |
| [RIPO](https://openreview.net/forum?id=nSRbKvrmsH) | ICML 2026 Regular | 用策略流形上的分布感知 clipping | 探索坍塌与 bias-variance | 不直接定位 Query 贡献 |

**判断**：这些方法改变“怎样更新”，不改变“哪个动作应得到什么信用”。如果
SAPR-RAG 只替换 clipping/ratio，论文会变成通用 GRPO 变体，任务针对性不足。

---

## 3. 单轮长推理中的信用分配

### 3.1 Monte Carlo prefix value

[VinePPO](https://openreview.net/forum?id=Myx2kJFzAn)（ICML 2025 Poster）
从中间 prefix 继续采样多个 continuation，以 Monte Carlo 成功率估计 token/step
value，替代训练不准的 value network。它在 MATH、GSM8K 上优于 PPO 等基线，
但额外 continuation 成本较高。

**对项目的映射**：

- 可从同一检索状态分叉多条 Query continuation，直接估计 Query value；
- 但每次分叉都触发 BGE+FAISS 和 Evidence Agent，成本显著高于数学 continuation；
- 不适合作为第一版实现，适合作为高质量上限。

### 3.2 隐式 step value

[GN-IVO](https://openreview.net/forum?id=eFXmrCun0c)（ICLR 2026 Poster）
通过组归一化的分布匹配从 policy 中隐式学习 step value，不需要显式 critic。
其理论主张是恢复 value 到一个加性常数，但评审仍质疑它是否真正实现了
within-trajectory credit，而非把序列权重传播到 prefix。

**对项目的启示**：论文必须用可观测的动作/状态指标验证“信用真的落到正确
Query”，不能只给新 advantage 公式和终点分数。

### 3.3 树结构与关键位置采样

- [AttnRL](https://openreview.net/forum?id=NCN8oUsiNf)（ICLR 2026 Poster）：
  用注意力选择值得分叉的推理位置，并动态分配采样预算，维持非零 advantage；
- [PS-GRPO](https://openreview.net/forum?id=96I8PGPALv)（NeurIPS 2025 Poster）：
  将 PRM 过程信号引入多模态数学 RL；
- [Dense Reward for Free](https://proceedings.mlr.press/v235/chan24a.html)
  （ICML 2024）：用 reward model attention 重新分配终局 reward，并证明与
  potential-based shaping 的联系。

这些工作说明“把 reward 变稠密”本身已非常拥挤，创新必须来自任务结构或更
可信的因果归因。

---

## 4. 多轮 Agent 的长程信用分配

### 4.1 分层 actor-critic

[ArCHer](https://proceedings.mlr.press/v235/zhou24t.html)
（ICML 2024）将语言 Agent 建模为层级 MDP：

- 高层 critic 在 turn 级学习长期 value；
- 低层策略在每个 utterance 内做 token 级更新；
- 用 TD 学习处理多轮延迟回报。

优点是理论结构清楚；缺点是需要额外 critic，系统复杂，和当前 critic-free
GRPO 架构差异较大。

### 4.2 同状态局部分组

[GiGPO](https://proceedings.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html)
（NeurIPS 2025）是与本项目最重要的基线之一：

- 宏观 advantage：完整轨迹组内比较；
- 微观 advantage：将重复出现的相同环境状态作为 anchor，对从同状态出发的
  不同动作重新分组；
- 无 critic、无额外 rollout；
- 在 ALFWorld、WebShop 和 search-augmented QA 上优于 GRPO。

**关键限制**：文本 Agent 的历史状态往往几乎不精确重复。论文通过相似状态分组
缓解，但阈值会带来“组规模—状态一致性”权衡。

**对 SAPR-RAG 的直接挑战**：

如果只做“同一问题内按 turn 分组”，很容易被视为 GiGPO 的弱化版。新的方法
必须说明为什么 RAG 的 evidence-gap 状态比 raw text anchor 更准确、更可验证。

### 4.3 树状 prefix/step credit

[Tree-GRPO](https://openreview.net/forum?id=ZpQwAFhU13)
（ICLR 2026 Poster）将 thought-action-observation 作为树节点，共享 prefix
进行分叉，并计算 intra-tree 与 inter-tree relative advantage。它在固定预算下
增加分支数量，并将 outcome reward 转成 step-level preference。

[TreePS-RAG](https://arxiv.org/abs/2601.06922)
（2026 arXiv，未确认顶会录用）进一步直接面向 Agentic RAG，用在线树结构和
descendant outcome 的 Monte Carlo value 构造 step-wise advantage。

**结论**：为每轮 Query 显式分叉并用后续成功率估值已经有直接先例。若采用树状
rollout，Tree-GRPO 和 TreePS-RAG 都必须成为主对照。

### 4.4 学习隐式过程 reward

[iStar](https://openreview.net/forum?id=ooROvpmxMV)
（ICLR 2026 Poster）从 trajectory preference 交替训练隐式 PRM，再把 step
advantage 与 episode advantage 融合。它不需要显式 step label 或额外 rollout，
但引入了在线 PRM 与交替优化复杂度。

**对项目的意义**：如果使用 learned evidence critic，必须证明它优于 iStar
这种通用隐式 PRM；首版不应轻易引入额外 critic。

### 4.5 进展势函数与信息增益

#### ΔBelief-RL

[Intrinsic Credit Assignment for Long Horizon Interaction](https://openreview.net/forum?id=SAJMlj9x39)
（ICML 2026 Regular）用每轮交互前后模型对正确目标的置信度变化：

\[
r_t^{belief}=p(y^*\mid s_{t+1})-p(y^*\mid s_t)
\]

为信息获取动作提供稠密进展信号。论文还将标准 GRPO 改为 turn-wise GRPO。

这与“新增 evidence potential”在数学形式上高度接近：两者都是
\(\Phi(s_{t+1})-\Phi(s_t)\)。

#### InfoPO

[InfoPO](https://openreview.net/forum?id=O7UHBoLrx4)
（ICML 2026 Regular）遮蔽某轮环境反馈，比较事实/反事实上下文下后续动作分布，
用信息增益奖励真正改变后续决策的 turn，并通过组内 reward 方差自适应融合
终局回报。

它与“删除本轮 evidence 后测 Answer 变化”的方案直接重合。若本项目采用类似
mask intervention，必须将 InfoPO 作为核心基线并给出 RAG 特有差异。

#### HPO

[Hindsight Policy Optimization](https://openreview.net/forum?id=iK3yDEvQ4y)
（ICML 2026 Regular）把 state-action 映射到语义 intent space，以成功轨迹形成
hindsight distribution，再用 Wasserstein/Kantorovich potential 给 step
分配低方差信用。在 SearchQA 七个 QA 数据集上验证，和本项目任务最接近。

它说明“用语义相似状态/动作共享成功信用”已有强顶会先例。

### 4.6 Milestone 与局部失败隔离

[BEACON](https://openreview.net/forum?id=Ga3AR4EF6R)
（ICML 2026 Regular）将长轨迹按 milestone 切段，在段内做 temporal shaping，
并融合局部与全局双尺度 advantage，避免远端失败污染早期正确动作。

其核心诊断“正确早期动作因终局失败而被错误惩罚”与 SAPR-RAG 完全一致。
区别是 RAG 的自然 milestone 可由“获得第一个/第二个支持证据”定义，而无需
额外 milestone detector。

### 4.7 Action Selection 与 Belief Tracking

[AREW / Information Self-Locking](https://openreview.net/forum?id=oPqm8k1CMQ)
（ICML 2026 Regular）把 active reasoning 拆成：

- Action Selection：是否提出能获取新信息的动作；
- Belief Tracking：是否将 observation 内化成更正确的任务理解。

它用方向性 critique 在原轨迹内重分配 advantage。该分解与本项目的
“Query 获取证据—Evidence/Answer 利用证据”非常接近，是创新边界中必须正面
讨论的工作。

---

## 5. Agentic RAG 专用过程优化

| 工作 | Venue | 信用粒度 | 监督来源 | 主要覆盖 | 留给本项目的缺口 |
|---|---|---|---|---|---|
| [Search-R1](https://openreview.net/forum?id=Rwhi91ideu) | COLM 2025 | 轨迹 | 最终答案 | 学会何时搜索与回答 | 中间 Query 信用稀疏 |
| [ReasonRAG](https://proceedings.neurips.cc/paper_files/paper/2025/hash/54e1381d0c0598127b90af4c940fd3d9-Abstract-Conference.html) | NeurIPS 2025 | 过程数据 | RAG-ProGuide | Query/Evidence/Answer 过程监督 | 主要是离线过程偏好，状态分布偏移 |
| [HiPRAG](https://openreview.net/forum?id=Gt4v9WBPzm) | ICLR 2026 | step | 在线知识感知 judge | over-search/under-search | 强在是否搜索，弱在证据如何改变答案 |
| [Search-P1](https://arxiv.org/abs/2602.22576) | arXiv 2026 | path | reference planner + soft outcome | 失败轨迹、路径覆盖 | 路径级分数仍可能混淆动作归因 |
| [TreePS-RAG](https://arxiv.org/abs/2601.06922) | arXiv 2026 | tree node | descendant outcome | 在线 step value | 分叉 rollout 成本、树结构复杂 |

### 关键判断

当前文献已覆盖：

- outcome reward；
- step/path reward；
- search necessity；
- tree/anchor-state micro advantage；
- belief change；
- masked-feedback counterfactual；
- Action Selection / Belief Tracking 分解。

所以以下命题都不足以单独作为新贡献：

```text
“我们给每轮 Query 一个 reward”
“我们用新增证据覆盖率做 reward”
“我们把最终 reward 拆到 turn”
“我们用树或相同状态做局部 GRPO”
“我们用答案置信度变化衡量信息增益”
```

---

## 6. 对原 Action-Causal GRPO 方案的重新定位

原方案：

```text
Query token  <- 本轮新增 evidence
Answer token <- 最终 F1
```

优点：

- 对当前实现问题高度对症；
- 比 sequence scalar 广播更合理；
- 可复用已有 `retrieved_steps` 和动作 mask；
- 能解释旧 reward-v3 为什么无效。

但从最新文献看，它与以下工作重叠：

| 原方案部分 | 最接近工作 |
|---|---|
| turn-level advantage | MT-GRPO、GiGPO、Tree-GRPO |
| 状态势函数差分 | ΔBelief-RL、BEACON |
| 信息获取动作奖励 | InfoPO、AREW |
| RAG 搜索过程 reward | ReasonRAG、HiPRAG、Search-P1 |
| evidence coverage 过程监督 | TreePS-RAG、Search-P1 |

因此它适合作为**诊断基线/第一版 pilot**，但不能直接当成最终论文算法。

---

## 7. 仍可能成立的 RAG 特有创新空隙

### 7.1 候选一：Evidence-Gap Grouped Credit

GiGPO 的微观分组依赖相同或相似环境状态。Agentic RAG 可定义一个可验证的
语义状态摘要：

\[
g_t=(\text{question},\ \text{uncovered gold evidence},\ \text{turn index})
\]

从相同 evidence gap 出发的 Query 才进入同一 micro group，比较其下一步
新增证据。相比 raw dialogue 相似度，这种状态：

- 与任务目标直接相关；
- 可解释；
- 不受表面措辞变化影响；
- 能明确表示“还缺哪一跳”。

候选 advantage：

\[
A^{query}_{i,t}
=A^{macro}_i
+\lambda
\frac{\Delta\Phi_{i,t}-\mu_{g_t}}{\sigma_{g_t}+\epsilon}
\]

风险：依赖 gold supporting facts，只适合训练；与 GiGPO/HPO 的差异需要通过
状态分组质量和消融证明。

### 7.2 候选二：Evidence-Mediated Credit Decomposition

把一轮搜索拆成三个可验证因果环节：

```text
Query acquisition:
  Query 是否检到新的必要证据

Evidence assimilation:
  新证据是否改变模型对正确答案/下一步动作的置信度

Answer realization:
  模型是否将已有证据转化为正确短答案
```

可构造：

\[
C_t^{query}=\Phi(E_{t+1})-\Phi(E_t)
\]

\[
C_t^{assim}
=
\log p(y^*\mid s_t,\mathrm{evidence}_t)
-
\log p(y^*\mid s_t,\mathrm{mask}(\mathrm{evidence}_t))
\]

\[
C^{answer}=\operatorname{F1}(\hat y,y^*)
\]

然后分别施加到 Query、Evidence/后续 reasoning、Answer token。

与现有工作的区别必须限定为：

- 比 ΔBelief-RL 更细：区分“获得信息”和“利用信息”；
- 比 InfoPO 更任务结构化：反事实变量是检索 evidence，而非一般用户反馈；
- 比 AREW 更可验证：使用真实检索和 gold evidence/answer，不依赖方向性 critique；
- 比 ReasonRAG 更 on-policy：信用在当前 student 访问状态上计算。

这是当前更有希望的 ICASSP 方法，但仍需先做文献和离线可辨识性审计。

### 7.3 候选三：只研究 zero-variance 的 RAG 特化修复

动态采样已被 DAPO 覆盖，不能只过滤全同分组。可研究：

> 当 final-answer group 全错时，是否可以用 evidence gap 对失败轨迹进行
> 二次分组，使“部分取得关键证据”的轨迹继续产生 Query 梯度。

该设计应作为候选一/二的自然结果，而不是单独的方法名。

---

## 8. 推荐的研究顺序

### 阶段 0：不训练的离线审计

使用 B 已保存的 rollout，统计：

1. final reward 零方差组中，有多少组的逐轮 evidence gain 有方差；
2. evidence gain 与最终 EM/F1 的相关性；
3. 找到 gold evidence 但 Answer 失败的比例；
4. 未找到 gold evidence 但 Answer 正确的比例；
5. 相同 evidence-gap 状态下可形成多大 micro group；
6. 每个数据集分别统计，避免 2Wiki 主导结论。

决策门：

- 若 zero-final-reward 组中至少 20% 可被 evidence gain 区分，动作信用值得实现；
- 若 evidence-gap micro group 大多小于 2，放弃分组法，转反事实 mediation；
- 若 evidence gain 与最终正确性相关很弱，不能以 gold coverage 作为核心势函数。

### 阶段 1：三个低成本基线

1. Dr.GRPO/DAPO-style normalization control；
2. 现有 sequence GRPO（B）；
3. 简单 action-masked marginal reward。

目的不是投稿，而是确认收益来自信用分配，不是长度归一化或动态采样技巧。

### 阶段 2：方法 pilot

只选择候选一或候选二，不同时实现。100/250 step 后检查：

- zero-std；
- 每轮 evidence gain；
- Query 重复率；
- Answer F1；
- max-turn；
- 梯度范数与输出长度。

### 阶段 3：正式对照

正式方法至少比较：

- B：标准 GRPO；
- DAPO/Dr.GRPO 风格强优化基线；
- GiGPO 或可复现的同状态 micro-advantage 基线；
- HiPRAG/ReasonRAG 的任务结果；
- 本方法及核心消融。

最终使用完整 dev 和 paired bootstrap。

---

## 9. 对方向二方案的最终建议

### 不建议直接做

```text
SaprMarginalRelevanceORM 返回每轮增量
  -> 求和
  -> 加到 sequence reward
```

该路线已经被 E07 否定，也没有解决信用位置丢失。

### 也不建议直接宣称

```text
Query 用 Δcoverage，Answer 用 F1
  -> Action-Causal GRPO
```

这个想法合理，但与最新顶会方法重叠过大。

### 建议先验证的最终候选

```text
Evidence-Mediated / Evidence-Gap GRPO

宏观：
  保留 final answer outcome

微观：
  Query acquisition credit
  + Evidence assimilation credit
  + Answer realization credit

归一化：
  按 evidence-gap 状态和动作类型计算局部 baseline
```

该方案的论文价值不在“更稠密的 reward”，而在：

> 对 Agentic RAG 的信息获取、信息吸收和答案实现进行可验证的因果分解，
> 并在 on-policy 状态上把信用分配回真正负责的动作。

---

## 10. 代表工作索引

### 正式录用：优先精读

1. [ArCHer, ICML 2024](https://proceedings.mlr.press/v235/zhou24t.html)
2. [Dense Reward for Free, ICML 2024](https://proceedings.mlr.press/v235/chan24a.html)
3. [VinePPO, ICML 2025](https://openreview.net/forum?id=Myx2kJFzAn)
4. [Search-R1, COLM 2025](https://openreview.net/forum?id=Rwhi91ideu)
5. [Dr.GRPO, COLM 2025](https://openreview.net/forum?id=5PAF7PAY2Y)
6. [DAPO, NeurIPS 2025](https://openreview.net/forum?id=2a36EMSSTp)
7. [GiGPO, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html)
8. [ReasonRAG, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/54e1381d0c0598127b90af4c940fd3d9-Abstract-Conference.html)
9. [PS-GRPO, NeurIPS 2025](https://openreview.net/forum?id=96I8PGPALv)
10. [GN-IVO, ICLR 2026](https://openreview.net/forum?id=eFXmrCun0c)
11. [GMPO, ICLR 2026](https://openreview.net/forum?id=nCEs0tSwc2)
12. [Tree-GRPO, ICLR 2026](https://openreview.net/forum?id=ZpQwAFhU13)
13. [HiPRAG, ICLR 2026](https://openreview.net/forum?id=Gt4v9WBPzm)
14. [iStar, ICLR 2026](https://openreview.net/forum?id=ooROvpmxMV)
15. [AttnRL, ICLR 2026](https://openreview.net/forum?id=NCN8oUsiNf)
16. [RIPO, ICML 2026](https://openreview.net/forum?id=nSRbKvrmsH)
17. [ΔBelief-RL, ICML 2026](https://openreview.net/forum?id=SAJMlj9x39)
18. [HPO, ICML 2026](https://openreview.net/forum?id=iK3yDEvQ4y)
19. [InfoPO, ICML 2026](https://openreview.net/forum?id=O7UHBoLrx4)
20. [AREW, ICML 2026](https://openreview.net/forum?id=oPqm8k1CMQ)
21. [BEACON, ICML 2026](https://openreview.net/forum?id=Ga3AR4EF6R)

### 预印本/未正式录用：用于跟踪，不作为顶会证据

1. [GSPO, CoRR 2025](https://openreview.net/forum?id=8hdHzwGLWY)
2. [MT-GRPO, arXiv 2025](https://arxiv.org/abs/2505.11821)
3. [Agent Lightning, arXiv 2025](https://arxiv.org/abs/2508.03680)
4. [TreePS-RAG, arXiv 2026](https://arxiv.org/abs/2601.06922)
5. [Search-P1, arXiv 2026](https://arxiv.org/abs/2602.22576)
6. [GRPO-λ, ICLR 2026 withdrawn](https://openreview.net/forum?id=iRWqcnBlLQ)
7. [TEMPO, ICLR 2026 withdrawn](https://openreview.net/forum?id=GRbI7kqA6S)

### 元调研入口

- [From Reasoning to Agentic: Credit Assignment in RL for LLMs](https://arxiv.org/abs/2604.09459)
- [Awesome Credit Assignment in LLM RL](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)

该 survey 为 2026 arXiv 预印本，可用于发现论文和建立分类，但 venue 与方法结论
仍应回到上述一手论文页面核验。
