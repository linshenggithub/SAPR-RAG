# 方向二：面向 Agentic RAG 的 Action-Causal GRPO

> **文献校准（2026-09-06）**：一般性的 turn-level advantage、势函数差分和
> 树状信用分配已有大量 2025–2026 顶会工作。本文件第 3 节的初始方案只能作为
> 诊断基线，不能直接作为最终创新。正式设计前必须先阅读
> [`../01_literature/related_work_drafts/grpo_long_horizon_credit_assignment_survey_20260906.md`](../01_literature/related_work_drafts/grpo_long_horizon_credit_assignment_survey_20260906.md)，
> 并优先考虑 evidence-gap 分组或 evidence-mediated 因果分解。

## 1. 论文命题

本方向不把“在 Agentic RAG 上使用 GRPO”作为贡献，而是研究：

> 如何针对多轮检索代理中异质动作、延迟回报和组内零方差问题，
> 将序列级 GRPO 改造成具有动作级因果信用分配的过程优化算法。

建议方法名暂定为：

```text
Action-Causal GRPO
动作因果组相对策略优化
```

备选名称：

```text
Action-Decomposed GRPO
State-Aware Process GRPO
```

核心贡献必须是 advantage 构造和 token 归因方式的变化，不能只是在最终 reward
中再加一个经验项。

## 2. 当前 GRPO 基线与问题证据

### 2.1 当前基线 B

B 从 E14 canonical SFT `checkpoint-4150` 起步，使用：

```text
R = 1.0 * answer_F1
  + 0.2 * cumulative_evidence_relevance
  + 0.05 * format
```

每题采样 8 条 on-policy 轨迹，先对整条轨迹算标量 reward，再在组内归一化，
最后将同一个 sequence advantage 广播给该轨迹的所有可训练 token。

ckpt1000 全量结果（EM / F1 / Cover-EM）：

| 数据集 | E14 SFT | B GRPO-only |
|---|---|---|
| HotpotQA | .4373/.5513/.4748 | .4629/.5837/.5026 |
| 2Wiki | .4051/.4513/.4188 | .5161/.5654/.5314 |
| MuSiQue | .1651/.2405/.1841 | .1808/.2794/.2056 |

训练数据与三个 dev 集规范化 question 精确重叠为 0，因此提升不是已知的数据
泄露造成的。

### 2.2 当前训练动态

B 的窗口平均训练指标：

| step | reward | F1 reward | reward std | 零方差组比例 | 平均输出长度 |
|---:|---:|---:|---:|---:|---:|
| 50 | .685 | .541 | .272 | .197 | 356 |
| 250 | .705 | .553 | .251 | .230 | 325 |
| 500 | .718 | .571 | .226 | .237 | 341 |
| 750 | .697 | .558 | .242 | .207 | 337 |
| 1000 | .757 | .602 | .219 | .280 | 329 |

训练稳定且有效，但约 20%–28% 的 rollout 组没有有效 reward 方差，组内
advantage 为零。

### 2.3 三个结构性问题

#### 问题一：异质动作共享同一个 advantage

Query、Evidence、Answer 对最终结果的因果作用不同，但当前实现把同一个
trajectory advantage 广播给所有 token：

```text
一次成功 Answer
  -> 此前所有 Query token 都得到同样正 advantage

一次失败 Answer
  -> 即使某轮 Query 找到了关键证据，也会被整体惩罚
```

这会把答案生成误差错误归因给检索动作，也会把偶然答对的收益分给无效 Query。

#### 问题二：累计 relevance 无法定位贡献轮次

`SaprRelevanceORM` 只看整条轨迹最终覆盖多少 gold evidence。即使知道最终覆盖
提升，也不知道是第几轮 Query 产生的。

已有 `SaprMarginalRelevanceORM` 虽然逐轮计算新增证据，但最终仍把各轮增量
求和成一个 sequence scalar，再广播给整条轨迹。因此 E07 的 reward-v3 仍未
真正解决动作级信用分配。

#### 问题三：组内零方差造成无效采样

当同题 8 条 rollout 的最终 reward 相同，GRPO 组内归一化后 advantage 为 0。
但“最终都答错”不代表各轮 Query 同样差：有的 Query 可能已找到部分关键证据，
只是 Answer 失败。序列级 reward 丢失了这部分可学习差异。

## 3. 方法：动作因果 advantage 分解

### 3.1 状态与动作

第 \(k\) 轮状态：

\[
s_k=(x,h_k,E_k)
\]

其中 \(E_k\) 是截至当前轮已覆盖的证据集合。Query 动作 \(q_k\) 触发检索环境
转移到 \(s_{k+1}\)。最终 Answer 动作生成答案 \(\hat y\)。

V1 只优化两类明确可归因的动作：

- Query span；
- Answer span。

Evidence Agent token 是否进入 actor loss 需先通过 loss mask 审计确认。V1 不给
Evidence span 新增 advantage，避免在因果归属未确认前引入噪声。

### 3.2 证据势函数

定义当前状态的证据覆盖势函数：

\[
\Phi(s_k)=\frac{|\mathrm{GoldHit}(E_k)|}{|\mathrm{GoldEvidence}|}
\]

第 \(k\) 次 Query 的直接过程收益为：

\[
r_k^Q=\Phi(s_{k+1})-\Phi(s_k)
\]

性质：

\[
\sum_k r_k^Q=\Phi(s_{K+1})-\Phi(s_1)
\]

因此它不是任意 reward trick，而是把现有最终累计 relevance 精确分解到真正
改变证据状态的 Query 动作上。重复 Query 或无新增证据的 Query 自然得到 0。

第一版不额外加入重复惩罚和 turn cost，避免同时改变多个变量。它们只作为后续
独立消融。

### 3.3 Answer 收益

Answer 动作使用最终答案质量：

\[
r^A=\operatorname{F1}(\hat y,y^*)
\]

格式信号只作用于产生对应协议标签的动作 token，不再作为整条轨迹的统一奖励。

### 3.4 两个候选 advantage 版本

#### G1：严格局部版

\[
A_t=
\begin{cases}
\operatorname{Norm}(r_k^Q), & t\in Query_k\\
\operatorname{Norm}(r^A), & t\in Answer\\
0, & \text{其他}
\end{cases}
\]

优点是因果归属最清楚。风险是 Query 只追求证据覆盖，忽略其对最终答案的长期
影响。

#### G2：局部加长程回报版

\[
A_t=
\begin{cases}
\alpha\operatorname{Norm}(r^A)
+\lambda\operatorname{Norm}(r_k^Q), & t\in Query_k\\
\operatorname{Norm}(r^A), & t\in Answer\\
0, & \text{其他}
\end{cases}
\]

其中 \(\alpha\) 保留 Query 对最终答案的长期责任，\(\lambda\) 提供直接过程
信用。建议先测：

```text
alpha = 0.25 / 0.5
lambda = 0.2
```

G1 是最干净的算法对照，G2 是更可能获得最终指标收益的正式候选。

### 3.5 分组与归一化

不同动作 reward 的量纲不同，不能先相加再统一归一化。建议：

- Answer F1：按同一问题的 8 条 rollout 归一化；
- Query marginal gain：按同一问题、同一轮次的可用 Query 动作归一化；
- 某组标准差为 0 时，对该分量返回 0，不制造虚假梯度；
- 分别记录 answer-zero-std 与 query-zero-std。

即使 8 条轨迹最终全错，只要它们的 Query 新增证据不同，Query span 仍能得到
训练信号。这正面针对当前 20%–28% 的 sequence zero-std。

## 4. 与已有 reward-v3 的本质区别

| 项目 | Reward-v3 | Action-Causal GRPO |
|---|---|---|
| 新增证据计算 | 逐轮 | 逐轮 |
| 最终保存形式 | 求和成一个序列标量 | 保留每轮 Query 的独立值 |
| advantage 归一化 | 序列级 | 动作类型/轮次级 |
| token 分配 | 同一值广播整条轨迹 | Query gain 只分给对应 Query token |
| Answer F1 | 同时影响所有动作 | 主要分给 Answer；G2 中少量长程项给 Query |
| 零方差缓解 | 有限 | final reward 相同时仍可由 Query gain 产生梯度 |

如果实现后仍把每轮增量求和到 sequence reward，本方向就没有方法创新。

## 5. 实现设计

### 5.1 默认路径必须完全兼容

新增显式开关，例如：

```text
advantage_mode=sequence          # 当前 B，默认
advantage_mode=action_causal     # 新方法
```

默认值必须保持 `sequence`，确保历史 B/C/D 结果可复现。

### 5.2 数据流

1. scheduler 已在 `rollout_infos.retrieved_steps` 保存每轮 Query 和 docs；
2. reward 侧逐轮计算 `delta_evidence_coverage`；
3. 将每轮 reward 与 Query turn 对齐；
4. 使用与 `build_teacher_action_mask` 同类的 token-span 对齐逻辑构造：
   - `query_action_mask[k]`；
   - `answer_action_mask`；
5. 在 batch 构造阶段生成 per-token advantage；
6. GRPO loss、ratio clipping 和 optimizer 不变。

### 5.3 预计修改范围

方向二会话拥有以下文件：

- `../ms-swift/swift/rl_core/advantage.py`
  - 新增动作 advantage 计算函数；
- `../ms-swift/swift/rlhf_trainers/grpo_trainer.py`
  - 读取逐轮 reward 并构造 per-token advantage；
- `../ms-swift/swift/rlhf_trainers/args_mixin.py`
  - 新增开关和系数；
- `03_sapr_rag/scripts/grpo/plugin.py`
  - 暴露逐轮 evidence gain，不再只返回求和标量；
- `03_sapr_rag/scripts/grpo/`
  - 新增 `run_icassp_action_causal_grpo_*.sh`，不覆盖历史 wrapper；
- `../ms-swift/tests/`
  - 动作 mask、归一化、零方差和 token 对齐单元测试。

### 5.4 必须覆盖的单元测试

1. 两轮 Query 中只有第二轮新增证据，只有第二轮 Query token 得正 advantage；
2. 重复 Query 无新增证据，局部 advantage 不为正；
3. Answer 正确但 Query 无效，Answer token 为正，Query 不被同等奖励；
4. Answer 错误但某轮 Query 命中关键证据，该 Query 仍保留过程信号；
5. final reward 全相同但 query gain 不同，Query advantage 非零；
6. 所有分量均零方差时不产生 NaN；
7. observation、padding 和 chat boundary token 不接收动作 advantage；
8. `advantage_mode=sequence` 与当前 B 的输出逐元素一致。

## 6. 实验矩阵

### 6.1 主实验

| 编号 | 方法 | 唯一变化 | 目的 |
|---|---|---|---|
| G0 | B：标准 GRPO | 序列 reward 广播 | 现有强基线 |
| G1 | Strict Action-Causal | Query=局部增量，Answer=F1 | 验证纯动作信用分配 |
| G2 | Hybrid Action-Causal | Query=长程 F1+局部增量 | 正式候选 |
| G3 | G2 + query cost | 仅无新增/重复 Query 扣费 | 可选消融 |

所有实验固定：

- E14 canonical SFT `checkpoint-4150`；
- 三源训练数据与顺序；
- 1000 step；
- LoRA、学习率、采样数、batch；
- Evidence Agent、Top-3、最多 6 turn；
- 同一完整 dev 评测口径。

### 6.2 执行阶段

#### G-A：离线轨迹审计

先用 B 已保存的 completions/rollout_infos 重算逐轮 gain，回答：

- 有多少失败轨迹实际命中过部分 gold evidence；
- final reward 零方差组中，有多少组的 query gain 有方差；
- 每轮新增 evidence 的分布；
- 重复 Query 与零 gain 的对应关系。

若 query gain 无法显著降低 zero-std，则先停止实现，重新设计软证据效用。

#### G-B：代码单测与 2-step smoke

必须验证默认 sequence 路径不变，新路径无 NaN、mask 和 turn 对齐正确。

#### G-C：100/250-step pilot

使用三源数据，避免只在单数据集上调参。比较：

- reward/F1；
- query gain；
- sequence/query zero-std；
- Query 重复率；
- 平均轮数；
- 输出长度和 grad norm。

#### G-D：1000-step 正式实验

只允许一个通过 pilot 门槛的配置进入正式训练，避免大规模参数搜索污染结论。

## 7. 评测指标与成功门槛

### 7.1 最终任务指标

- EM；
- F1；
- Cover-EM；
- 回答率；
- max-turn rate。

### 7.2 过程指标

- 每轮新增 gold evidence；
- 首次命中关键证据的平均轮次；
- 重复 Query 率；
- 无新增证据 Query 比例；
- sequence zero-std；
- query-action zero-std；
- 平均检索轮数；
- 找齐证据后继续检索率。

### 7.3 正式成功标准

将该方向作为论文核心，需要同时满足：

1. 相对 G0，三个数据集 macro-F1 或 macro-Cover-EM 明确提高；
2. 至少一个完整数据集的主要指标通过 paired bootstrap 显著性检验；
3. 另外两个数据集不存在超过 0.5pt 的系统性退化；
4. query zero-std 或无效 Query 比例明显下降；
5. 增益不能仅来自输出更短或回答率变化。

### 7.4 停止条件

出现以下任一情况，不进入 1000-step：

- 250 step 时过程指标与 G0 无差异；
- Query gain 仍在大多数样本上为 0；
- Answer F1 明显下降；
- max-turn、重复 Query 或输出长度恶化；
- 只有新增惩罚才能获得收益，去掉惩罚即消失。

## 8. 风险与备选

### 风险一：Gold evidence 依赖

当前 \(\Phi\) 使用 gold title/supporting sentence，只适用于训练时有支持证据标注
的数据。论文必须明确这是训练监督，推理不使用 gold。若要扩展到无 supporting
fact 数据，可将软证据效用作为后续工作，不应在首版同时引入 learned critic。

### 风险二：Query gain 仍然稀疏

若多数检索都无法精确命中 gold evidence，\(\Delta\Phi\) 仍为 0。备选顺序：

1. gold sentence 的软文本匹配；
2. gold title/answer mention 的分级分数；
3. 最后才考虑独立 learned evidence critic。

不要一开始引入外部 LLM judge，否则难以判断收益来自算法还是 judge。

### 风险三：只做 reward shaping 被判定为经验技巧

论文中必须突出：

- 势函数差分的 telescoping 性质；
- 异质动作上的因果归属；
- 动作类型/轮次级 baseline；
- 对 sequence zero-variance 的理论和实证缓解。

## 9. 论文叙事建议

1. **问题**：现有 GRPO 为整条 agent 轨迹分配单一标量，错误广播给异质动作；
2. **诊断**：20%–28% 组内零方差，旧 marginal reward 因回收到序列标量而无效；
3. **方法**：用证据势函数将 relevance 分解为逐轮 Query 增量，并将 Answer
   reward 只分配给 Answer span；
4. **性质**：过程 reward 总和保持最终 evidence coverage，减少任意 reward
   shaping；
5. **实验**：最终指标、过程指标、零方差率、动作消融和显著性。

## 10. 本方向会话的文件边界

本方向会话负责：

- 本文件；
- 上述 ms-swift advantage/trainer/args 变更；
- `plugin.py` 的动作级 reward 数据接口；
- Action-Causal GRPO 专属测试和 wrapper；
- 独立实验输出。

不修改方向一的 OPSD teacher prompt、OPSD 消融脚本和论文叙事文件。

所有新行为必须由显式开关启用，历史 sequence-GRPO、OPSD、DPO 默认路径不得
改变。
