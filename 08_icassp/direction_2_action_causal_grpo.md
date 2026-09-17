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

### 3.4.1 实现状态澄清（2026-09-08）

已完成的 E17 是 **additive prototype**，不是上面定义的完整 G2：

```text
E17: A_token = A_total_sequence + 0.2 * A_query_gain
```

其中 `A_total_sequence` 仍由 F1 + relevance + format 合成后广播给全部生成 token；
Answer F1 没有只路由到 Answer span，relevance 也同时存在于 sequence reward 和局部
query credit 中。因此 E17 全量无增益只能证伪“低权重全局附加”版本，不能证伪
G1/G2 的动作分解本身。

E18 pilot 先做一个更保守的小改动：只在 `A_total_sequence` 整组为 0 的 prompt 上
启用 query credit（`action_credit_gate=zero_outcome`, coef=0.5），验证局部信号能否
只填补 GRPO 死区而不干扰已有排序。若仍无增益，再实现严格 G1/G2 的 Answer/Query
分量独立归一化与 token 路由。

E18 的在线监控表明该门控过于保守：虽然约 18% 的 prompt 组被选中，但 125 个
generation 中只有 3 个产生非零动作项（2.4%），乘 0.5 后的全程平均绝对幅度仅
0.00314。因此 E19 直接实现上述完整 G2：Answer 使用独立归一化的 F1 advantage；
Query 使用 `alpha * A_F1 + lambda * A_query_gain`；其他 token 不接收任务
advantage。首个 pilot 取 `alpha=0.5, lambda=0.5`，在降低错误广播的同时保留一半
长程答案责任。

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
| G3 | Outcome-Signed Query Reweighting | 保留标准 GRPO，只按局部进展调整 Query 梯度幅度且不翻转符号 | E19 失败后的候选；E20 pilot 已否定 |
| G4 | Causal Return-to-Go GRPO | 将 terminal relevance 按剩余 evidence return 重分配到后续 Query 标签段 | E21 pilot 已否定 |
| G5 | DAPO Dynamic Sampling control | 不改 advantage；过滤并重采样零方差 prompt 组 | E22 pilot 已完成，未形成稳定增益 |
| G6 | Dr.GRPO length normalization control | 不改 advantage；固定长度分母，提高长轨迹相对权重 | E23 pilot 已否定 |

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

**E19 pilot 结果（2026-09-08）**：`alpha=0.5, lambda=0.5` 完成 250 step。
局部信用在 43.36% 的 prompt 组中非零，Query/Answer token advantage
绝对值均值分别为 0.3090/0.5025，机制信号有效。

固定 `hash1000, seed=20260908` 上，E19-250 相对同阶段 G0/B-250：

| 数据集 | ΔEM | ΔF1 | ΔCover-EM |
|---|---:|---:|---:|
| HotpotQA | +0.70pt | +0.30pt | +0.70pt |
| 2Wiki | +1.20pt | +1.25pt | +1.10pt |
| MuSiQue | +0.50pt | -0.09pt | +0.50pt |
| 宏平均 | +0.80pt | +0.49pt | +0.77pt |

单数据集 paired bootstrap 的 95% 差值区间仍跨 0，因此这里只判定为
“通过 pilot、进入正式训练”，不将其作为最终显著性结论。

#### G-D：1000-step 正式实验

只允许一个通过 pilot 门槛的配置进入正式训练，避免大规模参数搜索污染结论。
当前唯一进入正式训练的配置是 G2 `alpha=0.5, lambda=0.5`，入口为
`03_sapr_rag/scripts/grpo/run_canonical_sft_action_causal_s1000.sh`。
正式 run `action_causal_g2_a0.5_l0.5_s1000_20260908` 已于 2026-09-08 06:35 启动，
12:27 完成。checkpoint-1000 全量相对 B 的宏平均 EM/F1 分别下降
0.58/0.45pt，三数据集 EM/F1 均未提升，因此 G2 被否定。

#### G-E：Outcome-Signed Query Reweighting

E19 的核心问题是局部 gain 可以在最终错误轨迹上形成独立正优势，同时完整 G2
还删除了 B 中 relevance/format 对 Answer token 的约束。G3 改为：

```text
Query_k: A_seq + lambda * abs(A_seq) * clip(A_query_gain(k), -c, c)
其他:    A_seq
```

其中 `lambda=0.25, c=2.0`，并强制 `lambda*c<1`。因此：

- 标准 GRPO 的复合 reward、Answer 与非 Query token 全部保持不变；
- 成功轨迹中高 gain Query 被加强，低 gain Query 被减弱；
- 失败轨迹中高 gain Query 少受惩罚，低 gain Query 多受惩罚；
- 非零优势绝不跨 0，不会把失败轨迹中的局部动作变成正监督；
- sequence advantage 为 0 时保持 0，不声称解决 E18 已证伪的 zero-outcome 死区。

入口：`03_sapr_rag/scripts/grpo/run_canonical_sft_signed_query_reweight_pilot.sh`。

E20 的 checkpoint-250 在固定 hash1000 上相对 B-250 的宏平均 EM/F1/Cover-EM
分别为 −0.10/−0.34/−0.20pt。局部项覆盖约 69.6% completion token、平均改变量
0.0418 且无符号翻转，但 HotpotQA/2Wiki 检索覆盖几乎不变。失败原因不是局部项
太弱，而是即时 gain 与 B 的 terminal relevance 重复，并且只奖励直接命中，无法
给准备 bridge entity、后续才取得证据的 Query 分配长期信用。

#### G-F：Causal Return-to-Go GRPO

E21 不增加 reward，而是重分配 B 中已有的 relevance 分量。令
`gain_j = Phi(E_j)-Phi(E_{j-1})`，第 k 个 Query 的原始 return 为：

```text
R_query(k) = R_sequence - 0.2 * relevance_final
             + 0.2 * sum(gain_j, j >= k)
A_query(k) = GroupNorm(R_query(k))
```

设计约束：

- `sum(gain_j)` 必须与 `relevance_final` 一致，否则 fail-fast；
- Query 1 直接复用标准 `A_sequence`，与 B 逐值一致；
- 后续 turn-slot 无组内对比时回退 `A_sequence`；
- 只替换 `<query>...</query>` 标签段，Query 前 reasoning、Answer 与其他 token
  继续使用 `A_sequence`；
- 不引入额外系数，reward 权重仍为 F1 1.0 / relevance 0.2 / format 0.05。

该方法同时修复两个问题：标准 GRPO 会把历史已经取得的 evidence 继续奖励给后续
Query，而 E17/E20 的即时 gain 又无法奖励为下一跳创造条件的准备性 Query。E21 使用
future return，使 Query 只承担其执行时刻之后的可达收益。

入口：`03_sapr_rag/scripts/grpo/run_canonical_sft_causal_return_pilot.sh`。

E21 完成 250 step，固定 hash1000 sweep 选择 checkpoint-250。相对 B-250，
HotpotQA EM/F1 为 +0.30/+0.04pt，2Wiki 为 +0.60/−0.08pt，MuSiQue 为
−0.90/−0.89pt；宏平均 EM/F1/Cover-EM 分别为 +0.00/−0.31/−0.17pt。
方法在较短链数据上有局部 EM 增益，但明显伤害 MuSiQue 长链任务，因此未通过
pilot，不进入 1000-step。

#### G-G：DAPO Dynamic Sampling control

E22 不再修改 Query/Answer token 的优势。它完整保留 B 的复合 sequence advantage，
只丢弃组内复合 reward 标准差为 0 的 prompt group，并最多重采样 3 轮。
这可以验证 E17–E21 的失败究竟来自局部信用设计，还是 B 中 10%–50% 无有效梯度的
rollout 组降低了样本效率。

E22 是 DAPO 风格强优化基线，不作为新的信用分配方法。入口：
`03_sapr_rag/scripts/grpo/run_canonical_sft_dapo_dynamic_pilot.sh`。

E22 完成 250 step 后，固定 hash1000 sweep 选择 checkpoint-250。相对 B-250，
宏平均 EM/F1/Cover-EM 分别为 +0.20/+0.03/+0.20pt；HotpotQA 有小幅改善，
2Wiki F1 基本持平，MuSiQue EM/F1 均下降。动态采样确实将零方差组比例降至 0，
但额外 rollout 成本约使训练从 20.7 s/it 增至 33.0 s/it，且没有稳定 F1 增益，
因此不进入 1000-step。

#### G-H：Dr.GRPO length normalization control

E23 保留 B 的 reward、sequence advantage 与所有采样设置，只把 loss 从逐轨迹长度
平均改为 Dr.GRPO 固定长度归一化：

```text
B:   L = mean_i(sum_t L_i,t / T_i)
E23: L = sum_i,t L_i,t / (batch_size * max_completion_length)
```

B 的实际保存配置已确认 `loss_type=grpo`。其训练 completion 平均长度为 338.80
token，而 `max_completion_length=4096`；标准 GRPO 让短、长轨迹的总权重相同，
可能削弱 MuSiQue 等长链样本中多个动作的累计监督。E23 用固定分母消除该逐样本
长度归一化偏置，不引入新的 reward 或动作级启发式信用。

入口：`03_sapr_rag/scripts/grpo/run_canonical_sft_dr_grpo_pilot.sh`。250-step run
`dr_grpo_lengthnorm_r3_s250_20260911` 与 checkpoint-125/250 固定
`hash1000, seed=20260908` 评测均已完成。checkpoint-250 最优，但宏平均
EM/F1/Cover-EM 相对 B-250 分别下降 0.23/0.32/0.37pt；HotpotQA 仅 EM/Cover
小幅提高，2Wiki 与 MuSiQue 全面下降。提高长轨迹相对权重没有改善 MuSiQue，
因此 E23 不进入 1000-step。

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
