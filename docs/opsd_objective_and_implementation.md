# OPSD 目标函数与实现细节

**文档定位**：本文集中讲清 SAPR-RAG 中 OPSD（On-Policy Self-Distillation）的
**数学目标、逐 token 信号、动作级 mask、两种训练模式的 advantage 分解**，
以及与标准知识蒸馏的区别。

它是对以下文档的补充，不重复其内容：

- 数据 / rollout / reward 链路：[`grpo_opsd_pipeline_overview.md`](grpo_opsd_pipeline_overview.md)
- ms-swift 本地改动与迁移：[`ms_swift_local_patches.md`](ms_swift_local_patches.md)
- 实验配置与结果：[`experiment_tracker.md`](experiment_tracker.md)
- 外部教师 OPD 方案：[`opd_plan.md`](opd_plan.md)

关键代码位置：

| 内容 | 文件 |
|---|---|
| teacher 逐 token log-ratio（进 loss，k1） | `../ms-swift/swift/rl_core/advantage.py::compute_teacher_logratio` |
| teacher KL 监控（不进 loss，k3） | `../ms-swift/swift/rl_core/advantage.py::compute_teacher_kl_per_token` |
| 逐 token advantage 装配 | `../ms-swift/swift/rl_core/advantage.py::expand_advantage_to_per_token` |
| 分动作 token mask | `../ms-swift/swift/rlhf_trainers/gkd_helpers.py::build_teacher_action_mask` |
| advantage 写回与系数应用 | `../ms-swift/swift/rlhf_trainers/grpo_trainer.py` |
| teacher prompt 构造 | `03_sapr_rag/scripts/grpo/build_grpo_dataset_action_opsd.py` |

---

## 1. 记号

- 同一问题采样 `G` 条轨迹（GRPO 组），第 `i` 条轨迹标量任务奖励 `R_i`。
- `y_{i,t}`：第 `i` 条轨迹第 `t` 个 response token；`M_{i,t}`：completion mask（1=可训练 response token）。
- `log p_theta`：当前策略；`log p_old`：采样时旧策略；`log p_ref`：参考模型。
- `log p_S`、`log p_T`：student / teacher 在**同一采样 token** 上的对数概率。teacher 是**同一个模型**加上特权上下文的 stop-gradient 前向，不是独立大模型。
- `c_S`：student 上下文（推理时可见信息）；`c_T`：teacher 上下文 = `c_S` + 特权信息。
- `a(t) ∈ {query, evidence, answer}`：token `t` 所属动作；`beta_{a(t)}`：动作系数。

当前动作系数（`run_grpo_opsd.sh` 默认，与 experiment_tracker 一致）：

```text
beta_query    = 0.01
beta_evidence = 0.00
beta_answer   = 0.03
```

---

## 2. teacher 逐 token 信号：反向 KL 的单样本估计

### 2.1 signed log-ratio（k1，进 loss）

对 student 实际采样的 token `y_{i,t}`：

```text
s_{i,t} = log p_T(y_{i,t} | c_T, y_{i,<t}) - log p_S(y_{i,t} | c_S, y_{i,<t})
```

它是 teacher/student 在同一 token 上的对数概率差（signed log-ratio）。
`compute_teacher_logratio` 计算该值并对 response 之外 mask 为 0。

为什么是这个形式：OPSD 的理论目标是最小化**反向 KL** `KL(pi_theta || pi_T)`：

```text
KL(pi_theta || pi_T) = E_{y ~ pi_theta} [ log p_theta(y) - log p_T(y) ]
```

期望按 student 自己的分布取（on-policy），因此天然是反向 KL。
用**单个采样 token** 做无偏估计，就得到 `s_{i,t}`（取负即上式括号项）。
这是 OPD 原论文给出的 sampled-token policy-gradient 变体。

### 2.2 与全词表 KL / 标准蒸馏的区别

| | 标准蒸馏 GKD | 本项目 OPSD |
|---|---|---|
| 散度方向 | 正向 KL `KL(T‖S)`（也有反向变体） | 反向 KL `KL(S‖T)` |
| 期望按谁采样 | teacher | student（on-policy） |
| 每步覆盖 | 全词表 `sum_v` | 仅采样到的 1 个 token |
| teacher 需求 | 全词表分布 logits | 仅采样 token 的 logp |
| 使用方式 | 可微分布匹配 loss | 当作 per-token advantage 做 policy gradient |
| 行为 | mass-covering（求全） | mode-seeking（求精） |

要点：本项目**不是**用 teacher logits 监督 student logits 的分布蒸馏 loss，
而是取采样 token 的 log 概率差作为 RL advantage。省去 teacher 全词表传输，
天然嵌入 GRPO 的 per-token advantage；代价是单样本估计方差更大。

### 2.3 k3 只用于监控，不进 loss

`compute_teacher_kl_per_token` 计算 `exp(d) - d - 1`（`d = log p_T - log p_S`，
k3 估计），是非负的反向 KL 幅度估计，训练中应随收敛下降。它只写入
`teacher_kl` / `teacher_kl_scoped` 指标，**不参与梯度**。进梯度的始终是 k1 的
signed log-ratio。

---

## 3. 分动作 token mask：按整段 turn 归类

teacher 信号不是只加在 `<query>...</query>` 或 `<answer>...</answer>` 尖括号内，
而是按**整个 assistant turn** 归类后，作用于该 turn 全部可训练 response token。

`build_teacher_action_mask` 的 `classify_action` 规则：

```text
若该 turn 文本含 <answer> -> answer
否则含 <query>          -> query
否则含 <evidence>       -> evidence
否则                    -> 不监督
```

因此若某轮以 query 收尾，则该轮从 reasoning、`So the next query is`
到 `<query>...</query>` 的全部 response token 都记为 query 动作，统一施加
`beta_query`。observation、padding、chat 模板边界 token 不接收 teacher 信号。

被监督的对象是 **student 自己采样的 token**，不是 teacher prompt 里的参考文本；
参考文本只进入 `c_T` 影响 teacher 打分。

---

## 4. 逐 token advantage 装配（两种模式统一入口）

`expand_advantage_to_per_token` 产出每个 token 的 advantage：

```text
A_{i,t} = u * A_hat_i + beta_{a(t)} * s_{i,t} * M_{i,t}
```

- `A_hat_i` 是 GRPO 组内归一化的轨迹级 advantage：

```text
A_hat_i = ( R_i - mean(R_1..R_G) ) / std(R_1..R_G)
```

  它是**整条轨迹一个标量**，原样广播到该轨迹每个 token（同一轨迹所有 token 相同）。
- `u ∈ {0, 1}` 由 `use_base_advantage`（即 `opd_use_grpo_advantage`）控制。
- 第二项逐 token 不同，且仅在 query/answer token 非零。

外层是带 clip 的 GRPO/PPO 目标（两种模式共用，仅 `A_{i,t}` 组成不同）：

```text
r_{i,t}(theta) = exp( log p_theta(y_{i,t}) - log p_old(y_{i,t}) )

L = - (1 / sum M) * sum_{i,t} M_{i,t} * min( r_{i,t} * A_{i,t},
                                             clip(r_{i,t}, 1-eps, 1+eps) * A_{i,t} )
    + beta_KL * KL(pi_theta || pi_ref)
```

---

## 5. 两种训练模式

### 5.1 纯 OPSD（实验 D）

`ENABLE_REWARD=false` 且 `opd_use_grpo_advantage=false`，即 `u=0`、无 `R_i`：

```text
A_{i,t} = beta_{a(t)} * ( log p_T(y_{i,t}) - log p_S(y_{i,t}) ) * M_{i,t}
```

只有 query（0.01）/ answer（0.03）token 有信号，其余为 0。等价于用 teacher
log-ratio 作为每 token 的奖励优势做 policy gradient，不含任务 reward、不含组内
outcome advantage。

### 5.2 GRPO + OPSD（实验 E16）

`u=1`，同时保留任务 reward 与 teacher：

```text
A_{i,t} = A_hat_i + beta_{a(t)} * ( log p_T(y_{i,t}) - log p_S(y_{i,t}) ) * M_{i,t}
```

第一部分（粗粒度）：整条轨迹一个分，广播到每个 token。
第二部分（细粒度）：每个 token 各自的 teacher 信号，按动作区分。

### 5.3 纯 GRPO（对照 B）

`u=1`，teacher 系数全 0，只剩 `A_{i,t} = A_hat_i`。

| 模式 | 轨迹级 `A_hat_i` | teacher 逐 token |
|---|---|---|
| 纯 GRPO（B） | 开 | 关 |
| 纯 OPSD（D） | 关 | 开 |
| GRPO+OPSD（E16） | 开 | 开 |

---

## 6. 分动作 teacher 的信息边界与示例

不同动作使用不同特权视图，避免全知 teacher 破坏搜索策略：

| 动作 | student 可见 | teacher 额外可见 | 系数 |
|---|---|---|---:|
| query | 原问题、历史 query、已有 evidence | R3 成功轨迹的参考查询计划（**不含答案**） | 0.01 |
| evidence | 检索文档与上下文 | 当前不施加 teacher | 0 |
| answer | 完整可见历史与证据 | benchmark 标准短答案 + 支持证据 | 0.03 |

分类优先级：`answer > query > evidence`。

teacher prompt 由 `build_grpo_dataset_action_opsd.py` 的
`build_query_prompt` / `build_answer_prompt` 构造。真实示例
（问题：*The Oberoi family is part of a hotel company that has a head office in
what city?*，gold：Delhi）：

Query teacher（beta=0.01，只给参考检索计划，绝不含答案）：

```text
<privileged_query_guidance>
The following queries come from a successful real-retrieval trajectory for this
question. They do not contain the gold answer. Use them only as a search-plan
reference and still adapt the next query to the documents already observed.
1. What is the location of the head office of the hotel company associated with the Oberoi family?
</privileged_query_guidance>
```

Answer teacher（beta=0.03，给标准答案 + 支持证据，仅评 answer token）：

```text
<privileged_answer_guidance>
This information is available only when scoring final answer tokens. Preserve the
actual retrieval history above and use the verified evidence to produce the answer.
Gold answer(s): Delhi
Verified supporting evidence:
1. Title: Oberoi family
   - The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.
2. Title: The Oberoi Group
   - The Oberoi Group is a hotel company with its head office in Delhi.
When the accumulated evidence is sufficient, end with <answer>answer</answer>.
</privileged_answer_guidance>
```

这两段特权文本只在训练算 teacher logp 时拼进 `c_T`；student 上下文 `c_S` 不含，
推理阶段完全不出现，因此不增加推理成本、不泄露答案给部署模型。

---

## 7. 常见追问与准确表述

- teacher 与 student 是**同一模型**、同 tokenizer，仅上下文前缀不同，因此对同一
  串采样 token 的 logp 一一对齐，可直接相减。teacher 前向 stop-gradient。
- 进梯度的是 k1（signed log-ratio），k3 仅监控。
- 当前是 sampled-token 估计，不是全词表 KL；论文中需如实说明这是 OPD 原论文的
  policy-gradient 变体，不是本项目创新。
- OPSD 的“反向”来自 on-policy：信号只在 student 采样 token 上产生，期望按
  student 分布取。RLHF 里 `-beta * KL(pi_theta || pi_ref)` 同为反向 KL，原因相同。

---

## 8. 结论边界（与 experiment_tracker 对齐）

- 纯 OPSD（D）相对同起点 canonical SFT+DPO（E15）全面提升，可支撑
  “动作级 on-policy 自蒸馏是离线 DPO 的有效替代”。
- 但 `C - B ≈ 0`：OPSD 叠加在 GRPO 之上没有稳定独立增益，说明当前配置下
  teacher 信号与 GRPO reward 高度重叠。不能声称 OPSD 能进一步增强 GRPO。
