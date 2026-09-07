# 方向一：Action-Causal OPSD 替代离线 DPO

## 1. 论文命题

本方向不把 GRPO 作为方法主线，而是研究：

> 面向多轮 Agentic RAG，如何用无外部奖励函数的 on-policy 自蒸馏，
> 替代基于固定偏好对的离线 DPO，并通过动作级特权信息隔离减少状态分布偏移。

建议方法名暂定为：

```text
Action-Causal On-Policy Self-Distillation
动作因果在线自蒸馏
```

核心贡献不能只写成“将 OPSD 应用于 RAG”。必须证明：

1. Agentic RAG 的 Query、Evidence、Answer 是语义和因果作用不同的动作；
2. 原始 OPSD 使用统一特权上下文监督整条输出，可能把答案和未来证据泄漏给
   Query 策略；
3. 本方法按动作构造不同 teacher view，并只将 teacher 信号施加到对应 token；
4. 在不设计任务 reward、不使用外部强 teacher 的条件下，优于 DPO。

## 2. 为什么从 DPO 切换到 OPSD

### 2.1 DPO 在 Agentic RAG 中的结构性问题

DPO 使用固定的离线偏好对。训练轨迹由行为策略提前生成，训练时不会根据当前
模型的新 Query、新检索结果和新历史状态重新采样。部署时，Agent 的每次 Query
都会改变后续文档、Evidence 和 Answer 的状态分布，因此存在：

```text
离线偏好轨迹中的状态分布
    !=
当前模型自主检索时实际访问的状态分布
```

这会带来两类问题：

- **状态分布偏移**：DPO 没有监督当前模型真正会访问的错误状态；
- **轨迹级信用模糊**：整条 chosen/rejected 偏好无法精确指出错误来自 Query、
  Evidence 还是 Answer。

### 2.2 OPSD 的对应优势

纯 OPSD 直接从当前 student 采样多轮轨迹，再让同一个模型在特权信息条件下
评价这些 student token：

```text
student:
  只看推理时可用信息，生成 on-policy 轨迹

teacher:
  与 student 共享参数，但额外看到动作相关的特权信息

optimization:
  在 student 实际生成的 token 上使用 teacher/student log-ratio
```

它不依赖人工 reward，也不需要额外的 14B/70B teacher，直接针对当前策略
访问到的状态提供稠密 token 级监督。

## 3. 当前方法定义

### 3.1 状态与动作

第 \(k\) 轮状态记为：

\[
s_k=(x,h_k,e_k)
\]

其中 \(x\) 是原问题，\(h_k\) 是此前 Query/Answer 历史，\(e_k\) 是已获得证据。
当前协议中的主要动作是：

- Query：生成下一条检索查询；
- Evidence：Evidence Agent 对文档做证据抽取；
- Answer：输出最终短答案，同时隐式执行停止决策。

### 3.2 动作级 teacher view

| 动作 | student 可见信息 | teacher 特权信息 | 当前系数 |
|---|---|---|---:|
| Query | 原问题、已有历史、已有证据 | R3 成功轨迹聚合出的参考查询计划；不提供最终答案文本 | 0.01 |
| Evidence | 检索文档与当前上下文 | 当前不施加 teacher 信号 | 0 |
| Answer | 原问题、完整可见历史和证据 | benchmark 标准短答案及支持证据 | 0.03 |

动作分类优先级保持：

```text
answer > query > evidence
```

只在对应动作 token 上施加特权监督，chat template 边界 token 和 observation
不接收 teacher 信号。

### 3.3 纯 OPSD 训练目标

本方向不使用 F1、relevance、format 等 RL reward，也不使用 GRPO 组内
advantage。对 student 实际采样 token \(y_t\)，当前 sampled-token 目标为：

\[
A_t^{OPSD}
=
\beta_{a(t)}
\left[
\log p_T(y_t\mid c_T,y_{<t})
-
\log p_S(y_t\mid c_S,y_{<t})
\right]
\]

其中 \(a(t)\) 是 token 所属动作，\(\beta_{query}=0.01\)，
\(\beta_{answer}=0.03\)，\(\beta_{evidence}=0\)。

当前实现使用 OPSD 原论文给出的 sampled-token policy-gradient 变体，而非
主目标中的全词表 KL。论文中必须如实说明，不能把该目标形式写成本项目创新。

## 4. 已有实验证据

统一起点为 E14 canonical SFT `checkpoint-4150`，下表为
EM / F1 / Cover-EM：

| 方法 | HotpotQA | 2Wiki | MuSiQue |
|---|---|---|---|
| E14 SFT | .4373/.5513/.4748 | .4051/.4513/.4188 | .1651/.2405/.1841 |
| E15 SFT→DPO | .4140/.5281/.4304 | .4187/.4656/.4230 | .1585/.2459/.1676 |
| D SFT→纯分动作 OPSD | .4462/.5703/.5030 | .4948/.5548/.5270 | .1758/.2717/.2085 |

纯分动作 OPSD 相对 DPO 的绝对提升：

| 数据集 | EM | F1 | Cover-EM |
|---|---:|---:|---:|
| HotpotQA | +3.22pt | +4.22pt | +7.26pt |
| 2Wiki | +7.61pt | +8.92pt | +10.40pt |
| MuSiQue | +1.73pt | +2.58pt | +4.09pt |

这些结果支持“OPSD 可替代 DPO”，但尚不能证明“动作级设计优于原始 OPSD”。

## 5. 当前证据缺口

### 5.1 必须补的主对照

| 编号 | 实验 | 目的 | 优先级 |
|---|---|---|---|
| O0 | E14 SFT | 无后训练基线 | 已完成 |
| O1 | E15 DPO | 离线偏好优化基线 | 已完成 |
| O2 | Vanilla OPSD | 统一 teacher view 作用于所有生成动作 | 必须 |
| O3 | Answer-only OPSD | 只监督最终答案 | 必须 |
| O4 | Query-only OPSD | 只监督搜索策略 | 建议 |
| O5 | Query+Answer Action-Causal OPSD | 当前 D 实验 | 已完成 |

最小可发表归因：

```text
O5 - O1：on-policy 自蒸馏相对 DPO 的总体收益
O5 - O2：动作级特权隔离相对原始统一 OPSD 的收益
O5 - O3：Query teacher 的增量贡献
O5 - O4：Answer teacher 的增量贡献
```

如果 O5 不能稳定超过 O2 或 O3，则“Action-Causal”不能作为核心算法贡献，
论文最多只能主张 OPSD 在 Agentic RAG 上是一种有效的无 reward 替代方案。

### 5.2 Vanilla OPSD 的公平定义

Vanilla OPSD 必须与 O5 使用：

- 相同 E14 起点；
- 相同 277,839 条三源训练问题；
- 相同 student rollout、LoRA、学习率、采样数和 1000 step；
- 一份包含参考查询计划、支持证据和标准答案的统一 teacher prompt；
- teacher 信号作用于所有可训练 assistant token。

全局系数不能随意挑选。建议先按 O5 的平均绝对 teacher advantage 对齐，
再在小规模 pilot 中从 `0.01/0.02/0.03` 选择稳定值，避免仅因信号尺度不同
造成不公平比较。

### 5.3 DPO 比较的边界

E15 与 O5 使用不同训练目标和数据组织，当前结果可以作为系统级比较，但不是
严格的等算力因果对照。论文中应额外报告：

- 训练问题数与实际 rollout 数；
- 总生成 token；
- GPU 小时；
- teacher forward 开销；
- DPO preference pair 数量；
- 各方法是否使用 gold supporting facts。

不能在缺少这些统计时声称 OPSD “训练成本更低”。可以稳健声称的是：

> OPSD 不需要人工设计任务 reward，也不需要独立外部 teacher。

## 6. 实验执行顺序

### 阶段 O-A：实现与 smoke

1. 复用现有纯 OPSD 开关：
   `ENABLE_REWARD=false`、`OPD_USE_GRPO_ADVANTAGE=false`。
2. 为 O2/O3/O4 分别新增顶层 wrapper，不覆盖现有正式脚本。
3. 每个实验先跑 2-step smoke，检查：
   - teacher prompt 能构造；
   - teacher/student response token 完全对齐；
   - 目标动作 mask 非空；
   - `teacher_kl_scoped` 有限；
   - 非目标动作的 teacher 系数严格为 0。

### 阶段 O-B：短程筛选

每个新实验跑 100 或 250 step，只用于排除：

- loss/grad norm 异常；
- 输出长度发散；
- teacher KL 不生效；
- Answer 输出率明显下降；
- Query 重复率明显上升。

短程结果不进入论文主表。

### 阶段 O-C：正式训练

通过 smoke 的 O2/O3/O4 均跑 1000 step，每 250 step 保存。数据顺序、随机
种子、训练卡数和 rollout 配置与 O5 固定一致。

### 阶段 O-D：全量评测

对训练曲线选出的 checkpoint 在三个完整 dev 集上评测：

- HotpotQA：7405；
- 2Wiki：12576；
- MuSiQue：2417。

最终报告 EM、F1、Cover-EM、回答率、max-turn、平均轮数和空 evidence 率。
主结论必须使用 paired bootstrap 置信区间和双侧 p 值。

## 7. 成功与停止条件

### 成功条件

满足以下条件才将 Action-Causal OPSD 作为论文核心：

1. O5 在至少两个数据集上稳定优于 O2；
2. O5 相对 O3 的提升能证明 Query teacher 有实际价值；
3. O5 相对 E15 的主要指标提升通过显著性检验；
4. 三个数据集均无超过 0.5pt 的系统性退化；
5. 无训练/dev question 精确重叠。

### 停止条件

若 O5 与 O2/O3 基本持平，则停止扩大 OPSD 参数搜索。论文叙事降级为：

```text
无 reward 的 on-policy 自蒸馏是 DPO 的有效替代方案，
动作级隔离作为稳定性设计，而非主要性能贡献。
```

## 8. 论文结构建议

1. **问题**：离线 DPO 无法覆盖 Agentic RAG 自主检索产生的状态分布；
2. **观察**：不同动作需要不同特权信息，统一 teacher 会造成因果越界；
3. **方法**：Action-Causal OPSD；
4. **实验**：SFT、DPO、Vanilla OPSD、动作消融、完整 Action-Causal OPSD；
5. **分析**：状态分布、Query 行为、效率、失败类型；
6. **限制**：依赖训练集标准答案/支持证据，当前 sampled-token 目标并非全词表 KL。

## 9. 本方向会话的文件边界

本方向会话建议只负责：

- 本文件及 OPSD 专属论文材料；
- 新增 `run_icassp_opsd_*.sh` 顶层脚本；
- OPSD teacher prompt 数据构造与动作消融；
- OPSD 训练、评测与显著性统计。

为避免与方向二冲突，不应修改：

- `../ms-swift/swift/rl_core/advantage.py`；
- `../ms-swift/swift/rlhf_trainers/grpo_trainer.py`；
- Action-Causal GRPO 专属实现。

如确实需要修改共享核心，先记录需求，由两个方向统一协调后再改。
