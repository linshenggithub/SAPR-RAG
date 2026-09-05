# SAPR-OPD：面向 Agentic RAG 的外部教师 On-Policy Distillation 方案

**版本**：v1  
**日期**：2026-09-03  
**目标会议**：ICASSP  
**目标任务**：HotpotQA、2WikiMultiHopQA、MuSiQue 多跳问答  
**核心指标**：EM、token F1、Cover-EM  
**主要对照**：ReasonRAG、SFT、SFT+DPO、SFT+GRPO  
**当前状态**：方案设计，尚未开始实现或训练

---

## 0. 摘要

ReasonRAG 使用 MCTS 生成离线过程偏好，并通过 DPO 学习 Query
Generation、Evidence Extraction 和 Answer Generation。它比只使用最终答案奖励
更细粒度，但训练数据仍由固定教师策略和固定搜索状态产生。部署时，模型根据自己的
历史查询和检索结果继续决策；一旦早期查询偏离，模型会进入离线偏好数据没有覆盖的
状态，后续错误持续累积。

本方案提出 **SAPR-OPD**：先用 SFT 建立稳定的多轮检索协议，再让 student
在真实检索环境中生成 on-policy trajectory，由独立、冻结、能力更强的 teacher
在完全相同的学生访问状态上提供逐 token 密集监督。为避免 teacher 过度约束已经
正确的搜索路径，只对失败或低质量轨迹启用 teacher signal，并对 Query、Reasoning
和 Answer token 分别归一化。

核心假设是：

> Agentic RAG 的主要训练困难不仅是缺少过程监督，而且是监督状态分布与部署状态
> 分布不一致。SFT 解决冷启动，state-aligned OPD 在 student 实际访问的状态上
> 提供密集纠错，两者结合应比离线过程偏好和稀疏 outcome RL 更有效。

---

## 1. 论文故事

### 1.1 ReasonRAG 已经解决了什么

ReasonRAG 的贡献是把单一终答奖励拆成过程级偏好：

```text
Query Generation
Evidence Extraction
Answer Generation
```

通过 MCTS 构造同一状态下的 chosen/rejected action，再用 DPO 学习过程偏好。
它证明了 process supervision 对 Agentic RAG 有价值。

### 1.2 ReasonRAG 还没有解决什么

ReasonRAG 的监督是 **offline and teacher-induced**：

```text
固定 MCTS/teacher 访问状态
  -> 构造 preference pairs
  -> DPO
  -> 部署时 student 自己生成新状态
```

而 Agentic RAG 是闭环序列决策：

$$
s_{t+1}=f(s_t,a_t,\mathcal R(a_t)).
$$

其中 $a_t$ 是 student 生成的 query 或 answer，$\mathcal R$ 是检索环境。一次
query drift、实体遗漏或错误停止会改变之后全部输入状态。离线 DPO 能告诉模型在
已收集状态上哪个动作更好，却不能保证覆盖当前 policy 实际产生的新错误状态。

### 1.3 为什么普通 GRPO 不够

普通 GRPO 让 student 在自己的状态分布上训练，但主要依赖轨迹级 EM/F1 和
证据覆盖奖励：

- 最终奖励稀疏；
- 同题多条 rollout 经常同分，group advantage 接近零；
- 无法确定失败由哪一个 query、evidence 或 stop 决策引起；
- 为提高 evidence coverage 而持续检索，可能破坏回答率和停止行为。

本项目已有实验也显示：LoRA GRPO 与 SFT 基本持平；全参数 GRPO 虽提高答案
精确度，却降低回答率和 Cover-EM。

### 1.4 为什么使用 OPD

On-Policy Distillation 同时保留两项性质：

| 性质 | 来源 |
|---|---|
| 训练状态来自 student 当前 policy | 类似 RL，降低 train-inference mismatch |
| 每个生成 token 都有 teacher signal | 类似 KD，缓解稀疏 reward 和 credit assignment |

因此 OPD 对 Agentic RAG 的价值不是“teacher 知道标准答案”，而是：

> teacher 在 student 真正遇到的 query/history/retrieval state 上，告诉 student
> 当前生成分布应该如何调整。

### 1.5 与当前 OPSD 的区别

| 维度 | 当前 OPSD | 本方案 OPD |
|---|---|---|
| teacher | student 同源模型 | 独立冻结强模型 |
| teacher 输入 | gold answer、gold evidence、R3 plan 等特权提示 | 与 student 相同的真实因果上下文 |
| teacher 优势 | 特权信息 | 模型能力 |
| teacher 作用范围 | 按动作持续注入 | 优先只作用失败轨迹 |
| 主要风险 | 信息边界错配、自我确认 | teacher ceiling、算力和 token 对齐 |

本方案不把旧 OPSD 改名，而是建立一个独立的外部 teacher 对照。

---

## 2. 研究问题与假设

### RQ1：On-policy 状态是否重要

在相同 SFT 起点和 teacher 下，student-generated trajectory 上的 OPD 是否优于
固定 teacher trajectory 上的离线 KD？

**H1**：OPD 的 EM/F1 高于 off-policy KD，因为 teacher 能覆盖 student 实际
发生的 query drift、重复检索和错误停止状态。

### RQ2：选择性 teacher guidance 是否重要

只对失败轨迹施加 teacher KL，是否优于对全部轨迹统一施加 KL？

**H2**：failed-only OPD 优于 uniform OPD，因为正确轨迹保留自主探索，不会被
拉回 teacher 的单一路径。

### RQ3：动作平衡是否重要

Query、Reasoning、Answer 分别归一化的 OPD 是否优于全 token 平均？

**H3**：action-balanced OPD 能提高检索覆盖并减少重复查询，避免大量 answer /
reasoning token 淹没较短的 query token。

### RQ4：OPD 是否真正改善 Agentic 行为

EM/F1 提升是否伴随更好的检索覆盖和停止行为？

**H4**：有效方法应同时改善最终答案指标，并且不显著降低回答率或增加最大轮次率。

---

## 3. 方法

### 3.1 基本符号

- $x$：原始问题；
- $s_t$：第 $t$ 轮开始时的真实状态；
- $a_t$：student 在该轮生成的 Query、Reasoning 或 Answer token；
- $\mathcal R(a_t)$：检索器对 query 的真实返回；
- $\tau=(s_1,a_1,\ldots,s_T,a_T)$：student 完整 on-policy trajectory；
- $\pi_\theta$：待训练 student；
- $\pi_T$：独立冻结 teacher；
- $m_t$：模型生成 token mask，检索 observation token 为 0。

状态定义：

```text
s_t = {
  original_question,
  student_generated_query_history,
  actual_retrieved_top3,
  evidence_history,
  remaining_turn_budget
}
```

### 3.2 阶段 A：SFT 冷启动

直接复用：

```text
03_sapr_rag/saves/qwen2_5_7b/lora/sft/checkpoint-1650
```

SFT 已经完成：

- 多轮 `<query>/<evidence>/<answer>` 协议学习；
- 基本 query decomposition；
- 基本 evidence extraction；
- 从零样本约 45% 的 HotpotQA 最大轮次率降到约 10.7%。

首轮 OPD 不重新训练 SFT，避免同时改变数据、初始化和蒸馏方法。

### 3.3 阶段 B：Student on-policy rollout

Student 使用当前项目固定流程：

```text
Question
  -> student reasoning/query
  -> BGE + FAISS Top-3
  -> Evidence Agent
  -> 更新 query/evidence history
  -> 下一轮 query 或 answer
```

固定配置：

- 最大 6 轮；
- 每轮 Top-3；
- BGE `bge-base-en-v1.5`；
- `wiki18_extended` 语料；
- 同一 FAISS index；
- retrieval observation 作为 user message；
- retrieval/evidence 环境 token 不参与 loss。

### 3.4 State-Aligned External Teacher

Teacher 必须在 student 已生成的同一 token prefix 上计算：

$$
\log \pi_T(y_t|x,y_{<t},\mathcal R).
$$

必须满足：

1. teacher 与 student 看到同一个原问题；
2. teacher 看到 student 已生成的 query 和 history；
3. teacher 看到 student query 实际召回的相同 Top-3；
4. teacher 不看 gold answer、gold supporting facts 或 R3 query plan；
5. teacher 不重新生成另一条轨迹，只计算 student sampled token 的 log-prob；
6. 所有 observation token 使用相同 `response_loss_mask=0`；
7. student/teacher sampled-token 数量必须完全一致。

这保证 teacher signal 是部署状态下可解释的能力指导，而不是特权信息泄漏。

### 3.5 OPD 基础信号

在 student sampled token 上使用 signed teacher log-ratio：

$$
d_{i,t}
= \log\pi_T(y_{i,t}|s_{i,t})
- \log\pi_{\theta_{\mathrm{old}}}(y_{i,t}|s_{i,t}).
$$

它对应 sampled-token reverse-KL 的 policy-gradient 估计。若 teacher 比 student
更认可某 token，则 $d_{i,t}>0$；反之为负。

### 3.6 Failure-Selective Gate

第一版严格采用二值 EM gate：

$$
g_i=\mathbb 1[\operatorname{EM}(\hat y_i,y_i^*)=0].
$$

Teacher signal：

$$
A^{OPD}_{i,t}
= \beta g_i m_{i,t}d_{i,t}.
$$

含义：

- 轨迹回答正确：$g_i=0$，不接受 teacher 约束；
- 轨迹回答错误或未回答：$g_i=1$，接受 teacher 纠偏；
- 环境 observation：$m_{i,t}=0$，不更新。

第一版不使用 `1-F1` soft gate。原因是长答案即使事实错误也可能与 gold 有局部词元
重叠，导致错误轨迹只获得很弱的 teacher signal。

### 3.7 两种训练目标

#### 主实验：纯 Selective OPD

不使用 GRPO group advantage，只优化失败轨迹上的 teacher signal：

$$
A_{i,t}=A^{OPD}_{i,t}.
$$

该设置对应论文标题中的 `SFT + OPD`，能最干净地回答 OPD 本身是否有效。

#### 扩展实验：Selective OPD-RL

将 OPD 与现有任务奖励结合：

$$
A_{i,t}
=A_i^{GRPO}
+\beta g_i m_{i,t}d_{i,t}.
$$

其中：

$$
R_i =
1.0R_{F1}
+0.2R_{relevance}
+0.05R_{format}.
$$

主表应分别报告纯 OPD 与 OPD-RL，不能把两者合并后只称为 OPD。

### 3.8 Action-Balanced OPD

定义三类模型 token mask：

```text
m_query
m_reason
m_answer
```

每类 token 单独归一化：

$$
\mathcal L_{AB}
= \lambda_q
\frac{\sum_t m^q_t\ell_t}{\sum_t m^q_t+\epsilon}
+ \lambda_r
\frac{\sum_t m^r_t\ell_t}{\sum_t m^r_t+\epsilon}
+ \lambda_a
\frac{\sum_t m^a_t\ell_t}{\sum_t m^a_t+\epsilon}.
$$

初始建议：

```text
lambda_query  = 1.0
lambda_reason = 0.5
lambda_answer = 1.0
```

这里三个动作共享同一个 state-aligned teacher，不构造三个 privileged prompt。
该设计只解决 token 数量不平衡，不改变 teacher 的因果条件。

### 3.9 Reference KL

保留 student 相对 SFT 起点的弱 reference KL：

```text
beta_reference = 0.01 ~ 0.04
```

Teacher KL 与 reference KL 作用不同：

- teacher KL：将失败状态推向强 teacher；
- reference KL：防止整个策略偏离稳定 SFT 起点。

日志中必须分别记录，不能都命名为 `kl`。

---

## 4. Teacher 选择

### 4.1 硬约束

Teacher 必须：

- 与 student 使用相同 tokenizer 和词表；
- 支持对 student sampled token 返回 log-prob；
- 在相同 Agentic RAG pipeline 下运行；
- 冻结参数；
- 在目标任务上显著强于 SFT student；
- 不依赖 gold prompt 才能超过 student。

### 4.2 候选优先级

首选：

```text
Qwen2.5-14B-Instruct 或更强的同 tokenizer Agentic-RAG checkpoint
```

备选：

```text
Qwen2.5-32B-Instruct / 同族 search-RL checkpoint
```

不推荐作为第一版：

- API 黑盒 teacher：难以获得完整或 sampled-token log-prob；
- 不同 tokenizer teacher：student token 无法直接一一对齐；
- 当前 student 去掉 LoRA 后的 base model：能力未必更强，属于 self-distillation；
- 含 gold answer/evidence 的同模型 teacher：仍然回到 OPSD。

### 4.3 Teacher Ceiling Gate

正式训练前，teacher 必须在固定三数据集各 200 条上评测：

| 条件 | Go/No-Go |
|---|---|
| 宏平均 F1 比 SFT 高至少 5pt | Go |
| 每个数据集回答率不低于 SFT 2pt 以上 | Go |
| HotpotQA/2Wiki/MuSiQue 至少两个数据集 F1 明显更高 | Go |
| 只提高 EM，但 F1/Cover-EM 或停止行为明显更差 | No-Go |
| teacher 与 student 基本持平 | No-Go |

Teacher ceiling 不通过时，不能靠增大 $\beta$ 补救。

---

## 5. 数据方案

### 5.1 训练集

复用严格 train-derived 的三源数据：

| 数据集 | 数量 |
|---|---:|
| HotpotQA train | 90,447 |
| 2Wiki train | 167,454 |
| MuSiQue train | 19,938 |
| 合计 | 277,839 |

OPD 输入只保留：

```json
{
  "messages": [...],
  "golden_answers": [...],
  "gold_titles": [...],
  "gold_sup_sents": [...],
  "source": "hotpotqa | 2wiki | musique"
}
```

必须删除：

```text
teacher_prompt
teacher_query_prompt
teacher_evidence_prompt
teacher_answer_prompt
```

Gold answer 仅供 reward/gate 计算，不进入 teacher 或 student prompt。

### 5.2 采样策略

不能直接按原始数量混合，否则 2Wiki 占比约 60%。建议每个 optimizer step
按数据源均匀采样：

```text
HotpotQA : 2Wiki : MuSiQue = 1 : 1 : 1
```

每个 epoch 内允许重复采样较小的 MuSiQue，但验证集严格隔离。

### 5.3 验证集

固定：

- HotpotQA dev：7,405；
- 2Wiki dev：12,576；
- MuSiQue dev：2,417。

开发阶段使用每个数据集固定前 200 条，但 checkpoint 选择应使用固定、预先保存且
与最终主表区分的 selection split。

---

## 6. 工程实现

### 6.1 当前框架已有能力

相邻 `ms-swift` 已支持：

```text
--teacher_model <local_path>
--teacher_model_server <url>
--teacher_kl_coef <float>
```

并支持：

- student on-policy rollout；
- 本地冻结 teacher；
- 外部 teacher server；
- sampled-token teacher log-prob；
- observation loss mask；
- teacher log-ratio 注入 per-token advantage。

### 6.2 当前不能直接满足的部分

现有实现仍缺：

1. 按最终 EM/F1 构造 `teacher_sequence_gate_mask`；
2. 纯 OPD 时只使用 gate、关闭 GRPO base advantage；
3. state-aligned 外部 teacher 的完整 smoke；
4. action-balanced loss；
5. 正确/错误轨迹分别统计 teacher KL；
6. teacher/student tokenizer 和 sampled-token 对齐的启动前 fail-fast。

另外：

```text
teacher_action_scope=multi
```

当前只支持本地 self-distillation teacher，不能直接与外部
`teacher_model_server` 组合。第一版必须使用：

```text
teacher_action_scope=all
```

再通过同一 teacher 下的 token mask 做 action-balanced 统计，而不是多 teacher prompt。

### 6.3 建议代码改动

新增：

```text
03_sapr_rag/scripts/opd/
├── build_opd_dataset.py
├── evaluate_teacher_ceiling.sh
├── launch_teacher_server.sh
├── launch_sapr_opd.sh
├── run_opd_smoke.sh
└── audit_opd_alignment.py
```

修改：

```text
../ms-swift/swift/rlhf_trainers/grpo_trainer.py
../ms-swift/swift/rl_core/advantage.py
```

新增参数建议：

```text
--teacher_sequence_gate none|failed_em|failed_f1
--teacher_f1_gate_threshold 1.0
--opd_use_grpo_advantage true|false
--opd_action_balance true|false
--opd_query_weight 1.0
--opd_reason_weight 0.5
--opd_answer_weight 1.0
```

### 6.4 最小代码逻辑

在 reward 已计算、teacher log-prob 已取得后：

```python
failure_gate = (em_reward < 1.0).float().unsqueeze(1)
teacher_mask = completion_mask * failure_gate
teacher_signal = beta_opd * (
    teacher_per_token_logps - student_per_token_logps
) * teacher_mask

if opd_use_grpo_advantage:
    final_advantage = base_grpo_advantage[:, None] + teacher_signal
else:
    final_advantage = teacher_signal
```

必须先完成单元测试，再接真实模型。

### 6.5 必须新增的单元测试

1. 正确轨迹的 teacher mask 全为 0；
2. 错误轨迹的模型生成 token mask 为 1；
3. observation token 始终为 0；
4. 未回答轨迹归为失败；
5. 多别名答案的 EM gate 正确；
6. teacher/student sampled token 完全一致；
7. teacher/student tokenizer 不一致时启动失败；
8. pure OPD 的 base GRPO advantage 为 0；
9. action-balanced 三类 mask 不重叠且覆盖所有训练 token；
10. 无 query 或无 answer 时不会出现除零和 NaN。

---

## 7. 资源布局

### 7.1 8×H20 推荐布局

使用 14B teacher server：

| GPU | 角色 |
|---:|---|
| 0 | BGE + FAISS retrieval daemon |
| 1 | Qwen2.5-14B teacher log-prob server |
| 2-6 | 7B student LoRA training |
| 7 | student vLLM rollout |

Teacher server 需要支持：

- 接收完整 student trajectory；
- 对 prompt 中已存在的 student response token 返回 sampled-token log-prob；
- 禁止生成替代 response；
- 固定 temperature/eval mode；
- 返回 tokenizer/model revision。

如果 14B teacher 在 GPU1 上不能稳定处理最长轨迹：

1. 优先缩短 `max_completion_length` 到 2048；
2. 对历史 evidence 做确定性截断；
3. 再考虑 teacher tensor parallel；
4. 不优先使用 CPU offload，避免训练吞吐严重下降。

### 7.2 预计成本

OPD 每个 rollout 至少增加一次 teacher forward。相对 plain GRPO：

- student rollout 成本不变；
- retrieval 成本不变；
- 增加 teacher scoring；
- 不需要 teacher autoregressive generation。

必须记录：

```text
tokens/s
seconds/optimizer-step
teacher forward latency
GPU peak memory
teacher timeout/error rate
```

论文中应报告达到相同 EM/F1 所需的总训练 token 和 GPU hours，而不只报告 step 数。

---

## 8. 实验计划

### Phase 0：Teacher Ceiling

样本：

```text
HotpotQA 200
2Wiki 200
MuSiQue 200
```

比较：

```text
SFT student
SFT+DPO
候选 teacher-14B
候选 teacher-32B（如资源允许）
```

输出：

- EM/F1/Cover-EM；
- answered/max-turn/avg-turn；
- Top-3 evidence recall；
- 每题 teacher/student paired difference。

通过 Teacher Ceiling Gate 后才能继续。

### Phase 1：Token Alignment Smoke

规模：

```text
32 prompts × 2 rollouts × 1 optimizer step
```

验收：

- teacher/student token id 完全一致；
- teacher log-prob 全部有限；
- observation mask 正确；
- correct trajectory teacher scope ratio = 0；
- failed trajectory teacher scope ratio > 0；
- loss、gradient norm 无 NaN/Inf；
- LoRA rollout 与训练 policy 同步。

### Phase 2：小规模方法筛选

训练：

```text
每数据集 1,000 个 train prompt
三源均匀采样
100--300 optimizer steps
每题 4 rollouts
checkpoint every 25/50 steps
```

对照：

| ID | 方法 | 唯一变化 |
|---|---|---|
| O0 | SFT | 不训练 |
| O1 | SFT + off-policy KD | teacher trajectory |
| O2 | SFT + pure uniform OPD | 所有 student trajectory |
| O3 | SFT + pure selective OPD | 仅失败轨迹 |
| O4 | SFT + selective OPD-RL | 加 GRPO base advantage |
| O5 | SFT + action-balanced selective OPD-RL | 主方法完整体 |

筛选指标：三个固定 200 集合的宏平均 F1，EM 和 Cover-EM 作为共同约束。

进入下一阶段的条件：

- O3/O4/O5 至少一个相对 SFT 宏平均 F1 提高 1pt；
- 至少两个数据集 F1 正增益；
- 任一数据集 F1 不下降超过 1.5pt；
- 回答率下降不超过 1pt；
- 最大轮次率增加不超过 2pt；
- 无持续长输出、NaN、OOM 或 teacher timeout。

### Phase 3：完整训练

从 Phase 2 只选择一个最佳配置：

```text
train data: 277,839
max steps: 先 1,000，再根据趋势扩到 3,000
save steps: 250 或 500
selection: 三数据集固定 selection split
```

不应一次性承诺完整 epoch。当前数据下完整 epoch 成本过高，应使用固定 token budget
与 ReasonRAG/GRPO 对照。

### Phase 4：全量评测

候选 checkpoint 在三个完整 dev 集评测：

| 数据集 | N |
|---|---:|
| HotpotQA | 7,405 |
| 2Wiki | 12,576 |
| MuSiQue | 2,417 |

对每个本地 baseline 做同 ID paired bootstrap：

```text
20,000 resamples
95% CI
two-sided p-value
```

主结论要求：

- 宏平均 F1 优于 SFT+DPO；
- 至少两个数据集 F1 达到显著提升；
- HotpotQA 至少不低于当前 E12 checkpoint-1000；
- 回答率和最大轮次率没有明显退化。

---

## 9. Baseline 与消融矩阵

### 9.1 必要 Baseline

| Baseline | 作用 |
|---|---|
| ReasonRAG 论文结果 | 外部已发表基线 |
| 本地 ReasonRAG/DPO-only | 当前检索环境下的过程偏好基线 |
| SFT checkpoint-1650 | OPD 起点 |
| SFT+DPO checkpoint-395 | 当前强本地基线 |
| 严格 LoRA GRPO-control | 排除收益只来自 online RL |
| E12 SFT→分动作 OPSD checkpoint-1000 | 当前最好结果，不能因方法转向而隐藏 |
| Teacher | OPD 能力上限参考 |

### 9.2 核心消融

1. Off-policy KD vs on-policy distillation；
2. Uniform OPD vs failed-only OPD；
3. Pure OPD vs OPD-RL；
4. 全 token 平均 vs action-balanced；
5. external teacher vs self teacher；
6. EM binary gate vs F1 soft gate；
7. 关闭 Query token OPD；
8. 关闭 Answer token OPD；
9. teacher 14B vs 32B；
10. teacher entropy 高低分桶。

### 9.3 不允许同时变化的因素

主对比中必须固定：

- student SFT 起点；
- train question IDs；
- 检索器、索引和 corpus；
- Evidence Agent；
- Top-k 和 max turns；
- rollout temperature；
- optimizer、学习率和 LoRA rank；
- 总训练 token budget；
- checkpoint selection split。

---

## 10. 指标与诊断

### 10.1 最终答案指标

- EM；
- token F1；
- Cover-EM；
- LLM-acc，仅作为补充；
- paired bootstrap CI 与 p 值。

### 10.2 Agentic 行为指标

- Answered rate；
- Max-turn rate；
- Avg turns；
- Query exact-repeat rate；
- 每轮 Top-3 gold evidence recall；
- 完整 gold evidence recall；
- Empty evidence rate；
- 首轮与后续轮 query hit ratio。

### 10.3 OPD 专属指标

- teacher gate coverage；
- successful/failed trajectory teacher KL；
- Query/Reasoning/Answer token 数量；
- 各动作平均 teacher log-ratio；
- teacher entropy；
- teacher/student top-1 agreement；
- teacher scoring latency；
- 每步有效蒸馏 token 数；
- KL 与最终 EM/F1 改善的相关性。

### 10.4 训练稳定性

- completion length p50/p95/max；
- clipped ratio；
- loss mean/max；
- gradient norm p50/p95/max；
- reward zero-std ratio；
- OOM、HTTP timeout、NCCL 和 weight-sync 错误。

---

## 11. 风险与止损条件

### 风险 1：Teacher 不够强

**表现**：teacher ceiling 与 SFT 接近，或多跳数据集更差。  
**处理**：停止 OPD；更换 search-trained teacher，而不是调大 KL。

### 风险 2：Teacher 修不了 retrieval miss

**表现**：student 当前状态没有召回关键证据，teacher 只能在错误上下文中重新措辞。  
**处理**：单独统计“gold evidence 已召回/未召回”两组收益；若提升只发生在已召回组，
论文应定位为 evidence utilization，而不是 retrieval improvement。

### 风险 3：Reverse KL 导致模式坍缩

**表现**：query 多样性下降、重复率上升、teacher entropy 高的位置梯度异常。  
**处理**：降低 $\beta$；对高熵决策位加入 top-k forward KL；保留正确轨迹自由探索。

### 风险 4：外部 Teacher 成本过高

**表现**：teacher forward 成为吞吐瓶颈。  
**处理**：sampled-token log-prob、请求批处理、只处理失败轨迹、缓存 teacher score；
再考虑 OVD 式 verbal score。

### 风险 5：提升只来自答案风格

**表现**：EM 上升但 F1、Cover-EM、检索覆盖不升。  
**处理**：不能声称搜索策略改善；增加 Query token 消融与 retrieval diagnostics。

### 风险 6：OPD 不超过当前 E12

当前 E12 checkpoint-1000：

```text
HotpotQA EM       0.4086
HotpotQA F1       0.5379
HotpotQA Cover-EM 0.4984
```

OPD 若只超过 ReasonRAG、但没有超过该内部最好结果，仍可作为“更可信、无 privileged
context 的替代方案”，但不能声称达到项目最佳性能。论文必须同时报告 E12。

---

## 12. ICASSP 论文叙事

### 12.1 推荐题目

首选：

> **SAPR-OPD: State-Aligned Selective On-Policy Distillation for Agentic Retrieval-Augmented Generation**

备选：

> **Learning from On-Policy Search Failures for Multi-Hop Retrieval-Augmented Generation**

### 12.2 核心 Claim

> Offline process preference learning improves Agentic RAG on teacher-collected
> states, but does not supervise recovery under the evolving student's own state
> distribution. SAPR-OPD performs selective external-teacher distillation on
> student-generated retrieval trajectories, providing dense correction only when
> the trajectory fails.

### 12.3 三个贡献点

1. **Problem**：指出 Agentic RAG 离线过程监督中的 policy-induced state
   distribution shift。
2. **Method**：提出 state-aligned、failed-only、action-balanced external-teacher
   OPD。
3. **Evidence**：在三个多跳 QA 数据集上同时报告 EM/F1、检索覆盖、停止行为和
   显著性检验。

### 12.4 不应使用的表述

- 不说“首次将 OPD 用于 Agentic RAG”，ACL 2026 DGPO 已经做过；
- 不说“OPD 必然优于 RL”，DGPO 中 pure GKD 明显弱于 selective hybrid；
- 不把 gold answer 进入 teacher prompt 的 OPSD 称为 state-aligned OPD；
- 不把 EM 提升自动解释为 query improvement；
- 不隐藏当前 E12 的更强内部结果。

### 12.5 与 DGPO 的差异必须成立

仅复现 DGPO 不足以构成 ICASSP 方法贡献。至少需要证明一个 RAG 特定增量：

```text
DGPO: trajectory-level failed-only KL
SAPR-OPD: failed-only KL
          + exact state alignment under multi-turn retrieval
          + environment-token masking
          + action-balanced distillation
          + EM/F1/behavior joint evaluation
```

最关键消融是：

```text
DGPO-style failed-only OPD
vs
failed-only + action-balanced OPD
```

若 action-balanced 没有稳定收益，论文应收缩为严格复现与系统分析，不应硬写成新方法。

---

## 13. 里程碑

### M0：方案冻结

- [ ] 确认 student checkpoint；
- [ ] 确认 teacher 候选；
- [ ] 冻结 retrieval/eval pipeline；
- [ ] 分配新的实验 ID。

### M1：Teacher Ceiling

- [ ] 三数据集各 200 条；
- [ ] 检查 EM/F1/Cover-EM；
- [ ] 通过 Go/No-Go。

### M2：OPD 最小链路

- [ ] 独立 teacher log-prob server；
- [ ] sampled-token alignment；
- [ ] failed-only gate；
- [ ] 32×2 rollout smoke；
- [ ] 单元测试全部通过。

### M3：小规模消融

- [ ] O1 off-policy KD；
- [ ] O2 uniform OPD；
- [ ] O3 selective OPD；
- [ ] O4 selective OPD-RL；
- [ ] O5 action-balanced 完整体。

### M4：全量验证

- [ ] 三数据集全量 dev；
- [ ] paired bootstrap；
- [ ] badcase 与行为指标；
- [ ] 效率与算力统计。

### M5：论文材料

- [ ] 方法图；
- [ ] 主结果表；
- [ ] 消融表；
- [ ] teacher ceiling 表；
- [ ] query/evidence/stop case study；
- [ ] limitation 与 compute budget。

---

## 14. 最终决策标准

只有同时满足以下条件，才将 SAPR-OPD 作为 ICASSP 主方法：

1. 外部 teacher ceiling 明显高于 SFT；
2. selective OPD 稳定优于 uniform OPD；
3. 完整方法宏平均 F1 优于 SFT+DPO；
4. 至少两个数据集获得一致增益；
5. 回答率和最大轮次率不退化；
6. 增益不能完全由答案长度变化解释；
7. action-balanced 或其他 RAG 特定设计通过消融；
8. 训练与评测数据严格隔离；
9. 结果能在至少两个随机种子或全量 paired bootstrap 下成立。

若只满足“超过 ReasonRAG 论文值”，但没有超过本地 SFT+DPO 或 E12，则该方案不应
作为“性能最优方法”投稿，只能定位为更可信的外部教师训练机制分析。

---

## 15. 相关材料

- OPD 文献调研：`01_literature/related_work_drafts/opd_agentic_rag_survey.md`
- DGPO 深度笔记：`01_literature/paper_notes/2026_DGPO.md`
- 当前实验台账：`docs/experiment_tracker.md`
- GRPO/OPSD 代码链路：`docs/grpo_opsd_pipeline_overview.md`
- 当前 ms-swift 补丁：`docs/ms_swift_local_patches.md`
- DGPO 官方论文：https://aclanthology.org/2026.acl-long.1751/
- DGPO 官方代码：https://github.com/omron-sinicx/dgpo
- GKD 论文：https://openreview.net/forum?id=3zKtaqxLhW

