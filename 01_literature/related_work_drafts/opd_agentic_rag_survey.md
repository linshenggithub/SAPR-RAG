# OPD 用于 Agentic RAG 的顶会文献调研

**调研日期**：2026-09-03  
**目标**：判断是否应从当前 OPSD 路线转向外部教师驱动的 On-Policy Distillation（OPD），以及怎样以 HotpotQA、2WikiMultiHopQA、MuSiQue 的 EM/F1 为目标设计实验。  
**检索范围**：ACL Anthology、OpenReview、arXiv、论文官方项目页与官方代码仓库。

## 0. 结论先行

1. **严格满足“顶会正式发表 + Agentic RAG + OPD + 报告 QA EM/F1”的论文，目前只确认到一篇：ACL 2026 长文 DGPO。**它报告 EM，不报告 token F1。
2. DGPO 的有效设计不是“对所有学生 token 做教师 KL”，而是两阶段训练：
   - 先用正确 teacher-generated outputs 做离线 KD 冷启动；
   - 再让学生在真实检索环境中 rollout，答对时用 outcome RL，答错时才启用外部教师 KL。
3. DGPO 在 Qwen2.5 3B→0.5B 设置下，将七个 QA 数据集平均 EM 从 PPO 的 0.238、纯 GKD 的 0.240、普通 KD 的 0.298 提升到 0.329；HotpotQA EM 为 0.342，超过 3B teacher 的 0.340。
4. 对当前 SAPR-RAG，最值得复现的不是继续给同一个 7B 模型添加 privileged prompt，而是引入**独立、冻结、能力更强的外部 teacher**，采用“正确轨迹自由优化、错误轨迹选择性蒸馏”的 DGPO 结构。
5. 不能直接假设 DGPO 会提升 F1。论文只用 EM 训练和评测；本项目必须额外验证 token F1、Cover-EM、回答率、最大轮次率和检索覆盖率。

## 1. 术语与证据边界

### 1.1 本文中的 OPD

本文将 OPD 定义为：

```text
学生策略生成当前轨迹 y ~ pi_student
  -> 冻结的外部 teacher 在相同学生访问状态/前缀上提供分布或评价
  -> 用 token-level KL、log-ratio 或等价密集信号更新 student
```

它与当前项目 OPSD 的区别是：

| 维度 | 外部教师 OPD | 当前项目 OPSD |
|---|---|---|
| teacher 参数 | 独立、更强、冻结 | 与 student 同源或同权重 |
| teacher 优势来源 | 模型能力差距 | privileged prompt / gold 信息 |
| 主要风险 | teacher/student 词表与算力成本 | 特权信息错配、自我确认、动作因果泄漏 |
| 典型论文 | GKD、DGPO | Self-Distilled Reasoner、SDFT |

### 1.2 “顶会 + Agentic RAG + OPD”交集很小

检索到的论文分成三组：

1. **直接证据**：DGPO，ACL 2026，Agentic RAG，学生在线轨迹，外部 teacher KL，报告七个 QA 数据集 EM。
2. **方法基础或机制补充**：GKD、EOPD、SCoRe，均已发表顶会，但不是专门为 Agentic RAG 的 EM/F1 设计。
3. **高度相关但未正式顶会发表**：OVD、ATOD。它们分别直接覆盖 Web QA EM 和多轮搜索 Agent，但截至调研日只有 CoRR/arXiv 状态。

因此，不能写成“已有大量顶会工作证明 OPD 能提高 Agentic RAG 的 EM/F1”。更准确的表述是：

> ACL 2026 DGPO 已提供直接 EM 证据；ICLR/ICML 的 OPD 与 agent distillation 工作提供训练目标、稳定性和错误纠偏机制依据；F1 增益仍需由本项目自行验证。

## 2. 核心论文对比

| 论文 | Venue | 与 Agentic RAG 的关系 | 是否严格 OPD | 指标 | 对本项目价值 |
|---|---|---|---|---|---|
| DGPO | ACL 2026 Long | 直接训练多轮搜索 Agent | 是，错误轨迹选择性 teacher KL | EM、Hit Ratio、Search Steps | 最高，直接复现对象 |
| GKD | ICLR 2024 | 通用 OPD 基础，不是 RAG | 是 | ROUGE、BLEU、Accuracy | 定义 on-policy 数据与 KL 目标 |
| EOPD | ICML 2026 | 通用推理 OPD，不是 RAG | 是 | Avg@8、Pass@8 | 防止 reverse-KL 在高熵 token 上模式坍缩 |
| SCoRe | ICML 2026 | 多步 Agent/DeepSearch，相邻工作 | 否，学生中心纠错 + SFT + 短程 RL | Accuracy/EM 类结果 | 用“最早错误”缩短 credit assignment |
| SDFT | ICML 2026 Spotlight | 通用工具使用与持续学习 | 同模型自蒸馏 | Accuracy、forgetting | 说明同模型特权视图依赖强 ICL，但不解决本项目对 OPSD 的不信任 |
| OVD | CoRR 2026 | Web QA + Search Agent | 广义 OPD，teacher verbal score | Web QA 平均 EM | 黑盒 teacher、低显存方案，非顶会证据 |
| ATOD | arXiv 2026 | 多轮 Agent + Search-QA | 是，turn-aware OPD+RL | Success Rate | 轮次加权和退火方案，非顶会证据 |

## 3. 直接证据：DGPO

### 3.1 问题定义

[论文明确提出] 0.5B/1B 小模型直接做 Agentic RAG RL 时，早期学生轨迹质量接近零，导致奖励稀疏、探索质量差、训练容易崩溃。纯 on-policy GKD 同样会被低质量 student-generated outputs（SGOs）拖累。

这与 SAPR-RAG 的现象同构：

- 普通 LoRA GRPO 没有稳定超过 SFT；
- 同题采样组经常没有足够 reward variance；
- 全参数 GRPO 虽改变了策略，却破坏回答率和停止行为；
- 当前 OPSD 对所有或按动作 token 持续注入 teacher signal，仍可能过度约束正确轨迹。

### 3.2 两阶段方法

#### 阶段 A：正确教师轨迹冷启动

只保留 teacher 的正确轨迹，用 hard-label CE 与 forward KL 初始化 student：

$$
\mathcal{L}_{distill}
= \mathcal{L}_{CE}(\pi_g,\pi_\theta)
+ \lambda D_{KL}\left[\pi_g(\cdot|x)\|\pi_\theta(\cdot|x)\right].
$$

这里的目的不是最终优化，而是先让弱 student 获得基本的 `<think>/<search>/<answer>` 行为，避免一开始几乎全部 rollout 都是无效样本。

#### 阶段 B：学生在线 rollout + 选择性教师纠偏

学生在真实搜索环境中生成完整轨迹。最终答案正确时保留 RL 正奖励；最终答案错误时，才让 frozen teacher 提供 KL 纠偏：

$$
r_\phi(x,y)=
\begin{cases}
1, & y=y^*,\\
-\beta D_{KL}\left[\pi_\theta(y|x;\mathcal R)\|\pi_g(y|x;\mathcal R)\right],
& y\neq y^*.
\end{cases}
$$

官方代码的实际实现是：

```python
reward_condition = (token_level_scores < 0.1).all(dim=1).float()
reward_mask = reward_condition.unsqueeze(1).expand(-1, response_length)
token_level_rewards = token_level_scores - beta * kld * reward_mask
```

即只有 reward 接近 0 的整条失败轨迹启用 teacher KL。检索返回的外部 token 通过 `info_mask` 排除，不参与训练。

### 3.3 主结果

Qwen2.5-3B teacher → Qwen2.5-0.5B student，训练集为 NQ + HotpotQA，测试七个 QA 数据集：

| 方法 | HotpotQA EM | 2Wiki EM | MuSiQue EM | 七集平均 EM |
|---|---:|---:|---:|---:|
| PPO | 0.205 | 0.218 | 0.041 | 0.238 |
| GKD | 0.216 | 0.217 | 0.055 | 0.240 |
| 普通 KD | 0.286 | 0.284 | 0.091 | 0.298 |
| DGPO | **0.342** | **0.303** | **0.120** | **0.329** |
| Teacher-3B | 0.340 | 0.368 | 0.135 | 0.353 |

关键消融：

| 方案 | 七集平均 EM |
|---|---:|
| DGPO | **0.329** |
| 无 KD 冷启动 | 0.320，且约 step 800 后训练崩溃 |
| 对所有轨迹统一加 KL | 0.314 |
| KD 后只做 PPO，不用 teacher guidance | 0.306 |
| 先 PPO 再 KD | 0.286 |

[论文明确提出] “先 KD、后 RL”与“只对失败轨迹加 teacher KL”都不可缺少。  
[基于官方代码核验] 选择性 KL 的 gate 是整条轨迹最终 EM，而不是 query/evidence 级局部正确性。

### 3.4 对结果的谨慎解释

- DGPO 的所有主表指标是 EM，不是 token F1。
- HotpotQA 上 DGPO 仅比 3B teacher 高 0.2pt，不能据此声称稳定超越 teacher。
- 2Wiki 和 MuSiQue 仍低于 teacher，说明 teacher gap 在更难多跳任务上没有消失。
- Query rewriting 的 Hit Ratio 上 PPO 反而最好；DGPO 的最终 EM 提升不等价于每个局部搜索能力都最强。
- 每个设置只训练一次，论文未报告主表置信区间。
- 论文核心场景是 0.5B/1B compact student；本项目 7B student 的迁移收益不能直接外推。

## 4. 顶会方法基础

### 4.1 GKD：OPD 的基础形式

**论文**：*On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*，ICLR 2024。

[论文明确提出] GKD 让 student 在自身生成序列上学习 teacher 的 token distribution，以缓解固定 teacher outputs 带来的 train-inference mismatch。它允许在 teacher/offline 与 student/on-policy 序列间通过采样比例 $\lambda$ 混合，并比较 forward KL、reverse KL 和 generalized JSD。

对 SAPR-RAG 的启发：

- 必须让 teacher 评价 student 当前真实 rollout，而不是只训练固定教师轨迹；
- student rollout 已经偏离 teacher 路径时，OPD 仍可在其实际访问前缀上给密集反馈；
- divergence 不是无关紧要的超参：reverse KL 更偏 mode-seeking，forward KL 更偏 mode-covering；
- GKD 本身在 DGPO 的 Agentic RAG 实验中只达到 0.240 平均 EM，说明“直接把纯 OPD 套上去”并不够。

### 4.2 EOPD：高熵位置不要只用 reverse KL

**论文**：*Entropy-Aware On-Policy Distillation of Language Models*，ICML 2026。

[论文明确提出] 标准 reverse-KL OPD 在 teacher 高熵位置会降低生成多样性并产生不稳定信号。EOPD 在 teacher entropy 超过阈值时额外加入 top-k forward KL，在低熵位置保留 reverse KL。

对 Agentic RAG 的迁移意义：

- `<query>` 首 token、实体选择、继续搜索/回答的分叉通常是高不确定决策点；
- 只用 reverse KL 可能把 student 过早压到 teacher 的单一路径；
- 若实现外部 teacher OPD，应至少记录 teacher token entropy，并对 `<query>/<answer>` 决策位做独立审计；
- 第一阶段可不直接实现 EOPD，但应保存 top-k teacher logits，为后续切换 divergence 留接口。

### 4.3 SCoRe：只修最早错误，而不是模仿完整轨迹

**论文**：*Student-Centered Distillation Narrows the Agentic Gap Between Small and Large LLMs*，ICML 2026。

SCoRe 不是严格 token-level OPD，但与本项目更贴近：

1. student 先生成多步 agent 轨迹；
2. teacher 找到最早错误并只做局部修正；
3. 用修正轨迹做 SFT；
4. 从错误前的 verified prefix 启动短程 RL。

它解决的是长轨迹中完整 teacher imitation 的误差累积和 credit assignment。对 SAPR-RAG 可转化为：

- 找到首个 query drift、重复 query、证据遗漏或 premature answer；
- 只对该决策点之后的一小段轨迹启用 teacher；
- 不让 teacher 重写整条成功前缀；
- 将“失败轨迹选择性 OPD”进一步细化为“首错位置后的局部 OPD”。

## 5. 未正式顶会发表但值得跟踪

### 5.1 OVD：黑盒 teacher 的低显存替代

OVD 使用 teacher 输出的 0--9 离散 verbal score 替代全词表 token logits，在 Web QA 上报告最高 +12.9pt 平均 EM。它允许 GPT/Claude 类黑盒模型参与 on-policy 监督，也显著降低长轨迹 logits 存储成本。

但截至调研日 OpenReview 标注为 CoRR 2026，不能作为“顶会已发表”证据。若本项目拿不到稳定的 14B/32B 本地 teacher logits，可把 OVD 作为第二方案。

### 5.2 ATOD：多轮 Agent 的 OPD→RL 退火

ATOD 在训练早期以 OPD 为主，随后逐渐提高 RL 权重，并按 turn 的 teacher/student disagreement 与 uncertainty 重加权。它在 ALFWorld、WebShop、Search-QA 上优于纯 OPD 和 GRPO。

但它截至调研日仍是 arXiv 预印本，且报告 success rate 而非 HotpotQA/2Wiki/MuSiQue 的 EM/F1。适合借鉴调度方式，不适合作为正式已发表 baseline。

## 6. 对 SAPR-RAG 的推荐方案

### 6.1 推荐主方案：Selective External-Teacher OPD

不建议把当前 OPSD 简单改名为 OPD。应做真正的外部教师对照：

```text
student: 当前 SFT checkpoint-1650 或 E12 checkpoint-1000
teacher: 冻结的更强同 tokenizer 模型
rollout: student 在当前 BGE+FAISS+Evidence Agent 环境中生成
gate: 轨迹 answer EM/F1 是否失败
correct rollout: 只用 GRPO outcome/process reward
failed rollout: GRPO reward + teacher/student token KL
```

推荐第一阶段使用整轨迹失败 gate，以最大限度对齐 DGPO：

$$
g_i = \mathbb{1}[\mathrm{EM}(y_i,y_i^*)=0].
$$

$$
A_{i,t}^{final}
= A_i^{GRPO}
- \beta g_i D_{KL}
\left[\pi_\theta(\cdot|s_{i,t})\|\pi_T(\cdot|s_{i,t})\right].
$$

随后再验证 soft gate：

$$
g_i = 1-\mathrm{F1}(y_i,y_i^*),
$$

但不能一开始只用 F1 soft gate，因为长答案的部分词重叠可能把事实错误轨迹误判为“半正确”。

### 6.2 教师条件必须与部署状态一致

这是从 OPSD 转 OPD 最重要的改变：

- teacher 与 student 都看到同一个原问题；
- teacher 与 student 都看到 student 实际生成的 query/history；
- teacher 与 student 都看到同一批真实 Top-3 检索结果；
- teacher 不看 gold answer、gold supporting facts 或 R3 reference plan；
- 检索 observation token 全部 mask，不参与梯度；
- teacher 只提供“在当前真实状态下下一 token 应如何分布”的能力差。

这样能去掉当前 OPSD 最大的不可靠来源：teacher 因知道 gold 信息而处在部署时不可能出现的状态。

### 6.3 Teacher 选择

优先级：

1. **同 tokenizer、同模型族的 14B/32B instruction 或 search-RL teacher**：token 对齐最简单，可直接做 full/top-k KL。
2. 同 tokenizer 的 7B 强 checkpoint：成本较低，但 teacher-student 能力差可能不足。
3. 不同 tokenizer 或黑盒 API teacher：不能直接做 token-level KL，应转向 OVD 式 verbal/trajectory feedback，不能伪装成标准 OPD。

必须先做 teacher ceiling：

| Gate | 判定 |
|---|---|
| Teacher HotpotQA/2Wiki/MuSiQue EM/F1 明显高于 student | 可进入 OPD |
| Teacher 只在 EM 高、Cover/F1 或停止行为差 | 不能直接蒸馏整轨迹 |
| Teacher 与 student 基本持平 | OPD 上限不足，停止投入 |

### 6.4 最小实验矩阵

固定同一 SFT 起点、同一训练集、同一 rollout、同一检索器和总 token budget：

| 组别 | 方法 | 用途 |
|---|---|---|
| A | SFT baseline | 起点 |
| B | SFT + GRPO | 判断 outcome/process reward 本身 |
| C | SFT + pure GKD/OPD | 判断全量 teacher KL |
| D | SFT + DGPO-selective OPD | 主方法 |
| E | SFT + DGPO，但 teacher=student | 与原 OPSD 隔离 teacher 能力因素 |
| F | SFT + DGPO，随机/统一 KL gate | 验证失败选择机制 |

建议先做 HotpotQA train 1k prompt、每题 4 条 rollout、100--300 optimizer step；每 25/50 step 在固定 200 条上筛选，只在出现稳定趋势后做 7,405 全量评测。

### 6.5 必须报告的指标

最终任务：

- EM、token F1、Cover-EM；
- 同 ID paired bootstrap 95% CI 和双侧 p 值。

行为指标：

- answered rate、max-turn rate、avg turns；
- query exact-repeat rate；
- Top-3 gold evidence recall；
- empty evidence rate；
- 正确/错误轨迹各自的 teacher KL；
- teacher gate 覆盖率及每动作 KL。

训练稳定性：

- reward group zero-std ratio；
- completion length p50/p95/max；
- gradient norm p95/max；
- teacher/student disagreement 与 teacher entropy。

## 7. 为什么该方案比当前 OPSD 更可信

| 当前问题 | DGPO/外部 OPD 的对应修正 |
|---|---|
| teacher 与 student 同源，teacher 未必更强 | 使用独立冻结强 teacher，并先测 ceiling |
| privileged prompt 改变决策条件 | teacher/student 使用同一真实检索状态 |
| gold answer teacher 可能压制查询行为 | 只对失败轨迹启用 teacher，而非所有轨迹 |
| 动作级 mask 复杂且容易错位 | 第一阶段采用论文已验证的整轨迹 gate |
| 正确轨迹仍被 teacher 拉回其模式 | 正确轨迹不施加 teacher KL，保留探索 |
| reverse KL 可能导致搜索路径单一 | 记录 entropy，必要时引入 EOPD 的高熵 forward KL |

## 8. 不应过度承诺的部分

1. DGPO 证明的是 compact student 的 EM 提升，不是 7B student 的必然收益。
2. DGPO 没有报告 token F1，本项目必须独立验证。
3. DGPO 使用 PPO+critic；当前项目主要是 GRPO。迁移时要把“选择性 teacher KL gate”作为唯一变量，不能同时改变优化器、rollout 协议和 teacher。
4. 外部 teacher OPD 显著增加显存和 forward 成本；8×H20 上需要重新规划 train/rollout/teacher/retrieval 资源。
5. 如果 teacher 不能在 student 的真实多轮状态上稳定给出更优分布，OPD 会把 teacher 的搜索偏差一起蒸馏。

## 9. 推荐决策

**建议暂停继续扩大现有 OPSD 训练，先做 DGPO 风格的外部 teacher 小规模对照。**

Go/No-Go 顺序：

1. 固定 student 起点和评测链路。
2. 完成外部 teacher 三数据集 200 条 ceiling。
3. 验证 teacher/student 在同一学生 rollout token 上可严格对齐。
4. 跑 B/C/D 三组 100--300 step 小实验。
5. 只有 D 在固定 200 条同时改善 F1、Cover-EM且不降低回答率，才扩全量。

## 10. 一手来源

- DGPO，ACL 2026 Long：[ACL Anthology](https://aclanthology.org/2026.acl-long.1751/)、[arXiv](https://arxiv.org/abs/2508.20324)、[官方项目页](https://omron-sinicx.github.io/dgpo/)、[官方代码](https://github.com/omron-sinicx/dgpo)
- GKD，ICLR 2024：[OpenReview](https://openreview.net/forum?id=3zKtaqxLhW)、[arXiv](https://arxiv.org/abs/2306.13649)
- EOPD，ICML 2026：[OpenReview](https://openreview.net/forum?id=J5i09faOOf)、[代码](https://github.com/WLS04/EOPD)
- SCoRe，ICML 2026：[OpenReview](https://openreview.net/forum?id=KaTYG9LGJv)、[arXiv](https://arxiv.org/abs/2509.14257)、[代码](https://github.com/haruhi-sudo/SCoRe)
- SDFT，ICML 2026 Spotlight：[OpenReview](https://openreview.net/forum?id=qA6FgH0nnZ)
- OVD，CoRR 2026：[OpenReview](https://openreview.net/forum?id=nl0zkfqupK)、[arXiv](https://arxiv.org/abs/2601.21968)
- ATOD，arXiv 2026：[arXiv](https://arxiv.org/abs/2606.27814)

