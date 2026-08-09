# GRPO / OPSD Badcase 归因分析

**日期**：2026-08-09

## 1. 分析范围

对齐 HotpotQA dev 的 7,405 个相同 ID，比较：

- SFT
- SFT+DPO
- 严格 LoRA GRPO-control checkpoint-1000
- Full GRPO checkpoint-2500 / 3000 / 3660
- OPSD checkpoint-3000

逐题提取最终答案、查询序列、检索文档标题、supporting facts 命中、
重复查询、最大轮次和错误类型。

OPSD 使用 `SaprRagScheduler` 的 raw-document observation pipeline，其他主
baseline 使用 `agent_infer.py` 的独立 evidence extraction pipeline。因此
OPSD 与 SFT+DPO 的端到端分数不是完全同口径；首轮查询发生在 observation
之前，相关比较仍然有效。

## 2. 核心结论

| 优先级 | 归因 | 判断 |
|---|---|---|
| 1 | Reward 与端到端目标错位 | Full GRPO 的主因。检索命中提高，但不回答、重复查询和跑满轮次显著恶化。 |
| 2 | Relevance reward 数据与实现缺陷 | 混合训练集 65.0% 样本的 title/sentence 列不对齐；2Wiki 的 `gold_sup_sents` 100% 为空；answer-text fallback 又会虚增命中。 |
| 3 | OPSD teacher/student 动作不对齐 | Teacher 已看到 gold evidence，却对 student 必须执行的搜索动作打分，明显破坏子问题分解。 |
| 4 | 训练配置放大错误信号 | 每步只有 2 个 prompt、每组 7 个 generation；Full GRPO 后期约一半 group reward 零方差，全参更新对噪声更敏感。 |
| 5 | 基础训练链路故障 | 证据不足。LoRA control 与 SFT 几乎一致，说明链路可运行；问题主要是优化信号和泛化，而不是训练程序完全失效。 |

## 3. Full GRPO：检索变好，但终止行为变坏

| 指标 | SFT | LoRA GRPO-1000 | Full-2500 | Full-3000 |
|---|---:|---:|---:|---:|
| Cover-EM | 50.70% | 50.80% | 44.93% | 42.58% |
| 回答率 | 89.28% | 89.60% | 77.06% | 69.14% |
| 已回答样本 Cover-EM | 56.78% | 56.70% | 58.31% | 61.58% |
| 最大轮次率 | 10.71% | 10.36% | 22.86% | 30.76% |
| 含重复 query 的样本 | 21.20% | 20.90% | 30.01% | 47.83% |
| 两个 gold title 都被检索到 | 19.42% | 19.53% | 23.01% | 25.08% |
| 平均 title coverage | 42.63% | 42.68% | 46.38% | 48.35% |

Full GRPO 并没有整体降低检索命中。相反，训练越久，gold title coverage
越高；真正拖垮 Cover-EM 的是回答率。若保持 SFT 的回答率，只使用
Full-2500 已回答样本的 conditional Cover-EM，反事实 Cover-EM 约为
52.06%，高于 SFT 的 50.70%。

这说明 Full GRPO 学到的是“检索更多、答得更短”，没有学到“证据足够后
及时停止”。

### 3.1 题级回归分解

SFT 正确但 Full-2500 错误共有 1,030 题；Full-2500 新增正确 603 题，
净回归 427 题。

| Full-2500 回归类型 | 数量 | 占 1,030 回归题 |
|---|---:|---:|
| 已回答错误，缺少至少一个 gold title | 490 | 47.6% |
| 未回答，缺少至少一个 gold title | 276 | 26.8% |
| 已回答错误，两个 gold title 均已命中 | 181 | 17.6% |
| 未回答，两个 gold title 均已命中 | 83 | 8.1% |

至少 25.6% 的回归题已经拿到两个 gold title，仍因错误推理或不终止而失败。
全局检索指标又是上升的，因此“检索器变差”不能解释整体退化。

### 3.2 代表 badcase

**证据已齐但重复检索至超轮次**

```text
Question: Are both Aloinopsis and Eriogonum ice plants?
Gold: no

SFT queries:
1. Is Aloinopsis an ice plant?
2. Is Eriogonum an ice plant?
Answer: No, only Aloinopsis is an ice plant.

Full-2500 queries:
1. Is Aloinopsis an ice plant?
2-6. Is Eriogonum an ice plant?  # 同一 query 连续重复
Answer: None / max_turns_exceeded
```

**证据已齐但比较推理出错**

```text
Question: Were Sound Team and Dead by Sunrise both formed before 2010?
Gold: yes

Full-2500 已检索到两个实体及成立年份，仍重复第二个查询并回答 No。
```

**没有从第一跳实体推进到第二跳**

```text
Question: Getting Married in Buffalo Jump stars an actor who was born in what city?
Gold: Calgary, Alberta

SFT: 电影 -> Paul Gross -> 出生地，3 个 query 后回答正确。
Full-2500: 连续 6 次改写“谁主演该电影”，没有进入 Paul Gross 出生地这一跳。
```

## 4. Reward 的具体问题

当前总 reward：

```text
R = 1.0 * answer_F1 + 0.2 * relevance + 0.05 * format
```

### 4.1 没有终止、轮次和重复查询约束

- `format` 只检查最后是否有非空 `<answer>`，第 1 轮和第 6 轮回答得分相同。
- 没有 max-turn penalty。
- 没有每轮成本。
- 没有重复 query penalty。
- `relevance` 对所有历史检索文档取并集；增加检索轮次不会降低已有命中。
- sequence-level reward 被广播到整条轨迹，无法指出具体哪一个 query 有用。

Full GRPO 的训练曲线直接反映了该错位：

| 训练窗口 | Reward | F1 reward | Relevance | 平均轮次 | 零方差 group |
|---|---:|---:|---:|---:|---:|
| step 1-250 | 0.663 | 0.462 | 0.769 | 3.28 | 18.8% |
| step 2251-2500 | 0.875 | 0.659 | 0.838 | 3.63 | 46.4% |
| step 3501-3660 | 0.878 | 0.662 | 0.840 | 3.88 | 49.1% |

在线 reward 上升的同时平均轮次也上升，held-out 最大轮次率随 checkpoint
从 22.86% 增至约 30%。优化器在正确优化当前 reward，但当前 reward
不是最终想要的行为。

### 4.2 2Wiki supporting sentence 构造错误

`build_grpo_dataset_mixed_opsd.py::extract_gold` 只读取
`context.sentences` / `context.text`，但 2Wiki 使用 `context.content`。

结果：

| 数据检查 | 数量 |
|---|---:|
| 混合训练集总数 | 7,320 |
| `len(gold_titles) != len(gold_sup_sents)` | 4,757（65.0%） |
| 2Wiki `gold_sup_sents` 为空 | 3,660 / 3,660（100%） |
| HotpotQA title/sentence 数量不一致 | 1,097 / 3,660（30.0%） |

HotpotQA 的不一致来自先收集所有 supporting sentence、再单独去重 title，
导致 title 与 sentence 下标错位。

### 4.3 answer-text fallback 虚增 relevance

当 title 和 supporting sentence 都未命中时，当前实现只要任一检索文档
包含 gold answer，就把**每一个**缺失 gold fact 都判为命中。

在 held-out 轨迹上按当前 ORM 重放：

| 模型 | title/sentence evidence coverage | 当前 reward proxy coverage | fallback 虚增 |
|---|---:|---:|---:|
| SFT | 59.87% | 78.46% | +18.59pt |
| Full-2500 | 63.79% | 82.30% | +18.50pt |
| Full-3000 | 65.65% | 83.09% | +17.44pt |
| OPSD-3000 | 52.26% | 72.77% | +20.51pt |

Full-2500 有 32.8% 的样本在 evidence coverage 不完整时仍得到满 relevance
proxy。该 reward 难以区分“完成两跳检索”和“偶然检索到包含答案的单篇文档”。

## 5. OPSD：主要是子问题策略被 privileged teacher 改坏

### 5.1 查询行为发生结构性变化

| 指标 | SFT+DPO | OPSD-3000 |
|---|---:|---:|
| 首个 query 完全相同 | - | 6.05% |
| 完整 query 序列相同 | - | 2.03% |
| 两个 gold title 都命中 | 18.81% | 12.29% |
| 第 2 轮前两个 gold title 都命中 | 17.57% | 11.41% |
| Comparison 首 query 同时包含两个实体 | 3.41% | 53.06% |

OPSD 把原来的逐实体分解大量改成一个合并 query。例如：

```text
Question: Were Scott Derrickson and Ed Wood of the same nationality?

SFT+DPO:
1. What is the nationality of Scott Derrickson?
2. What is the nationality of Ed Wood?

OPSD:
1. What is the nationality of Scott Derrickson and Ed Wood?
2. What is the nationality of Ed Wood?
```

合并 query 往往只召回一个实体，导致第二轮前的双证据覆盖明显下降。

### 5.2 Teacher 的信息条件与 student 的动作条件不一致

Teacher prompt 已包含 gold supporting facts 和 gold answer，student 则必须
通过查询获得这些信息。当前 OPSD 对同一串 student-sampled query/answer
token 计算：

```text
A_t = A_GRPO + 0.1 * (logp_teacher_t - logp_student_t)
```

因此 teacher 会评价自己根本不需要执行的搜索动作。一个在“已知 gold”
条件下高概率的 token，不一定是在“未知 gold、必须搜索”条件下的好动作。

约 28%-38% 的 OPSD group 基础 GRPO reward 零方差；这些 group 中更新
主要由 teacher log-ratio 驱动。该项为 additive 且不裁剪，可以改变甚至
翻转环境 reward 的 token advantage。`teacher_kl` 中还存在少量极端离群点
（p99 约 1.13，10 个 step 大于 10），说明 teacher/student 条件分布并未
稳定对齐。

### 5.3 OPSD 分数仍有评测混杂

- OPSD 用 raw-document scheduler，SFT+DPO 主表用 evidence extraction pipeline。
- OPSD 尚未跑 LLM judge。
- Comparison 问题存在 Cover-EM 假阴。例如 gold 为 `yes`，回答
  “两人都是 American”在语义上正确，但因不含字面 `yes` 被判错。

所以当前可以确定“OPSD 查询策略退化”，但不能仅凭 Cover-EM 0.3869
精确量化其事实正确率下降幅度。

## 6. 训练方法是否有问题

训练实现没有明显证据表明梯度或 checkpoint 链路失效：

- LoRA GRPO-control 与 SFT 的查询、检索和 held-out 指标几乎一致。
- Full GRPO 的 train reward、F1 和 relevance 都按当前目标上升。
- Full GRPO 的已回答样本 conditional Cover-EM 也上升。

但配置会放大不良 reward：

- 每步仅 2 个不同 prompt，每个 prompt 7 个 generation。
- Full GRPO 更新全部参数，远比 LoRA 更容易改变动作策略。
- 后期约 49% group reward 零方差，有效训练信号显著变稀。
- LoRA control 只评估到 step 1000，约 0.27 epoch，不能证明完整一轮
  LoRA GRPO 一定无效；但其在线 reward 已无上升趋势。

结论是“训练方法是放大器，不是第一根因”。

## 7. 最小验证顺序

1. **先修数据和 reward，不立即重跑 full parameter**
   - 2Wiki 读取 `context.content`。
   - 保持 title 与 supporting sentence 一一对齐。
   - 删除 relevance 的 answer-text fallback，或单独降权记录。
   - 用保存的 completions 离线重放 reward，确认 group 区分度。

2. **增加终止行为 reward**
   - max-turn / 无答案明确负奖励。
   - 每轮轻微成本。
   - 重复标准化 query 负奖励。
   - relevance 改为“本轮新增 gold fact”而不是最终文档并集。
   - 证据已齐后继续查询增加额外惩罚。

3. **先跑 LoRA reward-v2 小规模对照**
   - SFT 起点。
   - 固定同一批 300-500 step。
   - 每 100 step 在固定 held-out 500 题评测。
   - Gate 同时检查 Cover-EM、回答率、max-turn 和重复 query。

4. **最后再恢复 OPSD**
   - 先用同一 scheduler 重跑 SFT+DPO 起点，建立公平 baseline。
   - 不让 privileged teacher 直接塑造 query token；优先只监督最终 answer，
     或只使用 student 已检索到的 evidence。
   - 优先尝试 sign-preserving 的乘法 reweight，而不是 additive log-ratio。
   - 对 teacher log-ratio 做裁剪，并单独监控 query/answer token。

5. **补事实正确率检查**
   - 对 SFT+DPO -> OPSD 的 1,272 个回归和 662 个增益样本做 paired
     LLM judge，至少先抽样 300 题，排除 `yes/no` 表述造成的 Cover-EM 假阴。

在 reward-v2 的固定 500 题评测中，只有同时满足“Cover-EM 不低于 SFT、
回答率下降不超过 2pt、max-turn 和重复查询不升高”时，才值得扩到完整
epoch 或全参数训练。
