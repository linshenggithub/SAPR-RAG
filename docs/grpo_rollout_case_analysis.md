# GRPO / OPSD rollout case 分析

**分析对象**：E02-E11 的训练 completion 与离线评测轨迹

**重点对象**：E11 查询/答案分动作 OPSD

**结论口径**：训练 completion 只用于解释优化信号和策略行为；最终效果仍以固定留出集评测为准。

## 1. 可用轨迹文件

| 实验 | 训练 `completions.jsonl` | 可替代的离线评测轨迹 | 备注 |
|---|---|---|---|
| E02 旧 GRPO | 当前缺失 | `data/eval_results/hotpotqa/grpo_v4_ckpt175_20260613_113015/merged.jsonl`；另有 2Wiki、MuSiQue 全量结果 | 训练时配置过 `log_completions=true`，原保存目录已不存在 |
| E03 旧全动作 OPSD | `03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_colocate_full/opsd_colocate_effect_pbs2_g7_manual/v0-20260805-203554/completions.jsonl` | `data/eval_results/hotpotqa/full_dev_opsd_batch64_20260806_192912/` | 约 51,240 条训练 rollout |
| E04 LoRA GRPO 对照 | `03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_colocate_full/grpo_control_sft_mixed_pbs2_g7_epoch1_20260807_140000/v0-20260807-135816/completions.jsonl` | `data/eval_results/hotpotqa/grpo_control_sft_mixed_ckpt1000_hotpotqa_full_traincfg_20260807_2231/merged.jsonl` | 约 17,738 条训练 rollout |
| E05 全参数 GRPO | 当前缺失；原文件写在训练节点临时目录 | `data/eval_results/hotpotqa/full_grpo_sweep_20260808_203755/ckpt2500/merged.jsonl` 等 | 可分析部署时完整轨迹，不能恢复训练采样组 |
| E06 Reward-v2 | 当前缺失；原文件写在训练节点临时目录 | `data/eval_results/hotpotqa/reward_v2_anti_repeat_w015_ckpt300_full_dev_20260809/merged.jsonl` | 可分析 7,405 条离线轨迹 |
| E07 Reward-v3 | 当前缺失 | 无成功落盘的 rollout 结果 | checkpoint sweep 在推理引擎初始化阶段失败 |
| E09 答案动作 OPSD 25 step | `03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/opsd_lorafix_evidence_alpha003_s25_20260811/v0-20260811-130405/completions.jsonl` | `data/eval_results/hotpotqa/opsd_lorafix_evidence_alpha003_ckpt25_full7405_20260811/merged.jsonl` | 去重后约 520 条训练 rollout |
| E10 答案动作 OPSD 100 step | `03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/opsd_lorafix_evidence_alpha003_s100_20260811/v0-20260811-165523/completions.jsonl` | `data/eval_results/hotpotqa/opsd_lorafix_evidence_alpha003_s100_ckpt100_full7405_20260811/merged.jsonl` | 去重后约 2,000 条训练 rollout |
| E11 分动作 OPSD | `03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/opsd_multi_q001_a003_3src_s3000_actionfix_tmux_20260812/v0-20260812-064707/completions.jsonl` | 尚无完整离线评测结果 | 约 702 MB；去重后约 120,000 条 rollout |

`completions.jsonl` 每行是一批采样，字段包括 `prompt`、`completion`、
各子奖励、`advantages` 和 `num_turns`。完整轨迹需要将 `prompt` 中已有的
多轮历史与末轮 `completion` 合并阅读。

E09、E10、E11 中存在连续重复记录，不能直接用文件行数乘每行样本数。
本报告按整批 `prompt + completion` 哈希去重。

## 2. E11 总体统计

### 2.1 分阶段行为

| step | F1 reward | 证据相关性 | 格式奖励 | 完全重复 query | 参数记忆兜底 | 跑满 6 轮 | 超长末轮 | 高证据但答错 | 格式失败 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-500 | 0.5385 | 0.4866 | 0.9645 | 27.02% | 26.74% | 7.18% | 0.00% | 6.14% | 3.55% |
| 501-1000 | **0.5582** | 0.5133 | 0.9644 | 25.45% | 23.20% | 7.56% | 0.02% | 5.93% | 3.56% |
| 1001-1500 | 0.5065 | 0.5495 | 0.8487 | 18.44% | 7.86% | 3.19% | 0.30% | 8.30% | 15.13% |
| 1501-2000 | 0.5045 | 0.5481 | 0.8615 | 12.50% | 2.74% | 1.27% | 0.55% | 9.39% | 13.85% |
| 2001-2500 | 0.5105 | 0.5462 | 0.8871 | 10.71% | 1.32% | 0.94% | 0.79% | 9.55% | 11.29% |
| 2501-3000 | 0.5156 | 0.5493 | 0.8851 | 10.80% | 1.24% | 0.83% | 1.09% | 8.75% | 11.48% |

“超长末轮”指单个末轮文本不少于 4,000 个字符。

这组曲线说明训练并非完全没有改变策略：

- 完全重复 query、参数记忆兜底和跑满轮次都明显下降。
- 证据相关性从 0.4866 上升到约 0.55。
- 但是 F1 在 501-1000 step 达峰后下降约 5 个点。
- 1000 step 后格式失败率从约 3.6% 跳到 11%-15%，并开始出现长串标签、
  乱码和重复文本。
- “高证据但答错”从约 6% 上升到约 9%，说明检索收益没有稳定转化为答案。

### 2.2 GRPO 组内信号

去重后的 15,000 个八路采样组中：

- 27.2% 的组总奖励完全相同，GRPO advantage 为零。
- 7.5% 的组八条答案全部错误且总奖励完全相同。
- 40.9% 的组同时包含正确和错误答案，理论上存在有效组内区分信号。

因此问题不是“所有 batch 都没有信号”，而是约四分之一的 batch 无法更新，
其余 batch 的奖励又混合了答案正确性、证据覆盖和宽松格式判定。

### 2.3 Query teacher 实际覆盖

按训练中实际出现的问题回查数据字段，约 18.5% 的 rollout 对应样本具有
R3 查询计划。其余样本没有 Query teacher，只接受普通 GRPO 和 Answer
teacher 信号。

| 数据源 | 有 Query teacher 的 rollout | 无 Query teacher 的 rollout |
|---|---:|---:|
| HotpotQA | 11,260 | 28,210 |
| 2Wiki | 4,790 | 66,620 |
| MuSiQue | 6,066 | 2,484 |

整体训练又以 2Wiki 为主，而 2Wiki 的 Query teacher 覆盖最低。因此 E11
虽然名义上是“查询/答案分动作 OPSD”，大多数实际更新仍不是查询蒸馏。

## 3. 代表性真实 case

### Case A：检索失败后用参数记忆猜测

问题：

```text
What is the place of birth of the director of film Shoot Twice?
```

标准链路：

```text
Shoot Twice -> director Nando Cicero -> born in Asmara
Gold answer: Asmara
```

同一采样组中的错误轨迹先重复询问电影导演，随后分别猜测导演是 Shane
Black、Chen Kaige、Takahiko Okada、Claude Chabrol 或 Najam Momtaz，
最终给出以下互相冲突的答案：

```text
California
Beijing, China
Unknown
Troyes
Pakistan
```

这些轨迹的 F1 和证据相关性均为 0，但只要末尾存在非空 `<answer>`，
格式奖励仍为 1。模型收到的是轨迹级负信号，无法知道错误首先发生在
“导演实体识别”，还是发生在“出生地回答”。

### Case B：字符串 F1 将语义正确答案判为 0

问题：

```text
Which country the composer of song Are You Ready For The Country (Song) is from?
```

轨迹：

```text
Query 1: Who is the composer of the song "Are You Ready For The Country"?
Evidence: Neil Young
Query 2: Which country is Neil Young from?
Evidence: Neil Young is a Canadian singer-songwriter
Answer: Canada
```

奖励：

```text
F1 = 0
Relevance = 1
Format = 1
```

数据集标准答案是 `Canadian`。当前 token F1 对 `Canada` 和 `Canadian`
没有共同 token，因此把语义正确答案当作完全错误。类似地，
`Indian` 对标准答案 `India` 也会得到 F1=0。

### Case C：答案碰巧正确，但证据链完全错误

问题：

```text
Which film has the director who is older,
For the Love of Mariastella or Tartar Invasion?
```

标准导演是 Michael Curtiz 和 Pino Mercanti。某条轨迹没有检索到支持证据，
却用参数记忆编造 Jiří Menzel 和 Michal Pichler，最后碰巧回答：

```text
<answer>Tartar Invasion</answer>
```

奖励：

```text
F1 = 1
Relevance = 0
Format = 1
总 reward = 1.05
```

这条轨迹会成为组内正样本，尽管推理链中的两个关键实体都不正确。当前主
reward 奖励最终短答案，不能约束答案必须由检索证据推出。

### Case D：格式奖励无法识别严重输出退化

问题：

```text
Here is My Heart is a 1934 musical comedy film staring an American
singer who received what in 1991?
```

标准答案是 `National Medal of Arts`。退化轨迹错误锁定 Bing Crosby，
输出 `Grammy Honorary Grammy Award`，随后生成约 16,000 字符的重复
`</query>` 标签。记录的奖励仍为：

```text
F1 = 0
Relevance = 0.5
Format = 1
```

原因是 `SaprFormatORM` 只收集能被正则匹配的 `<query>` 和 `<answer>`
事件，并检查最后一个可识别事件是否为非空 answer。它不检查：

- answer 标签是否嵌套；
- answer 后是否仍有大量 token；
- 是否存在成百上千个孤立闭合标签；
- 总长度是否异常；
- Unicode 乱码或模板泄漏。

### Case E：整组全错，GRPO 无法更新

问题：

```text
When was the actor who was part of "Billion Dollar Babies" born?
```

标准答案是 `February 4, 1948`，需要识别 Alice Cooper。八条采样分别回答：

```text
September 23, 1947
April 18, 1947
April 18, 1947
April 18, 1947
April 18, 1947
April 18, 1947
April 18, 1947
April 18, 1947
```

八条轨迹均为：

```text
F1 = 0
Relevance = 0
Format = 1
总 reward = 0.05
```

组内标准差为零，所以 GRPO 不产生纠正方向。增加采样数量只有在至少一条
轨迹越过正确实体或证据门槛时才有帮助。

### Case F：检索命中时，组内信号可以正常工作

问题：

```text
What is the date of birth of the director of film
She Wanted A Millionaire?
```

标准答案是 `December 2, 1892`。八条采样只有一条同时检索到导演
John G. Blystone 和出生日期，并得到：

```text
F1 = 1
Relevance = 1
Format = 1
总 reward = 1.25
```

其余轨迹回答不同日期或“无法确定”，reward 为 0-0.15。这个 case 表明，
当检索器和 query 恰好找到正确两跳证据时，现有 GRPO 组内排序是有效的；
真正困难在于多数失败组没有任何成功轨迹，以及 reward 对幸运答案和格式
退化的误判。

## 4. 为什么训练没有形成离线增益

按优先级排序：

1. **答案 reward 存在系统性误标。** `Canada/Canadian`、
   `India/Indian` 等语义等价表达被 token F1 判为完全错误。
2. **格式 reward 过于宽松。** 严重标签循环和超长乱码仍可能得到满分。
3. **结果正确不等于证据正确。** 无证据的幸运答案可以成为高 reward 正样本。
4. **证据 reward 与答案 reward 正在分离。** 1000 step 后证据相关性提高，
   F1 反而下降，高证据但答错的比例上升。
5. **过程归因仍然不足。** 累计证据 reward 广播到整条轨迹，不能判断具体
   哪条 query 引入了正确或错误实体。
6. **约四分之一采样组零方差。** 特别是整组都检索失败时，GRPO 没有方向。
7. **Query teacher 覆盖过低且数据源不均衡。** 约 81.5% 的实际 rollout
   没有查询教师信号，2Wiki 又占训练主体。
8. **1000 step 后出现优化稳定性问题。** 长输出、格式失败、KL 和梯度
   离群点同时增多；继续训练主要恢复了少量格式分数，没有恢复早期 F1。

因此更准确的描述不是“模型什么都没学到”，而是：

> 模型学会了减少显式重复查询、减少参数记忆兜底并提高证据覆盖，但当前
> reward 没有把这些变化稳定转换成正确答案；与此同时，长训练引入了输出
> 格式退化。

## 5. 下一步建议

1. 优先离线评测 E11 的 checkpoint-500 和 checkpoint-1000，不先使用
   checkpoint-3000。训练侧 F1 和格式在 501-1000 区间最好。
2. 将格式奖励改为严格协议解析：只允许一个最终 answer、禁止嵌套标签、
   answer 后只允许空白、加入长度上限和异常字符惩罚。
3. 为答案 reward 增加标准别名与规范化，至少覆盖国家名/国籍词、
   日期等格式和数据集自带 aliases；不能只依赖 token F1。
4. 增加证据支撑约束：高答案 reward 必须同时满足关键支持事实覆盖，
   或增加 answer-to-evidence entailment 检查，避免幸运猜中。
5. 对每轮 query 使用边际证据收益做动作级 credit，不再把最终累计覆盖
   广播给所有 query token。
6. 重新采样或分层组 batch，提高 Query teacher 样本占比，并分别报告
   HotpotQA、2Wiki、MuSiQue 以及有/无 Query teacher 的训练指标。
7. 每 250-500 step 固定评测同一小型留出集，以 F1、Cover-EM、回答率、
   格式失败率和长输出率联合早停，不能只看在线总 reward。
