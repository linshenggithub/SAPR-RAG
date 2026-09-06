# SAPR-RAG 实验总记录

**首次建立**：2026-05-30

**最后更新**：2026-09-06

**当前主线**：Canonical SFT 起点上的分动作 OPSD 与匹配 GRPO 对照

**用途**：统一记录实验动机、实现方法、控制变量、结果、可信度和产物位置，供复现、论文写作与后续交接使用。

---

## 阅读指南

### 可信度标记

| 标记 | 含义 | 使用规则 |
|---|---|---|
| A：主结论 | held-out 全量评测，数据无泄露，评测链路已核验 | 可进入论文主表或作为最终方法判断依据 |
| B：诊断结论 | 小样本、流程口径不同或只用于定位瓶颈 | 可指导下一步，但不能单独证明方法优于 baseline |
| C：无效/被替代 | 存在数据泄露、LoRA 未加载或关键流程错误 | 只保留故障分析价值，禁止进入论文结果 |
| P：待验证 | 方案或代码已准备，但尚无真实训练结果 | 不得写成实验结论 |

数值冲突时，证据优先级固定为：

```text
评测目录中的 metrics.json / paired_bootstrap*.json
> checkpoint 的 args.json / logging.jsonl
> 本文汇总表
> 会话记录或人工转述
```

### 当前总判断

1. SFT→Query/Answer 分动作 OPSD ckpt1000 在 HotpotQA 与 2Wiki 全量
   dev 上取得有效收益：HotpotQA 的 F1、Cover-EM 和 2Wiki 的
   EM、F1、Cover-EM 均显著超过 SFT+DPO；MuSiQue 未显著改善。
   ckpt500 已有较高 Cover-EM，但三个数据集的 EM/F1 均低于 ckpt1000。
2. 严格 LoRA GRPO-control 与 SFT 基本持平；全参数 GRPO 会增加检索但破坏终止行为。
3. Reward-v2/v3 没有稳定提升 EM/F1/Cover-EM；D1b 证明 Query 生成质量与 Top-3 召回是主要瓶颈。
4. 修复 LoRA rollout 后，Answer-only OPSD 在 25 step 有轻微正向趋势，但相对 SFT+DPO 不显著；扩到 100 step 后回落。
5. E12 的三数据集 ckpt500/1000 全量评测已完成；当前主线已切换到
   SFT→external-teacher selective OPD。
6. Canonical-answer SFT 证明旧 SFT 的 EM/F1 偏低主要来自 answer target
   风格错配：把 terminal `<answer>` 替换为原始 GT 短答案后，HotpotQA
   EM/F1 升至 0.4373/0.5513，超过 SFT+DPO 和 E12；但 2Wiki 与
   MuSiQue 仍未超过 E12/SFT+DPO，且 Cover-EM 低于旧 SFT，说明该修正
   主要改善答案格式对齐，不等价于全面提升 RAG 能力。
7. E13 external-teacher pure OPD 的 checkpoint-300 已完成三数据集全量
   评测。相对同起点旧 SFT，EM/F1 在三个数据集均提高，但 HotpotQA
   Cover-EM 从 0.5070 降至 0.5007；相对 SFT+DPO 与 E12，EM/F1 明显
   落后。该方法主要缩短了旧 SFT 的答案表达，尚未证明检索覆盖能力提升。

### 实验总表

| ID | 实验 | 起点与训练数据 | 唯一主要改动 | 评测与核心结果 | 可信度 | 详细记录/产物 |
|---|---|---|---|---|---|---|
| B00 | Zero-shot baseline | 原始 Qwen2.5-7B-Instruct，无后训练 | 直接测试模型执行多轮 RAG 协议的能力 | 三数据集均完成；HotpotQA Cover 0.268，max-turn 45.1% | A | `docs/midterm_results.md` |
| B01 | DPO-only baseline | 原始 Qwen2.5-7B + RAG-ProGuide 13,289 偏好对 | 不经过 R3 SFT，直接 DPO | 三数据集均完成；HotpotQA EM 0.3492 / F1 0.4563 / Cover 0.3999 | A | `docs/midterm_results.md` |
| E00 | SFT baseline | Qwen2.5-7B；R3 cold-start 178,061 个逐 step 样本；LoRA ckpt1650 | 学习 R3 多轮 query/reasoning/answer 轨迹 | HotpotQA 7405：EM 0.0971 / F1 0.2634 / Cover 0.5070 | A | `docs/midterm_results.md`；`data/eval_results/hotpotqa/20260608_175824/metrics.json` |
| E01 | SFT+DPO baseline | E00 + RAG-ProGuide 约 5k 偏好对；LoRA ckpt395 | DPO 对齐简洁答案与搜索行为 | HotpotQA 7405：EM 0.4008 / F1 0.5233 / Cover 0.4693 | A | `data/eval_results/hotpotqa/sft_dpo_20260610_145349/` |
| E02 | 旧 GRPO v4 | SFT；由 HotpotQA dev 派生的 7,321 条训练数据 | F1 + relevance + format GRPO | 与评测集同源，结果存在 dev leakage | C | `docs/midterm_results.md` 的“旧 GRPO dev 泄露” |
| E03 | 旧全动作 OPSD | SFT+DPO；HotpotQA/2Wiki train 各 3660 | gold evidence+answer teacher 作用于所有动作，训练 3660 step | HotpotQA：EM 0.2895 / F1 0.4026 / Cover 0.3869；同时存在动作错配、流程不一致及旧 LoRA 风险 | C | 下文“OPSD / GRPO Experiment Record” |
| E04 | 严格 LoRA GRPO-control | SFT；官方 train-derived HotpotQA/2Wiki 共 7320 | 关闭 teacher，仅验证 GRPO 本身 | HotpotQA：EM 0.1048 / F1 0.2716 / Cover 0.5080，与 SFT 基本持平 | A | `data/eval_results/hotpotqa/grpo_control_sft_mixed_ckpt1000_hotpotqa_full_traincfg_20260807_2231/` |
| E05 | 全参数 GRPO | SFT merged；同 E04 数据；ZeRO-3 | 从 LoRA 改为全参数更新 | 最佳 ckpt2500：EM 0.4003 / F1 0.5071 / Cover 0.4493；回答率降至 77.1% | A | 下文“Full-Parameter GRPO” |
| E06 | Reward-v2 anti-repeat | SFT merged；train-derived mixed 数据；LoRA | anti-repeat prompt + 重复 query 惩罚 0.15 + max-turn 修复 | ckpt300 full dev：EM 0.1086 / F1 0.2761 / Cover 0.5121；重复轨迹率 20.54%，无稳定收益 | B | 下文“Reward-v2”及对应评测目录 |
| E07 | Reward-v3 marginal evidence | SFT merged；同类 mixed 数据；LoRA 500 step | 首次命中 gold evidence 才奖励，全覆盖后继续检索扣分 | 固定 200 早期口径：EM 0.105 / F1 0.270 / Cover 0.520；训练 F1/Marginal 基本横盘 | B | 下文“Reward-v3” |
| E08 | D1b 检索上限诊断 | SFT 轨迹 + HotpotQA 前 200 | 对比原问题、模型 query、gold title 在不同 Top-k 的召回 | Top-3 完全召回：模型 query 20.5%，gold title 50.0% | B | `data/eval_results/hotpotqa/d1b_retriever_ceiling_200_20260811.json` |
| E09 | LoRA 修复后 Answer-only OPSD 25 step | SFT+DPO；100 条 pilot；LoRA | Evidence Agent 对齐；teacher 只作用 Answer；β=0.03 | HotpotQA：EM 0.4054 / F1 0.5264 / Cover 0.4690；相对 E01 增量不显著 | A | 下文“第一轮” |
| E10 | Answer-only OPSD 100 step | 与 E09 完全相同，仅训练延长至 100 step | 检验增益能否随 step 稳定扩大 | HotpotQA：EM 0.4032 / F1 0.5243 / Cover 0.4675；较 ckpt25 回落 | A | 下文“第二轮” |
| E11 | Query/Answer 分动作 OPSD | SFT+DPO；HotpotQA+完整 2Wiki+MuSiQue 共 277,839 条；LoRA | Query 看 R3 搜索计划；Answer 看 gold；独立动作系数 | 旧 worker 回收前运行至约 step1624，仅保存到 ckpt1500；未完成全量评测 | P | 下文“分动作新方案” |
| E12 | SFT→Query/Answer 分动作 OPSD | SFT ckpt1650；与 E11 相同的 277,839 条三源数据；LoRA | 跳过 DPO，只改变初始 adapter | ckpt500/1000 均完成三数据集全量评测；ckpt1000 在 HotpotQA、2Wiki 有效，MuSiQue 未显著改善 | A | 下文“E12 三数据集全量 checkpoint 对照” |
| E13 | SFT→external-teacher selective OPD | SFT ckpt1650；三源 train 277,839 条；14B SFT teacher | Student on-policy 同状态；仅 EM 失败轨迹施加 teacher token log-ratio；无 privileged prompt | 最终 run 在 step340 因 rollout 断连退出；checkpoint-300 三数据集全量完成。HotpotQA 0.2217/0.3789/0.5007，2Wiki 0.2094/0.3339/0.4579，MuSiQue 0.1018/0.1770/0.2073（EM/F1/Cover） | A | `docs/opd_plan.md`；下文“External-teacher selective OPD”；`data/eval_results/opd_ckpt300_fulldev_gpufaiss_20260905/` |
| E14 | Canonical-answer SFT | Qwen2.5-7B；R3 cold-start SFT 数据；LoRA ckpt4150 | 仅将 terminal `<answer>` 内容由 R3 长答案替换为原始训练集 GT 短答案，其余 query/history/evidence 不变 | 三数据集全量完成；HotpotQA EM 0.4373 / F1 0.5513 / Cover 0.4748；2Wiki EM 0.4051 / F1 0.4513 / Cover 0.4188；MuSiQue EM 0.1651 / F1 0.2405 / Cover 0.1841 | A | 下文“E14 Canonical-answer SFT”；`data/eval_results/sft_canonical_ckpt4150_3src_6gpu_20260904/` |
| E15 | Canonical SFT→DPO | E14 canonical SFT ckpt4150 起点；LoRA DPO（pref_beta 0.2, sigmoid）；ckpt451 | 在 canonical SFT 基础上做 1 epoch DPO 偏好对齐 | 三数据集全量完成；HotpotQA EM 0.4140 / F1 0.5281 / Cover 0.4304；2Wiki EM 0.4187 / F1 0.4656 / Cover 0.4230；MuSiQue EM 0.1585 / F1 0.2459 / Cover 0.1676 | A | 下文“E15 Canonical SFT→DPO”；`data/eval_results/sft_canonical_dpo_3src_6gpu_20260905/` |
| E16 | Canonical SFT→GRPO+分动作 OPSD | E14 canonical SFT ckpt4150；三源 train 277,839 条；LoRA | 复现 E12：GRPO reward + Query 0.01 / Answer 0.03 分动作 teacher，只替换 SFT 起点 | ckpt1000 三数据集全量完成；HotpotQA .4636/.5816/.5025；2Wiki .5154/.5659/.5307；MuSiQue .1837/.2786/.2089；相对 E14 全面提升（2Wiki 约 +11pt）；OPSD 独立贡献待 B/D 对照 | A | 下文“E16 Canonical SFT→分动作 OPSD” |
| B（E16 对照） | Canonical SFT→GRPO-only（关 teacher） | E14 canonical SFT ckpt4150；与 E16 同数据/采样/步数；LoRA | 与 E16 唯一差异：关闭全部 teacher（纯 GRPO reward） | ckpt1000 三数据集全量完成；HotpotQA .4629/.5837/.5026；2Wiki .5161/.5654/.5314；MuSiQue .1808/.2794/.2056 | A | 下文“B/D 对照与 OPSD 归因” |
| D（E16 对照） | Canonical SFT→纯分动作 OPSD（无 RL reward） | E14 canonical SFT ckpt4150；与 E16 同数据/采样/步数；LoRA | 与 E16 唯一差异：关闭 GRPO reward 与组内 advantage，仅保留 Query/Answer teacher log-ratio | ckpt1000 三数据集全量完成；HotpotQA .4462/.5703/.5030；2Wiki .4948/.5548/.5270；MuSiQue .1758/.2717/.2085 | A | 下文“B/D 对照与 OPSD 归因” |

### 外部天花板诊断（DeepSeek，非同口径参考）

以下 DeepSeek 结果**不属于** SAPR-RAG 主实验（模型未经本项目训练、口径各异），仅作为"SOTA 模型在这些多跳测试集上能到多少"的天花板参考，集中列出以便查阅。完整方法见 `docs/midterm_results.md` §5。

三档口径（由弱到强的证据供给）：

| 口径 | 模型 | 数据集 | N | EM | Cover-EM | F1 | 说明 |
|---|---|---|---:|---:|---:|---:|---|
| Closed-book zeroshot | deepseek-v4-pro | MuSiQue | 2,417 | 0.1411 | 0.1630 | 0.1961 | 只给问题、无检索，纯参数知识 |
| Agentic zeroshot（受控检索） | deepseek-chat | MuSiQue | 2,417 | 0.1920 | 0.2503 | 0.2589 | 同本项目 BGE+FAISS 检索器自主多轮；含 ~47% 系统错误率，为保守下限 |
| Oracle（完美证据） | deepseek-v4-flash | HotpotQA | 7,405 | 0.3098 | 0.3480 | 0.3916 | distractor 10 段，非开放域上限 |
| Oracle（完美证据） | deepseek-v4-flash | 2Wiki | 12,576 | 0.7407 | 0.8114 | 0.7992 | 支撑段落直供 |
| Oracle（完美证据） | deepseek-v4-flash | MuSiQue | 2,417 | 0.5180 | 0.5933 | 0.6154 | 支撑段落直供（V4-Pro 为 0.5722/0.6554/0.6671） |

关键读法：在 MuSiQue 上，受控 agentic zeroshot 的 Cover-EM 25.03%（保守下限）与本项目最好结果（E01 0.2069 / E12 0.2180）处于同一量级——公平检索口径下 SOTA 模型不显著领先本项目小模型；真正差距在 Oracle（完美证据）设置。产物见 `data/eval_results/ceiling/` 与 `data/eval_results/deepseek_agentic/`。

### HotpotQA 全量结果总表

以下主表统一使用 HotpotQA 完整 dev 7,405 条。固定 200 条 checkpoint
筛选结果不混入主表；ReasonRAG 为论文报告值，未报告 Cover-EM。

| 方法 | 起点 | EM | F1 | Cover-EM | 可信度与说明 |
|---|---|---:|---:|---:|---|
| ReasonRAG 论文基线 | 论文模型 | 0.3840 | 0.4890 | 未报告 | 外部基线 |
| Zero-shot | Qwen2.5-7B | 0.2040 | 0.2730 | 0.2680 | A |
| SFT | Base | 0.0971 | 0.2634 | **0.5070** | A |
| Canonical-answer SFT ckpt4150 | Base | **0.4373** | **0.5513** | 0.4748 | A；修正 final answer target 为原始 GT 短答案 |
| DPO-only | Base | 0.3492 | 0.4563 | 0.3999 | A；推理流程略有差异 |
| SFT+DPO | SFT | 0.4008 | 0.5233 | 0.4693 | A；主要本地基线 |
| 旧 GRPO ckpt125 | SFT | 0.1086 | 0.2742 | 0.5080 | C；训练集泄露 |
| 旧 GRPO ckpt175 | SFT | 0.1155 | 0.2824 | 0.5082 | C；训练集泄露 |
| 严格 LoRA GRPO-control ckpt1000 | SFT | 0.1048 | 0.2716 | 0.5080 | A；基本等于 SFT |
| 全参数 GRPO ckpt2500 | SFT | 0.4003 | 0.5071 | 0.4493 | A；最佳全参数 checkpoint |
| 全参数 GRPO ckpt3000 | SFT | 0.3824 | 0.4796 | 0.4258 | A；后期退化 |
| 全参数 GRPO ckpt3660 | SFT | 0.3854 | 0.4817 | 0.4265 | A；后期退化 |
| Reward-v2 ckpt300 | SFT | 0.1086 | 0.2761 | 0.5121 | B；raw-document 流程 |
| 旧全动作 OPSD ckpt3000 | SFT+DPO | 0.2895 | 0.4026 | 0.3869 | C；动作与流程错误 |
| 旧全动作 OPSD ckpt3660 | SFT+DPO | 0.2883 | 0.4014 | 0.3860 | C；动作与流程错误 |
| Answer-only OPSD ckpt25 | SFT+DPO | 0.4054 | 0.5264 | 0.4690 | A；增量不显著 |
| Answer-only OPSD ckpt100 | SFT+DPO | 0.4032 | 0.5243 | 0.4675 | A；增益回落 |
| SFT→分动作 OPSD ckpt500 | SFT，不经过 DPO | 0.3118 | 0.4602 | **0.5165** | A；Cover 已提高，但 EM/F1 尚未形成 |
| **SFT→分动作 OPSD ckpt1000** | **SFT，不经过 DPO** | 0.4086 | 0.5379 | **0.4984** | **A；OPSD 方法中最佳，Cover 显著高于 SFT+DPO** |

E12 ckpt1000 相对本地 SFT+DPO，在同一 7,405 个 ID 上进行 20,000 次
配对 bootstrap：

| 指标 | E12 | SFT+DPO | 差值 | 95% CI | 双侧 p 值 |
|---|---:|---:|---:|---:|---:|
| EM | 0.4088 | 0.4008 | +0.80pt | [-0.14, +1.72]pt | 0.0929 |
| F1 | 0.5380 | 0.5233 | +1.47pt | [+0.61, +2.33]pt | 0.0015 |
| Cover-EM | 0.4984 | 0.4693 | +2.92pt | [+1.99, +3.85]pt | 0.0001 |

因此，E12 是当前第一个在 HotpotQA 全量 dev 上同时提高 EM、F1 和
Cover-EM 的 OPSD 方案，其中 F1 与 Cover-EM 达到统计显著，EM 尚未
通过双侧 0.05 显著性阈值。该结果证明不经过 DPO 时，Query/Answer
分动作 OPSD 仍能产生独立收益。

### E12 三数据集全量 checkpoint 对照

评测统一使用 Evidence Agent、BGE+FAISS Top-3、最多 6 个 agent turn、
每轮最多 512 个生成 token。ckpt500 与 ckpt1000 均来自
SFT `checkpoint-1650` 起点、不经过 DPO 的同一训练线。

| 数据集 | 模型 | N | 回答率 | EM | F1 | Cover-EM | Max-turn | 空 evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | SFT+DPO | 7,405 | 96.57% | 0.4008 | 0.5233 | 0.4693 | 3.43% | 26.24% |
| HotpotQA | E12 ckpt500 | 7,405 | 97.52% | 0.3118 | 0.4602 | **0.5165** | 2.47% | 18.08% |
| HotpotQA | **E12 ckpt1000** | 7,405 | **97.66%** | **0.4086** | **0.5379** | 0.4984 | **2.21%** | **16.91%** |
| 2Wiki | SFT+DPO | 12,576 | 82.72% | 0.3915 | 0.4688 | 0.4452 | 17.26% | 41.45% |
| 2Wiki | E12 ckpt500 | 12,576 | 97.30% | 0.3855 | 0.4892 | **0.5484** | 2.70% | 29.22% |
| 2Wiki | **E12 ckpt1000** | 12,576 | **98.51%** | **0.4866** | **0.5655** | 0.5476 | **1.48%** | **26.10%** |
| MuSiQue | SFT+DPO | 2,417 | 83.08% | **0.1667** | 0.2477 | 0.2069 | 16.92% | 42.91% |
| MuSiQue | E12 ckpt500 | 2,417 | 92.93% | 0.1233 | 0.2243 | **0.2238** | 7.07% | 26.69% |
| MuSiQue | **E12 ckpt1000** | 2,417 | **94.99%** | 0.1547 | **0.2546** | 0.2180 | **4.92%** | **23.81%** |

ReasonRAG 论文外部基线：

| 数据集 | EM | F1 | Cover-EM |
|---|---:|---:|---:|
| HotpotQA | 0.384 | 0.489 | 未报告 |
| 2Wiki | 未报告 | 0.372 | 未报告 |
| MuSiQue | 未报告 | 0.321 | 未报告 |

同 ID、20,000 次配对 bootstrap 相对 SFT+DPO：

| 数据集 | checkpoint | EM 差值（双侧 p） | F1 差值（双侧 p） | Cover-EM 差值（双侧 p） |
|---|---|---:|---:|---:|
| HotpotQA | ckpt500 | -8.90pt (0.0001) | -6.31pt (0.0001) | +4.73pt (0.0001) |
| HotpotQA | ckpt1000 | +0.80pt (0.0929) | +1.47pt (0.0015) | +2.92pt (0.0001) |
| 2Wiki | ckpt500 | -0.60pt (0.1780) | +2.04pt (0.0001) | +10.32pt (0.0001) |
| 2Wiki | ckpt1000 | +9.50pt (0.0001) | +9.67pt (0.0001) | +10.23pt (0.0001) |
| MuSiQue | ckpt500 | -4.34pt (0.0001) | -2.34pt (0.0001) | +1.70pt (0.0226) |
| MuSiQue | ckpt1000 | -1.20pt (0.0678) | +0.69pt (0.2928) | +1.12pt (0.1266) |

checkpoint 轨迹呈现一致规律：

- ckpt500 已有较高 Cover-EM 和较低 Max-turn，但 EM/F1 较低，说明
  答案较常包含 gold 字符串，却尚未形成简洁精确的最终回答；
- 到 ckpt1000，HotpotQA 与 2Wiki 的 EM/F1 明显提高，同时保持较高
  Cover-EM，因此 ckpt1000 是该训练线的最佳 checkpoint；
- MuSiQue 从 ckpt500 到 ckpt1000 有恢复，但相对 SFT+DPO 的三项差值
  均未显著，且 F1 仍低于 ReasonRAG 论文值 0.321；
- 2Wiki ckpt1000 的 F1 为 0.5655，显著高于 SFT+DPO 0.4688 和
  ReasonRAG 论文值 0.372；HotpotQA ckpt1000 的 F1 为 0.5379，
  也高于 ReasonRAG 0.489。

权威产物：

- `data/eval_results/hotpotqa/sft_opsd_ckpt1000_full7405_20260902/full/checkpoint-1000/hotpotqa/metrics.json`
- `data/eval_results/hotpotqa/sft_opsd_ckpt1000_full7405_20260902/full/checkpoint-1000/hotpotqa/paired_bootstrap_vs_sft_dpo.json`
- `data/eval_results/2wikimultihopqa/sft_opsd_ckpt1000_full12576_20260903/full/checkpoint-1000/2wikimultihopqa/`
- `data/eval_results/musique/sft_opsd_ckpt1000_full2417_20260903/full/checkpoint-1000/musique/`
- `data/eval_results/action_opsd_sft_ckpt500_3src_full_20260903/full/checkpoint-500/`

### E14 Canonical-answer SFT

**实验 ID**：E14
**日期**：2026-09-04 -- 2026-09-05
**状态**：1 epoch 训练完成；三数据集 full dev 评测完成；可作为有效 SFT
数据修正结论。

研究问题：旧 SFT 的 EM/F1 显著低于 zero-shot，是否主要由 SFT 训练
target 中 `<answer>` 采用 R3 长解释答案、而非 benchmark canonical short
answer 导致？

唯一变量：保留 R3 cold-start 的多轮 `instruction`、query、history、
analysis、evidence 与 step 结构不变，仅将 terminal step 的
`<answer>...</answer>` 内容替换为原始训练数据集的 gold answer。构建时按
归一化 question 回连原始 train 集：

| 项目 | 数值 |
|---|---:|
| reasoning rows | 178,061 |
| terminal answer rows | 51,253 |
| 成功匹配 gold | 51,148 |
| 替换为 GT | 51,100 |
| 原本已一致 | 48 |
| 未匹配到 gold | 105 |

answer 长度从旧版 SFT 的平均 13.96 words / p50 12 words，下降到平均
2.23 words / p50 2 words；`<=5 words` 占比从 27.96% 升到 96.7%。

训练配置：

| 项目 | 配置 |
|---|---|
| base model | Qwen2.5-7B-Instruct |
| dataset | `sapr_reasoning_canonical,sapr_evidence_canonical` |
| 训练方式 | LoRA SFT, fp16 |
| worker | `4216626` |
| GPU | `CUDA_VISIBLE_DEVICES=4,5,6,7` |
| epoch | 1 |
| train examples | 274,168 |
| val examples | 2,770 |
| total steps | 4,284 |
| best checkpoint | `checkpoint-4150` |
| final train loss | 0.20599 |
| final eval loss | 0.14523 |

评测配置：`agent_infer.py --backend vllm`，同 SAPR-RAG 多轮 pipeline；
BGE+FAISS Top-3 retrieval daemon；最多 6 个 agent turn；reasoning
`max_tokens=512`，evidence `max_tokens=128`；6 GPU 分片评测后合并。

三数据集 full dev 结果：

| 数据集 | N | 回答率 | EM | F1 | Cover-EM | avg_turns | Max-turn | 空 evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 7,405 | 90.36% | **0.4373** | **0.5513** | 0.4748 | 2.457 | 9.62% | 18.41% |
| 2Wiki | 12,576 | 75.07% | 0.4051 | 0.4513 | 0.4188 | 3.481 | 24.92% | 31.89% |
| MuSiQue | 2,417 | 71.29% | 0.1651 | 0.2405 | 0.1841 | 3.736 | 28.71% | 26.53% |

与旧 SFT / SFT+DPO 的核心对比：

| 数据集 | 对比对象 | EM 差值 | F1 差值 | Cover-EM 差值 | 解读 |
|---|---|---:|---:|---:|---|
| HotpotQA | vs 旧 SFT | +34.02pt | +28.79pt | -3.22pt | 证实旧 SFT 的 EM/F1 主要受长答案 target 拖累 |
| HotpotQA | vs SFT+DPO | +3.65pt | +2.80pt | +0.55pt | 单靠 canonical SFT 已超过 DPO 后答案指标 |
| 2Wiki | vs 旧 SFT | +30.33pt | +19.98pt | -3.00pt | EM/F1 显著修复，但 Cover-EM 回落 |
| 2Wiki | vs SFT+DPO | +1.36pt | -1.75pt | -2.64pt | 不构成对 SFT+DPO 的全面优势 |
| MuSiQue | vs 旧 SFT | +11.59pt | +12.00pt | -0.70pt | EM/F1 修复明显 |
| MuSiQue | vs SFT+DPO | -0.16pt | -0.72pt | -2.28pt | 基本持平或略低，未形成新最佳 |

结论：E14 明确证明旧 SFT 的低 EM/F1 不是 SFT 学不到多跳 RAG，而是
final `<answer>` 训练目标偏长导致的指标错配。canonical answer target
能在 HotpotQA 上直接超过 SFT+DPO 和 E12 的 EM/F1，但 2Wiki 与 MuSiQue
未超过 E12/SFT+DPO，且三数据集 Cover-EM 普遍低于旧 SFT/E12。该实验适合
作为“答案格式对齐修复”的强证据；后续若继续做 DPO/OPSD，应优先以
E14 为新 SFT 起点重跑对齐实验，验证是否能同时保留短答案 EM/F1 与
OPSD 的 Cover-EM/行为收益。

权威产物：

- `03_sapr_rag/data/sft_build/out/sft_v2_reasoning_canonical.jsonl`
- `03_sapr_rag/data/sft_build/out/sft_v2_evidence_canonical.jsonl`
- `03_sapr_rag/scripts/train/sft_canonical_lora_fp16.yaml`
- `03_sapr_rag/scripts/train/logs/sft_canonical_lora_fp16_preload_20260903_205417/train.log`
- `03_sapr_rag/saves/qwen2_5_7b/lora/sft_canonical_fp16/checkpoint-4150/`
- `data/eval_results/sft_canonical_ckpt4150_3src_6gpu_20260904/`

### E15 Canonical SFT→DPO

**实验 ID**：E15
**日期**：2026-09-05
**状态**：1 epoch DPO 训练完成；三数据集 full dev 评测完成。

研究问题：在 E14 canonical SFT 基础上做偏好对齐（DPO），能否在保留
canonical short answer 带来的 EM/F1 的同时，进一步提升多跳表现或 Cover-EM。

训练配置：

| 项目 | 配置 |
|---|---|
| base model | Qwen2.5-7B-Instruct |
| 起点 adapter | `sft_canonical_fp16/checkpoint-4150`（E14） |
| dataset | `sapr_proguide_dpo`（chosen/rejected 偏好对） |
| 训练方式 | LoRA DPO；`pref_beta=0.2`，`pref_loss=sigmoid` |
| worker | `4216626`；GPU 1-7（7 卡） |
| epoch | 1（451 steps） |
| lr | 5.0e-6，cosine |
| cutoff_len | 2560 |
| val_size | 0.05 |
| final train loss | 1.2242 |
| final eval loss | 1.1704（step400） |
| best checkpoint | `checkpoint-451` |

训练曲线健康：train loss 1.557→1.022；eval loss 1.387→1.170 单调下降无过拟合；
rewards/margins 0.40→1.04 稳步扩大；rewards/accuracies 0.51→0.60，偏好对齐方向正确。

评测配置：`agent_infer.py --backend vllm`，同 SAPR-RAG 多轮 pipeline；
BGE+FAISS Top-3；最多 6 turn；reasoning `max_tokens=512`，evidence `max_tokens=128`；
6 GPU 分片评测后合并（rows=unique=expected 全部通过）。

三数据集 full dev 结果：

| 数据集 | N | 回答率 | EM | F1 | Cover-EM | avg_turns | Max-turn | 空 evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 7,405 | 95.00% | 0.4140 | 0.5281 | 0.4304 | 2.231 | 5.00% | 26.77% |
| 2Wiki | 12,576 | 82.89% | 0.4187 | 0.4656 | 0.4230 | 3.266 | 17.09% | 40.19% |
| MuSiQue | 2,417 | 83.12% | 0.1585 | 0.2459 | 0.1676 | 3.321 | 16.88% | 39.80% |

与 E14 / SFT+DPO(E01) 的核心对比：

| 数据集 | 对比对象 | EM 差值 | F1 差值 | Cover-EM 差值 | 解读 |
|---|---|---:|---:|---:|---|
| HotpotQA | vs E14 | -2.33pt | -2.32pt | -4.44pt | DPO 未带来增益，三指标均回落 |
| HotpotQA | vs SFT+DPO(E01) | +1.32pt | +0.48pt | -3.89pt | EM/F1 略优，Cover-EM 更低 |
| 2Wiki | vs E14 | +1.36pt | +1.43pt | +0.42pt | 三指标同步小幅提升 |
| 2Wiki | vs SFT+DPO(E01) | +2.72pt | -0.32pt | -2.22pt | EM 明显更高，Cover-EM 略低 |
| MuSiQue | vs E14 | -0.66pt | +0.54pt | -1.65pt | 基本持平，F1 微升 |
| MuSiQue | vs SFT+DPO(E01) | -0.82pt | -0.18pt | -3.93pt | 未形成优势 |

结论：canonical SFT 基础上做 DPO 的收益按数据集分化——仅在 2Wiki 上三指标
同步小幅提升；HotpotQA 整体回落（尤其 Cover-EM -4.44pt），MuSiQue 基本持平。
DPO 训练本身健康（margins/accuracies 均正向），但当前偏好数据（`sapr_proguide_dpo`）
对短答案 canonical 起点的增益有限，且普遍压低 Cover-EM。E14 canonical SFT 仍是
HotpotQA 上的答案指标最佳，E15 未能全面超越 E14。后续若继续偏好对齐，建议针对性
构造更贴合多跳证据链的偏好对，并监控 Cover-EM 回落。

权威产物：

- `03_sapr_rag/scripts/train/dpo_canonical_lora.yaml`
- `03_sapr_rag/scripts/train/run_dpo_canonical.sh`
- `03_sapr_rag/saves/qwen2_5_7b/lora/sft_canonical_dpo/checkpoint-451/`
- `data/eval_results/sft_canonical_dpo_3src_6gpu_20260905/`
- `data/eval_results/sft_canonical_dpo_3src_6gpu_20260905/compare_dpo_vs_baselines.md`

### E16 Canonical SFT→分动作 OPSD

**实验 ID**：E16
**日期**：2026-09-05（训练）/ 2026-09-06（评测）
**状态**：1000-step 正式训练在 `worker4216626` 完成，`checkpoint-1000` 已完成三数据集全量评测。

研究问题：E12 的 Query/Answer 分动作 OPSD 在修复答案目标后的 E14
canonical SFT 强起点上是否仍能提升 EM/F1/Cover-EM；以及相对匹配
GRPO-only control，teacher 信号是否有独立收益。

本实验复现 E12 在 `checkpoint-1000` 验证有效的目标，唯一核心变化是
将起点从旧 SFT `checkpoint-1650` 替换为 E14
`sft_canonical_fp16/checkpoint-4150`：

| 项目 | 配置 |
|---|---|
| 起点 | E14 `sft_canonical_fp16/checkpoint-4150` |
| 数据 | `hotpotqa_2wiki_musique_train_multi_opsd.jsonl`，277,839 条 |
| 更新方式 | LoRA，学习率 `1e-6` |
| GRPO reward | F1 / relevance / format，权重 `1.0 / 0.2 / 0.05` |
| teacher | Query 0.01 / Evidence 0 / Answer 0.03 |
| rollout | GPU7；Evidence Agent；最多 6 turn |
| 检索 | GPU0；BGE+FAISS Top-3 |
| 训练 | GPU2-6；per-device batch 2；grad accumulation 4 |
| 采样 | 8 generations；steps-per-generation 8 |
| 长度 | max completion 4096 |
| 步数 | 1000；每 250 step 保存 |
| 截断惩罚 | 关闭，与原始有效 E12 ckpt1000 保持一致 |
| run | `opsd_canonical_sft_q001_a003_3src_s1000_20260905` |

启动后前 6 step 未见 NaN、OOM 或服务错误。step 3 同时记录到
Query scoped KL `0.1351` 和 Answer scoped KL `0.0542`，证明两类
teacher 信号在 canonical SFT 起点下均实际生效。训练曲线上
`checkpoint-1000` 为最优点（reward、F1 分项、format 均最高，输出长度稳定），
选作评测 checkpoint。

`checkpoint-1000` 三数据集全量结果（口径与 E14/E15 一致）：

| 数据集 | 样本 | EM | F1 | Cover-EM | 相对 E14 EM / F1 / Cover |
|---|---:|---:|---:|---:|---|
| HotpotQA | 7405 | 0.4636 | 0.5816 | 0.5025 | +2.63 / +3.03 / +2.77 |
| 2Wiki | 12576 | 0.5154 | 0.5659 | 0.5307 | +11.03 / +11.46 / +11.19 |
| MuSiQue | 2417 | 0.1837 | 0.2786 | 0.2089 | +1.86 / +3.81 / +2.48 |

（E14 基线：HotpotQA 0.4373/0.5513/0.4748；2Wiki 0.4051/0.4513/0.4188；
MuSiQue 0.1651/0.2405/0.1841。）E16 在三数据集上相对 E14 全面提升，
2Wiki 提升最大（约 +11pt）。这度量的是整套 `GRPO+OPSD` 后训练收益。

归因边界：E16 同时包含 GRPO 与 OPSD，单独比较 E16 与 E14 只能得到
整套后训练收益。后续必须补跑相同起点、数据、reward、采样和步数但关闭
teacher 的 GRPO-only control；只有 `E16 - GRPO control` 才能解释为
OPSD 的独立贡献。此外已补跑纯 OPSD（无 RL reward）对照（见下文 D），
用于隔离自蒸馏本身的贡献。

权威产物：

- `03_sapr_rag/scripts/grpo/run_canonical_sft_multi_opsd_s1000.sh`
- `03_sapr_rag/scripts/grpo/logs/opsd_canonical_sft_q001_a003_3src_s1000_20260905/`
- `03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/opsd_canonical_sft_q001_a003_3src_s1000_20260905/`
- `data/eval_results/e16_opsd_canonical_ckpt1000_3src_20260905/`

### B/D 对照与 OPSD 归因

**实验 ID**：B（GRPO-only 对照）、D（纯 OPSD 对照）
**日期**：2026-09-06
**状态**：两者 1000-step 训练与 ckpt1000 三数据集全量评测均完成。

目的：E16（C）同时含 GRPO reward 与分动作 OPSD teacher，单独 `C - E14`
只能得到整套后训练收益。为隔离各成分，补跑两个与 E16 完全对齐、
只差一个开关的对照（同起点 E14 ckpt4150、同三源数据、同采样、
同 1000 step、同 Query 0.01 / Answer 0.03 动作系数）：

- **B（GRPO-only）**：关闭全部 teacher，仅 GRPO reward（F1/relevance/format）。
  run `grpo_control_canonical_sft_3src_s1000_20260906`（worker4216626）。
- **D（纯 OPSD）**：`ENABLE_REWARD=false` + `opd_use_grpo_advantage=false`，
  关闭 GRPO reward 与组内 advantage，只保留 Query/Answer teacher log-ratio
  作为逐 token advantage（对应 OPSD 原论文 arXiv:2601.18734 的纯自蒸馏形式）。
  run `pure_opsd_canonical_sft_q001_a003_3src_s1000_20260906`（worker4220660）。

四模型 ckpt1000 全量结果（EM / F1 / Cover-EM）：

| 模型 | HotpotQA | 2Wiki | MuSiQue |
|---|---|---|---|
| E14 SFT（基线） | .4373/.5513/.4748 | .4051/.4513/.4188 | .1651/.2405/.1841 |
| B GRPO-only | .4629/.5837/.5026 | .5161/.5654/.5314 | .1808/.2794/.2056 |
| D 纯 OPSD | .4462/.5703/.5030 | .4948/.5548/.5270 | .1758/.2717/.2085 |
| C=E16 GRPO+OPSD | .4636/.5816/.5025 | .5154/.5659/.5307 | .1837/.2786/.2089 |

归因矩阵（EM 维度，单位 pt）：

| 对比 | 含义 | HotpotQA | 2Wiki | MuSiQue |
|---|---|---:|---:|---:|
| B − E14 | GRPO 贡献 | +2.6 | +11.1 | +1.6 |
| D − E14 | 纯 OPSD 贡献 | +0.9 | +9.0 | +1.1 |
| C − B | OPSD 叠加在 GRPO 之上 | +0.1 | −0.1 | +0.3 |
| C − D | GRPO 叠加在 OPSD 之上 | +1.7 | +2.1 | +0.8 |

结论：

1. GRPO（B）在 E14 强起点上贡献了绝大部分增益，2Wiki 尤为显著（+11pt）。
2. 纯 OPSD（D）本身相对 E14 有正向提升，但整体弱于 GRPO。
3. **OPSD 叠加在 GRPO 之上（C − B）几乎无额外贡献**（三数据集 −0.1~+0.3pt），
   说明在该配置下 teacher 信号与 GRPO 的增益高度重叠，未提供 GRPO 之外的
   新信息。因此当前不能声称分动作 OPSD teacher 在 canonical 强起点上有
   稳健的独立收益。
4. GRPO 叠加在纯 OPSD 之上（C − D）仍有小幅正向增益，进一步印证主要驱动
   力来自 GRPO reward 而非 teacher。

数据无泄露核对（2026-09-06）：

B/C/D 共用训练集 `hotpotqa_2wiki_musique_train_multi_opsd.jsonl`（277,839 行，
规范化去重后 277,785 个唯一问题）。训练集无稳定 id 字段，dev 集 id 为独立
重编号（`dev_N`），故以规范化 question 文本（去 `Question:` 前缀、压缩空格、
转小写、去尾问号）做精确匹配。三个 dev 集与训练集重叠均为 0：

| dev 集 | dev 唯一问题 | 与全体训练重叠 | 与同源训练重叠 |
|---|---:|---:|---:|
| HotpotQA | 7405 | 0（0.00%） | 0 |
| 2Wiki | 12576 | 0（0.00%） | 0 |
| MuSiQue | 2411 | 0（0.00%） | 0 |

因此 B 相对 E14 的提升（尤其 2Wiki +11pt）不是训练/测试泄露所致，与历史上
因 dev-derived 训练数据被判 C 的 E02/旧 GRPO 有本质区别，GRPO 在 canonical
强起点上的有效性成立且干净。限制：精确文本匹配只能捕获完全相同的问题，
无法识别"同题但表述改写"的潜在近重复；多跳 QA 问题表述高度特异，0 精确重叠
已是较强证据。

权威产物：

- B 训练：`03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/grpo_control_canonical_sft_3src_s1000_20260906/`
- B 评测：`data/eval_results/b_grpo_control_canonical_ckpt1000_3src_20260906/`
- D 启动：`03_sapr_rag/scripts/grpo/run_canonical_sft_pure_opsd_s1000.sh`
- D 训练：`03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/pure_opsd_canonical_sft_q001_a003_3src_s1000_20260906/`
- D 评测：`data/eval_results/D_pure_opsd_ckpt1000_3src_20260906/`

### External-teacher selective OPD

**实验 ID**：E13
**日期**：2026-09-03 至 2026-09-06
**状态**：代码、teacher SFT、三数据集 ceiling、2-step smoke、checkpoint-300
三数据集全量评测均已完成。正式 run 在 step340 因 rollout server 连接
中断退出，最后一个完整 checkpoint 为 300。

该方案不再给 teacher 注入 gold answer、gold evidence 或 R3 query plan。
Teacher 和 student 看到同一条 student on-policy 多轮轨迹；gold answer
只通过独立 `sapr_em` reward 判定轨迹是否失败。纯 OPD 目标为：

```text
gate_i = 1[EM_i < 1]
A_t = gate_i * 0.01 * (logp_teacher_t - logp_student_t)
```

环境 observation token 保持 mask；`opd_use_grpo_advantage=false`，因此
不混入 GRPO advantage。训练数据由原三源数据删除全部 `teacher_*`
字段得到，共 277,839 条。

14B teacher 先用与 7B 相同的 ReasonRAG SFT 数据做 300-step LoRA
协议训练：

| 项目 | 结果 |
|---|---:|
| train loss | 0.3007 |
| eval loss | 0.1835 |
| checkpoint | `03_sapr_rag/saves/qwen2_5_14b/lora/sft_teacher/checkpoint-300` |

固定三数据集各 50 条的 teacher ceiling：

| 数据集 | 14B teacher F1 | 7B SFT F1 | 差值 |
|---|---:|---:|---:|
| HotpotQA | 0.3025 | 0.2436 | +0.0589 |
| 2Wiki | 0.3524 | 0.1763 | +0.1761 |
| MuSiQue | 0.2628 | 0.1568 | +0.1060 |
| 宏平均 | 0.3059 | 0.1922 | +0.1137 |

HotpotQA teacher 回答率比 7B SFT 低 2/50，其余两个数据集更高；
考虑 50 条样本的离散粒度，pilot gate 使用最大 5pt 回答率下降容忍。
全量结果不能沿用该容忍度。

Teacher 输出风格审计显示它仍明显偏向完整句子，而非 canonical 短答案：

| 数据集 | 已回答数 | 平均答案 token | 中位数 | EM | Cover-EM |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 45/50 | 9.36 | 9 | 0.12 | 0.56 |
| 2Wiki | 43/50 | 11.58 | 8 | 0.16 | 0.60 |
| MuSiQue | 41/50 | 8.17 | 3 | 0.14 | 0.26 |

HotpotQA 与 2Wiki 的 `Cover-EM - EM` 均为 44pt，说明 teacher 经常包含
正确短答案，但附带解释性文本。训练期间 teacher 只返回 student sampled
token 的 logprob，不生成独立 completion；`completions.jsonl` 保存的是
student rollout，不能直接作为 teacher 文本审计。

2-step smoke：

- run：`opd_14b_sft_failed_em_smoke_v6_worker4220660`
- loss：`0.0272 -> 0.0387`
- grad norm：`0.0794 -> 0.0955`
- teacher KL：`0.7538 -> 1.0005`
- 两步截断率均为 0，checkpoint-1/2 均已保存。

最终正式训练：

| 项目 | 配置 |
|---|---|
| run | `opd_sft14b_failed_em_spg2_s500_20260904` |
| student | Qwen2.5-7B + SFT `checkpoint-1650` |
| teacher | Qwen2.5-14B + SFT `checkpoint-300`，冻结 |
| 数据 | 三源 train 277,839 条，无 `teacher_*` 字段 |
| 设备 | GPU0 retrieval；GPU1 teacher；GPU7 rollout；GPU2/3/4/5/6 train |
| 优化 | pure OPD；failed-EM gate；teacher coef 0.01 |
| batch | per-device 1；grad accumulation 4；steps-per-generation 2；5 generations |
| 计划 | 500 step；每 50 step 保存 |
| 实际终点 | step340 连接中断；最后完整 checkpoint 为 300 |

训练过程健康指标：

- checkpoint-300 前 loss 从约 0.036 降至约 0.027，grad norm 从约
  0.11 降至约 0.05；
- reward 的 20-step 均值从约 0.366 升至约 0.489；
- failed-EM gate ratio 从约 0.98 降至约 0.69；
- completion 平均长度从约 367 token 降至约 293 token；
- 未发现 NaN、OOM 或梯度发散；
- step340 在同步 LoRA adapter 到 rollout vLLM 时出现
  `RemoteDisconnected`，训练进程退出；checkpoint-300 完整可用。

checkpoint-300 三数据集全量评测使用统一 `agent_infer.py` 流程：
BGE + FAISS GPU fp32 Top-3、Evidence Agent、最多 6 轮。各结果文件均完成
行数与唯一 ID 校验。

| 数据集 | N | 回答率 | EM | F1 | Cover-EM | Avg turns | Max-turn | 空 evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 7,405 | 89.97% | 0.2217 | 0.3789 | 0.5007 | 2.505 | 10.02% | 20.97% |
| 2Wiki | 12,576 | 75.84% | 0.2094 | 0.3339 | 0.4579 | 3.587 | 24.16% | 35.82% |
| MuSiQue | 2,417 | 71.58% | 0.1018 | 0.1770 | 0.2073 | 3.746 | 28.38% | 31.00% |

相对同起点旧 SFT：

| 数据集 | EM 差值 | F1 差值 | Cover-EM 差值 |
|---|---:|---:|---:|
| HotpotQA | +12.46pt | +11.55pt | -0.63pt |
| 2Wiki | +10.76pt | +8.24pt | +0.91pt |
| MuSiQue | +5.26pt | +5.65pt | +1.62pt |

相对强基线，E13 未形成总体优势：

- HotpotQA 相对 SFT+DPO：EM -17.91pt、F1 -14.44pt、Cover-EM
  +3.14pt；相对 E12 ckpt1000：EM -18.69pt、F1 -15.90pt、
  Cover-EM +0.23pt。
- 2Wiki 相对 SFT+DPO：EM -18.21pt、F1 -13.49pt、Cover-EM
  +1.27pt；相对 E12 ckpt1000 三项均下降。
- MuSiQue 相对 SFT+DPO：EM -6.49pt、F1 -7.07pt、Cover-EM
  +0.04pt；相对 E12 ckpt1000 三项均下降。

HotpotQA 答案风格分析进一步表明：

- 1,642 条（22.17%）为 exact match；
- 2,066 条（27.90%）包含 gold 但因额外文本不满足 EM，平均答案长度
  12 token；
- 1,205 条（16.27%）仅部分 token 重合；
- 1,749 条（23.62%）与 gold 无重合；
- 743 条（10.03%）未回答；
- 总体平均答案长度为 7.96 token，介于旧 SFT 的 11.9 token 与
  Canonical SFT 的 2.1 token 之间。

因此，E13 的主要收益是缩短旧 SFT 的长句答案并改善 EM/F1，而不是稳定
提高证据覆盖。当前 sampled-token OPD 只对 student 已采样 token 加权，
无法直接提供 student 未生成的更优 query/action；下一轮应以 Canonical
SFT 为起点、训练 canonical short-answer teacher，并评估 teacher-action
OPD（在 student 状态上生成并筛选替代动作）。

评测有效性说明：

- 有效主结果目录为
  `data/eval_results/opd_ckpt300_fulldev_gpufaiss_20260905/`；
- 该评测使用 GPU FAISS fp32，三数据集均无 ReadTimeout、OOM 或
  rollout 异常；
- `data/eval_results/opd_ckpt300_fulldev_agentinfer_20260905/` 中旧结果
  全部为 CPU FAISS ReadTimeout 失败记录，指标全零，标记为无效，不得引用；
- 训练阶段误用了 FAISS CPU 默认配置，主要影响吞吐；索引仍为同一
  `IndexFlatIP` fp32 精确索引。

权威产物：

- `data/eval_results/teacher_14b_sft_ceiling_50/ceiling_gate.json`
- `03_sapr_rag/saves/qwen2_5_7b/lora/opd/opd_14b_sft_failed_em_smoke_v6_worker4220660/`
- `03_sapr_rag/saves/qwen2_5_7b/lora/opd/opd_sft14b_failed_em_spg2_s500_20260904/v0-20260904-023807/checkpoint-300/`
- `03_sapr_rag/scripts/opd/logs/opd_sft14b_failed_em_spg2_s500_20260904/`
- `data/eval_results/opd_ckpt300_fulldev_gpufaiss_20260905/`

### 三数据集基础实验矩阵

基础 4 个 setting 已在 HotpotQA、2Wiki、MuSiQue 全量 dev 上完成。
`Cover-EM` 反映字符串包含关系，`LLM-acc` 由 DeepSeek judge 判断事实等价；
二者同时报告是因为 DPO 会显著改变答案长度和表述风格。

| 数据集 | Setting | N | Cover-EM | LLM-acc | EM | F1 | Max-turn rate |
|---|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | Zero-shot | 7,405 | 0.2680 | 0.3380 | 0.2040 | 0.2730 | 45.1% |
| HotpotQA | SFT | 7,405 | **0.5070** | **0.6070** | 0.0971 | 0.2634 | 10.7% |
| HotpotQA | Canonical-answer SFT | 7,405 | 0.4748 | 未评 | **0.4373** | **0.5513** | 9.6% |
| HotpotQA | DPO-only | 7,405 | 0.3999 | 0.5356 | 0.3492 | 0.4563 | 未统一记录 |
| HotpotQA | SFT+DPO | 7,405 | 0.4693 | 0.6060 | **0.4008** | **0.5233** | **3.4%** |
| 2Wiki | Zero-shot | 12,576 | 0.1114 | 0.1178 | 0.0803 | 0.1049 | 66.6% |
| 2Wiki | SFT | 12,576 | **0.4488** | 0.4431 | 0.1018 | 0.2515 | 27.9% |
| 2Wiki | Canonical-answer SFT | 12,576 | 0.4188 | 未评 | **0.4051** | 0.4513 | 24.9% |
| 2Wiki | DPO-only | 12,576 | 0.4061 | 0.4249 | 0.3496 | 0.4194 | 未统一记录 |
| 2Wiki | SFT+DPO | 12,576 | 0.4452 | **0.4705** | **0.3915** | **0.4688** | **17.3%** |
| MuSiQue | Zero-shot | 2,417 | 0.0956 | 0.1129 | 0.0728 | 0.1070 | 63.6% |
| MuSiQue | SFT | 2,417 | 0.1911 | 0.2081 | 0.0492 | 0.1205 | 33.4% |
| MuSiQue | Canonical-answer SFT | 2,417 | 0.1841 | 未评 | 0.1651 | 0.2405 | 28.7% |
| MuSiQue | DPO-only | 2,417 | 0.1452 | 0.1957 | 0.1200 | 0.1935 | 未统一记录 |
| MuSiQue | SFT+DPO | 2,417 | **0.2069** | **0.2462** | **0.1667** | **0.2477** | **16.9%** |

基础矩阵的核心解释：

- SFT 最显著的贡献是学会多轮 RAG 协议和及时停止，三数据集 max-turn
  均大幅下降。
- SFT+DPO 在 HotpotQA 上 Cover-EM 比 SFT 低，但 LLM-acc 基本相同；
  EM/F1 的大幅提高主要反映回答更简洁、与 gold 字符串更对齐。
- 2Wiki 和 MuSiQue 的 LLM-acc 在 SFT+DPO 后继续提高，说明 DPO 没有
  破坏 MuSiQue 能力，尽管 DPO 数据本身不含 MuSiQue。
- Canonical-answer SFT 直接修复了旧 SFT 的答案长度错配：HotpotQA
  EM/F1 超过 SFT+DPO，2Wiki/MuSiQue 的 EM/F1 也大幅高于旧 SFT；
  但 LLM-judge 尚未补评，且 Cover-EM 不如旧 SFT/E12 稳定。

### 主结果可比性

| 对比 | 是否可直接比较 | 原因 |
|---|---|---|
| E01 SFT+DPO vs E09/E10 | 是 | 同一 7405 ID、相同 Evidence Agent/Top-3 评测口径，且 LoRA 已修复 |
| E00 SFT vs E04 LoRA GRPO-control | 是 | held-out train-derived 数据，无 teacher，用于检验 GRPO 本身 |
| E01 vs E05 Full GRPO | 可比较最终任务指标 | 同为 held-out 7405，但训练起点和更新参数范围不同，需同时报告行为指标 |
| E06/E07 vs E09/E10 | 不建议直接比较 | Reward-v2/v3 主要采用 raw-document 流程；E09/E10 使用独立 Evidence Agent |
| E02 与任何实验 | 否 | HotpotQA dev 泄露 |
| E03 与修复后 OPSD | 否 | teacher 作用范围、pipeline 和 LoRA 状态均不同 |

### 后续新增实验的固定记录格式

每个新实验必须在本文件中记录以下字段，缺失字段标记为“未知”，不能靠会话记忆补齐：

```text
实验 ID / 日期 / 状态
研究问题与假设
起始模型与 checkpoint
训练数据来源、规模、是否与评测集隔离
Student 单题 pipeline
相对上一实验唯一改变的变量
reward/teacher 公式与系数
训练方式（LoRA/全参数）、step、batch、采样数、学习率
检索器、Top-k、Evidence Agent、max-turn
固定小样本结果与全量结果
对照组、置信区间和显著性
训练日志、checkpoint、results、metrics 的绝对或项目相对路径
已知异常、是否可用于主结论
一句话结论与下一步决策
```

---

## 历史 ClosureRAG 规划（未执行，不属于当前实验主线）

**日期**：2026-05-30

**状态**：仅规划，未形成可用实验结果

---

### Run Log

| Run ID | Gate | Date | Dataset | N | System | EM | F1 | Premature Stop Rate | Unsupported Claim Rate | Bridge Entity Recall | Avg Steps | Status |
|--------|------|------|---------|---|--------|-----|-----|---------------------|----------------------|---------------------|-----------|--------|
| - | G0 | - | HotpotQA | 50 | ReasonRAG baseline | - | - | - | - | - | - | pending |
| - | G0 | - | HotpotQA | 50 | ClosureRAG-prompt (Board+Stop) | - | - | - | - | - | - | pending |
| - | G1 | - | HotpotQA | 80 | +Claim Gate | - | - | - | - | - | - | pending |
| - | G2 | - | HotpotQA | 200 | Full prompt + heuristic | - | - | - | - | - | - | pending |
| - | G2 | - | HotpotQA | 200 | ReasonRAG + more steps | - | - | - | - | - | - | pending |
| - | G2 | - | HotpotQA | 200 | Post-hoc verification | - | - | - | - | - | - | pending |
| - | G3 | - | HotpotQA | 500 | ClosureRAG-trained | - | - | - | - | - | - | pending |
| - | G3 | - | 2Wiki | 500 | ClosureRAG-trained | - | - | - | - | - | - | pending |
| - | G4 | - | All 4 | full | Full system + baselines | - | - | - | - | - | - | pending |

---

### Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-30 | SAPR-RAG v1 → ClosureRAG v2 | Phase 3-4 review: v1 四模块同等贡献 novelty 不硬，收缩为 Board + Claim Gate + Stop Closure |
| 2026-05-30 | 砍掉 branch rollback | 审稿人建议：复杂度高，会被要求和 MCTS 对比 |
| 2026-05-30 | Evidence selection 降级为 heuristic | 审稿人建议：LLM judge 打分太慢太贵，不是核心贡献 |
| 2026-05-30 | Board schema 从 6 字段砍到 4 字段 | 审稿人建议：最小充分状态，减少出错面 |

---

### Data Artifacts

| Artifact | Description | Status |
|----------|-------------|--------|
| Board annotations (500 trajectories) | Phase 1 数据积累 | pending |
| Claim-evidence pairs (~600) | Phase 1 数据积累 | pending |
| Closure labels (slot/claim/chain) | Phase 1 数据积累 | pending |
| Human validation (100-200) | 人工校验 Board + claim 标注 | pending |

---

### Key Files

- Proposal: `refine-logs/FINAL_PROPOSAL_v2.md`
- Experiment Plan: `refine-logs/EXPERIMENT_PLAN_v2.md`
- Phase 3 Novelty: `idea-stage/PHASE3_NOVELTY_VERIFICATION.md`
- Phase 4 Review: `idea-stage/PHASE4_CRITICAL_REVIEW.md`
- Literature Landscape: `idea-stage/LITERATURE_LANDSCAPE.md`
- ReasonRAG Improvement Ideas: `idea-stage/REASONRAG_IMPROVEMENT_IDEAS.md`

---

## OPSD / GRPO Experiment Record

**Date**: 2026-08-09
**Scope**: SAPR-RAG OPSD、严格 LoRA GRPO-control 与全参数 GRPO 的训练和 HotpotQA held-out 评测。

**实验 ID**：E03（旧全动作 OPSD）、E04（严格 LoRA control）、E05（全参数 GRPO）

> E03 使用“gold teacher 评价全部动作”的旧设计，并早于 Swift rollout
> LoRA 漏挂问题的最终修复；其数值只用于解释失败模式，可信度为 C。
> E04/E05 使用 train-derived held-out 数据，可作为对应方法的有效结果。

### Method Conclusions

| Topic | Conclusion |
|---|---|
| OPSD mechanism | Teacher and student score the same student-sampled response tokens. Student uses the normal prompt / online RAG context; teacher uses a privileged `teacher_prompt` containing gold evidence and gold answer. |
| Loss integration | Current implementation does not add a standalone KL loss. It injects teacher log-ratio into token-level GRPO advantage: `A_t = A_GRPO + alpha * (logp_teacher_t - logp_student_t)`. |
| Official ms-swift support | ms-swift supports `GRPO + teacher` as OPD-RL and supports OPSD via `teacher_prompt`. It is a supported path, not a universal recommendation for all GRPO tasks. |
| Why use OPSD here | SAPR-RAG has gold supporting facts and gold answers, so privileged teacher prompts are natural for RAG trajectory supervision. |
| BGE-on-GPU decision | Not changed for this run. Current stable retrieval service is `BGE CPU + FAISS GPU`; making `BGE GPU + FAISS GPU` requires a unified environment with both CUDA torch and H20-compatible `faiss-gpu=1.14.3`. |

### Training Run

| Item | Value |
|---|---|
| Run | `opsd_colocate_effect_pbs2_g7_manual` |
| Dataset | `data/grpo/hotpotqa_2wiki_train_opsd.jsonl` |
| Samples | 7320 total, balanced HotpotQA / 2Wiki |
| GPUs | GPU0 retrieval service, GPU1-7 colocate GRPO/vLLM |
| Effective prompts per update | 2 prompts/update (`7 GPUs * per_device_train_batch_size=2 / num_generations=7`) |
| Total steps | 3660 |
| Epochs | 1.0 |
| Runtime | 16h 22m 26s |
| Avg step time | 16.1s/it |
| Final checkpoint | `03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_colocate_full/opsd_colocate_effect_pbs2_g7_manual/v0-20260805-203554/checkpoint-3660` |
| Training log | `03_sapr_rag/scripts/grpo/logs/opsd_colocate_effect_pbs2_g7_manual.log` |

### Training Curve Summary

250-step smoothed online reward metrics from the training log:

| Step range | Reward | F1 reward | Relevance | Format | Avg turns | Mean length | `frac_reward_zero_std` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-250 | 0.796 | 0.599 | 0.754 | 0.920 | 3.18 | 290.6 | 0.316 |
| 1001-1250 | 0.763 | 0.569 | 0.742 | 0.906 | 3.21 | 297.5 | 0.286 |
| 2251-2500 | 0.779 | 0.582 | 0.759 | 0.900 | 3.22 | 294.6 | 0.278 |
| 2501-2750 | 0.789 | 0.591 | 0.761 | 0.927 | 3.14 | 286.0 | 0.306 |
| 2751-3000 | 0.811 | 0.612 | 0.770 | 0.912 | 3.17 | 292.8 | 0.300 |
| 3001-3250 | 0.798 | 0.595 | 0.784 | 0.923 | 3.12 | 285.8 | 0.304 |
| 3251-3500 | 0.800 | 0.601 | 0.770 | 0.909 | 3.20 | 294.6 | 0.316 |
| 3501-3660 | 0.822 | 0.624 | 0.760 | 0.923 | 3.14 | 291.7 | 0.375 |

Interpretation:

| Observation | Interpretation |
|---|---|
| Online reward is noisy and not monotonic. | Expected for on-policy GRPO with small effective prompt batch. |
| Late windows are slightly higher than mid-run. | Training did not collapse, but online reward alone is not sufficient for checkpoint selection. |
| `frac_reward_zero_std` remains around 0.28-0.38. | Many groups still have weak intra-group reward contrast, so GRPO signal is sparse. |
| Fixed evaluation favored `checkpoint-3000` over final on 200 samples. | Checkpoint selection must rely on fixed-set / full-dev eval, not only online reward. |

### HotpotQA 200 Strict Evaluation

Strict evaluation extracts only final `<answer>...</answer>` from the final assistant message.

| Checkpoint | N | Answered | EM | Cover EM | F1 | Avg turns | Avg latency | Result path |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-250` | 200 | 173 | 0.320 | 0.390 | 0.4242 | 2.035 | 1.50s | `data/eval_results/hotpotqa/opsd_alpha0p1_ckpt250_lora_rollout_rag_http200_20260805_200200/metrics.strict.json` |
| `checkpoint-3000` | 200 | 175 | 0.325 | 0.405 | 0.4237 | 2.070 | 0.55s | `data/eval_results/hotpotqa/opsd_alpha0p1_ckpt3000_lora_rollout_rag_http200_20260806_162349/metrics.strict.json` |
| `checkpoint-3660` | 200 | 171 | 0.305 | 0.385 | 0.4070 | 2.105 | 0.55s | `data/eval_results/hotpotqa/opsd_alpha0p1_ckpt3660_lora_rollout_rag_http200_20260806_165003/metrics.strict.json` |

Interpretation:

| Finding | Note |
|---|---|
| `checkpoint-3000` is best among the 200-sample strict evals. | It beats final by +2.0 EM points and +2.0 cover EM points on this subset. |
| The gain over early checkpoints is small. | `checkpoint-3000` vs `checkpoint-250` is only +0.5 EM point on 200 samples, equal to one question. |
| Full-dev eval is needed. | 200 samples have wide uncertainty; full HotpotQA dev narrows the confidence interval. |

### Inference Throughput Benchmark

Benchmark target: `checkpoint-3000`, HotpotQA first 200 samples, one rollout server, `max_tokens=512`.

| Batch size | N | Wall time | Throughput | Avg latency (`batch_dt / batch`) | Errors |
|---:|---:|---:|---:|---:|---:|
| 8 | 200 | 105s | 1.905 samples/s | 0.523s | 0 |
| 16 | 200 | 79s | 2.532 samples/s | 0.382s | 0 |
| 32 | 200 | 68s | 2.941 samples/s | 0.325s | 0 |
| 64 | 200 | 59s | 3.390 samples/s | 0.282s | 0 |

Decision:

| Decision | Rationale |
|---|---|
| Use `batch_size=64` for full HotpotQA dev eval. | Highest measured throughput among tested values. |
| Keep `max_tokens=512`. | Matches previous 200-sample checkpoint eval and avoids changing answer budget. |
| Do not switch BGE to GPU for this run. | Environment is not yet unified for CUDA torch + H20-compatible FAISS GPU. |

### Full HotpotQA Dev Evaluation

Full-dev evaluation completed with `batch_size=64`.

| Checkpoint | N | Answered | EM | Cover EM | F1 | Avg turns | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `checkpoint-3000` | 7405 | 6321 (85.36%) | 0.2895 | 0.3869 | 0.4026 | 2.122 | completed |
| `checkpoint-3660` | 7405 | 6311 (85.23%) | 0.2883 | 0.3860 | 0.4014 | 2.122 | completed |

OPSD full-dev did not preserve the 200-sample ranking advantage: `checkpoint-3000`
and final are effectively tied, and both are below the SFT+DPO starting point on
Cover-EM (`0.4693`). The strict HTTP artifact records no explicit max-turn
exceptions, so its `max_turns_rate=0` is not compared with `agent_infer.py`
behavior metrics.

### Strict LoRA GRPO-Control

**实验 ID**：E04

This control removes the privileged teacher signal and the leaked dev-derived
training set. It starts from SFT and trains on a balanced official
train-derived HotpotQA/2Wiki dataset.

| Item | Value |
|---|---|
| Training data | 7320 samples: HotpotQA train 3660 + 2Wiki train 3660 |
| Tuner | LoRA |
| OPSD | disabled |
| Evaluated checkpoint | `checkpoint-1000` (training stopped early for held-out evaluation) |
| Evaluation | HotpotQA dev, 7405 unique IDs, 0 cohort exceptions |

| Setting | Cover EM | EM | F1 | Answered | Avg turns | Max-turn rate | Empty evidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT | 0.5070 | 0.0971 | 0.2634 | 89.28% | 2.513 | 10.71% | 21.29% |
| LoRA GRPO-control ckpt1000 | 0.5080 | 0.1048 | 0.2716 | 89.60% | 2.508 | 10.36% | 20.61% |

The result is essentially tied with SFT. It confirms that the corrected
train-derived GRPO path is valid, but does not establish a meaningful
held-out gain.

### Full-Parameter GRPO

**实验 ID**：E05

Policy and reference models both start from the SFT LoRA merged into the base
model. Training uses ZeRO-3 on GPU1-7 and completes one epoch (`3660` steps)
over the same 7320-sample balanced train-derived dataset.

| Setting | Cover EM | EM | F1 | Answered | Avg turns | Max-turn rate | Empty evidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT | 0.5070 | 0.0971 | 0.2634 | 89.28% | 2.513 | 10.71% | 21.29% |
| SFT+DPO | 0.4693 | 0.4008 | 0.5233 | 96.57% | 2.151 | 3.43% | 26.20% |
| Full GRPO ckpt2500 | 0.4493 | 0.4003 | 0.5071 | 77.06% | 3.162 | 22.86% | 18.72% |
| Full GRPO ckpt3000 | 0.4258 | 0.3824 | 0.4796 | 69.14% | 3.735 | 30.76% | 21.16% |
| Full GRPO ckpt3660 | 0.4265 | 0.3854 | 0.4817 | 69.79% | 3.704 | 30.16% | 20.69% |

Paired bootstrap uses the same 7405 IDs, 10000 resamples and seed `20260808`.

| Checkpoint vs SFT | Cover-EM delta (95% CI) | F1 delta (95% CI) | Answer-rate delta (95% CI) |
|---|---:|---:|---:|
| ckpt2500 | -5.77pt [-6.83, -4.70] | +24.37pt [+23.38, +25.36] | -12.22pt [-13.18, -11.29] |
| ckpt3000 | -8.12pt [-9.20, -7.02] | +21.62pt [+20.61, +22.61] | -20.14pt [-21.16, -19.08] |
| ckpt3660 | -8.05pt [-9.12, -6.95] | +21.83pt [+20.82, +22.82] | -19.49pt [-20.53, -18.45] |

`checkpoint-2500` is the best full-parameter checkpoint, but it is still
`-2.00pt` Cover-EM versus SFT+DPO (95% CI `[-3.05, -0.96]`) and `-1.62pt`
F1 (95% CI `[-2.60, -0.64]`). Its EM is effectively tied with SFT+DPO.

### Current Conclusion

| Finding | Evidence |
|---|---|
| Corrected LoRA GRPO is neutral. | ckpt1000 Cover-EM is 0.5080 versus SFT 0.5070. |
| OPSD is ineffective in the current setup. | Full-dev Cover-EM falls to about 0.386 from the SFT+DPO starting point 0.4693. |
| Full GRPO changes answer style but hurts end-to-end behavior. | Mean answer length falls from 13.29 to 2.35-2.55 words, while answer rate falls to 69-77% and max-turn rate rises to 23-31%. |
| The current reward is misaligned with termination. | Relevance reward encourages continued retrieval; format weight is only 0.05 and there is no explicit turn/max-turn penalty. |

The next GRPO iteration should add an explicit termination reward and turn
penalty, reduce relevance weight, and select checkpoints on fixed held-out
Cover-EM plus answer/max-turn behavior rather than online reward alone.

---

## Reward-v2：anti-repeat 与终止约束

**实验 ID**：E06

**日期**：2026-08-09

**研究问题**：重复 query 和跑满轮次是否是 GRPO 无增益的主因；提高重复惩罚并修复 max-turn reward 后，能否同时降低重复检索并提高最终答案指标。

### 方法

| 项目 | 设置 |
|---|---|
| 起点 | SFT checkpoint-1650 合并模型 |
| 训练方式 | LoRA，rank 16，学习率 `1e-6` |
| 数据 | `data/grpo/hotpotqa_2wiki_train_reward_v2.jsonl` |
| 训练步数 | 500，checkpoint-100/200/300/400/500 |
| 单题流程 | query → BGE+FAISS Top-3 原始文档 → 下一轮 query/answer |
| 重复约束 | Prompt 明确说明检索确定性；归一化后完全相同 query 运行时拦截 |
| Reward | F1 / relevance / format / turn cost / repeat query / max turn |
| 权重 | `1.0 / 0.15 / 0.05 / 0.02 / 0.15 / 0.50` |
| max-turn 修复 | 按 agent turn 与实际 retrieval 次数的关系重写触发条件 |

Reward-v2 仍使用累计 relevance：只要轨迹最终覆盖 gold evidence 就给分，
重复命中同一证据不会区分“首次新增”与“重复出现”。这正是 Reward-v3
后续替换为 marginal relevance 的原因。

### 结果

同一 raw-document 流程下的固定前 200 条对照：

| 模型 | EM | F1 | Cover-EM | Answer rate | Avg queries | Exact repeat |
|---|---:|---:|---:|---:|---:|---:|
| SFT merged | 0.105 | 0.2835 | **0.545** | 90.5% | 2.165 | **13.5%** |
| Reward-v2 ckpt300 | 0.110 | 0.2739 | 0.520 | 89.5% | 2.210 | 15.0% |

HotpotQA full dev 的 ckpt300 结果：

| N | EM | F1 | Cover-EM | Answer rate | Avg turns | Max-turn rate | Repeat trajectory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7405 | 0.1086 | 0.2761 | 0.5121 | 89.68% | 2.484 | 10.28% | 20.54% |

### 结论

- 增强重复惩罚和运行时拦截没有让模型真正学会避免重复；固定 200 条中
  exact-repeat 从 SFT 的 13.5% 升到 15.0%。
- full-dev 的 EM/F1/Cover-EM 相对 SFT 只有小幅数值变化，且没有配对
  显著性证据，不能认为 Reward-v2 带来稳定提升。
- 问题不只是惩罚力度：累计 relevance 对首次发现和重复发现同一证据
  区分不足，因此转向 Reward-v3 的“逐轮新增证据”奖励。

权威产物：

- 训练日志：`03_sapr_rag/scripts/grpo/logs/grpo_reward_v2_anti_repeat_w015_maxturnfix_s500_20260809_112257.log`
- 固定 200 对照：`data/eval_results/hotpotqa/rawdoc_sft_vs_reward_v2_ckpt300_200_20260809/`
- full-dev：`data/eval_results/hotpotqa/reward_v2_anti_repeat_w015_ckpt300_full_dev_20260809/metrics.with_repeat.json`

---

## Reward-v3（新增证据奖励）实验记录

**实验 ID**：E07

**日期**：2026-08-10

**范围**：在统一"原始文档流程"下，从 SFT 合并模型起点跑 Reward-v3 500-step LoRA GRPO 小规模对照实验。

**数据完整性说明**：此前在控制节点检查节点本地临时目录，误判训练节点
上的 checkpoint 已丢失；后续连接训练节点后确认 checkpoint 仍在。
但是当前共享工作区没有落盘可核验的 Reward-v3 最终 `metrics.json`，
只保留训练日志、早期 200 条会话数值和两次失败 sweep 的状态文件。
因此本节可信度维持 B，不能把“补评曾执行”当作可复现的最终结果。

### 与上一轮 Reward-v2 的差异

| 项目 | 说明 |
|---|---|
| 新增证据奖励 | 用 `sapr_marginal_relevance` 取代旧 `sapr_relevance`：gold 证据仅首次命中给分，重复命中不加分，全覆盖后仍检索则扣分（`gamma=0.9`，`after_full_penalty=0.10`） |
| 完全重复 query 硬拦截 | 运行时归一化后若与本轨迹已出现的 query 完全一致，不再调检索，直接返回提示并标记 `exact_duplicate=True` |
| 训练/评估流程统一 | 统一走原始文档流程（query → top-3 原始文档回填），不再走 evidence agent 抽取 |
| 检索服务 | BGE CPU + FAISS GPU0（H20 兼容 `faiss-gpu 1.14.3 + CUDA 12.9`），训练占用 GPU1-7 |

### 训练配置

| 项目 | 取值 |
|---|---|
| Run | `grpo_reward_v3_marginal_w015_s500_20260810_121225` |
| 起点 | SFT LoRA 合并模型 `qwen2_5_7b_sft_ckpt1650_merged` |
| Tuner | LoRA |
| 奖励项 | `sapr_f1 sapr_marginal_relevance sapr_format sapr_turn_cost sapr_repeat_query sapr_max_turn` |
| 奖励权重 | `1.0 0.15 0.05 0.02 0.15 0.50` |
| 训练步数 | 500（跑满，每 100 step 存一次 checkpoint） |
| GPU 分工 | GPU0 检索服务，GPU1-7 GRPO/vLLM |
| 训练日志（保留，可核验） | `03_sapr_rag/scripts/grpo/logs/grpo_reward_v3_marginal_w015_s500_20260810_121225.log` |
| checkpoint / 评估产物 | 训练节点上已确认保留并完成补评 |

### 训练曲线（据训练日志复核，100-step 分段均值，未乘权重）

| step 段 | reward | F1 | Marginal | Format | TurnCost | Repeat | MaxTurn | 平均轮数 | completion 长度 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-100 | 0.286 | 0.269 | 0.603 | 0.920 | -1.318 | -0.351 | -0.080 | 3.29 | 357 |
| 101-200 | 0.301 | 0.268 | 0.621 | 0.939 | -1.339 | -0.331 | -0.060 | 3.33 | 355 |
| 201-300 | 0.297 | 0.275 | 0.628 | 0.920 | -1.322 | -0.344 | -0.080 | 3.31 | 350 |
| 301-400 | 0.323 | 0.289 | 0.623 | 0.933 | -1.351 | -0.306 | -0.066 | 3.34 | 358 |
| 401-500 | 0.299 | 0.285 | 0.598 | 0.926 | -1.371 | -0.384 | -0.074 | 3.37 | 359 |
| 全程 | 0.301 | 0.277 | 0.615 | 0.928 | -1.340 | -0.343 | -0.072 | 3.33 | — |

全程 `frac_reward_zero_std=0.024`（组内奖励对比整体不塌缩，方差主要来自 F1）。

训练曲线解读：

| 观察 | 解读 |
|---|---|
| 总 reward 全程横在 0.29-0.32，无上升趋势 | 主信号 F1 从 0.269 微升到 0.285，基本没被 GRPO 推动 |
| Marginal（新增证据奖励）稳定在 0.60 附近、不上升 | 检索到新 gold 证据的能力从 SFT 起点起就没再提高 |
| Repeat 惩罚末段反而更负（-0.35 → -0.38） | 训练未能让模型学会不重复；靠运行时硬拦截兜底，未在奖励层面收敛 |
| 平均轮数缓慢升到 3.37 | TurnCost 权重仅 0.02，压不住多检索倾向 |

### 早期离线评估会话记录

以下表格只保留早期会话记录。由于最终补评 metrics 未同步到当前工作区，
这些数字只能用于判断大致趋势，不能进入论文主表：

| 模型 | EM | cover-EM | F1 | 重复率 | 来源 |
|---|---:|---:|---:|---:|---|
| SFT merged | 0.105 | 0.545 | 0.284 | 0.135 | 会话记录，产物在 |
| Reward-v2 ckpt300 | 0.110 | 0.520 | 0.274 | 0.150 | 会话记录 |
| Reward-v3 ckpt500 | 0.105 | 0.520 | 0.270 | 0.135 | 早期会话口径，当前无本地最终 metrics |

- ckpt300/400 的 sweep 评估当时因 vLLM 显存不足全部跳过，未产出 metrics；磁盘仅存 rollout 日志（`data/eval_results/hotpotqa/reward_v3_ckpt_sweep_200_*`）。
- ckpt500 的 checkpoint 后来在训练节点上确认仍在；此前是检查节点错误导致的误判。

### 诊断性结论（当前不可进入主表）

| 诊断判断 | 依据与限制 |
|---|---|
| Reward-v3 修好了 v2 的"重复变多"副作用 | 重复率从 v2 的 0.150 回到 SFT 水平 0.135（评估数字待核验） |
| 但主指标（EM/cover-EM/F1）未获提升 | 训练侧 F1 全程横盘（日志可核验）；评估侧持平/略低于 SFT（待核验） |
| 瓶颈可能不在奖励设计，而在检索召回上限或 SFT 起点饱和 | F1/Marginal 均不随训练上升，KL 极小 |

**后续状态**：训练曲线和早期评测均未显示稳定收益，研究方向已转向
action-scoped OPSD。若未来需要引用 Reward-v3 数值，必须先把 worker
上的最终结果同步到共享目录并重新生成 `metrics.json`。

---

## D1b：Query 质量与检索器上限诊断

**实验 ID**：E08

**日期**：2026-08-11

**可信度**：B（HotpotQA 前 200 条诊断，不是最终任务评测）

### 研究问题与方法

Reward-v3 没有提升新增证据覆盖，可能有两种原因：

1. FAISS/BGE 检索器本身无法在 Top-3 找到 gold 文档；
2. 检索器具备能力，但模型生成的 query 不够好。

固定同一 BGE+FAISS 索引，对每道题分别使用三种输入检索：

- `question`：直接用原问题；
- `model_query`：使用 SFT 真实轨迹生成的子查询；
- `gold_title`：直接使用 gold supporting title，作为接近 oracle 的查询。

每种输入分别统计 Top-3/5/10/20 的 gold title 平均覆盖率和“所有 gold
title 均召回”的完全召回率。

### 结果

| Query 来源 | Top-3 平均覆盖 | Top-3 完全召回 | Top-5 完全召回 | Top-10 完全召回 | Top-20 完全召回 |
|---|---:|---:|---:|---:|---:|
| 原问题 | 27.25% | 4.0% | 12.0% | 18.0% | 26.5% |
| SFT 模型子查询 | 44.75% | 20.5% | 26.0% | 37.5% | 43.5% |
| Gold title | 69.50% | **50.0%** | 59.0% | 65.0% | 69.5% |

### 结论与决策

- 模型子查询明显优于直接用原问题，说明 Agentic decomposition 有效。
- 同样固定 Top-3，模型 query 的完全召回率只有 20.5%，而 gold-title
  query 可达到 50.0%；Query 质量是当前检索覆盖的主要可优化空间。
- Top-20 能提高诊断上限，但主实验仍固定 Top-3，以保持与 ReasonRAG
  检索预算一致；不采用 Top-20+rerank 作为主表方案。
- 该结果直接推动后续从“继续堆 reward/step”转向 Query 级特权蒸馏，
  即 E11 分动作 OPSD。

权威产物：

- `data/eval_results/hotpotqa/d1b_retriever_ceiling_200_20260811.json`
- `03_sapr_rag/scripts/eval/d1b_retriever_ceiling.py`

---

## Action-scoped OPSD：LoRA 修复、25/100-step 实验与分动作新方案

**日期**：2026-08-11 至 2026-08-12

### Swift rollout 漏挂 LoRA：问题、影响与修复

`swift rollout` 虽然解析了 `--adapters`，但
`SwiftRolloutDeploy.get_infer_engine()` 没有向 `GRPOVllmEngine` 传入
`args.adapters`。日志会显示 LoRA 路径，实际 rollout 却与无 LoRA
基础模型逐 token 一致。因此，所有依赖该 Swift server 静态加载 LoRA
checkpoint 的旧评估结果均不能继续作为有效 OPSD/GRPO 结论。

修复方式是在构造 rollout inference engine 时显式透传
`adapters=args.adapters`。修复后，同一 247-token prompt 的 78 个生成
token 与 direct vLLM + SFT+DPO 完全一致；固定 200 条恢复到
`EM 0.360 / F1 0.4820 / Cover-EM 0.425`。后续 25-step 和 100-step
实验均使用修复后的 LoRA rollout 链路。

### 共同训练设置

| 项目 | 设置 |
|---|---|
| 起点 | Qwen2.5-7B + SFT+DPO LoRA `checkpoint-395` |
| 更新方式 | LoRA 增量训练，不是全参数训练 |
| 训练数据 | `data/grpo/hotpotqa_2wiki_train_pilot_opsd.jsonl`，100 条 pilot |
| 在线流程 | reasoning/query → BGE+FAISS Top-3 → 独立 Evidence Agent → 下一轮 query/answer |
| OPSD 范围 | Answer-only；只有以 `<answer>` 结束的模型 turn 接收 teacher 信号 |
| Teacher 信息 | gold answer + gold supporting evidence |
| Teacher 系数 | `teacher_kl_coef=0.03` |
| GRPO reward | F1 / relevance / format，权重 `1.0 / 0.2 / 0.05` |
| 采样 | `num_generations=8`，`steps_per_generation=8` |
| 学习率 | `1e-6` |
| 检索口径 | BGE + FAISS Top-3，无 reranker |

这里的“Evidence”表示 rollout 中启用了独立 Evidence Agent，**不表示
Evidence Agent 的输出 token 接收了 OPSD**。这两轮实验都是
Answer-only OPSD。

OPSD 仍采用逐 token advantage 注入：

```text
A_t = A_GRPO + 0.03 * (logp_teacher_t - logp_student_t)
```

基础 GRPO advantage 作用于所有 completion token；teacher log-ratio
只作用于 Answer action mask。25-step 首个采样批记录到
`teacher_action_scope_ratio=0.3805`、`teacher_kl_scoped=0.1360`，
证明修复后 teacher 信号非零且确实限制在 Answer token。

### 第一轮：修复后 Answer-only OPSD，25 step

**实验 ID**：E09

HotpotQA 全量 dev 结果：

| 模型 | N | Answered | EM | F1 | Cover-EM | Avg turns | Max-turn rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReasonRAG 论文 | - | - | 0.3840 | 0.4890 | 未报告 | - | - |
| 本地 SFT+DPO | 7405 | 96.57% | 0.4008 | 0.5233 | 0.4693 | 2.151 | 3.43% |
| 修复后 OPSD ckpt25 | 7405 | 7144 (96.48%) | **0.4054** | **0.5264** | 0.4690 | 2.135 | 3.50% |

同一 7405 个 ID、20000 次配对 bootstrap：

| 指标 | 相对 SFT+DPO | 95% CI | 单侧 p 值 | 结论 |
|---|---:|---:|---:|---|
| EM | +0.46pt | [-0.12, +1.05] | 0.0648 | 正向趋势，未达到 0.05 显著性 |
| F1 | +0.31pt | [-0.26, +0.86] | 0.1432 | 正向趋势，不显著 |
| Cover-EM | -0.03pt | [-0.63, +0.57] | 0.5410 | 基本持平 |

相对 ReasonRAG 论文值，EM `+2.14pt`，F1 `+3.74pt`。候选自身
95% bootstrap 下界为 EM `0.3943`、F1 `0.5162`，均高于 ReasonRAG。
但相对本地 SFT+DPO 的增量不显著，因此不能把超过 ReasonRAG 的主要
贡献归因给 Answer-only OPSD；主要贡献仍来自 SFT+DPO 起点。

权威产物：

- `data/eval_results/hotpotqa/opsd_lorafix_evidence_alpha003_ckpt25_full7405_20260811/metrics.json`
- `data/eval_results/hotpotqa/opsd_lorafix_evidence_alpha003_ckpt25_full7405_20260811/paired_bootstrap_vs_sft_dpo.json`
- checkpoint：`03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/opsd_lorafix_evidence_alpha003_s25_20260811/v0-20260811-130405/checkpoint-25`

### 第二轮：扩展到 100 step

**实验 ID**：E10

保持起点、数据、Evidence Agent、reward、LoRA 和
`teacher_kl_coef=0.03` 不变，仅把训练扩展到 100 step，并保存
checkpoint-25/50/75/100。

固定 200 条 checkpoint sweep：

| Checkpoint | EM | F1 | Cover-EM | Answered | Avg turns | Empty evidence |
|---|---:|---:|---:|---:|---:|---:|
| ckpt25 | 0.355 | 0.4737 | 0.425 | 193/200 | 2.020 | 22.77% |
| ckpt50 | 0.355 | 0.4773 | 0.435 | 194/200 | 2.030 | 21.92% |
| ckpt75 | 0.355 | 0.4773 | 0.435 | 193/200 | 2.055 | 22.38% |
| ckpt100 | 0.355 | **0.4802** | **0.435** | 194/200 | 2.035 | 21.87% |

固定 200 条上，50 step 后 EM/Cover-EM 已进入平台，F1 到 100 step
仅缓慢增加。为避免小样本误判，对 ckpt100 继续运行 HotpotQA 全量 dev：

| 模型 | N | Answered | EM | F1 | Cover-EM | Avg turns | Max-turn rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 本地 SFT+DPO | 7405 | 96.57% | 0.4008 | 0.5233 | 0.4693 | 2.151 | 3.43% |
| OPSD ckpt25 | 7405 | 96.48% | **0.4054** | **0.5264** | **0.4690** | 2.135 | 3.50% |
| OPSD ckpt100 | 7405 | 7142 (96.45%) | 0.4032 | 0.5243 | 0.4675 | 2.125 | 3.52% |

ckpt100 相对 SFT+DPO 的 20000 次配对 bootstrap：

| 指标 | 相对 SFT+DPO | 95% CI | 单侧 p 值 | 结论 |
|---|---:|---:|---:|---|
| EM | +0.24pt | [-0.34, +0.84] | 0.2181 | 不显著 |
| F1 | +0.10pt | [-0.46, +0.67] | 0.3655 | 不显著 |
| Cover-EM | -0.18pt | [-0.77, +0.42] | 0.7266 | 不显著 |

结论：Answer-only OPSD 的 25-step 结果有轻微正向趋势，但扩展到
100 step 后没有稳定放大，反而略有回落。继续简单增加训练步数缺少依据；
下一步应改变 teacher 信息分配方式，而不是继续延长 Answer-only 训练。

权威产物：

- 固定 200 sweep：`data/eval_results/hotpotqa/opsd_lorafix_evidence_alpha003_s100_direct200_20260811/summary_metrics.json`
- ckpt100 全量：`data/eval_results/hotpotqa/opsd_lorafix_evidence_alpha003_s100_ckpt100_full7405_20260811/metrics.json`
- 配对检验：`data/eval_results/hotpotqa/opsd_lorafix_evidence_alpha003_s100_ckpt100_full7405_20260811/paired_bootstrap_vs_sft_dpo.json`
- checkpoint：`03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/opsd_lorafix_evidence_alpha003_s100_20260811/v0-20260811-165523/checkpoint-100`

### 新方案：Query / Evidence / Answer 分动作 OPSD

**实验 ID**：E11

**状态**：代码、完整三源数据、1/2-step multi-action smoke 均已完成。
3000-step 正式训练正在训练节点上运行；尚未完成训练和三数据集
全量评测，因此以下中途指标只用于健康检查，不是效果结论。

#### 设计原则

| 动作 | Student 部署上下文 | Teacher 额外信息 | 因果约束 |
|---|---|---|---|
| Query | 原问题 + 当前真实查询/检索历史 | R3 成功轨迹中的有序参考查询计划 | 不允许看 gold answer 或 gold supporting facts |
| Evidence | 当前 query + 当前实际 Top-3 | 当前 Top-3 中可核验的 gold/SFT evidence | 必须作为独立 Evidence Agent auxiliary batch |
| Answer | 当前真实查询/evidence 历史 | gold answer + 已验证 supporting evidence | 只作用于最终 Answer token |

Query teacher 看到的是 R3 在真实检索环境中成功回答该问题的
`[query1, query2, ...]` 搜索计划，不是人工 gold subquery。每条 R3
轨迹在 parquet 中按 step 展开；构造器会按原问题重新分组并保留有序的
多个子查询，只删除同一问题中的完全重复 query。后续轮仍必须结合当前
BGE Top-3 状态决定下一条 query，不能机械复制“第 N 条参考 query”。

Teacher 信号使用独立动作系数：

```text
A_t = A_GRPO
    + beta_action(t) * (logp_teacher_t - logp_student_t)

beta_query    = 0.01
beta_evidence = 0.00  # 独立 auxiliary batch 完成前保持关闭
beta_answer   = 0.03
```

所有 completion token 始终保留 GRPO advantage；只有存在对应标注且
命中动作 mask 的 token 才叠加 OPSD。缺少 R3 查询计划的样本自动令
Query mask 为 0，不丢弃样本，也不伪造 Query teacher。

#### R3 数据审计

`data/raw/r3_coldstart.parquet`：

| 指标 | 数值 |
|---|---:|
| 逐 step 样本 | 178,061 |
| 唯一问题 | 51,328 |
| Query step | 126,808 |
| 至少有一个 Query 的问题 | 50,342 |
| 平均 Query step / 问题 | 2.47 |
| 每题总 step 中位数 | 3 |
| 每题总 step 最大值 | 11 |

最终三源训练文件：

`data/grpo/hotpotqa_2wiki_musique_train_multi_opsd.jsonl`

| 数据集 | 训练样本 | 含 R3 Query plan | Query 覆盖率 |
|---|---:|---:|---:|
| HotpotQA train | 90,447 | 25,377 | 28.1% |
| 完整 2Wiki train | 167,454 | 10,832 | 6.5% |
| MuSiQue train | 19,938 | 14,085 | 70.6% |
| 合计 | 277,839 | 50,294 | 18.1% |

所有 277,839 条样本均有 gold answer、gold title 和 Answer teacher
prompt。仅 8 条 MuSiQue 样本的 Answer evidence 因 token budget 截断。
2Wiki 的 `context` 与 `supporting_facts` 在 parquet 中是 JSON 字符串，
构造器已先透明解码再提取证据，避免把完整 2Wiki 错判为空证据。

#### 已完成实现

- ms-swift 数据契约支持
  `teacher_query_prompt`、`teacher_evidence_prompt`、
  `teacher_answer_prompt`。
- `teacher_action_scope=multi` 支持 Query/Evidence/Answer 分别
  teacher forward，再按动作 token mask 合并 log-prob。
- 支持 `teacher_query_kl_coef`、`teacher_evidence_kl_coef`、
  `teacher_answer_kl_coef`。
- Query 特权追加到首个 user turn，保证从第一轮 query 起可见。
- Answer 特权追加到最后一个 user turn，同时保留真实检索 observation。
- 无该动作标注的样本使用 identity fallback，动作系数 mask 自动归零。
- 数据构造脚本：
  `03_sapr_rag/scripts/grpo/build_grpo_dataset_action_opsd.py`。
- 完整训练集准备脚本：
  `03_sapr_rag/scripts/grpo/prepare_action_opsd_train_data.py`。
- 200 条数据契约 smoke：
  `data/grpo/hotpotqa_2wiki_action_opsd_smoke_100.jsonl`；
  HotpotQA 100 条中 82 条有 R3 Query prompt，2Wiki 100 条中 76 条有。
- 1-step 与 2-step 真实 Query+Answer multi-action smoke 已通过；
  Query/Answer teacher log-ratio 非零，且动作 mask 只落在对应 token。
- 同一 turn 同时出现 `<query>` 与 `<answer>` 时按调度器语义唯一分类为
  Answer，避免两个 OPSD mask 重叠。
- 36 项单元测试、`py_compile`、shell 语法检查和 `git diff --check`
  已通过。

#### 正式训练

| 项目 | 配置 |
|---|---|
| run | `opsd_multi_q001_a003_3src_s3000_actionfix_tmux_20260812` |
| 起点 | SFT+DPO `checkpoint-395` |
| 训练设备 | GPU2-6 |
| rollout | GPU7，Evidence Agent 开启 |
| 检索 | GPU0，BGE+FAISS Top-3，无 reranker |
| 动作系数 | Query 0.01 / Evidence 0 / Answer 0.03 |
| batch | per-device 2，gradient accumulation 4，8 generations |
| 训练步数 | 3000 |
| checkpoint | 每 500 step，最多保留 8 个 |
| 保活方式 | `tmux` 会话 `opsd_multi_s3000` |

完整一轮约需 55,568 optimizer step。按实测约 25.5 秒/step，完整一轮
约需 394 小时，明显超过 Worker 96 小时生命周期。当前 3000-step 方案
约覆盖 0.054 epoch，预计 21 小时，给 checkpoint sweep 和三数据集全量
评测预留时间；因此不能声称完成了完整 epoch。

截至 step 125 的健康检查：

- 全程平均 loss `0.0300`、reward `0.6636`、F1 reward `0.5201`、
  relevance `0.4775`、format `0.9593`；
- Query scoped KL `0.0681`（19 个采样批次），Answer scoped KL
  `0.1514`（63 个采样批次），两个 teacher 信号均实际生效；
- step 113-124 的梯度范数均值 `0.246`、最大 `0.396`，显存稳定在
  `25 GiB`；
- 未发现 NaN、OOM、通信错误或 rollout HTTP 错误；
- 最终 turn 同时含 Query/Answer 的格式违规率从 step 1-50 的 `4.95%`
  降至 step 51-100 的 `3.35%`，暂未出现输出退化。

以上是训练健康度，不代表离线 EM/F1/Cover-EM 已提升。

实时审计侧录：

- 详细现场日志与 checkpoint 审计记录在
  `docs/e11_action_opsd_live_audit.md`。
- 截至 `2026-08-12 15:47 +0800`，训练推进到 `1126/3000`，
  已保存 `checkpoint-500` 与 `checkpoint-1000`。
- step 1039 出现 `7494` token 长输出，step 1040/1046 出现局部
  loss/gradient 峰值；随后训练恢复，step 1047-1126 窗口 loss 最大
  `0.0478`、grad norm p95 为 `3.04`，step 1126 loss 为 `0.0409`、
  grad norm 为 `0.603`。
- 日志显存高水位已升至 `84.57 GiB`，实时 GPU2 显存约
  `90161/97871 MiB`。当前未见 NaN、OOM、Traceback 或进程退出，
  但 checkpoint-1500 前需重点监控长输出和显存余量。

#### 未完成项

1. 继续训练至 step 3000，并审计 checkpoint-500/1000/1500/2000/2500/3000。
2. 在三数据集各固定 200 题上做 checkpoint sweep，以三数据集宏平均
   F1 选择最佳 checkpoint。
3. 用相同 Evidence Agent + BGE/FAISS Top-3 流程运行 HotpotQA 7405、
   2Wiki 12576、MuSiQue 2417 全量 dev。
4. 与 E01 SFT+DPO 和 ReasonRAG 对比；HotpotQA 使用同 ID 配对
   bootstrap。
5. Evidence OPSD 仍需独立 auxiliary batch；完成前保持
   `beta_evidence=0`。

评测入口：

`03_sapr_rag/scripts/eval/eval_action_opsd_3src.sh`
