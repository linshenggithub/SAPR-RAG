# SAPR-RAG ICASSP：GRPO 上的 OPSD 动作项加法诊断

日期：2026-09-23。状态：**全量完成，协议校验通过；不作为目标动作消融**。

> **口径说明**：本诊断保留了 GRPO outcome advantage，比较
> Outcome-only、GRPO+Query OPSD、GRPO+Answer OPSD 与 Full。它回答的是 OPSD
> 对 GRPO 的边际贡献，不是最终要求的 `SFT + Query-only OPSD` /
> `SFT + Answer-only OPSD` 纯 OPSD 动作消融。后者必须关闭 reward 和 GRPO
> advantage，另行训练与评测。

## 1. 实验目标

本诊断在完全匹配的 canonical SFT 起点和 GRPO 训练配置下，测量 Query OPSD 与
Answer OPSD 叠加到 GRPO 后的边际贡献。四组方法为：

| 方法 | Outcome advantage | Query OPSD | Answer OPSD |
|---|---:|---:|---:|
| Outcome-only | ✓ | × | × |
| Query-only | ✓ | ✓ | × |
| Answer-only | ✓ | × | ✓ |
| Full SAPR-RAG | ✓ | ✓ | ✓ |

Query-only 与 Answer-only 均从 E14 canonical SFT `checkpoint-4150` 起步，训练
1000 step。除动作系数外，模型、数据、reward、rollout、Evidence Agent、LoRA、batch、
采样数、长度、学习率、检索器和 checkpoint 选择规则均与 Full E16 一致。

```text
Query-only:  beta_q=0.01, beta_ans=0
Answer-only: beta_q=0,    beta_ans=0.03
Full:        beta_q=0.01, beta_ans=0.03
```

评测统一使用 S5-K3 强制最终回答协议，覆盖 HotpotQA、2WikiMultiHopQA 和 MuSiQue
完整 dev 集。四组均固定比较 checkpoint-1000，不按方法分别挑选最佳 checkpoint。

## 2. 执行与验收

- Query-only：worker4300716，1000/1000 step，09:39 完成；
- Answer-only：worker4307895，1000/1000 step，10:07 完成；
- 两组均保存 checkpoint-250/500/750/1000；
- 新增评测结果：3 个模型 × 22,398 题 = 67,194 条；
- 加上已有 Full 结果，共形成 4 模型 × 3 数据集的 12 个评测单元；
- 所有结果 ID 严格对齐，`answer_rate=100%`；
- 格式失败率、服务失败率和逐行错误数均为 0；
- 10,000 次分层 paired bootstrap 已完成，共 5 组预注册比较；
- `validation.json` 状态为 `pass`。

训练动作信号在正式 run 中确实生效：

- Query-only：
  `teacher_kl_scoped_query=0.1970`，
  `teacher_action_scope_ratio_query=0.5388`；
- Answer-only：
  `teacher_kl_scoped_answer=0.1003`，
  `teacher_action_scope_ratio_answer=0.2538`；
- 两个关闭分支的系数均为 0，GRPO advantage 保持开启。

## 3. 最终质量结果

### 3.1 三数据集宏平均

宏平均是三个数据集的非加权平均。

| 方法 | EM | F1 | Cover-EM |
|---|---:|---:|---:|
| Outcome-only | 0.3873 | 0.4794 | 0.4150 |
| Query-only | 0.3849 | 0.4770 | 0.4127 |
| Answer-only | 0.3875 | 0.4791 | 0.4157 |
| Full SAPR-RAG | 0.3872 | 0.4785 | 0.4152 |

四组宏 F1 最大差距只有 0.0024。Query-only 略低于 Outcome-only；Answer-only、
Full 与 Outcome-only 基本持平。

### 3.2 逐数据集 F1

| 方法 | HotpotQA | 2WikiMultiHopQA | MuSiQue | Macro |
|---|---:|---:|---:|---:|
| Outcome-only | 0.5838 | 0.5625 | 0.2919 | 0.4794 |
| Query-only | 0.5833 | 0.5642 | 0.2834 | 0.4770 |
| Answer-only | 0.5858 | 0.5607 | 0.2909 | 0.4791 |
| Full SAPR-RAG | 0.5842 | 0.5639 | 0.2874 | 0.4785 |

Query-only 在 2Wiki 上比 Outcome-only 高 0.0017，但在 MuSiQue 上低 0.0085；
Answer-only 在 HotpotQA 上高 0.0020，但其他两集略低。没有一个动作分支在三个数据集
上表现出一致增益。

## 4. 配对显著性

下表报告三数据集分层重采样后的宏 F1 差值。

| 比较 | Delta F1 | 95% CI | 双侧 p 值 |
|---|---:|---:|---:|
| Query-only - Outcome-only | -0.0024 | [-0.0063, 0.0015] | 0.2270 |
| Answer-only - Outcome-only | -0.0002 | [-0.0039, 0.0036] | 0.9305 |
| Full - Outcome-only | -0.0009 | [-0.0044, 0.0027] | 0.6247 |
| Full - Query-only | +0.0015 | [-0.0025, 0.0055] | 0.4484 |
| Full - Answer-only | -0.0007 | [-0.0043, 0.0030] | 0.7111 |

EM 与 Cover-EM 的五组比较同样全部不显著。现有样本规模下，没有证据表明 Query
OPSD、Answer OPSD 或二者联合能在 GRPO 之上提高端到端答案质量。

## 5. 行为与成本

| 方法 | Avg Searches | Avg RPC | Forced Rate | Repeat Query |
|---|---:|---:|---:|---:|
| Outcome-only | 2.599 | 2.398 | 7.46% | 7.65% |
| Query-only | 2.571 | 2.367 | 6.79% | 7.85% |
| Answer-only | 2.552 | 2.360 | 6.86% | 7.45% |
| Full SAPR-RAG | 2.566 | 2.371 | 6.80% | 7.53% |

三个带 OPSD 的模型都比 Outcome-only 少约 0.03–0.05 次平均逻辑检索，同时强制回答率
低约 0.6 个百分点。这是描述性的行为变化，但尚未对行为指标做独立配对显著性检验，
且没有转化为答案质量提升，因此不能把它表述成已验证的质量—成本 Pareto 改进。

## 6. 结论与论文表述

本诊断给出一个清晰的负结果：

1. Query OPSD 没有在 Outcome-only GRPO 之上带来显著质量增益；
2. Answer OPSD 没有在 Outcome-only GRPO 之上带来显著质量增益；
3. Full SAPR-RAG 与强 Outcome-only 基线统计上相当；
4. 两个动作项会轻微改变检索行为，但其独立价值尚未体现在 EM/F1/Cover-EM。

因此论文不能写“AS-OPSD 显著增强 GRPO”。准确表述应为：

> 完整目标相对 SFT 和纯 OPSD 有提升，但在严格匹配的训练与统一推理协议下，与强
> Outcome-only GRPO 基线相当。Query/Answer 动作级蒸馏改变了少量检索行为，尚未带来
> 可验证的端到端质量收益。

这也说明后续研究重点不应继续放在扩大默认动作 KL 系数或选择性汇报单一数据集，而应
转向更直接的动作级信用分配、同状态 Search/Answer 效用监督，或验证 teacher 信号是否
与 outcome advantage 提供了高度冗余的信息。该结论不能替代关闭 GRPO 后的纯 OPSD
Query-only / Answer-only 消融。

## 7. 产物与复现

正式结果：

```text
data/eval_results/icassp_action_ablation_ckpt1000_20260923/
├── outcome_only/
├── query_only/
├── answer_only/
├── validation.json
├── summary.json
├── summary.csv
└── paired_bootstrap/
    └── index.json
```

训练入口：

```text
03_sapr_rag/scripts/grpo/run_canonical_sft_query_opsd_s1000.sh
03_sapr_rag/scripts/grpo/run_canonical_sft_answer_opsd_s1000.sh
```

汇总入口：

```text
03_sapr_rag/scripts/eval/summarize_icassp_action_ablation.py
```

训练产物：

```text
03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/
├── query_opsd_canonical_sft_q001_a000_3src_s1000_20260923/
└── answer_opsd_canonical_sft_q000_a003_3src_s1000_20260923/
```

执行期间 worker4300716 的默认 NVML 动态库软链接指向 0 字节文件。Query-only 进程仅在
自身环境中预加载同机有效的 `libnvidia-ml.so.535.261.03`，没有修改系统软链接；该修复
只影响 vLLM 平台识别，不改变训练或评测算法。
