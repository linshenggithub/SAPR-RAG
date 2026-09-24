# ICASSP 纯 OPSD Query／Answer 动作消融报告

日期：2026-09-23 至 2026-09-24
状态：**两组 1000-step 训练、四模型三数据集全量评测和 10,000 次配对 bootstrap 均已完成**

## 1. 研究问题与结论

本实验回答一个严格归因问题：在不混入 GRPO outcome advantage 的前提下，
Query OPSD 与 Answer OPSD 分别对端到端效果和检索行为贡献什么。

核心结论：

1. **Query-only OPSD 不提升端到端答案质量。** 相对 SFT，三数据集宏平均
   EM/F1 分别下降 1.79/0.90 个百分点，均达到统计显著；Cover-EM 提高
   2.41 个百分点，但伴随更多检索、更多重复 Query 和更高强制回答率。
2. **Answer-only OPSD 保持答案质量并稳定提高覆盖。** 相对 SFT，宏平均
   EM 下降 0.50 个百分点、F1 提高 0.22 个百分点，二者均不显著；
   Cover-EM 提高 1.47 个百分点且显著。
3. **双分支 OPSD 没有显著优于 Answer-only。** 双分支相对 Answer-only 的
   宏 EM/F1/Cover-EM 差值均不显著，说明当前纯 OPSD 的主要有效成分来自
   Answer 分支，Query 分支没有提供稳定互补收益。
4. 结合此前 Outcome-only 与 SAPR-RAG 的匹配对照，当前端到端 EM/F1 增益的
   主要来源仍是 GRPO outcome reward；动作级 OPSD 更适合表述为改变证据覆盖和
   终止行为的辅助信号，而不是独立提升答案准确率的主因。

## 2. 严格控制变量

四组均从 canonical-answer SFT `checkpoint-4150` 出发；新增的两组与已有
双分支纯 OPSD 使用同一训练数据、rollout、Evidence Agent、LoRA、batch、
学习率、长度限制、采样数和 1000 个 optimizer step。只改变动作分支系数：

| 方法 | GRPO reward | GRPO advantage | Query OPSD | Answer OPSD |
|---|---:|---:|---:|---:|
| SFT | × | × | × | × |
| SFT + Query-only OPSD | × | × | 0.01 | × |
| SFT + Answer-only OPSD | × | × | × | 0.03 |
| SFT + OPSD-only | × | × | 0.01 | 0.03 |

正式评测统一采用：

- HotpotQA dev 7,405 条、2WikiMultiHopQA dev 12,576 条、MuSiQue dev
  2,417 条；
- `top_k=3`、`max_searches=5`、调度轮数 `max_turns=6`；
- Evidence Agent 开启；
- 检索预算耗尽后使用 `answer_only_system_prefill_v2` 强制回答；
- 固定比较 Query/Answer/双分支的 `checkpoint-1000`，不按数据集挑选
  checkpoint。

## 3. 训练验收

| 项目 | Query-only | Answer-only |
|---|---:|---:|
| optimizer steps | 1,000 | 1,000 |
| wall time | 6h 30m | 5h 19m |
| 平均 train loss | 0.01721 | 0.02936 |
| grad norm 范围 | [0.01465, 0.20860] | [0.01343, 0.64532] |
| 非零分支 scoped KL 均值 | Query 0.10498 | Answer 0.24234 |
| 非零分支 scope ratio 均值 | Query 0.69896 | Answer 0.38071 |
| 另一动作分支记录数 | Answer 0 | Query 0 |
| reward / reward std | 恒为 0 / 恒为 0 | 恒为 0 / 恒为 0 |
| 保存 checkpoint | 250/500/750/1000 | 250/500/750/1000 |

两组均无 NaN、OOM 或训练中断。另一动作分支的统计记录严格为 0，且 task
reward 与 reward std 全程为 0，证明本实验没有误混入 GRPO reward 或另一动作
teacher 信号。

## 4. 分数据集主结果

以下每张表只展示一个数据集，数值来自同一强制回答协议下的全量评测。

### 4.1 HotpotQA

| 方法 | N | EM | F1 | Cover-EM |
|---|---:|---:|---:|---:|
| SFT | 7,405 | **0.4527** | **0.5747** | 0.4906 |
| SFT + Query-only OPSD | 7,405 | 0.4375 | 0.5681 | **0.5149** |
| SFT + Answer-only OPSD | 7,405 | 0.4427 | 0.5712 | 0.5051 |
| SFT + OPSD-only | 7,405 | 0.4455 | 0.5721 | 0.5036 |

相对 SFT 的 10,000 次配对 bootstrap：Query-only 的 EM 显著下降
1.51pt（95% CI [-2.40, -0.62]，双侧 `p=0.0006`），F1 下降 0.66pt
但不显著；Answer-only 的 EM 下降 1.00pt（`p=0.0170`），F1 差异不显著。
三种 OPSD 设置的 Cover-EM 均显著高于 SFT。

### 4.2 2WikiMultiHopQA

| 方法 | N | EM | F1 | Cover-EM |
|---|---:|---:|---:|---:|
| SFT | 12,576 | **0.5061** | 0.5576 | 0.5200 |
| SFT + Query-only OPSD | 12,576 | 0.4649 | 0.5405 | **0.5418** |
| SFT + Answer-only OPSD | 12,576 | 0.4929 | 0.5561 | 0.5293 |
| SFT + OPSD-only | 12,576 | 0.4986 | **0.5611** | 0.5344 |

Query-only 相对 SFT 的 EM/F1 分别显著下降 4.12/1.71pt
（均为 `p=0.0002`），Cover-EM 显著提高 2.18pt。Answer-only 的 EM
下降 1.32pt（`p=0.0008`），F1 持平，Cover-EM 提高 0.93pt
（`p=0.0144`）。双分支相对 Answer-only 在本数据集提高约 0.5pt，
但该局部收益没有扩展成跨数据集宏平均优势。

### 4.3 MuSiQue

| 方法 | N | EM | F1 | Cover-EM |
|---|---:|---:|---:|---:|
| SFT | 2,417 | 0.1659 | 0.2701 | 0.1891 |
| SFT + Query-only OPSD | 2,417 | 0.1684 | 0.2669 | **0.2151** |
| SFT + Answer-only OPSD | 2,417 | **0.1742** | **0.2816** | 0.2094 |
| SFT + OPSD-only | 2,417 | 0.1676 | 0.2762 | 0.2069 |

Answer-only 相对 SFT 的 F1 提高 1.15pt，但双侧 `p=0.0632`，尚未达到
0.05 显著性阈值。三种 OPSD 设置的 Cover-EM 均显著提高；EM 差异均不显著。

## 5. 宏平均与统计检验

宏平均是三个数据集的等权平均，不按样本数加权。

| 方法 | EM | F1 | Cover-EM |
|---|---:|---:|---:|
| SFT | **0.3749** | 0.4675 | 0.3999 |
| SFT + Query-only OPSD | 0.3569 | 0.4585 | **0.4239** |
| SFT + Answer-only OPSD | 0.3699 | 0.4696 | 0.4146 |
| SFT + OPSD-only | 0.3706 | **0.4698** | 0.4150 |

10,000 次按数据集分层、逐题配对 bootstrap：

| 候选 − 对照 | 指标 | 差值 | 95% CI | 双侧 p |
|---|---|---:|---:|---:|
| Query-only − SFT | EM | -1.79pt | [-2.39, -1.20]pt | 0.0002 |
| Query-only − SFT | F1 | -0.90pt | [-1.46, -0.33]pt | 0.0024 |
| Query-only − SFT | Cover-EM | +2.41pt | [+1.80, +3.02]pt | 0.0002 |
| Answer-only − SFT | EM | -0.50pt | [-1.04, +0.04]pt | 0.0762 |
| Answer-only − SFT | F1 | +0.22pt | [-0.31, +0.75]pt | 0.4134 |
| Answer-only − SFT | Cover-EM | +1.47pt | [+0.89, +2.05]pt | 0.0002 |
| OPSD-only − SFT | EM | -0.43pt | [-0.97, +0.11]pt | 0.1146 |
| OPSD-only − SFT | F1 | +0.24pt | [-0.28, +0.76]pt | 0.3886 |
| OPSD-only − SFT | Cover-EM | +1.50pt | [+0.94, +2.06]pt | 0.0002 |
| OPSD-only − Query-only | EM | +1.36pt | [+0.89, +1.86]pt | 0.0002 |
| OPSD-only − Query-only | F1 | +1.13pt | [+0.66, +1.61]pt | 0.0002 |
| OPSD-only − Query-only | Cover-EM | -0.90pt | [-1.42, -0.39]pt | 0.0012 |
| OPSD-only − Answer-only | EM | +0.06pt | [-0.26, +0.39]pt | 0.6965 |
| OPSD-only − Answer-only | F1 | +0.02pt | [-0.31, +0.33]pt | 0.9237 |
| OPSD-only − Answer-only | Cover-EM | +0.03pt | [-0.32, +0.39]pt | 0.8335 |

## 6. 行为指标

所有组回答率均为 100%，强制回答有效率均为 100%，格式失败率、服务失败率和
超检索预算样本数均为 0。

| 方法 | 平均逻辑检索数 | 平均检索 RPC | 强制回答率 | 重复 Query 率 | 空 evidence 率 |
|---|---:|---:|---:|---:|---:|
| SFT | 2.661 | 2.354 | 11.21% | 11.39% | 22.74% |
| Query-only | **2.733** | **2.377** | **14.88%** | **12.84%** | **24.41%** |
| Answer-only | 2.426 | 2.185 | 7.19% | 9.87% | 18.73% |
| OPSD-only | 2.456 | 2.205 | 7.60% | 10.14% | 19.30% |

Query-only 的 Cover-EM 增益来自更积极的搜索，但更高的重复率、空 evidence
率和预算耗尽率表明其检索效率下降；这与 EM/F1 的显著回落一致。Answer-only
则使模型更早形成终止答案，同时降低无效检索，因而在保持 F1 的同时提高
Cover-EM。双分支的行为和质量均接近 Answer-only，进一步说明 Answer 分支主导
当前纯 OPSD 的实际作用。

## 7. 表述边界

本实验支持以下论文表述：

> 在严格关闭 outcome reward 的动作消融中，Answer-scoped OPSD 显著提高
> evidence coverage，同时保持宏平均 F1；Query-scoped OPSD 虽提高覆盖率，
> 但以更多低效检索和显著下降的 EM/F1 为代价。将两者联合并未显著优于
> Answer-only，说明当前 Query teacher 尚未形成稳定互补贡献。

本实验不支持“Query OPSD 提升端到端答案准确率”或“Query+Answer OPSD 显著
优于 Answer-only”的说法。旧 E14/D 数值没有强制回答，不能与本报告的统一
强制回答数值直接混表；需要比较方法时应整体采用本报告重跑后的四组结果。

## 8. 复现与权威产物

- 训练数据 SHA256：
  `e8242e2dde241a7285aaf9166c3ac01869d53df35482727be8020f809db6e898`
- 训练时 `plugin.py` SHA256：
  `49184a8b8b3a46de125a741526c0fbfa494d08889cf7979ac922cabe7578314d`
- SAPR-RAG 起始代码：`a636dabf7e98870e1bde8e60424a8757c8979220`
- ms-swift：`1dbd1bf64a46bd6bb710d9ace05d529ff071cd1f`
- Query-only 入口：
  `03_sapr_rag/scripts/grpo/run_canonical_sft_pure_query_opsd_s1000.sh`
- Answer-only 入口：
  `03_sapr_rag/scripts/grpo/run_canonical_sft_pure_answer_opsd_s1000.sh`
- 汇总与统计：
  `03_sapr_rag/scripts/eval/summarize_icassp_pure_opsd_ablation.py`
- 评测与统计根目录：
  `data/eval_results/icassp_pure_opsd_action_ablation_20260924/`
- 机器可读验收：`validation.json`（`status=pass`）
- 汇总：`summary.csv`、`summary.json`
- 配对检验：`paired_bootstrap/`（6 组比较，每组 10,000 次）

完整评测共 `4 × 22,398 = 89,592` 条结果。四模型在每个数据集上的 question
ID 集合严格一致。
