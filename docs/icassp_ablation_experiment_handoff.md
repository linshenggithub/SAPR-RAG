# SAPR-RAG ICASSP 消融实验补做清单与执行交接

日期：2026-09-24。状态：**推理预算消融与纯 OPSD Query/Answer 消融均已完成**。
已完成的 GRPO-additive 诊断见
[`icassp_ablation_experiment_report.md`](icassp_ablation_experiment_report.md)。
本清单要求的纯 OPSD 动作消融最终结果见
[`icassp_pure_opsd_action_ablation_report.md`](icassp_pure_opsd_action_ablation_report.md)。

完成摘要：两组匹配训练均完成 1000 step；四模型在三个 full dev 上统一重评，
共 89,592 条结果；协议与 ID 校验通过，并完成六组 10,000 次配对 bootstrap。

目标读者：能够访问 SAPR-RAG、配套 ms-swift checkout、模型、数据、检索服务和 GPU 的执行 AI。
本文的目的，是在投稿截止前用最少但可归因的实验验证当前完整目标：

\[
A_{i,t}=\hat A_i+\beta_qm_{i,t}^{q}d_{i,t}^{q}
+\beta_{ans}m_{i,t}^{ans}d_{i,t}^{ans}.
\]

完整方法在论文中可统一称为 **SAPR-RAG**；动作级训练机制暂称
**Action-Scoped On-Policy Self-Distillation（AS-OPSD）**。名称不能改变实验事实：
`\hat A_i` 是 outcome-level GRPO advantage，另外两项分别是 Query 和 Answer
动作范围内的 OPSD 信号。

本轮执行基线：`a636dabf7e98870e1bde8e60424a8757c8979220`（`main`）。
执行过程中保留并隔离了其他实验的本地改动。

## 0. 先给结论：缺什么、不缺什么

### 0.0 2026-09-23 非目标诊断记录

- 已完成的 Query-only 与 Answer-only 均**保留 GRPO outcome advantage**，属于
  “GRPO 上叠加单个 OPSD 动作项”的加法诊断，不是本文要求的纯 OPSD 动作消融；
- checkpoint-250/500/750/1000 全部保存；
- Outcome-only、Query-only、Answer-only 使用统一 S5-K3 强制回答协议重新完成三数据集
  全量评测，共 67,194 条新增结果；
- 加上已有 Full 结果，四组均为 22,398 题且 question ID 严格对齐；
- 回答率 100%，格式失败、服务失败和逐行错误均为 0；
- 10,000 次分层 paired bootstrap 的五组比较全部不显著；
- 宏 F1：Outcome-only 0.4794、Query-only 0.4770、Answer-only 0.4791、
  Full 0.4785；
- 结论：在 GRPO 基础上叠加 Query/Answer OPSD 会轻微改变检索行为，但没有在强
  Outcome-only 基线上带来可验证的端到端质量收益；
- 以上结果只能作为 GRPO-additive 诊断，不能冒充
  `SFT + Query-only OPSD` / `SFT + Answer-only OPSD` 消融。

### 0.1 已完成新增训练的两组

已补齐两个与 OPSD-only 基线严格匹配的动作分支消融：

1. **SFT + Query-only OPSD**：关闭 GRPO，只开启 Query OPSD。
2. **SFT + Answer-only OPSD**：关闭 GRPO，只开启 Answer OPSD。

旧 E09/E10 的 Answer-only OPSD 从 SFT+DPO 起点训练，数据、步数与当前 canonical
SFT 主线不匹配，**不能**冒充这里的 Answer-only 对照。

### 0.2 已复用的训练

以下四组已有可信的 1000-step 训练和三数据集评测记录。先核实 checkpoint、args、
metrics 和评测目录存在；一致时直接汇总，不重训：

| 论文显示名 | 实验记录 | 三个优势项 | 现有入口 |
|---|---|---|---|
| SFT | E14 | 全关 | `launch_sft_canonical_ckpt4150_6gpu_eval.sh` |
| OPSD-only | D | 仅 Query+Answer OPSD | `run_canonical_sft_pure_opsd_s1000.sh` |
| Outcome-only / w/o AS-OPSD | B | 仅 `\hat A_i` | `run_canonical_sft_grpo_control_s1000.sh` |
| SAPR-RAG | C=E16 | 三项全开 | `run_canonical_sft_multi_opsd_s1000.sh` |

权威数值与产物位置见 `docs/experiment_tracker.md`。截至核对版本，EM/F1/Cover-EM：

| 方法 | HotpotQA | 2WikiMultiHopQA | MuSiQue |
|---|---|---|---|
| E14 SFT | .4373/.5513/.4748 | .4051/.4513/.4188 | .1651/.2405/.1841 |
| B Outcome-only | .4629/.5837/.5026 | .5161/.5654/.5314 | .1808/.2794/.2056 |
| D OPSD-only | .4462/.5703/.5030 | .4948/.5548/.5270 | .1758/.2717/.2085 |
| C=E16 SAPR-RAG | .4636/.5816/.5025 | .5154/.5659/.5307 | .1837/.2786/.2089 |

这些结果显示 `C-B` 很小且不一致。不得提前声称 OPSD 显著增强 GRPO，也不得隐藏
Outcome-only。新增 Query/Answer 消融的任务是解释信号作用，不是保证制造正结果。

### 0.3 后续可选工作

纯 OPSD P0 训练、全量评测、行为统计与 bootstrap 已完成。仍可按论文篇幅和资源预算
补充 checkpoint 收敛曲线；推理深度和 Top-k 的全量敏感性已由独立推理预算实验覆盖。

训练数据量、更多随机种子、大范围 beta sweep 都是低优先级，不得抢占前述实验资源。

## 1. 论文必须形成的两张核心消融表

### 1.1 优化信号总消融

| Method | Outcome advantage | Query OPSD | Answer OPSD |
|---|---:|---:|---:|
| SFT | × | × | × |
| OPSD-only | × | ✓ | ✓ |
| Outcome-only / w/o AS-OPSD | ✓ | × | × |
| SAPR-RAG | ✓ | ✓ | ✓ |

这一表回答 outcome optimization、纯 self-distillation 及联合目标的关系。前三组不是
可以删掉的普通 baseline，而是完整目标的归因对照。

### 1.2 Query／Answer 动作消融

| Method | Outcome advantage | Query OPSD | Answer OPSD |
|---|---:|---:|---:|
| SFT | × | × | × |
| SFT + Query-only OPSD | × | ✓ | × |
| SFT + Answer-only OPSD | × | × | ✓ |
| SFT + OPSD-only | × | ✓ | ✓ |

这一表在不混入 outcome advantage 的前提下，直接验证两个 OPSD 动作项，是当前
最高优先级的新增训练。

## 2. P0：Query-only 与 Answer-only 匹配训练（已完成）

### 2.1 唯一允许改变的配置

两组都从 E14 canonical SFT `checkpoint-4150` 起步，使用与 D（OPSD-only）相同的
三源 `hotpotqa_2wiki_musique_train_multi_opsd.jsonl`、rollout、Evidence Agent、
模型、LoRA、batch、采样数、长度、学习率、reference、检索器和 1000 steps。

共同开关：

```text
ENABLE_OPSD=true
ENABLE_REWARD=false
OPD_USE_GRPO_ADVANTAGE=false
TEACHER_ACTION_SCOPE=multi
TEACHER_EVIDENCE_KL_COEF=0.0
ENABLE_TRUNCATION_REWARD=false
ACTION_CREDIT_MODE=off
ADVANTAGE_MODE=sequence
```

唯一差异：

| 组 | `TEACHER_QUERY_KL_COEF` | `TEACHER_ANSWER_KL_COEF` |
|---|---:|---:|
| Query-only | `0.01` | `0.0` |
| Answer-only | `0.0` | `0.03` |
| D OPSD-only（已有） | `0.01` | `0.03` |

不要使用 E09/E10 的旧 Answer-only wrapper，不要保留 GRPO reward/advantage，也不要
同时开启 E17–E23 的动作信用、return-to-go、动态采样或长度归一化。

### 2.2 已新增入口

从 `run_canonical_sft_pure_opsd_s1000.sh` 复制并逐项审计，建议新增：

```text
03_sapr_rag/scripts/grpo/run_canonical_sft_pure_query_opsd_s1000.sh
03_sapr_rag/scripts/grpo/run_canonical_sft_pure_answer_opsd_s1000.sh
```

脚本名符合仓库规范。除了两个动作系数、`RUN_NAME`、端口和经用户确认的 GPU 分配，
与 D 不应有其他实验差异。复制后用机器可读方式比较展开配置，并把 diff 写入
`experiment_note.md`。

建议 run name：

```text
pure_query_opsd_canonical_sft_q001_a000_3src_s1000_YYYYMMDD
pure_answer_opsd_canonical_sft_q000_a003_3src_s1000_YYYYMMDD
```

不得照抄历史固定的 GPU2–7 或端口。执行前查看 GPU/进程/端口归属并取得用户对资源、
GPU 小时和并行方式的确认。不要停止别人的服务，不要为了赶时间静默减少模型、样本、
steps、生成数或最大长度。

### 2.3 训练前强制检查

执行 AI 首次反馈必须给出：

- SAPR-RAG 与 ms-swift SHA/dirty diff、实际 `swift.__file__`；
- `docs/ms_swift_local_patches.md` 所列补丁是否一致；
- E14 adapter 和两份训练 JSONL 是否存在、大小及哈希；
- 检索 `/health` 的语料/索引/BGE/FAISS 字段；
- 可用 GPU、显存、端口、磁盘和预计 wall time；
- 展开后的 Query-only／Answer-only／D 配置差异。

已有完整 ms-swift 补丁基于
`1dbd1bf64a46bd6bb710d9ace05d529ff071cd1f`；按迁移手册核验，不在有本地改动的
依赖 checkout 上盲目重复应用。

先完成 CPU/unit tests、launcher `DRY_RUN=true` 和 1–2 optimizer-step smoke。确认：

- LoRA 真正加载到 rollout，且训练更新会同步；
- Query-only 的 Answer teacher coverage/advantage 为零；
- Answer-only 的 Query teacher coverage/advantage 为零；
- 非零分支的 mask、token 对齐、teacher log-ratio 和学生梯度有限；
- observation、padding 不接收 teacher 信号；
- 输出目录是唯一新目录，不覆盖旧 checkpoint。

任何一项失败都停止，不用 loss 下降代替实现正确性。

### 2.4 checkpoint 与停止规则

保存 250/500/750/1000，但核心消融固定比较 **checkpoint-1000**，与 SFT/D 一致。
不得为每个方法分别挑最好 checkpoint 后放进同一消融表。

若因截止时间只跑到 250，应明确标为 pilot，并与 D 的 250-step 匹配 checkpoint
比较，不能把 250-step 新方法与 1000-step 完整结果并表。资源不足时报告并请求用户
选择，不自动改变方案。

## 3. P0：统一正式评测与行为统计（已完成）

### 3.1 新模型全量评测（已完成）

Query-only 和 Answer-only 的 checkpoint-1000 必须在完全相同 pipeline 上评测：

```text
HotpotQA dev
2WikiMultiHopQA dev
MuSiQue dev
top_k=3
max_turns=6（当前语义下至多约 5 次检索）
Evidence Agent=true
evidence_max_tokens=128
VLLM_MAX_MODEL_LEN=8192
```

可复用 `03_sapr_rag/scripts/eval/eval_action_opsd_3src.sh`，但先核对 checkpoint root、
adapter、数据路径和当前权威评测口径。不要用选择集挑 checkpoint；本实验固定 1000。

至少输出：EM、F1、Cover-EM、回答率、平均决策轮数、平均逻辑检索数、重复 Query
率、max-turn rate、格式失败率、服务失败率。对每个问题保存 id，使用
`paired_bootstrap.py` 与 SFT 及 OPSD-only 做配对差值和 95% CI；若脚本只默认比较
SFT+DPO，先参数化 reference 文件，不能把错误 reference 的输出放入论文。

### 3.2 现有四组是否需要重推理

先比较其结果 manifest：评测数据哈希、retriever/corpus/index、top-k、max-turns、
Evidence Agent、模型长度、answer extractor、score.py SHA 和请求失败处理。全部一致就
复用现有结果；任一关键项不同，统一重推理，不重训。

### 3.3 行为指标可以优先从已有结果重算

如果 `results.jsonl`／`trajectories.jsonl` 已保存 query、turn 和 termination 信息，直接
重算以下指标，不浪费 GPU：

```text
avg_searches = 每题实际逻辑检索动作数的均值
repeat_query_rate = 规范化后重复 query 数 / query 总数
max_turn_rate = 达到评测最大决策轮数的问题比例
answer_rate = 成功抽取 <answer> 的问题比例
format_failure_rate = 无合法 query/answer 或协议冲突比例
```

同时报告分母、数据集和失败样本处理。不能把服务错误删除后只对成功样本算均值。

## 4. P1：最大检索深度敏感性（仅推理）

### 4.1 设置

论文横轴应写“Maximum Retrieval Steps”，建议：

```text
max_searches ∈ {1, 3, 5}
```

当前 `run_direct_rollout_eval.py` 的参数是 decision turns，scheduler 在最后 turn 不再执行
检索，因此需要映射并通过一条轨迹验证：

```text
max_searches=1 -> max_turns=2
max_searches=3 -> max_turns=4
max_searches=5 -> max_turns=6
```

不要不加检查地把 `--max_turns 1/3/5` 直接称为 1/3/5 次检索。

### 4.2 比较对象

至少评 SAPR-RAG Full；更有归因力的主图同时评 Outcome-only，形成两条曲线。所有点
固定 top-k=3、checkpoint-1000、同一问题集合和 decoding。报告 EM/F1、平均实际
检索数、回答率和 max-turn rate。

如果完整 dev 的 2 models × 3 settings × 3 datasets 超出剩余预算，先给用户准确成本
估计并冻结一个按 id/seed 分层的固定子集；不得执行中途看结果再调整样本。子集结果
只能标为诊断/消融，不冒充主表全量。

## 5. P1：Top-k 敏感性（仅推理）

### 5.1 设置

复刻 ReasonRAG 的常见设置：

```text
top_k ∈ {1, 3, 5}
```

固定 `max_turns=6`、checkpoint、数据、检索索引、Evidence Agent 和 decoding。至少评
Full；资源允许再加入 Outcome-only。报告 EM/F1/Cover-EM、平均检索数、reasoner 与
Evidence Agent token/延迟。Top-k 增大可能改变 Evidence Agent 输入与开销，不能只报
质量不报成本。

### 5.2 现有脚本必须先参数化

核对版本的 `eval_action_opsd_3src.sh` 在启动 rollout 时硬编码：

```text
SAPR_TOP_K=3
```

并在评测时硬编码：

```text
--max_turns 6
```

因此单纯从外层设置 `SAPR_TOP_K` 或 `MAX_TURNS` **不会生效**。修改脚本时新增默认值
保持历史行为，例如：

```text
TOP_K=${TOP_K:-3}
MAX_TURNS=${MAX_TURNS:-6}
```

再分别传入 rollout 环境和 `run_direct_rollout_eval.py`。增加正整数校验，将两值写入
config、输出目录和 metrics manifest。先用一条样本验证 `/health`/日志、返回文档数和
最大实际检索数；否则不能启动批量评测。

## 6. P2：训练收敛曲线（优先复用）

目标图：三数据集 macro F1（或分别画 F1）随 optimizer step 变化。建议使用固定：

```text
checkpoint ∈ {250, 500, 750, 1000}
```

至少画 SFT、Query-only、Answer-only、OPSD-only；GRPO 相关组放在独立总目标表。
先读取现有 eval 目录和 metrics，不重复推理已有 checkpoint。缺失的 checkpoint 在同一
固定 validation 子集上补评，不允许每条方法用不同题集。横轴可以是 steps；只有可靠
记录了 GPU hours 才画 GPU hours，不能从别的机器估算后当实测。

这张图是训练效率/稳定性分析，不替代最终 checkpoint 的组件消融。

## 7. P3：仅在 P0/P1 完成后的可选实验

### 7.1 beta 敏感性

若仍有训练预算，只做一个预注册的小范围缩放，不做网格搜索。建议固定比例
`beta_q:beta_ans=1:3`，测试共同 scale：

```text
0.5×: 0.005 / 0.015
1.0×: 0.010 / 0.030（已有 Full）
2.0×: 0.020 / 0.060
```

三组必须同 steps；短 pilot 与 Full-1000 不能混表。若没有预算，坦诚写默认系数沿用
开发集选择或既有设置，不伪造敏感性。

### 7.2 低优先级，不在截止前默认执行

- 25%/50%/100% 训练数据量；
- 第二/第三随机种子；
- Top-k 与 max-turn 的笛卡尔积；
- 新 reward、Evidence teacher、长度惩罚或更多动作项；
- 重新运行已明确失败的 E17–E23。

这些都需要用户单独确认资源，不属于本文自动授权范围。

## 8. 执行顺序、算力门槛与并发

严格按以下顺序：

1. 只读核验已有 E14/D 产物和协议；B/C 只用于另一张总目标表。
2. 实现两个新 wrapper 和评测脚本参数化，完成测试/dry-run。
3. 用户确认 GPU 预算后，跑 Query-only／Answer-only smoke。
4. 完成两组 1000-step 匹配训练；资源不允许则停止并报告。
5. 固定 checkpoint-1000，统一三数据集全量评测和行为统计。
6. 补 max-searches 与 top-k 推理敏感性。
7. 汇总已有 checkpoint 曲线。
8. 只有前述交付完整且用户批准，才做 beta sensitivity。

不默认跨机器分布式训练。并行运行前检查检索服务吞吐、rollout GPU、train GPU、端口
和磁盘，避免两个 run 共用同一输出目录或通信端口。检索服务已有健康实例时复用，不能
擅自 stop/restart；服务配置不一致则先报告。

## 9. 结果目录与交付物

每个 run 使用唯一目录，并按仓库约定至少保存：

```text
run_config.yaml
raw_output.jsonl
trajectories.jsonl
metrics.json
badcases.jsonl
experiment_note.md
```

额外保存：代码 SHA、ms-swift SHA/patch SHA256、数据/评测 id 哈希、检索 health、
adapter/checkpoint、展开配置、启动命令、GPU hours、wall time、峰值显存、失败与重试。

最终报告：

```text
docs/icassp_pure_opsd_action_ablation_report.md
```

报告必须区分：

- `real_result`：完整、可审核、协议一致的结果；
- `debug_result`：smoke 或固定小子集；
- `expected_result`：尚未运行的计划；
- `failed_run`：代码、服务、资源或方法失败。

不要提交 checkpoint、原始大 JSONL、私有日志、绝对个人路径、密钥或内部服务信息。
只提交允许公开的代码、配置、小型汇总、统计结果和迁移说明。

## 10. 验收标准与论文表述边界

一次实验只有同时满足以下条件才可进入论文：

- 与声明的对照仅有预定变量差异；
- 全量或预注册固定子集，样本数/失败数透明；
- checkpoint 选择规则一致；
- 推理协议、数据、检索器和评分代码有哈希或 SHA；
- EM/F1 之外包含检索行为与成本；
- 关键比较提供 paired bootstrap 95% CI；
- 负结果、服务失败和不显著差异没有被隐藏。

纯 OPSD 动作表只回答 Query 与 Answer teacher 各自的作用，不能混入 GRPO advantage。
现有 GRPO-additive 诊断应单独报告，不能替代该表。

## 11. 最终执行结果

- Query-only 和 Answer-only 均完成 1000-step 匹配训练，保存
  checkpoint-250/500/750/1000；
- Query-only 仅记录 Query teacher 信号，Answer-only 仅记录 Answer teacher 信号，
  两组 reward 与 reward std 全程为 0；
- SFT、Query-only、Answer-only、OPSD-only 在统一 S5-K3 强制回答协议下完成
  HotpotQA、2WikiMultiHopQA、MuSiQue 全量评测；
- 89,592 条结果无逐行错误、格式错误、服务错误或超预算行为，question ID 严格对齐；
- 六组比较均完成 10,000 次分层配对 bootstrap；
- 结论与完整数值已写入
  `docs/icassp_pure_opsd_action_ablation_report.md`。
