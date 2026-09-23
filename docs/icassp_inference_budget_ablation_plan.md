# SAPR-RAG ICASSP 推理预算消融实验计划

日期：2026-09-22。状态：**已完成并通过全量验收**。正式结果与统计解释见
[`icassp_inference_budget_ablation_report.md`](icassp_inference_budget_ablation_report.md)。

本计划只评测已经训练完成的 SAPR-RAG Full（C/E16）`checkpoint-1000`，不新增训练，
不比较其他 checkpoint，不改变模型参数。实验包含两类推理敏感性消融：

1. 最大检索次数：`max_searches ∈ {0,1,2,3,4,5,6,7}`；
2. 单次检索文档数：`top_k ∈ {1,3,5}`。

目标是回答：SAPR-RAG 的答案质量与推理成本如何随检索深度和检索宽度变化，以及默认
配置是否位于合理的质量—成本折中点。

## 1. 固定模型与唯一自变量

固定 checkpoint：

```text
03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/
opsd_canonical_sft_q001_a003_3src_s1000_20260905/
v0-20260905-151754/checkpoint-1000
```

该 checkpoint 是从 E14 canonical SFT `checkpoint-4150` 出发，联合启用：

```text
GRPO outcome advantage
+ Query OPSD  beta_q   = 0.01
+ Answer OPSD beta_ans = 0.03
```

固定不变：

- 模型、LoRA adapter、tokenizer；
- HotpotQA、2WikiMultiHopQA、MuSiQue 完整 dev 集及题目顺序；
- BGE 模型、FAISS 索引、语料版本与正文截断；
- Evidence Agent 开启，`evidence_max_tokens=128`；
- `VLLM_MAX_MODEL_LEN=8192`；
- reasoner `temperature=0.0`、`top_p=1.0`；
- answer extractor、`score.py` 和失败处理；
- 最终回答协议；
- batch size、服务重试与超时设置。

每类实验只允许改变表中指定变量。

## 2. 统一终止协议：检索耗尽后强制回答

旧 SAPR 评测使用 `max_turns=6` 截断。若模型最后仍未输出 `<answer>`，样本直接以
`answer=None` 计零；该语义与 ReasonRAG 的最后一轮强制 `answer_generation` 不一致。
本消融不能复用旧结果，所有配置必须使用以下统一协议重新推理。

### 2.1 `max_searches` 的定义

`max_searches` 表示一题允许的最大**逻辑检索动作数**：

- 模型生成一个合法 `<query>...</query>` 即消耗一次；
- 精确重复 query 也消耗一次，但可不重复发 RPC；
- 技术重试不增加逻辑检索数，单独记录 RPC 与重试次数；
- 非法格式不伪装成检索，也不无限重采样。

实际检索次数可以小于上限，因为模型可提前回答。

### 2.2 推理状态机

```text
question
  -> 模型提前生成合法 <answer>：结束
  -> 模型生成合法 <query> 且 searches < max_searches：
       searches += 1
       检索 -> Evidence Agent -> 累积 evidence -> 下一轮
  -> searches == max_searches：
       禁止继续检索
       追加固定 final-answer instruction
       生成一次最终答案
       结束
```

固定 final-answer instruction：

```text
The retrieval budget is exhausted. Based on the original question and all
evidence accumulated so far, provide the best final answer now. Do not issue
another retrieval query. End with <answer>answer</answer>.
```

强制回答轮：

- 不执行检索；
- 不读取 gold；
- 只使用原问题和截至该时刻 student 已见的历史/evidence；
- 若仍无合法 `<answer>`，记 format failure，EM/F1/Cover-EM 均为 0；
- 保存 `forced_answer=true`，不得把强制回答伪装成模型主动停止。

### 2.3 边界配置

- `max_searches=0`：闭卷强制回答，不执行任何检索；
- `max_searches=1`：最多检索一次，然后强制回答；
- `max_searches=7`：最多七次逻辑检索，然后强制回答。

不得简单将旧 `--max_turns` 设置成 `1…8` 冒充本实验。必须显式维护
`logical_search_count` 并为最终回答保留独立调用。

## 3. 实验矩阵

### 3.1 最大检索次数消融

固定 `top_k=3`：

| 配置名 | `max_searches` | `top_k` |
|---|---:|---:|
| S0-K3 | 0 | 3 |
| S1-K3 | 1 | 3 |
| S2-K3 | 2 | 3 |
| S3-K3 | 3 | 3 |
| S4-K3 | 4 | 3 |
| S5-K3 | 5 | 3 |
| S6-K3 | 6 | 3 |
| S7-K3 | 7 | 3 |

`S0-K3` 中 `top_k` 不会实际使用，仅为配置完整性保留。

### 3.2 Top-k 消融

固定 `max_searches=5`：

| 配置名 | `max_searches` | `top_k` |
|---|---:|---:|
| S5-K1 | 5 | 1 |
| S5-K3 | 5 | 3 |
| S5-K5 | 5 | 5 |

两类实验共享 `S5-K3`，因此实际只有 **10 个唯一配置**，不是 11 个。

### 3.3 评测规模

每个唯一配置评测三个完整 dev 集：

| 数据集 | 样本数 |
|---|---:|
| HotpotQA | 7,405 |
| 2WikiMultiHopQA | 12,576 |
| MuSiQue | 2,417 |
| 合计 | 22,398 |

总计：

```text
10 个唯一配置 × 3 个数据集 = 30 个数据集运行
10 × 22,398 = 223,980 个 question-config 样本
```

如果时间不足，只能在运行前冻结一个按 ID/hash 选择的固定子集，并明确标注
`diagnostic_subset`；不得运行中看结果再调整样本，也不得把子集结果放进论文主表。

## 4. 需要实现的最小代码改动

### 4.1 评测 scheduler

新增默认关闭的 eval-only 模式，建议配置：

```text
SAPR_FORCE_FINAL_ANSWER=true
SAPR_MAX_SEARCHES=<0..7>
```

实现要求：

1. 显式记录逻辑 query 次数，不根据 `num_turns` 猜测；
2. 达到搜索预算后不再调用 retriever；
3. 强制回答只使用当时可见历史，不回流未来信息；
4. 提前回答保持原行为；
5. 重复 query 消耗逻辑预算，`search_executed=false`；
6. 技术重试不消耗新预算；
7. 输出中保存终止原因和强制回答标记。

建议新增行为字段：

```json
{
  "logical_search_count": 3,
  "actual_retrieval_rpc_count": 3,
  "max_searches": 3,
  "top_k": 3,
  "forced_answer": true,
  "forced_answer_valid": true,
  "finish_reason": "forced_answer_after_search_budget"
}
```

### 4.2 参数化现有评测脚本

`03_sapr_rag/scripts/eval/eval_action_opsd_3src.sh` 当前硬编码：

```text
SAPR_TOP_K=3
--max_turns 6
```

需改为显式参数并保持历史默认值：

```bash
TOP_K="${TOP_K:-3}"
MAX_SEARCHES="${MAX_SEARCHES:-5}"
FORCE_FINAL_ANSWER="${FORCE_FINAL_ANSWER:-true}"
```

要求：

- `TOP_K` 只接受正整数；
- `MAX_SEARCHES` 接受整数 `0…7`；
- 输出目录、manifest、日志都包含 `s${MAX_SEARCHES}_k${TOP_K}`；
- 未解析变量、重复目录、已有结果冲突直接报错；
- `DRY_RUN=true` 打印完整展开配置，不启动服务。

### 4.3 评分与行为统计

`run_direct_rollout_eval.py` 不再用 `num_turns >= max_turns` 推断耗尽，应优先读取
scheduler 返回的：

```text
finish_reason
forced_answer
logical_search_count
actual_retrieval_rpc_count
```

强制回答成功的样本必须正常参与 EM/F1；只有强制回答仍无合法答案才记零。

## 5. 强制验证，不能直接跑全量

### 5.1 CPU/unit tests

至少覆盖：

1. `max_searches=0`：检索 client 调用次数严格为 0，执行一次强制回答；
2. `max_searches=1`：最多一个合法 query，然后强制回答；
3. `max_searches=7`：逻辑 query 数不超过 7；
4. 模型提前回答：不触发额外强制回答；
5. 重复 query：逻辑次数增加、RPC 次数不增加；
6. 最后一轮强制回答不得生成或执行 query；
7. 修改强制回答之后的伪造 future observation，不改变其输入；
8. service error 与模型格式错误分开；
9. 默认关闭时旧 scheduler 行为不变；
10. `top_k=1/3/5` 确实控制返回文档数量。

### 5.2 单题 GPU smoke

先对 `S0-K3`、`S1-K3`、`S5-K3`、`S7-K3`、`S5-K1`、`S5-K5` 各跑至少一题，
人工核查：

- checkpoint-1000 LoRA 真实加载；
- search/evidence/forced-answer 时序正确；
- `S0` 没有检索；
- 搜索预算未超限；
- 强制回答输入包含此前证据但不含 gold；
- Top-k 返回文档数真实变化；
- 输出 JSONL 行为字段与 trace 一致。

### 5.3 固定小样本吞吐 smoke

用固定 hash 选择每集 20 题，共 60 题，运行全部 10 个唯一配置。记录：

- GPU 小时与 wall time；
- 检索 QPS；
- 推理失败率；
- 峰值显存；
- 预计全量完成时间。

任一配置服务失败率超过 5%、搜索计数不正确或强制回答泄漏 gold，则停止全量。

## 6. 正式运行顺序

按以下顺序执行：

1. 完成 scheduler、参数化和 unit tests；
2. 单题 smoke；
3. 60 题固定小样本 smoke；
4. 运行共享默认点 `S5-K3`；
5. 运行深度消融 `S0/S1/S2/S3/S4/S6/S7-K3`；
6. 运行宽度消融 `S5-K1`、`S5-K5`；
7. 完整性检查后统一评分；
8. 相对 `S5-K3` 做 paired bootstrap；
9. 生成论文表格、曲线和失败案例。

允许多个 rollout server 并行，但必须：

- 复用唯一健康 retrieval daemon；
- 每个 server 使用独立 GPU、端口和输出目录；
- 不硬编码旧 GPU/端口；
- 不停止他人的服务；
- 并行前先用 60 题 smoke 验证检索吞吐没有成为瓶颈。

## 7. 指标与统计

### 7.1 主要质量指标

- 每数据集 EM、F1、Cover-EM；
- 三数据集 macro EM、macro F1、macro Cover-EM；
- 主要观察指标：macro F1。

### 7.2 行为与成本指标

- `answer_rate`；
- `forced_answer_rate`；
- `forced_answer_valid_rate`；
- forced-answer 子集 EM/F1；
- 平均逻辑检索数；
- 平均实际 RPC 数；
- 达到搜索预算比例；
- 重复 query 率；
- 格式失败率、服务失败率；
- reasoner / Evidence Agent token 数；
- 延迟 P50/P95 和 GPU 小时。

### 7.3 配对统计

固定 `S5-K3` 为默认参考点。对其他配置按 question ID 做 paired bootstrap，建议
10,000 或 20,000 次，分别报告：

```text
Delta EM
Delta F1
Delta Cover-EM
95% CI
```

同时报告相邻深度差值：

```text
S1-S0, S2-S1, ..., S7-S6
```

不能只报告最佳点；所有预注册配置必须进入结果表。

## 8. 论文展示形式

### 8.1 最大检索次数表

| Max Searches | EM | F1 | Cover-EM | Avg Searches | Forced Answer Rate |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1445 | 0.2219 | 0.1613 | 0.000 | 100.00% |
| 1 | 0.1825 | 0.2742 | 0.2082 | 0.998 | 99.84% |
| 2 | 0.2736 | 0.3791 | 0.3303 | 1.932 | 93.39% |
| 3 | 0.3545 | 0.4469 | 0.3870 | 2.325 | 39.28% |
| 4 | 0.3729 | 0.4640 | 0.4020 | 2.497 | 16.80% |
| 5 | 0.3872 | 0.4785 | 0.4152 | 2.566 | 6.80% |
| 6 | 0.3908 | 0.4811 | 0.4179 | 2.616 | 3.55% |
| 7 | 0.3913 | 0.4832 | 0.4191 | 2.634 | 2.53% |

正文建议画两张对齐横轴的曲线：

1. `max_searches -> macro F1 / EM`；
2. `max_searches -> avg logical searches / latency`。

### 8.2 Top-k 表

| Top-k | EM | F1 | Cover-EM | Avg Searches | Latency P95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3394 | 0.4280 | 0.3625 | 2.729 | 1.085 |
| 3 | 0.3872 | 0.4785 | 0.4152 | 2.566 | 1.597 |
| 5 | 0.4117 | 0.5055 | 0.4420 | 2.532 | 0.884 |

## 9. 结果解释边界

- `max_searches=0` 是 closed-book 强制回答，不代表 RAG 系统；
- 检索次数增加但 F1 不升，说明额外搜索未转化为有效证据利用；
- 检索次数减少且 F1 基本不降，说明存在推理成本冗余；
- Top-k 增大可能同时提高召回和引入噪声，必须联合成本解释；
- forced-final-answer 会改变旧评测口径，因此不能把新结果与旧截断计零结果直接做
  因果比较；
- 本实验只证明同一训练 checkpoint 对推理预算的敏感性，不证明训练目标各组件的独立
  贡献；
- 不根据曲线事后选择一个点替换论文主模型，再把它描述成预设默认配置。

## 10. 产物与验收

建议输出根目录：

```text
data/eval_results/icassp_inference_budget_full_ckpt1000_20260922/
```

目录结构：

```text
s0_k3/{hotpotqa,2wikimultihopqa,musique}/
s1_k3/{hotpotqa,2wikimultihopqa,musique}/
...
s7_k3/{hotpotqa,2wikimultihopqa,musique}/
s5_k1/{hotpotqa,2wikimultihopqa,musique}/
s5_k5/{hotpotqa,2wikimultihopqa,musique}/
summary.json
summary.csv
paired_bootstrap/
runtime_manifest.json
split_ids.json
experiment_note.md
```

每个配置必须保存：

- `results.jsonl`、`metrics.json`；
- 展开配置与启动命令；
- checkpoint、代码 SHA、数据哈希、检索 health；
- 样本数、失败数与重试；
- GPU/wall time、显存峰值；
- 至少一个提前回答、一个强制回答成功、一个强制回答失败案例。

最终新增：

```text
docs/icassp_inference_budget_ablation_report.md
```

只有 10 个配置全部完成、协议检查通过、结果可按 ID 配对且失败样本处理透明，才可将
表格和曲线作为 ICASSP 正式消融结果。

## 11. 完成记录

- 10 个唯一配置、30 个数据集运行全部退出 0；
- 223,980 条正式结果全部通过协议与完整性检查；
- `answer_rate=100%`，格式失败与服务失败均为 0；
- 10,000 次分层 paired bootstrap 已完成，共 16 组比较；
- 正式报告：[`icassp_inference_budget_ablation_report.md`](icassp_inference_budget_ablation_report.md)；
- 结果根目录：`data/eval_results/icassp_inference_budget_full_ckpt1000_20260922/`。
