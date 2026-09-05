# SAPR-RAG 后训练实验说明：从 SFT、DPO 到 GRPO、OPSD 与 OPD

**面向读者**：了解大模型后训练基本概念，但不熟悉本项目具体训练与评测实现的研究者
**最后核验日期**：2026-09-05
**事实依据**：`docs/experiment_tracker.md`、训练配置、实际训练参数和评测产物

## 0. 摘要

本项目研究复杂问答中的多步 Agentic RAG。模型不是一次回答问题，而是反复执行“分析状态 -> 生成检索查询 -> 读取证据 -> 继续检索或给出答案”。后训练目标不只包括最终答案正确率，还包括问题分解、证据获取、避免重复检索和及时停止。

目前最可靠的结论是：

1. **SFT 是多轮 RAG 行为能力的主要来源**；E14 进一步证明，旧 SFT
   的 EM/F1 偏低主要来自最终答案目标过长，而不是模型没有学会检索。
2. **Canonical-answer SFT 是当前最强的 HotpotQA 起点**：E14 的
   EM/F1 为 0.4373/0.5513；在其上继续 DPO 的 E15 仅改善 2Wiki，
   HotpotQA 和 MuSiQue 没有稳定收益。
3. **普通 LoRA GRPO 没有稳定超过对应 SFT**。全参数 GRPO 会明显改变
   策略，但同时破坏回答率和停止行为。
4. **E12 查询/答案分动作 OPSD 在旧 SFT 起点上有效**：HotpotQA 的
   F1/Cover-EM 和 2Wiki 的 EM/F1/Cover-EM 显著超过 SFT+DPO；MuSiQue
   没有显著改善。
5. E12 的截断惩罚复现实验证明它能防止后期策略崩坏，但不能进一步提高
   EM/F1；它是稳定性保护，不是主要增益来源。
6. **E13 external-teacher selective OPD 尚无离线结论**：正式训练在
   step340 因模型服务连接中断停止。
7. 当前正在运行 **E16：E14 canonical SFT + GRPO + 分动作 OPSD**。
   只有补齐完全匹配的 GRPO-only control，才能把增益归因给 OPSD teacher。

项目当前的核心问题是：分动作 teacher 在强 canonical SFT 起点上是否仍有
独立收益，以及如何在不破坏因果信息边界的前提下，把不同监督分配给查询、
证据和答案动作。

### 0.1 术语使用规则

本文按以下规则使用中英文术语：论文方法名、训练算法缩写、代码字段名和输出标签保留原文；普通叙述和本项目自定义实验名使用中文。固定术语第一次出现时同时给出中文解释。

| 保留术语 | 本文中的中文含义 |
|---|---|
| Agentic RAG | 智能体式检索增强生成：模型可以主动决定何时检索、检索什么以及何时停止并回答 |
| SFT | 监督微调（Supervised Fine-Tuning）：模仿高质量示范数据中的目标输出 |
| DPO | 直接偏好优化（Direct Preference Optimization）：提高偏好输出相对非偏好输出的概率 |
| GRPO | 组相对策略优化（Group Relative Policy Optimization）：对同一问题采样多条轨迹并在组内计算相对优势 |
| OPSD | 本项目中的在线策略自蒸馏：同源教师利用动作相关特权信息，评价学生在线采样的同一串 token |
| OPD | 在线策略蒸馏（On-Policy Distillation）：独立冻结教师在学生实际访问的状态上评价学生 token；E13 使用该定义 |
| LoRA | 低秩适配：冻结大部分基础模型参数，只训练低秩增量参数 |
| student / teacher | 学生策略 / 教师策略。前者是在真实部署信息下生成轨迹的待训练模型；后者在训练时利用动作特权信息或独立强模型能力评价学生输出 |
| rollout | 在线轨迹采样：模型边生成查询、边调用检索器，直至回答或达到最大轮次的完整交互过程 |
| prompt | 提示词。代码字段名如 `teacher_prompt` 保留原文，正文统一写“教师提示词” |
| gold | 数据集提供的标准标注，例如标准答案、标准支持文档和标准支持句；不是模型生成结果 |
| query / evidence / answer | 查询 / 证据 / 答案。仅在 `<query>` 等输出标签、代码字段和论文原始动作名中保留英文 |
| checkpoint | 训练检查点，即某个训练步保存的模型参数 |
| policy / reference model | 策略模型 / 参考模型。GRPO 用二者的 KL 散度约束策略不要偏离起点过远 |
| advantage | 优势值，表示某条采样轨迹相对同题其他轨迹更好或更差的程度 |
| chosen / rejected | DPO 中的偏好输出 / 非偏好输出。二者对应同一输入状态下的两个候选动作 |
| batch / gradient accumulation | 单次送入每张设备的样本数 / 梯度累积次数；二者共同影响一次参数更新的有效样本量 |
| generations | GRPO 对同一个问题采样的轨迹数量，用于组内比较相对好坏 |
| step / epoch | 参数更新步 / 完整遍历一遍训练数据。`checkpoint-500` 表示第 500 步保存的训练检查点 |
| LoRA rank / alpha / dropout | LoRA 低秩维度 / 缩放系数 / 随机失活率，决定可训练增量参数的容量和正则强度 |
| beta / gamma | 本文中的 `beta` 可能指 DPO 强度、GRPO 的 KL 约束或 OPSD 教师系数，必须结合所在小节；`gamma` 是边际证据奖励的轮次折扣 |
| train / dev | 训练集 / 验证集。本文将 dev 上的最终测评分数称为验证集结果 |
| EM / F1 | 完全匹配 / token（词元）级 F1。EM 要求归一化后的预测与标准答案完全相同；F1 衡量二者词元重叠 |
| Cover-EM | 覆盖式精确匹配：标准答案是否作为连续词元子序列出现在预测中 |
| LLM-acc | 大模型裁判准确率：由 DeepSeek-V3 判断预测与标准答案是否事实等价 |

`messages`、`golden_answers`、`gold_titles`、`gold_sup_sents` 等内容属于实际代码字段，因此保留英文代码名，但每次讨论其语义时使用中文说明。

## 1. 公共模型、推理协议与评测环境

### 1.1 本文中的 ReasonRAG 与 R3-RAG 分别指什么

本文使用的两个简称对应以下两篇论文，不是泛指同名或相似缩写的方法：

1. **ReasonRAG**：Wenlin Zhang 等人的 *Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning*，发表于 NeurIPS 2025。论文提出 ReasonRAG 和 RAG-ProGuide，通过 MCTS 自动构造查询生成（Query Generation）、证据抽取（Evidence Extraction）、答案生成（Answer Generation）三类过程级偏好，并使用 DPO 训练 Agentic RAG。论文链接：[OpenReview](https://openreview.net/forum?id=h3LlJ6Bh4S)，预印本：[arXiv:2505.14069](https://arxiv.org/abs/2505.14069)。本文所说的“ReasonRAG 基线”“ReasonRAG 推理流程”和“RAG-ProGuide”均指这篇工作及其官方实现。
2. **R3-RAG**：Yuan Li 等人的 *R3-RAG: Learning Step-by-Step Reasoning and Retrieval for LLMs via Reinforcement Learning*，发表于 *Findings of the Association for Computational Linguistics: EMNLP 2025*。论文先通过冷启动教模型交替进行推理与检索，再用答案正确性和文档相关性奖励做强化学习。论文链接：[ACL Anthology](https://aclanthology.org/2025.findings-emnlp.554/)，预印本：[arXiv:2505.23794](https://arxiv.org/abs/2505.23794)。本文的“R3 冷启动数据”“R3 参考查询计划”均来自这篇工作发布的数据，而不是本项目自行生成的 R3 轨迹。

为避免歧义，本文中的 R3-RAG 不指同样简称为“R3”的检索器强化学习方法，也不指 R3AG 研讨会。

### 1.2 基座与 ReasonRAG 对齐范围

所有主要实验使用 `Qwen2.5-7B-Instruct`。多轮协议与 ReasonRAG 对齐：

```text
需要检索：So the next query is <query>...</query>
证据抽取：Based on the query, the relevant evidence is <evidence>...</evidence>.
证据充分：So the answer is <answer>...</answer>
```

对齐项包括基座模型、动作标签、最多 6 轮、每轮 Top-3 文档预算及主要评测集。但项目主要使用 BGE 和自建扩展语料索引，ReasonRAG 论文及公开实现还存在 E5 等配置。因此，项目内部相同检索环境下的比较是主要证据；与论文绝对分数的比较不是严格单变量对照。

### 1.3 两种实际使用的推理流程

**流程 A：双角色证据抽取**

```text
原问题 -> 推理智能体生成查询 -> BGE+FAISS Top-3
       -> 证据抽取智能体提炼证据 -> 写入历史
       -> 推理智能体继续查询或输出答案
```

推理智能体输入原问题和历史查询/证据；证据抽取智能体输入当前查询和
Top-3 文档。SFT、SFT+DPO、E09/E10、E12–E16 均使用或按该流程评测。

**流程 B：原始文档回填**

```text
原问题 -> 查询 -> Top-3 原始文档作为下一轮用户观察信息
       -> 继续生成查询或答案
```

该流程不经过独立证据抽取智能体，主要用于 E02–E07 等早期 GRPO/OPSD、
第二版奖励（Reward-v2）和第三版奖励（Reward-v3）。两种流程的输入分布
不同，不能默认直接横比。

### 1.4 检索器、索引和语料

| 项目 | 配置 |
|---|---|
| 查询编码器 | `bge-base-en-v1.5` |
| 向量维度 | 768 |
| 索引 | FAISS `IndexFlatIP`，L2 归一化后等价于余弦相似度 |
| 语料 | FlashRAG `wiki18_100w` + ReasonRAG `RAG_extend_corpus` |
| 规模 | 22,352,695 个 Wikipedia 文本片段 |
| 每轮文档数 | Top-3 |
| 最大轮次 | 6 |

文档和查询使用同一个 BGE 编码器；文档侧不加前缀，查询侧添加 BGE 检索指令前缀。详见 `docs/index_build.md`。

### 1.5 评测基准与指标

| 数据集 | 数据划分 | 题数 | 特点 |
|---|---:|---:|---|
| HotpotQA | 验证集 | 7,405 | 典型双跳，含比较型/桥接型 |
| 2WikiMultiHopQA | 验证集 | 12,576 | 多实体关系链，终止较难 |
| MuSiQue | 验证集 | 2,417 | 2-4 跳组合问题，分解更难 |

R3 SFT 数据包含三个数据集的训练集；RAG-ProGuide DPO 数据包含 PopQA、HotpotQA、2Wiki，不含 MuSiQue。因此 MuSiQue 可观察 DPO 是否遗忘 SFT 能力，但不是对整个训练流程完全未见的分布外（OOD）。

主要指标为 EM、token 级 F1、Cover-EM、DeepSeek-V3 LLM-acc，以及回答率、最大轮次率、平均轮数、空证据率。EM/F1 对答案风格敏感，必须结合 Cover-EM、LLM-acc 和行为指标解释。

### 1.6 三个问答数据集的原始字段及含义

本节说明当前使用的三个多跳问答数据集**下载后的原始字段**，即训练/评测数据构造脚本读取的输入字段。本项目再把这些原始字段转换成统一的训练字段（`messages`、`golden_answers`、`gold_titles`、`gold_sup_sents` 等），转换逻辑见第 4.1 节和第 5.2 节；本节只解释原始输入本身。

三个数据集的原始存储格式不同：HotpotQA 使用 FlashRAG 预处理后的 JSONL；2Wiki 从 `xanhho/2WikiMultihopQA` 的 `train.parquet` 逐行转成 JSONL，其中 `context`、`supporting_facts`、`evidences` 是 **JSON 字符串**，需要再次解码；MuSiQue 使用 `bdsaglam/musique` 官方 `musique_ans_v1.0_train.jsonl`。数据准备见 `03_sapr_rag/scripts/grpo/prepare_action_opsd_train_data.py`。

**HotpotQA**（`data/raw/hotpotqa/train.jsonl`，FlashRAG 预处理版）

| 原始字段 | 类型 | 含义 |
|---|---|---|
| `id` | str | 样本编号，如 `train_0` |
| `question` | str | 多跳问题 |
| `golden_answers` | list[str] | 标准答案，已预处理为列表 |
| `metadata.type` | str | 问题类型，如 `comparison`（比较型）、`bridge`（桥接型） |
| `metadata.level` | str | 难度，`easy` / `medium` / `hard` |
| `metadata.supporting_facts` | dict | 标准支持句定位，结构为 `{title:[...], sent_id:[...]}`，即支持句所在文档标题与句子下标 |
| `metadata.context` | dict | 候选文档，结构为 `{title:[...], sentences:[[...]]}`，每个标题对应一个句子列表 |

**2WikiMultihopQA**（`data/raw/2wikimultihopqa_full/train.jsonl`）

| 原始字段 | 类型 | 含义 |
|---|---|---|
| `_id` | str | 样本编号（32 位十六进制） |
| `type` | str | 问题类型，如 `compositional`、`comparison`、`bridge_comparison`、`inference` |
| `question` | str | 多跳问题 |
| `context` | JSON 字符串 | 解码后为 `list[[title, [sentences...]]]`，候选文档标题与句子列表 |
| `supporting_facts` | JSON 字符串 | 解码后为 `list[[title, sent_id]]`，标准支持句的标题与句子下标 |
| `evidences` | JSON 字符串 | 解码后为 `list[[主语, 关系, 宾语]]`，标准证据三元组（知识图谱式关系标注） |
| `answer` | str | 标准答案，单个字符串（无别名字段） |

**MuSiQue**（`data/raw/musique/train.jsonl`，`musique_ans_v1.0` 版）

| 原始字段 | 类型 | 含义 |
|---|---|---|
| `id` | str | 样本编号，前缀标明跳数，如 `2hop__482757_12019` |
| `question` | str | 组合式多跳问题 |
| `paragraphs` | list[dict] | 候选段落，每个含 `idx`（段落下标）、`title`（标题）、`paragraph_text`（正文）、`is_supporting`（是否为支持段落） |
| `question_decomposition` | list[dict] | 子问题分解，每个含 `id`、`question`（子问题）、`answer`（子答案）、`paragraph_support_idx`（对应支持段落的 `idx`） |
| `answer` | str | 最终标准答案 |
| `answer_aliases` | list[str] | 标准答案的别名列表，可能为空 |
| `answerable` | bool | 该问题是否可答 |

**辅助数据：R3-RAG 冷启动**（`data/raw/r3_coldstart.parquet`）

R3-RAG 冷启动数据不是问答评测集，而是用于 SFT 与 E11 查询教师的轨迹数据（来源见第 1.1、2.1、5.6 节）。本项目只读取其中的 `instruction` 与 `output` 两列，并用正则从 `instruction` 中抽取 `The question:` 后的问题、从 `output` 中抽取 `The retrieval query:` 后的检索查询，按问题聚合成有序参考查询计划。

> 字段对齐说明：`answer` / `answer_aliases`（2Wiki、MuSiQue）与 `golden_answers`（HotpotQA）在本项目内统一映射为 `golden_answers`；`supporting_facts` / `context`（HotpotQA、2Wiki）与 `paragraphs` / `question_decomposition`（MuSiQue）用于恢复 `gold_titles`、`gold_sup_sents`。具体转换与信息隔离规则见第 4.1 节。

## 2. SFT：先学会多轮 RAG 协议

### 2.1 数据来源与改造

| 数据类型 | 数量 | 原始来源 | 本项目改动 |
|---|---:|---|---|
| 推理动作样本 | 178,061 | R3-RAG 冷启动 | 将 R3 步骤转为 ReasonRAG 标签协议 |
| 证据抽取样本 | 98,877 | R3 中去重的查询/文档组合 | 用 DeepSeek-V3 提炼证据，构造独立监督样本 |

R3-RAG 冷启动数据不是本项目从零生成。它由 R3-RAG 使用 GPT-4o 和真实检索轨迹构造，覆盖 HotpotQA、2Wiki、MuSiQue 的训练问题。每行对应一个教师强制训练步骤（teacher forcing，即训练时给定真实历史）：输入包含此前历史，输出是当前分析和查询，或最终答案。

本项目将 R3 的纯文本格式转写成与 ReasonRAG/DPO 一致的 `<query>/<evidence>/<answer>` 协议，避免 SFT 和 DPO 学习两套不兼容格式。

### 2.2 推理动作 SFT 输入与输出

第 t 个动作的输入：

```text
系统：模型可以检索，需要分解问题，并以 <query> 或 <answer> 结束
用户：Question + 前 t-1 步的查询和证据历史
```

输出为“分析 + 下一条 `<query>`”，或“分析 + 最终 `<answer>`”。因此 SFT 学习的是**状态到下一动作的行为克隆**，不只是最终答案。

### 2.3 证据抽取 SFT 输入与输出

R3 没有独立证据抽取智能体动作。早期尝试直接截断检索原文作为证据，质量不足，最终废弃。正式方案对每个唯一 `(query, reference documents)`，即查询与参考文档组合，调用 DeepSeek-V3，生成一句简洁证据。98,877 个唯一组合来自约 21.7 万次历史出现，通过哈希缓存去重。

```text
输入：当前查询 + 检索文档
输出：<evidence>简洁证据</evidence>，或 <evidence>None</evidence>
```

这部分属于**本项目基于外部轨迹二次构造的数据**。

### 2.4 训练配置与结果

| 配置项 | 实际设置 |
|---|---|
| 框架 / 基座 | LLaMA-Factory / Qwen2.5-7B-Instruct |
| 方式 | LoRA，rank=16，alpha=32，dropout=0.05，全部线性层 |
| 数据 | 178,061 条推理动作样本 + 98,877 条证据抽取样本 |
| 最大序列长度 / 学习率 | 2,048 / `1e-4` |
| 调度 | 余弦学习率调度，预热比例=0.03 |
| 计划 | 1 轮训练，约 2,142 步 |
| 实际采用 | 训练检查点 checkpoint-1650，约 0.77 轮 |

训练没有跑完整一轮，但第 1450-1650 步的验证损失仅从 0.1813 降到
0.1794，已进入平台。因此 `checkpoint-1650` 被作为 E01–E13 的历史
SFT 起点；E14 完成答案目标修正后，`sft_canonical_fp16/checkpoint-4150`
成为 E15/E16 的新版统一起点。

| 数据集 | 零样本 Cover-EM | SFT Cover-EM | 零样本最大轮次率 | SFT 最大轮次率 |
|---|---:|---:|---:|---:|
| HotpotQA | 0.2680 | **0.5070** | 45.1% | **10.7%** |
| 2Wiki | 0.1114 | **0.4488** | 66.6% | **27.9%** |
| MuSiQue | 0.0956 | **0.1911** | 63.6% | **33.4%** |

SFT 的主要贡献是学会多轮协议、问题分解和及时停止。HotpotQA EM 从 0.2040 降到 0.0971，不代表事实正确率下降：SFT 答案更长，严格 EM 大量假阴。DeepSeek-V3 裁判的 HotpotQA 事实正确率从 0.338 提高到 0.607。

### 2.5 E14：Canonical-answer SFT

E14 只修改 SFT terminal step 的答案目标：保留 R3 的问题、查询、历史、
分析、证据和 step 结构，把最终 `<answer>` 内容替换为原始训练集的
canonical gold short answer。该改动用于消除旧 SFT 长解释答案与
EM/F1 评测口径之间的错配。

| 数据审计项 | 数量 |
|---|---:|
| reasoning rows | 178,061 |
| terminal answer rows | 51,253 |
| 成功回连 gold | 51,148 |
| 实际替换 | 51,100 |
| 未匹配 gold | 105 |

答案平均长度由 13.96 words 降至 2.23 words，`<=5 words` 占比由
27.96% 提高到 96.7%。训练使用 Qwen2.5-7B-Instruct、LoRA
rank=16/alpha=32/dropout=0.05、fp16、1 epoch，共 4,284 step；根据
验证损失选择 `sft_canonical_fp16/checkpoint-4150`。

| 数据集 | N | 回答率 | EM | F1 | Cover-EM | 最大轮次率 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 7,405 | 90.36% | **0.4373** | **0.5513** | 0.4748 | 9.62% |
| 2Wiki | 12,576 | 75.07% | 0.4051 | 0.4513 | 0.4188 | 24.92% |
| MuSiQue | 2,417 | 71.29% | 0.1651 | 0.2405 | 0.1841 | 28.71% |

E14 相对旧 SFT 的 HotpotQA EM/F1 分别提高 34.02/28.79pt，并直接超过
SFT+DPO 和 E12 的 HotpotQA EM/F1。与此同时，2Wiki/MuSiQue 尚未超过
E12，三个数据集的 Cover-EM 也低于旧 SFT/E12。因此 E14 是答案格式对齐
修复和后续 DPO/GRPO/OPSD 的新统一起点，而不是所有指标上的最终模型。

主要产物：

- 数据：`03_sapr_rag/data/sft_build/out/sft_v2_reasoning_canonical.jsonl`
- 配置：`03_sapr_rag/scripts/train/sft_canonical_lora_fp16.yaml`
- checkpoint：`03_sapr_rag/saves/qwen2_5_7b/lora/sft_canonical_fp16/checkpoint-4150/`
- 三源评测：`data/eval_results/sft_canonical_ckpt4150_3src_6gpu_20260904/`

## 3. DPO：学习过程级偏好与简洁答案

### 3.1 数据来源与偏好/非偏好输出

DPO 数据直接来自 ReasonRAG 官方 RAG-ProGuide：约 5,000 个 PopQA、HotpotQA、2Wiki 训练问题，使用 GPT-4o + MCTS 生成候选轨迹，节点奖励记录为带步数折扣的 `F1 × 0.9^step`（其中 step 表示 MCTS 树中的步骤深度），最终发布为 13,289 个偏好/非偏好输出对。

偏好关系不是本项目重新标注的。本项目仅做字段映射、过滤空/相同输出，并统一为 SFT 使用的标签协议。实际训练文件保留完整 13,289 对。

```text
输入 x：系统提示词 + 问题 + 已有检索历史
chosen y+（偏好输出）：MCTS 过程奖励更高的下一步
rejected y-（非偏好输出）：同一状态下奖励较低的下一步
```

| 偏好输出的动作类型 | 数量 |
|---|---:|
| 答案生成（Answer Generation） | 5,689 |
| 证据抽取（Evidence Extraction） | 4,305 |
| 查询生成（Query Generation） | 3,295 |

多数样本比较同类动作，也包含“应该回答还是继续查询”等跨动作偏好。因此 DPO 不只学习答案风格，也学习查询、证据抽取和停止决策。

### 3.2 两个实验设置与训练配置

- **仅 DPO**：从原始基座直接做 DPO，对应 ReasonRAG“无 SFT、直接 DPO”路线。答案指标已统一重算，但推理轮次定义与本项目流程不同。
- **SFT+DPO**：从 SFT 训练检查点 checkpoint-1650 继续 DPO，验证“先学协议，再学偏好”。

| 配置项 | SFT+DPO 实际设置 |
|---|---|
| 方式 | 在 SFT LoRA 上继续进行 LoRA 训练，rank=16，alpha=32，dropout=0.05 |
| 损失 | 标准 sigmoid DPO 损失，强度系数 beta=0.2 |
| 数据 | 13,289 对，最大序列长度=2,560 |
| 学习率 / 有效批量 | `5e-6` / 32 |
| 调度 | 余弦学习率调度，预热比例=0.03 |
| 训练 | 1 轮训练，395 步 |

chosen 与 rejected 的隐式奖励差从 0.116 增至 0.554，说明偏好优化确实生效。

### 3.3 结果

| 数据集 | 实验设置 | Cover-EM | LLM-acc | EM | F1 | 最大轮次率 |
|---|---|---:|---:|---:|---:|---:|
| HotpotQA | SFT | 0.5070 | 0.6073 | 0.0971 | 0.2634 | 10.7% |
| HotpotQA | 仅 DPO | 0.3999 | 0.5356 | 0.3492 | 0.4563 | 不同口径 |
| HotpotQA | SFT+DPO | 0.4693 | 0.6062 | **0.4008** | **0.5233** | **3.4%** |
| 2Wiki | SFT | 0.4488 | 0.4431 | 0.1018 | 0.2515 | 27.9% |
| 2Wiki | 仅 DPO | 0.4061 | 0.4249 | 0.3496 | 0.4194 | 不同口径 |
| 2Wiki | SFT+DPO | 0.4452 | **0.4705** | **0.3915** | **0.4688** | **17.3%** |
| MuSiQue | SFT | 0.1911 | 0.2081 | 0.0492 | 0.1205 | 33.4% |
| MuSiQue | 仅 DPO | 0.1452 | 0.1957 | 0.1200 | 0.1935 | 不同口径 |
| MuSiQue | SFT+DPO | **0.2069** | **0.2462** | **0.1667** | **0.2477** | **16.9%** |

仅 DPO 有效，但不如先通过 SFT 建立多轮能力。旧 SFT+DPO 大幅提高
EM/F1，主要因为答案更短、更贴近标准答案；HotpotQA LLM-acc 与旧 SFT
持平，2Wiki 和 MuSiQue 则提高。它还进一步降低三个数据集的最大轮次率，
是 E14 之前综合行为最稳定的起点。

### 3.4 E15：Canonical SFT→DPO

E15 从 E14 `sft_canonical_fp16/checkpoint-4150` 继续训练，使用与 E01
同类的 `sapr_proguide_dpo` 偏好数据。配置为 LoRA DPO、
`pref_beta=0.2`、sigmoid loss、学习率 `5e-6`、最大长度 2,560，
训练 1 epoch（451 step），采用 `checkpoint-451`。

| 数据集 | N | 回答率 | EM | F1 | Cover-EM | 最大轮次率 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 7,405 | 95.00% | 0.4140 | 0.5281 | 0.4304 | 5.00% |
| 2Wiki | 12,576 | 82.89% | 0.4187 | 0.4656 | 0.4230 | 17.09% |
| MuSiQue | 2,417 | 83.12% | 0.1585 | 0.2459 | 0.1676 | 16.88% |

相对 E14，E15 仅在 2Wiki 上同步提高 EM/F1/Cover-EM
（+1.36/+1.43/+0.42pt）；HotpotQA 三项均下降
（-2.33/-2.32/-4.44pt），MuSiQue 基本持平。训练曲线正常且偏好 margin
持续扩大，说明问题不是 DPO 未生效，而是当前 RAG-ProGuide 偏好对对
canonical short-answer 起点的边际价值有限。

主要产物：

- 配置：`03_sapr_rag/scripts/train/dpo_canonical_lora.yaml`
- checkpoint：`03_sapr_rag/saves/qwen2_5_7b/lora/sft_canonical_dpo/checkpoint-451/`
- 三源评测：`data/eval_results/sft_canonical_dpo_3src_6gpu_20260905/`

## 4. GRPO：在线优化答案、检索覆盖和格式

### 4.1 普通 GRPO 的训练流程与奖励

GRPO 训练文件不是评测基准可以直接送入训练框架的原始格式，而是本项目从问题、答案和支持证据标注中转换得到。每行字段来源如下：

| GRPO 字段 | 是否为原始字段 | 来源与构造方式 | 用途 |
|---|---|---|---|
| `messages` | 否 | 本项目写入固定的推理智能体系统提示词，并把原始 `question` 包装成用户消息 `Question: ...`；其中不含人工查询、标准答案或标准支持证据 | 学生策略在线采样轨迹的首轮输入 |
| `golden_answers` | 语义来自原始标注，字段名可能经过标准化 | HotpotQA 和旧版 2Wiki 已预处理为 `golden_answers`；E11 的完整 2Wiki 和 MuSiQue 从原始 `answer` 及 `answer_aliases` 转为答案列表 | 计算答案 F1 奖励；构造 OPSD 答案教师提示词 |
| `gold_titles` | 否，是派生字段 | HotpotQA 和旧版 2Wiki 从 `metadata.supporting_facts.title` 去重获得；完整 2Wiki 解码顶层 `supporting_facts` 后读取 `(title, sent_id)`；MuSiQue 用 `paragraph_support_idx` 回查支持段落标题 | 计算标准支持文档覆盖率 |
| `gold_sup_sents` | 否，是派生字段 | HotpotQA/2Wiki 用 `(title, sent_id)` 回查同题 `context`；MuSiQue 用 `paragraph_support_idx` 回查 `paragraphs` 段落文本；同一标题的多句用换行合并 | 计算支持证据文本命中；构造答案教师提示词 |
| `source` | 否 | 构造时写入 `hotpotqa`、`2wiki` 或 `musique` | 审计训练数据组成 |

这里的“构造”只是把评测基准已有标注转成奖励函数需要的统一结构，**没有调用模型重新生成标准答案、标准支持文档标题或标准支持句**。这与 SFT 中使用 DeepSeek-V3 蒸馏证据、DPO 中使用 MCTS 构造偏好对不同。

必须区分三类信息：

1. `messages` 只含系统提示词和原始问题，是学生策略可见的初始输入。
2. `golden_answers/gold_titles/gold_sup_sents` 只在训练端计算奖励，普通 GRPO 的学生策略看不到这些标准标注。
3. 学生策略实际看到的 Top-3 文档不预存在训练文件中，而是在模型生成查询后，从统一的 BGE+FAISS 语料库在线检索得到。原始数据集的 `context` 只用于恢复标准支持句，不等于在线返回的文档。

E02 使用相同转换逻辑，但错误地从 HotpotQA 验证集构造训练数据。E04-E07 使用 HotpotQA 和 2Wiki 官方训练集，并按固定随机种子做平衡采样。

训练时，一个问题会采样多条完整交互轨迹。同一问题的多条轨迹组成“同题采样组”，奖励在组内归一化，得到每条轨迹的相对优势值，再用于更新策略模型。

早期奖励为：

```text
总奖励 = 1.0 × 答案 F1 奖励
        + 0.2 × 证据相关性奖励
        + 0.05 × 格式奖励
```

- **答案 F1 奖励**：最终 `<answer>` 与标准答案的 token 级 F1。
- **证据相关性奖励**：整条轨迹检索到的文档覆盖了多少标准支持事实。早期实现按整条轨迹累计，不能指出具体哪轮查询有效。
- **格式奖励**：允许前面出现多个 `<query>`，但最后一个协议动作必须是非空 `<answer>`。

GRPO 的 `beta=0.04` 是策略模型相对参考模型的 KL 约束系数，用来限制策略偏离起点；它与 OPSD 的教师信号系数不是同一参数。

### 4.2 E02：旧版 GRPO（v4 格式奖励修复）

**验证目的**：SFT 之后加入在线答案奖励、检索证据奖励和格式奖励，能否进一步提高多跳问答能力。

| 配置项 | 设置 |
|---|---|
| 起点 / 更新方式 | SFT 训练检查点 checkpoint-1650 / LoRA |
| 数据 | 7,321 条 HotpotQA 样本 |
| 奖励权重 | 答案 F1 1.0 / 证据相关性 0.2 / 格式 0.05 |
| 单卡批量 / 每题采样轨迹数 / 梯度累积 | 2 / 8 / 4 |
| 学习率 / 最大轮次 | `1e-6` / 6 |
| 计划 / 实际 | 计划 1,220 步；第 234 步崩溃，主要评测 checkpoint-175 |

后来确认，这 7,321 条训练数据由 **HotpotQA 验证集**派生，而非官方训练集。训练奖励又使用了这些题目的标准答案和支持事实，因此 HotpotQA 结果存在验证集泄露，不能作为留出评测证据。

跨数据集结果只保留诊断价值：2Wiki checkpoint-175 的 Cover-EM 为 0.4573，略高于 SFT 的 0.4488；MuSiQue 为 0.1986，略高于 SFT 的 0.1911，但均低于 SFT+DPO 的综合表现。

### 4.3 E04：严格 LoRA GRPO 对照实验（关闭教师信号）

**验证目的**：去除验证集泄露和 OPSD 教师信号后，单独判断普通 GRPO 是否有效。

训练数据从官方训练集重新构造：HotpotQA 3,660 条、2Wiki 3,660 条，共 7,320 条，与 HotpotQA 验证集隔离。

| 配置项 | 设置 |
|---|---|
| 起点 / 更新方式 | SFT 训练检查点 checkpoint-1650 / LoRA，rank=16，alpha=32 |
| 奖励权重 | 答案 F1 1.0 / 证据相关性 0.2 / 格式 0.05 |
| 单卡批量 / 每题采样轨迹数 / 梯度累积 | 2 / 7 / 1 |
| 每次更新的不同问题数 | 约 2 个 |
| 学习率 / 策略 KL 系数 | `1e-6` / 0.04 |
| 评测检查点 | checkpoint-1000，约 0.27 轮训练 |

| 实验设置 | Cover-EM | EM | F1 | 回答率 | 平均轮数 | 最大轮次率 |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 0.5070 | 0.0971 | 0.2634 | 89.28% | 2.513 | 10.71% |
| LoRA GRPO 对照实验 | 0.5080 | 0.1048 | 0.2716 | 89.60% | 2.508 | 10.36% |

结论：严格 LoRA GRPO 与 SFT 基本持平。训练链路有效，但没有实质泛化增益。

### 4.4 E05：全参数 GRPO

**验证目的**：LoRA 的可训练容量是否限制了 GRPO。相对 E04 保持数据、奖励和采样配置不变，只将更新范围改为全部模型参数。

| 配置项 | 设置 |
|---|---|
| 起点 / 更新方式 | 合并 SFT LoRA 后的完整模型 / 全参数 ZeRO-3 |
| 数据 | 与 E04 相同的 7,320 条 |
| 单卡批量 / 每题采样轨迹数 | 2 / 7 |
| 学习率 / 策略 KL 系数 | `1e-6` / 0.04 |
| 训练长度 | 3,660 步，完整 1 轮 |

| 训练检查点 | Cover-EM | EM | F1 | 回答率 | 平均轮数 | 最大轮次率 |
|---|---:|---:|---:|---:|---:|---:|
| checkpoint-2500 | **0.4493** | **0.4003** | **0.5071** | 77.06% | 3.162 | 22.86% |
| checkpoint-3000 | 0.4258 | 0.3824 | 0.4796 | 69.14% | 3.735 | 30.76% |
| checkpoint-3660 | 0.4265 | 0.3854 | 0.4817 | 69.79% | 3.704 | 30.16% |

全参数 GRPO 提高了标准支持文档标题覆盖率，也提高了“只在已回答样本上计算的 Cover-EM”，并使答案变短；但端到端行为退化，表现为回答率下降、最大轮次率和重复查询率上升。

这不是模型没有学到奖励，而是奖励目标错位：早期证据相关性奖励按整条轨迹累计，不惩罚过度检索；同时缺少明确的轮次成本、重复查询和最大轮次惩罚。

E04/E05 使用的早期混合数据后来还发现支持句与标题对齐缺陷；旧版证据相关性奖励又把“检索文本包含标准答案”作为兜底命中条件，进一步降低了奖励区分度。这直接推动了后续第二版和第三版奖励设计。

### 4.5 E06：第二版奖励，加入反重复与终止约束

“第二版奖励（Reward-v2）”是项目内部实验编号，不是已有论文方法。其目的在于检验全参数 GRPO 的失败是否主要由重复查询和无法终止导致。

```text
总奖励 = 1.00 × 答案 F1 奖励
        + 0.15 × 累计证据覆盖奖励
        + 0.05 × 格式奖励
        + 0.02 × 轮次成本惩罚
        + 0.15 × 重复查询惩罚
        + 0.50 × 最大轮次惩罚
```

- **累计证据覆盖奖励**：统计整条轨迹最终覆盖了多少标准支持事实；首次发现和重复发现同一事实没有区别。
- **轮次成本惩罚**：第一条查询不扣分，从第二条查询开始按数量扣分。
- **重复查询惩罚**：规范化后完全相同的查询会扣分。
- **最大轮次惩罚**：跑满最大轮次且仍未回答时扣分。

| 配置项 | 设置 |
|---|---|
| 起点 / 更新方式 | SFT LoRA 合并模型 / LoRA，rank=16 |
| 学习率 / 策略 KL 系数 | `1e-6` / 0.04 |
| 单卡批量 / 每题采样轨迹数 | 2 / 7 |
| 训练长度 | 500 步，每 100 步保存 |

固定 200 题、原始文档回填流程下的对照：

| 模型 | EM | F1 | Cover-EM | 回答率 | 平均查询数 | 完全重复率 |
|---|---:|---:|---:|---:|---:|---:|
| SFT LoRA 合并模型 | 0.105 | **0.2835** | **0.545** | 90.5% | 2.165 | **13.5%** |
| 第二版奖励 checkpoint-300 | **0.110** | 0.2739 | 0.520 | 89.5% | 2.210 | 15.0% |

HotpotQA 全量验证集上，checkpoint-300 为 EM 0.1086 / F1 0.2761 / Cover-EM 0.5121，与 SFT 基本持平，重复行为也没有改善。这说明问题不只是缺少重复惩罚，还涉及轨迹级奖励无法指出具体哪轮查询无效。

### 4.6 E07：第三版奖励，改为边际新增证据

“第三版奖励（Reward-v3）”同样是项目内部实验编号。它只替换第二版中的证据项，其他主要配置保持不变。

- **边际新增证据奖励**：每条标准支持事实只在首次命中时得分。
- 越早首次命中，奖励越高；轮次折扣 `gamma=0.9`。
- 全部支持事实已经覆盖后，继续查询每次额外扣 0.10。

该设计要验证：区分“首次取得新证据”和“重复取得旧证据”，能否改善查询策略。

训练 500 步后，答案 F1 奖励和边际新增证据奖励均基本横盘，平均轮数反而缓慢上升。当前共享产物缺少可复核的最终全量指标文件，因此该实验只保留诊断可信度。早期固定 200 题约为 EM 0.105 / F1 0.270 / Cover-EM 0.520，没有超过 SFT 的趋势。

### 4.7 GRPO 总体判断

GRPO 系列排除了三个简单解释：失败不是因为 LoRA 容量不足，不只是因为缺少重复和终止惩罚，也不只是因为累计证据奖励定义粗糙。

更深层的问题是：轨迹级奖励对具体查询的局部归因太弱；同题采样组内经常没有奖励差异；终止和持续检索目标相互冲突；同时查询质量限制了 Top-3 证据召回。

## 5. OPSD：用特权信息评价学生策略轨迹

### 5.1 方法定义

OPSD 中有两个条件不同的策略视图：

- **学生策略（student）**：只能看到部署时真实可用的信息，即原问题、自己此前生成的查询和在线检索结果。
- **教师策略（teacher）**：训练时额外看到特权信息，用于评价学生策略已经生成的 token；教师策略不会重新生成另一条轨迹。

训练过程是：学生策略先在线采样完整轨迹；教师策略再对**同一串学生 token**计算逐 token 对数概率；最后把教师与学生的对数概率差加入 GRPO 优势值：

```text
A_t = A_GRPO
    + beta_action(t) × [log p_teacher(y_t) - log p_student(y_t)]
```

其中 `p_teacher` 和 `p_student` 是代码和公式中的固定符号，分别表示教师策略和学生策略给当前 token 的概率；`beta_action(t)` 是当前动作对应的教师信号系数。

关键问题不是教师看到的信息越多越好，而是教师评价某种动作时，是否遵守该动作的因果信息边界。例如，已经看到标准答案的教师不适合评价学生下一步应该搜索什么。

### 5.2 OPSD 数据相对普通 GRPO 多了什么

所有 OPSD 实验都复用普通 GRPO 的基础字段：`messages` 供学生策略采样，`golden_answers/gold_titles/gold_sup_sents` 供基础奖励计算。OPSD 额外增加的是本项目构造的“教师视图”，不是评测基准原生字段。

| 实验 | 新增字段或机制 | 教师信息来源 | 数据性质 |
|---|---|---|---|
| E03 旧全动作 OPSD | `teacher_prompt` | 将原始问题、由支持事实派生的标准支持证据、标准答案拼成教师提示词 | 底层标注来自评测基准；提示词和使用方式由本项目构造 |
| E09/E10 仅答案动作 OPSD | 教师提示词 + 答案动作掩码 | 与 E03 使用相同的标准答案和支持证据，但教师与学生的对数概率差只作用于最终答案 token | 动作掩码由本项目实现 |
| E11 查询教师 | `teacher_query_prompt` | 从 R3-RAG 冷启动数据中按原问题聚合成功轨迹的检索查询，去除同题完全重复查询，组成有序参考查询计划 | 查询内容来自 R3-RAG；聚合、提示词和动作隔离由本项目构造 |
| E11 答案教师 | `teacher_answer_prompt` | 将评测基准的答案及别名、由支持证据标注恢复的证据拼成教师提示词 | 底层标注来自评测基准；提示词由本项目构造 |
| E11 证据教师 | 当前没有有效教师提示词 | 代码只保留接口，`teacher_evidence_kl_coef=0`，尚未构造独立辅助训练批次 | 未启用，不能声称已有证据动作 OPSD 数据 |

学生策略始终在普通初始提示词和在线 Top-3 检索环境中生成轨迹。教师提示词只改变训练时“如何评价学生 token”，不会用标准支持证据替换学生的真实检索结果。

### 5.3 E03：旧版全动作 OPSD

- 学生策略起点：SFT+DPO checkpoint-395；
- 数据：HotpotQA/2Wiki 官方训练集各 3,660 条；
- 教师提示词：标准支持证据 + 标准答案；
- 教师信号系数：0.1，作用于全部查询和答案 token；
- 基础奖励：答案 F1 1.0 / 证据相关性 0.2 / 格式 0.05；
- LoRA rank=16，学习率 `1e-6`，单卡批量=2，每题采样 7 条轨迹；
- 训练 3,660 步，完整 1 轮。

HotpotQA checkpoint-3000 的结果为 EM 0.2895 / F1 0.4026 / Cover-EM 0.3869，显著低于 SFT+DPO。错误案例显示，比较型问题中大量逐实体查询被改成合并查询，双证据覆盖下降。

根因是教师策略已经看到答案和支持证据，本身不需要搜索，却仍评价学生策略的查询 token。对“已知答案”的教师而言高概率的搜索动作，不一定适合“未知答案、必须检索”的学生策略。

该实验还存在推理流程混杂和旧版在线采样服务未实际加载 LoRA 的问题，因此可信度为 C，只保留失败机理分析价值。

### 5.4 在线轨迹采样服务的 LoRA 加载修复

旧版 Swift 在线采样服务虽然记录了 LoRA 适配器参数，却没有把 adapter 传给实际推理引擎，导致生成结果等同于基础模型。修复后，通过比较同一提示词的逐 token 输出，确认在线采样服务与直接加载 SFT+DPO LoRA 的推理结果一致。E09/E10 及之后实验均使用修正后的链路。

这说明在线策略（on-policy）后训练必须验证三者一致：参与梯度更新的训练策略、负责生成轨迹的在线采样策略、用于离线评测的策略。不能只根据启动日志判断 LoRA 已生效。

### 5.5 E09/E10：仅答案动作 OPSD

**验证假设**：标准答案和标准支持证据适合评价最终答案，但不适合评价查询。把教师信号限制在最终答案 token，能否避免搜索策略被破坏？

| 配置项 | 设置 |
|---|---|
| 起点 | SFT+DPO checkpoint-395 |
| 数据 | HotpotQA/2Wiki 预实验 100 条 |
| 推理流程 | Top-3 检索 + 独立证据抽取智能体 |
| 教师信息 | 标准答案 + 标准支持证据，仅评价最终答案 token |
| 教师信号系数 | 0.03 |
| 基础奖励 | 答案 F1 1.0 / 证据相关性 0.2 / 格式 0.05 |
| 更新方式 | LoRA，rank=8，alpha=32，学习率 `1e-6` |
| 策略 KL 系数 | 0.04 |
| 每题采样轨迹数 / 单卡批量 / 梯度累积 | 8 / 1 / 4 |
| 每轮采样后复用的更新步数 | 8 |

E09 训练 25 步；E10 仅将训练长度延长至 100 步，并保存第 25/50/75/100 步的检查点。

| 模型 | EM | F1 | Cover-EM | 回答率 | 平均轮数 | 最大轮次率 |
|---|---:|---:|---:|---:|---:|---:|
| SFT+DPO | 0.4008 | 0.5233 | **0.4693** | 96.57% | 2.151 | 3.43% |
| 仅答案动作 OPSD 25 步 | **0.4054** | **0.5264** | 0.4690 | 96.48% | 2.135 | 3.50% |
| 仅答案动作 OPSD 100 步 | 0.4032 | 0.5243 | 0.4675 | 96.45% | 2.125 | 3.52% |

同一 7,405 个样本的配对自助法显著性检验显示：第 25 步相对 SFT+DPO 的 EM/F1/Cover-EM 增量为 +0.46/+0.31/-0.03 个百分点，均不显著；第 100 步为 +0.24/+0.10/-0.18 个百分点，同样不显著。

答案动作掩码避免了 E03 的明显退化，但增益没有随训练步数稳定放大。

### 5.6 E11：查询/答案分动作 OPSD

E11 针对因果错配，为查询和答案动作分配不同的教师信息：

| 动作 | 学生策略可见信息 | 教师额外信息 | 教师信号系数 |
|---|---|---|---:|
| 查询 | 原问题 + 实际检索历史 | R3-RAG 成功轨迹的有序参考查询计划 | 0.01 |
| 证据 | 当前查询 + 实际 Top-3 | 计划只使用 Top-3 内可核验证据 | **0.00，未启用** |
| 答案 | 实际检索历史 | 标准答案 + 已核验标准支持证据 | 0.03 |

查询教师不能看到标准答案或标准支持事实。R3-RAG 参考查询计划只是成功轨迹的搜索参考，不是人工定义的唯一正确子问题序列，也不要求学生策略机械复制。答案教师只评价最终答案 token。

证据教师暂时关闭，因为证据抽取智能体是独立生成调用。未来必须构造独立辅助训练批次，不能把证据动作监督直接混入推理智能体的动作掩码。

#### 数据来源

| 数据集 | 训练样本 | 有 R3-RAG 参考查询计划 | 覆盖率 |
|---|---:|---:|---:|
| HotpotQA | 90,447 | 25,377 | 28.1% |
| 2Wiki | 167,454 | 10,832 | 6.5% |
| MuSiQue | 19,938 | 14,085 | 70.6% |
| 合计 | **277,839** | **50,294** | **18.1%** |

评测基准的问题、答案和支持证据来自公开数据；参考查询计划从 R3-RAG 发布轨迹按问题重新聚合；分动作教师提示词、信息隔离和缺失标注回退机制由本项目构造。没有参考查询计划的样本仍参与普通 GRPO 和答案动作 OPSD，只将查询教师掩码置零。

#### 训练配置与状态

| 配置项 | 设置 |
|---|---|
| 起点 / 更新方式 | SFT+DPO checkpoint-395 / LoRA，rank=8，alpha=32 |
| 基础奖励 | 答案 F1 1.0 / 证据相关性 0.2 / 格式 0.05 |
| 查询 / 证据 / 答案教师系数 | 0.01 / 0 / 0.03 |
| 策略 KL 系数 / 学习率 | 0.04 / `1e-6` |
| 每题采样轨迹数 / 单卡批量 / 梯度累积 | 8 / 2 / 4 |
| 每轮采样后复用的更新步数 | 8 |
| 计划 | 3,000 步，每 500 步保存 |

完整一轮约需 55,568 个优化步；3,000 步只覆盖约 0.054 轮训练。该实验旨在先验证短程分动作教师信号，而不是声称完成全量训练。

E11 后续因原 worker 回收而停止，约运行到 step1624，最后完整保存
`checkpoint-1500`。该实验从 SFT+DPO 起点训练，未完成三数据集全量
评测，因此仍不能用于隔离分动作 OPSD 的收益。E12 随后改为直接从 SFT
起点重跑，形成有效主结果。

### 5.7 E12：SFT→查询/答案分动作 OPSD

E12 跳过 DPO，从旧 SFT `checkpoint-1650` 直接启动分动作 OPSD。相对
E11，三源训练数据、在线 Evidence Agent、奖励、teacher 信息和动作系数
保持不变，唯一核心变化是初始 adapter。

| 配置项 | 设置 |
|---|---|
| 训练数据 | HotpotQA 90,447 + 2Wiki 167,454 + MuSiQue 19,938，共 277,839 条 |
| 更新方式 | LoRA，rank=16，alpha=32，dropout=0.05 |
| 基础奖励 | F1 1.0 / 累计 evidence relevance 0.2 / format 0.05 |
| teacher 系数 | Query 0.01 / Evidence 0 / Answer 0.03 |
| 采样 | 8 generations；steps-per-generation 8 |
| batch | 每卡 2；梯度累积 4；GPU2-6 训练 |
| 环境 | GPU7 rollout；独立 Evidence Agent；GPU0 BGE+FAISS Top-3 |
| 长度 / 学习率 | max completion 4096 / `1e-6` |

原始训练计划 3,000 step，实际在 step1858 主动停止。原因是
`checkpoint-1500` 已出现长输出、格式损坏和回答率下降；保留的完整
checkpoint 为 500/1000/1500，`checkpoint-1000` 被登记为最佳点。

三个数据集全量验证集：

| 数据集 | checkpoint | N | 回答率 | EM | F1 | Cover-EM | Max-turn |
|---|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 500 | 7,405 | 97.52% | 0.3118 | 0.4602 | 0.5165 | 2.47% |
| HotpotQA | **1000** | 7,405 | **97.66%** | **0.4086** | **0.5379** | 0.4984 | **2.21%** |
| 2Wiki | 500 | 12,576 | 97.30% | 0.3855 | 0.4892 | **0.5484** | 2.70% |
| 2Wiki | **1000** | 12,576 | **98.51%** | **0.4866** | **0.5655** | 0.5476 | **1.48%** |
| MuSiQue | 500 | 2,417 | 92.93% | 0.1233 | 0.2243 | **0.2238** | 7.07% |
| MuSiQue | **1000** | 2,417 | **94.99%** | **0.1547** | **0.2546** | 0.2180 | **4.92%** |

`checkpoint-1000` 相对 SFT+DPO 的同 ID、20,000 次配对自助法结果：

| 数据集 | EM 差值（双侧 p） | F1 差值（双侧 p） | Cover-EM 差值（双侧 p） |
|---|---:|---:|---:|
| HotpotQA | +0.80pt (0.0929) | +1.47pt (0.0015) | +2.92pt (0.0001) |
| 2Wiki | +9.50pt (0.0001) | +9.67pt (0.0001) | +10.23pt (0.0001) |
| MuSiQue | -1.20pt (0.0678) | +0.69pt (0.2928) | +1.12pt (0.1266) |

因此 E12 证明了“GRPO + 分动作 OPSD”整体方案在 HotpotQA 和 2Wiki
有效，但当时没有完全匹配的 GRPO-only control，不能把全部增益归因于
teacher。MuSiQue 没有显著改善；三源自然采样中 MuSiQue 只占 7.2%，
1000 step 预计只覆盖约 359 个 MuSiQue 问题。

#### 截断惩罚稳定性复现

为修复原始 `checkpoint-1500` 的策略崩坏，另从相同旧 SFT 起点训练
1,500 step，并在 GRPO 总 reward 中增加：

```text
R = F1 + 0.2 * relevance + 0.05 * format + 0.5 * truncation
truncation = -1（达到生成上限）或 0（未截断）
```

截断奖励先参与组内 GRPO advantage 计算，不是在 `A_GRPO` 之后直接
叠加 token advantage。稳定版 `checkpoint-1500` 的结果为：

| 数据集 | EM | F1 | Cover-EM | 回答率 |
|---|---:|---:|---:|---:|
| HotpotQA | 0.4061 | 0.5388 | 0.5047 | 97.96% |
| 2Wiki | 0.4825 | 0.5635 | 0.5476 | 98.23% |
| MuSiQue | 0.1556 | 0.2546 | 0.2218 | 94.54% |

它相对原始 `checkpoint-1000` 的三数据集 EM/F1 基本持平，仅 HotpotQA
Cover-EM 提高 0.62pt（双侧 p=0.0385）。结论是截断惩罚能防止后期
退化，但没有带来新的 EM/F1 增益。

主要产物：

- 原始训练：`03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/opsd_sft_q001_a003_3src_s3000_20260902/`
- ckpt500 三数据集评测：`data/eval_results/action_opsd_sft_ckpt500_3src_full_20260903/`
- ckpt1000 HotpotQA：`data/eval_results/hotpotqa/sft_opsd_ckpt1000_full7405_20260902/`
- ckpt1000 2Wiki：`data/eval_results/2wikimultihopqa/sft_opsd_ckpt1000_full12576_20260903/`
- ckpt1000 MuSiQue：`data/eval_results/musique/sft_opsd_ckpt1000_full2417_20260903/`
- 截断稳定版训练：`03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/opsd_sft_q001_a003_trunc05_3src_s1500_20260903/`

### 5.8 E13：External-teacher selective OPD

E13 与 OPSD 不同：teacher 是独立冻结的 Qwen2.5-14B，而不是同源
student 的特权视图；teacher 与 7B student 看到完全相同的 student
on-policy 查询、检索和证据历史，不看 gold answer、gold evidence 或
R3 query plan。Gold answer 只用于失败轨迹 gate：

```text
gate_i = 1[EM_i < 1]
A_t = gate_i * 0.01 * (log p_teacher(y_t) - log p_student(y_t))
```

当前使用 pure OPD，`opd_use_grpo_advantage=false`；F1/relevance/format
仅用于日志和 gate 辅助，不进入基础 GRPO advantage。

14B teacher 先使用与 7B 相同的 R3 SFT 协议训练 300-step LoRA，
`checkpoint-300` 的 train/eval loss 为 0.3007/0.1835。固定三数据集
各 50 条的 ceiling 中，14B teacher 相对 7B SFT 的 F1 宏平均提高
11.37pt，通过正式训练门槛。

正式优化后采用 `opd_sft14b_failed_em_spg2_s500_20260904`：

| 配置项 | 设置 |
|---|---|
| student | 旧 SFT `checkpoint-1650` |
| teacher | 14B SFT `checkpoint-300`，冻结 |
| 数据 | 三源 277,839 条，删除全部 `teacher_*` 字段 |
| 训练目标 | pure OPD；failed-EM gate；teacher coef 0.01 |
| 采样 | 5 generations；steps-per-generation 2 |
| 计划 | 500 step；每 50 step 保存 |

训练推进到 step340 后，模型服务 HTTP 连接被对端关闭，分布式训练退出；
完整 checkpoint 保存到 step300。此前 loss、grad norm、teacher KL 和
截断率均正常，但尚未运行离线全量评测，因此 E13 仍为 **P：待验证**，
不能写成 external-teacher OPD 已产生收益。

主要产物：

- 方案：`docs/opd_plan.md`
- teacher：`03_sapr_rag/saves/qwen2_5_14b/lora/sft_teacher/checkpoint-300/`
- ceiling：`data/eval_results/teacher_14b_sft_ceiling_50/ceiling_gate.json`
- 正式训练：`03_sapr_rag/saves/qwen2_5_7b/lora/opd/opd_sft14b_failed_em_spg2_s500_20260904/`
- 日志：`03_sapr_rag/scripts/opd/logs/opd_sft14b_failed_em_spg2_s500_20260904/`

### 5.9 E16：Canonical SFT→GRPO+分动作 OPSD

E16 用 E14 `sft_canonical_fp16/checkpoint-4150` 替换 E12 的旧 SFT
起点，重新验证已经观察到收益的 Query/Answer 分动作 OPSD。除训练
上限改为 1,000 step、每 250 step 保存外，目标函数复现原始有效 E12：

```text
R = F1 + 0.2 * relevance + 0.05 * format
A_t = A_GRPO
    + 0.01 * QueryMask_t  * (log p_query_teacher - log p_student)
    + 0.03 * AnswerMask_t * (log p_answer_teacher - log p_student)
```

Evidence teacher 保持关闭，Evidence Agent、Top-3、最大 6 turn、8 条
rollout、steps-per-generation 8、学习率 `1e-6` 与 E12 一致。为严格
复现原始有效 `checkpoint-1000`，E16 不启用截断惩罚。

正式 run 为
`opsd_canonical_sft_q001_a003_3src_s1000_20260905`，运行在
`worker4216626`：GPU0 检索、GPU2-6 训练、GPU7 rollout。截至本次核验
已稳定推进超过 step100；Query/Answer scoped KL 均非零，未见 NaN、
OOM、截断或服务错误。

E16 同时包含 GRPO 与 OPSD。单独比较 E16 与 E14 只能得到整套后训练
收益；后续必须补跑相同起点、数据、reward、采样和步数但关闭 teacher
的 GRPO-only control。只有 `E16 - matched GRPO control` 才能解释为
OPSD teacher 的独立贡献。

主要产物：

- 启动入口：`03_sapr_rag/scripts/grpo/run_canonical_sft_multi_opsd_s1000.sh`
- 训练：`03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/opsd_canonical_sft_q001_a003_3src_s1000_20260905/`
- 日志：`03_sapr_rag/scripts/grpo/logs/opsd_canonical_sft_q001_a003_3src_s1000_20260905/`

## 6. 结果总表与可信度

### 6.1 基础矩阵

| ID | 方法 | HotpotQA EM/F1/Cover-EM | 2Wiki EM/F1/Cover-EM | MuSiQue EM/F1/Cover-EM | 可信度 |
|---|---|---|---|---|---|
| B00 | 零样本 | .2040/.2730/.2680 | .0803/.1049/.1114 | .0728/.1070/.0956 | A |
| E00 | SFT | .0971/.2634/.5070 | .1018/.2515/.4488 | .0492/.1205/.1911 | A |
| B01 | 仅 DPO | .3492/.4563/.3999 | .3496/.4194/.4061 | .1200/.1935/.1452 | A；行为口径不同 |
| E01 | SFT+DPO | **.4008/.5233/.4693** | **.3915/.4688/.4452** | **.1667/.2477/.2069** | A |
| E14 | Canonical-answer SFT | **.4373/.5513/.4748** | .4051/.4513/.4188 | .1651/.2405/.1841 | A；新版 SFT 起点 |
| E15 | Canonical SFT→DPO | .4140/.5281/.4304 | .4187/.4656/.4230 | .1585/.2459/.1676 | A；仅 2Wiki 小幅改善 |

### 6.2 GRPO / OPSD

| ID | 实验 | HotpotQA EM/F1/Cover | 2Wiki EM/F1/Cover | MuSiQue EM/F1/Cover | 判断 |
|---|---|---|---|---|---|
| E02 | 旧版 GRPO | checkpoint-175 使用 dev 派生训练数据 | 未作为主结论 | 未作为主结论 | C，验证集泄露 |
| E03 | 全动作 OPSD | .2895/.4026/.3869 | 未完成 | 未完成 | C，动作错配、流程和旧 LoRA 混杂 |
| E04 | LoRA GRPO control | .1048/.2716/.5080 | 未完成 | 未完成 | A，HotpotQA 与旧 SFT 持平 |
| E05 | 全参数 GRPO ckpt2500 | .4003/.5071/.4493 | 未完成 | 未完成 | A，终止行为退化 |
| E06 | Reward-v2 ckpt300 | .1086/.2761/.5121 | 未完成 | 未完成 | B，重复未改善 |
| E07 | Reward-v3 ckpt500 | 200 条约 .105/.270/.520 | 未完成 | 未完成 | B，最终产物不完整 |
| E09 | Answer-only OPSD 25 步 | .4054/.5264/.4690 | 未完成 | 未完成 | A，正向但不显著 |
| E10 | Answer-only OPSD 100 步 | .4032/.5243/.4675 | 未完成 | 未完成 | A，较第 25 步回落 |
| E11 | SFT+DPO→分动作 OPSD | 未完成 | 未完成 | 未完成 | P，worker 回收前停止 |
| E12 | SFT→分动作 OPSD ckpt1000 | .4086/.5379/.4984 | **.4866/.5655/.5476** | .1547/.2546/.2180 | A，HotpotQA/2Wiki 有效 |
| E12-T | E12 + 截断惩罚 ckpt1500 | .4061/.5388/.5047 | .4825/.5635/.5476 | .1556/.2546/.2218 | A，防退化但无新增 EM/F1 |
| E13 | External-teacher selective OPD | 未评测 | 未评测 | 未评测 | P，step340 服务中断 |
| E16 | Canonical SFT→GRPO+分动作 OPSD | 训练中 | 训练中 | 训练中 | P，匹配 GRPO control 待补 |

A 表示留出集全量评测且链路已核验；B 表示诊断/小样本或产物不完整；C 表示存在泄露或关键流程问题；P 表示方法已实现但尚无结果。

## 7. 当前失败归因与方法启示

### 7.1 起点质量与后训练贡献必须分开

早期实验中，HotpotQA 超过 ReasonRAG 的主要贡献确实来自 SFT+DPO；
E09/E10 的 Answer-only OPSD 增量不显著。但 E12 后来证明，从旧 SFT
直接进行 Query/Answer 分动作 OPSD，也能在 HotpotQA 和 2Wiki 获得
显著收益。

E14 又进一步说明，SFT 最终答案目标本身是强混杂因素：只把 R3 长答案
替换为 canonical short answer，HotpotQA EM/F1 就提高到
0.4373/0.5513。因此后续不能再跨 SFT 版本直接归因后训练算法；E15/E16
必须分别与 E14 比较。

### 7.2 GRPO 的问题不是“训练没有生效”

全参数 GRPO 明显改变了检索覆盖、答案长度和在线训练奖励，说明策略模型确实发生了变化。失败主要来自目标错位：证据相关性奖励鼓励继续检索，轨迹级奖励无法把收益归因到具体查询，终止和持续检索目标相互冲突，而且同题采样组内经常缺少有效奖励差异。

因此，继续增大学习率、增加训练步数或更新更多参数，不足以解决当前问题。

### 7.3 查询质量是最明确的瓶颈

固定同一个 BGE+FAISS 检索器，在 HotpotQA 前 200 题上进行检索上限诊断：

| 查询来源 | Top-3 标准证据平均覆盖率 | Top-3 标准证据完全召回率 |
|---|---:|---:|
| 直接使用原问题 | 27.25% | 4.0% |
| SFT 模型生成的查询 | 44.75% | 20.5% |
| 直接使用标准支持文档标题查询 | 69.50% | 50.0% |

模型查询明显优于直接检索原问题，说明问题分解有效；但与接近理想上限的标题查询仍有明显差距。这是 E11 引入查询教师的直接依据。

### 7.4 OPSD 的核心是因果信息边界

旧版 OPSD 说明特权信息不是越多越好。当前设计原则是：

- 查询教师可以知道成功轨迹如何搜索，但不能看到标准答案；
- 答案教师可以看到标准答案和支持证据，但只能评价答案 token；
- 证据教师必须限制在学生策略当轮实际得到的候选文档中。

### 7.5 E16 必须补齐匹配 GRPO 对照

E12 和 E16 的训练目标都同时包含 GRPO advantage 与分动作 teacher
log-ratio。`E16 - E14` 只能衡量整套 `GRPO+OPSD` 后训练，不能单独证明
OPSD 有效。严格归因需要三组：

```text
E14：Canonical SFT
Control：E14 + GRPO，关闭全部 teacher
E16：E14 + 相同 GRPO + Query/Answer OPSD
```

Control 与 E16 必须使用相同数据顺序、reward、rollout、Evidence Agent、
步数、batch 和学习率；唯一差异是 teacher 是否开启。只有
`E16 - Control` 才是 OPSD 的独立贡献。

## 8. 实验限制与希望讨论的问题

### 8.1 必须正视的限制

1. SFT/DPO 数据主要复用外部工作。本项目贡献在协议统一、证据蒸馏和后续方法设计，不能把全部数据生成表述为自建。
2. R3-RAG 和 RAG-ProGuide 都使用 GPT-4o/MCTS，教师数据风格可能限制学生策略的探索空间。
3. BGE + 扩展 wiki18 与 ReasonRAG 论文检索配置不完全相同。
4. 证据抽取流程与原始文档回填流程都实际使用过，跨流程分数差异不能直接归因于训练算法。
5. E04/E05 的早期证据相关性奖励数据和实现存在已知缺陷。
6. E12/E16 的 1,000 step 仅覆盖约 0.018 epoch；短程无提升时，需要区分
   方法无效、监督覆盖不足和训练预算不足。
7. Query teacher 总覆盖率只有 18.1%，且 2Wiki 仅 6.5%；不同数据集实际
   接收的 Query OPSD 强度不均衡。
8. 证据动作 OPSD 尚无独立辅助训练批次，不能声称已经完成查询、证据、
   答案三类动作蒸馏。
9. E13 在 step340 因服务连接中断停止且尚未离线评测，不能把训练健康度
   写成 external-teacher OPD 的效果结论。
10. E16 的匹配 GRPO-only control 尚未完成，当前不能把未来的
    `E16 - E14` 增量全部归因于 OPSD。

### 8.2 希望重点讨论的方法问题

1. 查询教师应该来自单条成功轨迹、多条可替代路径，还是显式的下一跳信息增益？
2. 学生策略发生检索分叉后，如何避免机械模仿 R3-RAG 参考查询计划？
3. 是否需要逐步优势值、从当前步骤到轨迹结束的累计回报，或独立的查询过程奖励模型，以改善局部奖励归因？
4. 是否应该单独建模停止动作的“证据充分性”，而不是只通过最终答案和最大轮次间接学习？
5. 证据抽取智能体是否应从自由生成改为原文片段选择、引用标注或候选证据排序？
6. 查询动作使用轨迹偏好学习、答案动作使用特权信息蒸馏，是否比统一的“GRPO 加教师信号”更合理？
7. 是否应冻结唯一标准推理流程，并重新评测全部 SFT、DPO、GRPO、OPSD 检查点？

## 9. 项目内事实依据

本文不引用任何机器绝对路径。关键项目相对文件：

- 总台账：`docs/experiment_tracker.md`
- 基础结果：`docs/midterm_results.md`
- SFT/DPO 方案：`docs/sft_dpo_plan.md`
- GRPO/OPSD 流程：`docs/grpo_opsd_pipeline_overview.md`
- 正式检索服务部署：`docs/retrieval_service_gpu_runbook.md`
- 失败归因：`docs/grpo_opsd_badcase_attribution.md`
- SFT/DPO 数据构造：`03_sapr_rag/data/sft_build/`
- SFT/DPO 配置：`03_sapr_rag/scripts/train/`
- GRPO 奖励与调度器：`03_sapr_rag/scripts/grpo/plugin.py`
- 分动作 OPSD 数据：`03_sapr_rag/scripts/grpo/build_grpo_dataset_action_opsd.py`
- E12 三数据集评测：`data/eval_results/action_opsd_sft_ckpt500_3src_full_20260903/`
- E12 截断稳定版入口：`03_sapr_rag/scripts/grpo/run_sft_multi_opsd_trunc_s1500.sh`
- E13 external-teacher OPD：`docs/opd_plan.md`、`03_sapr_rag/scripts/opd/`
- E14 canonical SFT：`03_sapr_rag/scripts/train/sft_canonical_lora_fp16.yaml`
- E15 canonical DPO：`03_sapr_rag/scripts/train/dpo_canonical_lora.yaml`
- E16 canonical SFT→OPSD：`03_sapr_rag/scripts/grpo/run_canonical_sft_multi_opsd_s1000.sh`
- 评测指标：`03_sapr_rag/scripts/eval/score.py`
- 三源 OPSD 评测：`03_sapr_rag/scripts/eval/eval_action_opsd_3src.sh`
- 检索索引：`docs/index_build.md`
