# SAPR-RAG 中期方案：SFT + DPO 双阶段训练

> 起草于 2026-06-06。本文档是当前方案的"事实底稿"——背景、决策、规模、成本、待办全部固化在这里。任何后续讨论以本文为准。

## 0. 目标与边界

- **目标**：在中期答辩前产出可见的、完整的 RAG 多跳推理训练管线（SFT + DPO + 评估），让评委直观看到工作量。
- **不是目标**：方法新颖性、超过 SOTA、为审稿人服务。
- **核心赌注**：复现+扩展 ReasonRAG 的训练范式（双角色动作体系），用 R3-RAG 现成 cold-start 数据补一个 SFT 阶段，把 SFT-only / DPO-only / SFT+DPO 三种训练路线都跑出来摆对照表。

## 1. 来龙去脉（为什么是这个方案）

### 1.1 弃用上一版 reranker 路线
- 之前路线：自己造 (qid, step_idx) 级别的 reranker 训练数据，目标是在多跳检索每一步学一个 state-aware reranker。
- 卡点：
  1. step_gold 在集合型 subquery 上有 answer leakage（"演员表"被塌缩成单个演员）
  2. BGE 的输入只有 subquery，方法核心贡献模糊
  3. 重排序的实际增益在 smoke 数据上不明显
- 决定：**reranker 留作后续模块，不进中期答辩；** 当前主线切到 SFT+DPO。

### 1.2 为什么选 R3-RAG + ReasonRAG 拼接

|  | R3-RAG | ReasonRAG | 我们 |
|---|---|---|---|
| 公开数据 | cold-start SFT 数据 178k 行 ✓ | RAG_ProGuide DPO 数据 13.3k 偏好对 ✓ | 直接复用 |
| 训练阶段 | SFT + RL | **直接 DPO，无 SFT** | SFT(基于 R3) + DPO(基于 ProGuide) |
| 框架 | LLaMA-Factory | LLaMA-Factory | LLaMA-Factory |
| 基座 | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct |
| 语料 | Wikipedia 2018 (NQ-Tevatron 100词切) | Wikipedia 2018 | KILT Wikipedia 2018 (100词切) |
| 检索器 | E5/BGE/BM25 都测过 | E5 | **BGE-large-en-v1.5**（已建 64GB Flat 索引） |

**关键事实**：ReasonRAG 论文明确说"无 SFT 直接 DPO"。我们补上 SFT 阶段，本身就是个独立可看的工作量。

### 1.3 数据格式冲突 → 必须转写
- R3 用纯文本 step：`Step k: The problem analysis: ... The retrieval query: ... The retrieval documents: ...`
- ReasonRAG 用 tag：`<query>...</query>` / `<evidence>...</evidence>` / `<answer>...</answer>`
- SFT 输出格式必须跟 DPO 数据完全对齐，否则 SFT 学的东西 DPO 阶段全打掉。
- 解决方案：**转写脚本** [r3_to_reasonrag_sft.py](../03_sapr_rag/data/sft_build/r3_to_reasonrag_sft.py) 把 R3 step 翻译成 ReasonRAG tag 协议。

## 2. 核心设计决策（已拍板）

### 2.1 双角色动作体系（与 ReasonRAG 对齐）

ReasonRAG 原始有 5 套 system prompt，简化为 **2 套**（详见 ReasonRAG pipeline.py 第 155-205 行）：

| 角色 | system prompt | 学的能力 |
|---|---|---|
| **推理 agent** | `BEGIN_REASONING_PROMPT`（首/后续合并） | 看状态 → 决定下一步动作（出 query 或 answer） |
| **证据 extractor** | `DOCUMENT_ANALYSIS_PROMPT` | 看 query+文档 → 提炼 evidence |

不分"首轮/后续"两套推理 prompt，原因：(1) 输出协议完全相同；(2) ProGuide DPO 数据自己就混用；(3) R3 数据没有"首/后"标签，强分会引入噪声。

### 2.2 LLaMA-Factory 字段映射（已纠正）

之前错误：把 system prompt 塞进 `instruction`，导致它被当成 user 输入。
**正确写法**（alpaca 三段 + 独立 system 列）：

```json
{
  "system":      "<ReasonRAG system prompt>",
  "instruction": "Question: ... Previous Thoughts: ...",
  "input":       "",
  "output":      "...So the next query is <query>...</query>"
}
```

需要在 `dataset_info.json` 里注册条目并指明 `columns: {prompt: instruction, system: system, response: output}`。

### 2.3 evidence 来源：DeepSeek 蒸馏（方案 B），不再用截断原文

| 方案 | 做法 | 状态 |
|---|---|---|
| ~~A：截断原文 300 字~~ | smoke 阶段用过，质量差 | 已废弃 |
| **B：DeepSeek-V3 提炼**（实施中） | 用 ReasonRAG `DOCUMENT_ANALYSIS_PROMPT` 调 DeepSeek，每 (query, ref) 提炼一句 evidence | **正在跑全量缓存** |

### 2.4 cache 去重设计

- **cache key** = `sha256(query + "\n--R3REF--\n" + reference)[:32]`
- 同一 trajectory 的 step 在 R3 teacher-forcing 展开里会出现多次（一条 6 步轨迹的 Step 1 会出现 6 次：1 次 output + 5 次 historical instruction）
- **9.88 万唯一对** vs **21.7 万次出现**，平均复用 2.2 次 → 省掉一半 API 调用
- 后续转写脚本用同一 cache_key 反查，每个 step 出现都拿到同一份 evidence，**不丢信息**

### 2.5 "脑补步"全保留（A 方案）

R3 数据里 12% 的 step 含 "utilizing the model's parameter knowledge..."（检索失败时模型靠参数知识填空）。其中 6.7% 直接脑补出 final answer。

**决策**：全保留，不过滤。理由：
- 工作量目标下，简化优先
- R3 原作者的设计就是教 fallback 鲁棒性
- 后续 DPO 阶段的 process reward 可以矫正这部分

写进报告 limitation 章节即可。

## 3. 数据规模与产出

### 3.1 SFT 阶段数据
| 类型 | 数量 | 来源 |
|---|---|---|
| 推理样本（reasoning） | **~17.8 万** | R3 每行 → 1 条 |
| 证据抽取样本（evidence） | **~9.9 万** | 每唯一 cache → 1 条 |
| **合计** | **~27.7 万** | 按 90/10 切 train/dev |

### 3.2 DPO 阶段数据
| 类型 | 数量 | 来源 |
|---|---|---|
| 偏好对（chosen/rejected） | **13,289** | ReasonRAG RAG_ProGuide 直接复用 |

## 4. 评估实验设计：4-setting 对比表

| Setting | SFT | DPO | 备注 |
|---|---|---|---|
| **Zero-shot** | ✗ | ✗ | Qwen2.5-7B-Instruct 直接推理 |
| **SFT-only** | ✓ | ✗ | 我们补出来的新对照点 |
| **DPO-only** | ✗ | ✓ | 复现 ReasonRAG |
| **SFT+DPO** | ✓ | ✓ | 我们的主推路线 |

**评估基准**：HotpotQA / 2WikiMultiHopQA / MuSiQue dev set
**指标**：EM / F1
**报告**：4 setting × 3 数据集 = 12 个数字摆对照表

## 5. 实施进度

### 5.1 已完成
- [x] R3 数据下载持久化：`SAPR-RAG/data/raw/r3_coldstart.parquet`（141MB, 17.8 万行）
- [x] ProGuide 数据下载持久化：`SAPR-RAG/data/raw/proguide_dpo.parquet`（15MB, 13,289 偏好对）
- [x] 转写脚本 v1（smoke）：[r3_to_reasonrag_sft.py](../03_sapr_rag/data/sft_build/r3_to_reasonrag_sft.py)（100 条样例验证 OK）
- [x] evidence 提炼脚本：[evidence_distill.py](../03_sapr_rag/data/sft_build/evidence_distill.py)
- [x] **evidence cache 全量跑完**：98,877 次 API 调用 / 29.5 分钟 / 零错误 / 实际花费 $15.4 / 文件 167MB
- [x] **人工抽检 evidence 质量**：用户随机查阅 ~10 条均判定为合格，按二项分布 95% 置信下界 70%+，**通过，不重跑**

### 5.2 待办（按顺序）
- [ ] **改造转写脚本**：
  - 修字段映射（`system` 独立列、`instruction`=Question+History）
  - 接入 evidence cache（替换截断原文）
  - 同时产出推理样本 + 证据抽取样本
  - 全量跑产 27.7 万行 jsonl
- [ ] **写 `dataset_info.json`** 注册 sapr_reasoning + sapr_evidence
- [ ] **写 LLaMA-Factory SFT 配置**（基座 Qwen2.5-7B-Instruct，LoRA 还是 full FT 待定）
- [ ] **跑 SFT 训练 + 评估**
- [ ] **跑 DPO 训练**（基于 SFT checkpoint，用 ProGuide）
- [ ] **跑评估 4-setting 对比表**
- [ ] 撰写中期报告（数据章 / 训练章 / 评估章 / 局限章）

### 5.3 资源约束（待用户拍板）
- GPU 资源：可用几张？决定 LoRA vs Full FT
- 训练规模：全量 27.7 万 vs 采样
- evidence 提炼是否需要后续升级（如换更大模型重提炼）

## 6. 关键代码与数据位置

```
SAPR-RAG/
├── data/raw/                                        # 原始数据持久化
│   ├── r3_coldstart.parquet                         # R3 SFT 源（17.8 万行）
│   └── proguide_dpo.parquet                         # ReasonRAG DPO 数据
│
├── 03_sapr_rag/data/sft_build/
│   ├── r3_to_reasonrag_sft.py                       # 转写脚本（待改造）
│   ├── evidence_distill.py                          # evidence 蒸馏脚本
│   ├── dataset_info.json                            # LLaMA-Factory 数据集注册
│   └── out/
│       ├── evidence_cache.jsonl                     # 全量 evidence 缓存（生成中）
│       ├── sft_smoke.jsonl / sft_smoke_pretty.json  # smoke 100 条样例
│       └── evidence_distill.log                     # 后台日志
│
└── docs/sft_dpo_plan.md                             # 本文档
```

```
ReasonRAG/                                # GitHub 仓库已 clone
├── pipeline/reasonrag_pipeline.py        # 5 套 system prompt 来源
├── training_config/qwen_dpo.yaml         # DPO 配置参考
└── data_generation.py                    # MCTS 偏好对生成（参考用）

R3-RAG/                                   # GitHub 仓库已 clone
├── data/construct/02sample_from_dataset/main.py  # cold-start 拼接逻辑（每步 3 篇）
└── startup/retriever_service.py          # 默认 E5-base-v2 + Wikipedia-NQ
```

## 7. Limitations（写报告用）

1. **脑补步保留**：12% 的 SFT 样本含"utilizing parameter knowledge"，模型可能学到检索失败时直接编 answer 的捷径；后续依赖 DPO 矫正。
2. **evidence 一致性**：DeepSeek 蒸馏的 evidence 与 ReasonRAG 真实推理时模型自己抽的 evidence 不完全同分布。
3. **检索器漂移**：R3 cold-start 用 E5，我们 SFT 后推理用 BGE-large-en-v1.5，输入侧文档分布有微差。
4. **基准重叠**：HotpotQA / 2Wiki / MuSiQue 的训练集已被 R3 cold-start 用过 → 严格说 SFT-only 在这三个数据集上不算 zero-shot；DPO（ProGuide）数据来源未公开严格切分，可能也存在污染。报告里如实说明。
