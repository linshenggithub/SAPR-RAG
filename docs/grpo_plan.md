# SAPR-RAG 强化学习阶段方案：GRPO（ms-swift）

> 起草于 2026-06-08。本文档是 GRPO 阶段的"事实底稿"——选型、reward 设计、训练时序、风险、待办全部固化在这里。后续讨论以本文为准。
> 前置文档：[sft_dpo_plan.md](./sft_dpo_plan.md)（SFT 阶段）、[handoff_2026-06-07.md](./handoff_2026-06-07.md)（baseline + badcase 归档）。

## 0. 目标与边界

- **目标**：在 SFT checkpoint 基础上加一个 GRPO 阶段，把 cover_em / F1 相对 SFT-only 再推高，并为中期答辩提供"上了 RL"的故事线（DeepSeek-R1 同款范式，比 DPO 更有讲头）。
- **不是目标**：超过 SOTA、方法新颖性。
- **替代关系**：**用 GRPO 替代原计划的 DPO**。DPO 不再做（见 §1.2）。最终对照表从"Zero-shot / SFT / DPO / SFT+DPO"改为 **"Zero-shot / SFT-only / SFT+GRPO"**。
- **核心赌注**：reward shaping 能精准打击已归档的两类失败（hallucinated_hop、检索召回错），把 SFT 的"格式遵循 + 早停"增益再叠加一层"答案正确性"增益。

## 1. 选型决策（已拍板）

### 1.1 框架：ms-swift（不是 LLaMA-Factory）

**为什么不用 LLaMA-Factory**：已查证（2026-06-08），LLaMA-Factory 截至 v0.9.5 的 RLHF 方法清单为 PPO / DPO / KTO / ORPO / SimPO，**无原生 GRPO**。业界真实做法（如 DR-Tulu）是"LLaMA-Factory 跑 SFT + 另一框架跑 GRPO"。

**为什么用 ms-swift**：
- 原生支持 GRPO，内置 vllm rollout 集成 + reward 函数模板
- 团队已熟悉，上手成本最低
- 支持直接加载 LoRA adapter 作为初始 policy，与现有 SFT 产物无缝衔接

| 框架 | GRPO | 备注 |
|---|---|---|
| LLaMA-Factory | ❌ | 当前 SFT 用它，但不做 GRPO |
| **ms-swift** | ✅ | **本阶段选用** |
| Axolotl / TRL / verl / OpenRLHF | ✅ | 备选，不用 |

### 1.2 弃用 DPO 的理由
- DPO 需要构造 chosen/rejected step-level 偏好对（RAG_ProGuide 那套），数据工程量大
- GRPO 不需要偏好对，直接用 reward function，且故事性更强
- 资源有限，集中投一条 RL 路线

## 2. Reward 设计（核心）

### 2.0 实证依据（2026-06-08，基于 SFT 全量结果 + corpus 扫描）

reward 设计不再凭直觉，下面三轮实测（脚本 [retrieval_recall.py](../03_sapr_rag/scripts/eval/retrieval_recall.py)）直接定下了 relevance reward 的形态与约束。对齐 R3-RAG（EMNLP 2025 Findings，OpenRLHF+PPO，reward = answer correctness + document relevance + format）的"过程奖励"思想，但我们用纯规则的 gold-supporting 命中作信号，不引入 LLM verifier（保持可复现 + 对齐评估口径）。

**(a) SFT 检索对 gold supporting_facts 的命中率（全量 7405 题）**

| 口径 | 文档级命中率 | 题级全覆盖率 |
|---|---|---|
| title 精确匹配 | 0.424 | 0.194 |
| title + text fallback | 0.604 | 0.370 |

- title+text 联合口径比纯 title 高 18 个点 → **corpus 标题错位脏数据严重，reward 必须用联合口径**，否则给正确检索发错误负奖励。
- 答对组(n=3754) 文档级命中 0.722 vs 答错组(n=3651) 0.488 → **检索命中与答对强相关，relevance 信号有判别力**。

**(b) 答对组按检索覆盖度细分（n=3754）**

| 类别 | 数量 | 占比 |
|---|---|---|
| 全覆盖 | 1873 | 49.9% |
| 部分覆盖 | 1614 | 43.0% |
| 零命中（一篇 gold 都没检到却答对） | 267 | 7.1% |

- 零命中 267 题再切：**187 题(70%)答案其实出现在检索到的非-gold 文档里**（多为 HotpotQA 单篇绑定 + 脏数据漏判，不是真"无关文档蒙对"）；**80 题(30%≈全体 2%)纯参数化记忆答对**。
- 结论：真正靠记忆绕过检索的只有 ~2%，relevance reward 误伤面很小；但 answer reward 仍需保留主导地位，否则会教模型"蒙对就行"退化成 closed-book。

**(c) gold supporting facts 在检索 corpus 中的物理可达性（扫全量 2235 万条）**

| 指标 | 数值 |
|---|---|
| gold title 存在于 corpus | 12436/13783 = **90.2%** |
| 题级：至少缺 1 篇 gold | 1313 = **17.7%** |
| 题级：全部 gold 都不可检索 | 78 = **1.1%** |

- 缺失全是长尾页面（赛事/球队/年份页），符合 wiki18 dump 特性，非管线 bug。
- **物理不可检索的题仅 1.1%** → relevance reward 惩罚"不可能任务"的噪声极小，可在训练集预过滤剔除。
- 17.7% 缺 ≥1 篇 → **relevance reward 必须用连续命中比例，不能用"全覆盖才给分"的硬阈值**，否则这 17.7% 被一刀切判死。

**(d) SFT 答错组（n=3651）败因归因 + cover_em 误伤量化**

| 失败环节 | 数量 | 占答错组 |
|---|---|---|
| 检索零命中 | 982 | 26.9% |
| 检索部分覆盖 | 1805 | 49.4% |
| 检索全覆盖却答错（推理/抽取） | 864 | 23.7% |

- **检索相关失败 = 2787 题（76.3%）**，纯推理失败仅 23.7% → **检索召回是 GRPO 的首要优化目标，relevance reward 直接对症**（这是 reward 设计的核心依据，不是赌）。
- cover_em 误伤（纯规则估计）：反向包含 3.0% + 高 token-F1 但 cover_em=0 2.9%，并集 **4.9%** → 纯推理失败实际更低，进一步强化"钱花在检索上"；主指标仍用 cover_em（5% 低估对所有 setting 一致，相对对比可信，不引 LLM-judge）。

### 2.1 设计原则
- **主 reward 用连续值（F1）**，不用稀疏的 EM/cover_em 做主信号——GRPO 的 group-relative baseline 在稀疏 0/1 reward 上会退化
- **answer reward 保持主导**——实证 (b)：仅 ~2% 靠纯记忆答对，但若 answer 不主导会退化成 closed-book
- **辅助 reward 给小权重**，仅作兜底约束，避免 reward hacking
- **relevance 用 gold-supporting 命中、连续比例、联合口径**——实证 (a)(c) 直接约束：联合口径抗脏数据、连续比例消化 17.7% 部分可达题
- **不奖励"检索行为本身"**（次数/非空）——会激励无意义检索 / 主动死循环；奖励的是"检索到对答案有贡献的文档"，不是"检索了"

### 2.2 Reward 公式（草稿，待 sanity 后调 λ）

```
R_total = R_answer + λ_rel · R_relevance + λ_fmt · R_format

R_answer    = F1(pred_answer, gold)                      # 主信号，连续 0-1，复用 score.py:f1_score
R_relevance = hit / num_gold_supporting                  # 连续命中比例 0-1，过程奖励
R_format    = 1.0 if 轨迹严格符合 <query>/<evidence>/<answer> 标签协议 else 0.0
```

`R_relevance` 的命中判据 = **三级 OR**（任一成立即算命中该篇 gold）：
1. 检索 doc.title 与 gold supporting title 归一化精确匹配（HTML 解码 + dash 统一 + 小写）
2. gold supporting 句子文本出现在某检索 doc 正文里（抗标题错位脏数据）
3. （可选）最终 gold answer 文本出现在某检索 doc 正文里（抗 HotpotQA 单篇绑定）

建议初始权重：
- `λ_fmt = 0.05`（很小，SFT 已基本解决格式，仅兜底；过大会让输出僵化）
- `λ_rel = 0.2`（中等过程奖励；因误伤面小[实证 b]可略大，但不能盖过 F1）
- F1 主导

**训练集预处理**：剔除/mask 掉 78 题"全部 gold 不可检索"（实证 c），避免给 relevance 喂纯噪声负信号。

> 备注：原占位的 `R_traceable`（反 hallucinated_hop）被 `R_relevance` 取代。relevance 用正向"检到有贡献文档"引导，比负向惩罚幻觉实体更稳、更有数据支撑；hallucinated_hop 由 answer reward(答错自然低分) + relevance(没检到则过程分低) 间接覆盖。

### 2.3 明确不采用的 reward（反面清单）
| 设计 | 为什么不用 |
|---|---|
| 奖励"检索次数 / 检索了就加分" | 鼓励无意义查询、主动死循环 |
| 奖励"evidence 非空" | 鼓励抽假 evidence |
| EM/cover_em 做主 reward | 太稀疏，group baseline 退化 |
| relevance 用 title 精确匹配硬阈值 | 实证 (a)：脏数据低估 18 点；实证 (c)：17.7% 题天然达不到全覆盖 |
| LLM-as-verifier 打相关性（R3-RAG 原法） | 不可复现 + rollout 多一次 LLM 推理 + 与评估口径不一致 |

### 2.4 reward 函数复用点
- `F1`：直接复用 [score.py:f1_score](../03_sapr_rag/scripts/eval/score.py#L57) + [normalize_answer](../03_sapr_rag/scripts/eval/score.py#L19)，与评估口径严格一致（关键：训练 reward 和最终指标同口径）
- 格式校验：复用 [agent_infer.py](../03_sapr_rag/scripts/eval/agent_infer.py) 的 `RE_QUERY/RE_ANSWER/RE_EVIDENCE` 正则（L167-169）
- relevance 命中：复用 [retrieval_recall.py](../03_sapr_rag/scripts/eval/retrieval_recall.py) 的 `load_gold` / `collect_retrieved` / `norm_title` / `norm_text` + 三级 OR 判据；gold supporting 来自 `metadata.supporting_facts.title` + `metadata.context`
- **reward 用 Python 自定义函数写**（不是 shell），便于解析 multi-turn history + 取 trace 里每步 retrieve 的 docs

## 3. 训练时序与工程现实

### 3.1 GRPO rollout 与 SFT 的本质区别
- SFT：只读静态轨迹，离线
- GRPO：**online**，每个 step 要对每个 prompt 采样 K 条完整 multi-turn 轨迹（reason→retrieve→evidence × N 轮），每条轨迹里的 retrieve 都要**实时打 FAISS**
- 单步训练耗时 ≈ K × 完整 agent loop 时间。K 常取 4-8

### 3.2 检索必须先 daemon 化（强制前置）
- 现状：每个推理进程各自 load 一份 64GB CPU FAISS Flat 索引，8 进程并发把内存带宽打满（已实测 0.25 q/s，见 handoff §0.6）
- GRPO 训练时 rollout 也要检索，如果还是每进程一份索引，**会再次爆内存带宽**，训练吞吐惨不忍睹
- **前置任务**：把检索做成独立 daemon（单进程持有一份索引 + RPC/HTTP 接口），所有 rollout worker 共享。这件事在纯推理时"不划算"（之前讨论过），但**在 GRPO 训练时是刚需**

### 3.3 SFT checkpoint 衔接
- SFT 产物是 LoRA adapter：`03_sapr_rag/saves/qwen2_5_7b/lora/sft`
- ms-swift GRPO 直接加载该 LoRA 作为初始 policy，**不需要先 merge 回 base**

### 3.4 vllm 版本
- ms-swift 的 GRPO rollout 对 vllm 版本敏感，先 pin 一个已知可跑通版本，避免与当前推理 stack 冲突

## 4. 分阶段路线（里程碑）

| 阶段 | 内容 | 产出 | 时间预估 |
|---|---|---|---|
| **P0** | 等 zeroshot 跑完，铺 2Wiki/MuSiQue 的 SFT+zeroshot，拿到完整对照表 | 4-dataset × 2-setting 指标 | 1-2 天 |
| **P1** | 检索 daemon 化（单 FAISS 进程 + RPC，rollout/eval 共用） | 检索服务 + 客户端 | 1 天 |
| **P2** | ms-swift 装环境 + GRPO 最小 demo（100 条 sanity，能跑通不崩、reward 有信号） | 跑通的最小 pipeline | 1-2 天 |
| **P3** | HotpotQA 全量 GRPO + reward shaping 迭代（调 λ） | SFT+GRPO checkpoint | 5-7 天 |
| **P4** | 全量评估，与 SFT-only 对照，归档 | 最终对照表 | 1 天 |

### 里程碑判据
- P2 通过：GRPO 能复现 SFT baseline 水平（不退步）
- P3 通过：cover_em 比 SFT-only 高 ≥ 2-3 个点
- 论文级提升（>5 点）：不保证

## 5. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| reward hacking（模型钻 reward 空子） | 指标虚高但实际变差 | 反面清单 §2.3 + 小辅助权重 + 人工抽查 rollout |
| GRPO 不收敛 / 炼一周没结果 | 中期答辩开天窗 | 先用 SFT-only 兜底（已有 baseline），GRPO 作为加分项而非主线 |
| FAISS 内存带宽再次成瓶颈 | 训练吞吐过低 | P1 检索 daemon 化作为强制前置 |
| ms-swift × vllm 版本冲突 | 环境装不起来 | pin 版本，准备隔离的 conda env |
| reward 与评估口径不一致 | 训练目标偏离评估 | F1 reward 直接复用 score.py，同口径 |

## 6. 待办（TODO）

- [ ] P0：zeroshot 跑完 → 出 HotpotQA Zero-shot/SFT 对照
- [ ] P0：铺 2Wiki / MuSiQue（SFT + zeroshot）
- [ ] P1：检索 daemon 化（FAISS server + client）
- [ ] P2：ms-swift 环境 + GRPO 最小 demo
- [ ] P2：实现自定义 reward 函数（F1 + relevance + format，三级 OR 命中判据）
- [ ] P2：训练集预过滤——剔除 78 题"全部 gold 不可检索"
- [ ] P3：HotpotQA 全量 GRPO + λ 调参
- [ ] P4：全量评估 + 归档到 handoff
