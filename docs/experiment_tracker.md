# SAPR-RAG 实验总记录

**首次建立**：2026-05-30

**最后更新**：2026-08-12

**当前主线**：SFT+DPO 起点上的分动作 OPSD

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

1. HotpotQA 上超过 ReasonRAG 的主要贡献来自 SFT+DPO，不是当前 Answer-only OPSD。
2. 严格 LoRA GRPO-control 与 SFT 基本持平；全参数 GRPO 会增加检索但破坏终止行为。
3. Reward-v2/v3 没有稳定提升 EM/F1/Cover-EM；D1b 证明 Query 生成质量与 Top-3 召回是主要瓶颈。
4. 修复 LoRA rollout 后，Answer-only OPSD 在 25 step 有轻微正向趋势，但相对 SFT+DPO 不显著；扩到 100 step 后回落。
5. 当前下一步是 Query/Answer 分动作 OPSD；Evidence OPSD 在独立 auxiliary batch 完成前保持关闭。

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
| E11 | Query/Answer 分动作 OPSD | SFT+DPO；HotpotQA+完整 2Wiki+MuSiQue 共 277,839 条；LoRA | Query 看 R3 搜索计划；Answer 看 gold；独立动作系数 | 3000-step 正式训练进行中；训练与全量评测尚未完成，不能下效果结论 | P | 下文“分动作新方案” |

### 三数据集基础实验矩阵

基础 4 个 setting 已在 HotpotQA、2Wiki、MuSiQue 全量 dev 上完成。
`Cover-EM` 反映字符串包含关系，`LLM-acc` 由 DeepSeek judge 判断事实等价；
二者同时报告是因为 DPO 会显著改变答案长度和表述风格。

| 数据集 | Setting | N | Cover-EM | LLM-acc | EM | F1 | Max-turn rate |
|---|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | Zero-shot | 7,405 | 0.2680 | 0.3380 | 0.2040 | 0.2730 | 45.1% |
| HotpotQA | SFT | 7,405 | **0.5070** | **0.6070** | 0.0971 | 0.2634 | 10.7% |
| HotpotQA | DPO-only | 7,405 | 0.3999 | 0.5356 | 0.3492 | 0.4563 | 未统一记录 |
| HotpotQA | SFT+DPO | 7,405 | 0.4693 | 0.6060 | **0.4008** | **0.5233** | **3.4%** |
| 2Wiki | Zero-shot | 12,576 | 0.1114 | 0.1178 | 0.0803 | 0.1049 | 66.6% |
| 2Wiki | SFT | 12,576 | **0.4488** | 0.4431 | 0.1018 | 0.2515 | 27.9% |
| 2Wiki | DPO-only | 12,576 | 0.4061 | 0.4249 | 0.3496 | 0.4194 | 未统一记录 |
| 2Wiki | SFT+DPO | 12,576 | 0.4452 | **0.4705** | **0.3915** | **0.4688** | **17.3%** |
| MuSiQue | Zero-shot | 2,417 | 0.0956 | 0.1129 | 0.0728 | 0.1070 | 63.6% |
| MuSiQue | SFT | 2,417 | 0.1911 | 0.2081 | 0.0492 | 0.1205 | 33.4% |
| MuSiQue | DPO-only | 2,417 | 0.1452 | 0.1957 | 0.1200 | 0.1935 | 未统一记录 |
| MuSiQue | SFT+DPO | 2,417 | **0.2069** | **0.2462** | **0.1667** | **0.2477** | **16.9%** |

基础矩阵的核心解释：

- SFT 最显著的贡献是学会多轮 RAG 协议和及时停止，三数据集 max-turn
  均大幅下降。
- SFT+DPO 在 HotpotQA 上 Cover-EM 比 SFT 低，但 LLM-acc 基本相同；
  EM/F1 的大幅提高主要反映回答更简洁、与 gold 字符串更对齐。
- 2Wiki 和 MuSiQue 的 LLM-acc 在 SFT+DPO 后继续提高，说明 DPO 没有
  破坏 MuSiQue 能力，尽管 DPO 数据本身不含 MuSiQue。

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
