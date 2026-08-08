# Experiment Tracker v2 — ClosureRAG

**Date**: 2026-05-30
**Status**: Planning phase, awaiting Gate 0

---

## Run Log

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

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-30 | SAPR-RAG v1 → ClosureRAG v2 | Phase 3-4 review: v1 四模块同等贡献 novelty 不硬，收缩为 Board + Claim Gate + Stop Closure |
| 2026-05-30 | 砍掉 branch rollback | 审稿人建议：复杂度高，会被要求和 MCTS 对比 |
| 2026-05-30 | Evidence selection 降级为 heuristic | 审稿人建议：LLM judge 打分太慢太贵，不是核心贡献 |
| 2026-05-30 | Board schema 从 6 字段砍到 4 字段 | 审稿人建议：最小充分状态，减少出错面 |

---

## Data Artifacts

| Artifact | Description | Status |
|----------|-------------|--------|
| Board annotations (500 trajectories) | Phase 1 数据积累 | pending |
| Claim-evidence pairs (~600) | Phase 1 数据积累 | pending |
| Closure labels (slot/claim/chain) | Phase 1 数据积累 | pending |
| Human validation (100-200) | 人工校验 Board + claim 标注 | pending |

---

## Key Files

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
