# E11 Action-Scoped OPSD Live Audit

This file records live training evidence for E11 without replacing the
canonical experiment ledger in `docs/experiment_tracker.md`.

## Run

- Run: `opsd_multi_q001_a003_3src_s3000_actionfix_tmux_20260812`
- Worker: training node (identifier intentionally omitted)
- tmux: `opsd_multi_s3000`
- Auto eval tmux: `opsd_multi_eval_after_train`
- Output:
  `03_sapr_rag/saves/qwen2_5_7b/lora/grpo_opsd_action_scoped/opsd_multi_q001_a003_3src_s3000_actionfix_tmux_20260812/v0-20260812-064707`
- Dataset:
  `data/grpo/hotpotqa_2wiki_musique_train_multi_opsd.jsonl`
- Scope and coefficients: Query `0.01`, Evidence `0.00`, Answer `0.03`
- Devices: training GPU2-6, rollout GPU7, retrieval GPU0

The requested complete epoch is not feasible on this worker. The full
three-source training set would require about 55,568 optimizer steps, while
this run is intentionally capped at 3000 steps to leave time for checkpoint
sweep and full-dev evaluation before worker expiry.

## 2026-08-12 Live Status

Latest checked state:

- Time: `2026-08-12 15:43 +0800`
- Latest log step: `1124/3000`
- Latest metrics: loss `0.03978586`, grad norm `0.50828505`, KL `0.88094182`
- Log memory high-water mark: `84.57 GiB`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Error scan: no Traceback, CUDA OOM, Killed, or NaN found in current train
  and rollout logs. Early NCCL lines are initialization INFO, not failures.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 80% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 90161 / 97871 | 0% |
| 3 | train | 83747 / 97871 | 100% |
| 4 | train | 80885 / 97871 | 100% |
| 5 | train | 87593 / 97871 | 100% |
| 6 | train | 78467 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 0% |

GPU2 has only about 7.5 GiB free at the instant sampled. Memory has not risen
since the previous check, but this remains the main operational risk.

## Metric Windows

| Step window | Loss mean | Loss max | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-500 | 0.0365 | 0.8754 | 3540.41 | 0.6840 | 0.5385 | 0.4866 | 0.9645 | 0.0732 | 0.1345 | 1254 | 0.0000 | 27.26 |
| 501-1000 | 0.0394 | 0.0559 | 62.84 | 0.7090 | 0.5582 | 0.5133 | 0.9644 | 0.0829 | 0.2093 | 6555 | 0.0015 | 51.41 |
| 1001-1123 | 0.0499 | 0.6517 | 5628.60 | 0.7244 | 0.5646 | 0.5701 | 0.9143 | 0.0725 | 0.1799 | 7494 | 0.0512 | 84.57 |

Interpretation:

- Query and Answer scoped teacher signals are active.
- Rewards are not collapsing; F1 and relevance rewards are numerically higher
  in the 501-1123 range than in the first 500 steps, but these are online
  training rewards and cannot prove offline EM/F1 gains.
- Format reward dropped in the 1001-1123 window because very long completions
  became more frequent.

## Notable Events

- `checkpoint-500` is complete: trainer state step 500, adapter around 78 MiB,
  392 LoRA tensors, tensors finite.
- `checkpoint-1000` is complete: trainer state step 1000, adapter around 78 MiB,
  392 LoRA tensors, tensors finite.
- Step 1039 produced a very long completion: max length `7494`, clipped ratio
  `0.0375`.
- Step 1040 and 1046 then showed local gradient spikes:
  - step 1040: loss `0.48484054`, grad norm `229.85403442`
  - step 1046: loss `0.65170282`, grad norm `5628.59667969`
- After step 1046, the run recovered. By step 1124 loss is `0.03978586` and
  grad norm is `0.50828505`.
- Step 1103 raised the log memory high-water mark from `71.69 GiB` to
  `84.57 GiB`, likely due to repeated long completions in the 4k-5k token
  range after an earlier 7.5k token case.

Current conclusion:

The run is still valid to continue, but the 1000-step onward segment is no
longer "clean and boring". It should be treated as stable-after-spike: no
NaN/OOM and no persistent loss explosion, but significant long-output and
memory pressure must be watched until checkpoint-1500.

## Next Checks

1. Continue monitoring memory and latest loss/grad/KL until `checkpoint-1500`.
2. If realtime GPU memory approaches about 95 GiB on GPU2 or GPU5, ask for
   approval before stopping and restarting from the latest complete checkpoint
   with more conservative generation length.
3. After checkpoint-1500, audit adapter integrity, recent metric window,
   long-output rate, repeated-query rate, and Evidence Agent none rate.
4. Let `opsd_multi_eval_after_train` continue waiting for training completion;
   it will run checkpoint sweep, full dev evaluation, and paired bootstrap.
