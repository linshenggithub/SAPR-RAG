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

- Time: `2026-08-12 15:47 +0800`
- Latest log step: `1126/3000`
- Latest metrics: loss `0.04090714`, grad norm `0.60263944`, KL `0.83476059`
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
| 1047-1126 | 0.0404 | 0.0478 | 14.79 | 0.6997 | 0.5445 | 0.5517 | 0.8978 | 0.0732 | 0.1961 | 4935 | 0.0688 | 84.57 |

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
- After step 1046, the run recovered. In the step 1047-1126 window, loss max
  is `0.0478`, grad norm p95 is `3.04`, and grad norm max is `14.79`.
  By step 1126 loss is `0.04090714` and grad norm is `0.60263944`.
- Step 1103 raised the log memory high-water mark from `71.69 GiB` to
  `84.57 GiB`, likely due to repeated long completions in the 4k-5k token
  range after an earlier 7.5k token case.

Current conclusion:

The run is still valid to continue, but the 1000-step onward segment is no
longer "clean and boring". It should be treated as stable-after-spike: no
NaN/OOM and no persistent loss explosion, but significant long-output and
memory pressure must be watched until checkpoint-1500.

## 2026-08-12 17:37 +0800 Check

Latest checked state:

- Latest log step: `1234/3000`
- Latest metrics: loss `0.05453975`, grad norm `1.76999807`, KL `1.14095467`
- Log memory high-water mark: `89.21 GiB`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Error scan: no Traceback, CUDA OOM, Killed, RuntimeError, HTTP 5xx, or
  numeric NaN/Inf found in non-completion run logs and recent logging rows.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 0% |
| 3 | train | 35489 / 97871 | 100% |
| 4 | train | 95017 / 97871 | 100% |
| 5 | train | 54087 / 97871 | 100% |
| 6 | train | 48851 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 98% |

Process-level GPU query attributes the GPU4 high memory to the active training
process group, not an obvious unrelated leftover process. GPU4 is therefore
the current operational risk: only about 2.8 GiB is free at the instant
sampled.

Recent metric windows:

| Step window | Loss mean | Loss max | Grad p95 | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1150-1230 | 0.0642 | 1.6728 | 53.39 | 8544.73 | 0.6461 | 0.4918 | 0.5600 | 0.8447 | 0.0808 | 0.2323 | 7961 | 0.1184 | 89.21 |
| 1200-1230 | 0.0493 | 0.2428 | 493.94 | 811.06 | 0.6465 | 0.4944 | 0.5544 | 0.8242 | 0.0826 | 0.2491 | 7961 | 0.1325 | 89.21 |
| 1220-1230 | 0.0633 | 0.2428 | 811.06 | 811.06 | 0.6045 | 0.4545 | 0.5456 | 0.8175 | 0.1488 | 0.2311 | 7961 | 0.1350 | 89.21 |

Notable recent events:

- Step 1225 had a local spike: loss `0.24277671`, grad norm `811.05541992`.
  It recovered by step 1226: loss `0.03681803`, grad norm `0.40071872`.
- Step 1229 produced the longest completion so far in this run window:
  max completion length `7961`, clipped ratio `0.1625`; the next logged step
  recovered to loss `0.03999431`, grad norm `0.50205803`.
- Step 1233 had max completion length `5524`, clipped ratio `0.1`, with
  normal loss and grad norm.

Current conclusion:

The run remains valid to continue. The main concern is no longer an active
loss/gradient failure, but repeated long completions plus very high live memory
on one training GPU. Continue watching closely until `checkpoint-1500`; do not
claim this segment is clean, only that it has recovered after local spikes and
has not hit NaN/OOM.

## 2026-08-12 17:50 +0800 Check

Latest checked state:

- Latest log step: `1248/3000`
- Latest metrics: loss `0.04052184`, grad norm `1.05114019`, KL `0.9249294`
- Log memory high-water mark: `89.21 GiB`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Error scan: no Traceback, CUDA OOM, Killed, RuntimeError, HTTP 5xx, or
  numeric NaN/Inf found in non-completion run logs and recent logging rows.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 85% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 0% |
| 3 | train | 35489 / 97871 | 100% |
| 4 | train | 95017 / 97871 | 100% |
| 5 | train | 54087 / 97871 | 100% |
| 6 | train | 48851 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 86% |

GPU4 remains the active memory risk, but it did not increase compared with the
previous live check.

Recent metric windows:

| Step window | Loss mean | Loss max | Grad p95 | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1230-1248 | 0.0585 | 0.3638 | 2858.94 | 2858.94 | 0.5727 | 0.4209 | 0.5412 | 0.8722 | 0.0570 | 0.2465 | 5524 | 0.1056 | 89.21 |
| 1200-1248 | 0.0530 | 0.3638 | 493.94 | 2858.94 | 0.6188 | 0.4668 | 0.5494 | 0.8422 | 0.0712 | 0.2481 | 7961 | 0.1224 | 89.21 |
| 1150-1248 | 0.0634 | 1.6728 | 53.93 | 8544.73 | 0.6326 | 0.4788 | 0.5565 | 0.8497 | 0.0745 | 0.2349 | 7961 | 0.1161 | 89.21 |

Notable recent event:

- Step 1235 had a local spike: loss `0.36376807`, grad norm `2858.93603516`,
  max completion length `4312`, clipped ratio `0.1375`. It did not become a
  consecutive failure: by step 1248 loss is `0.04052184` and grad norm is
  `1.05114019`.

Current conclusion:

Continue the run. This is still a high-memory segment with intermittent local
gradient spikes, but no current evidence supports stopping before
`checkpoint-1500`.

## 2026-08-12 17:58 +0800 Check

Latest checked state:

- Latest log step: `1254/3000`
- Latest metrics: loss `0.04500557`, grad norm `0.45747888`, KL `0.93192823`
- Log memory high-water mark: `89.21 GiB`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Error scan: no Traceback, CUDA OOM, Killed, RuntimeError, HTTP 5xx, or
  numeric NaN/Inf found in non-completion run logs and recent logging rows.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 0% |
| 3 | train | 35489 / 97871 | 100% |
| 4 | train | 95017 / 97871 | 100% |
| 5 | train | 54087 / 97871 | 100% |
| 6 | train | 48851 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 0% |

GPU4 remains at the same high memory level as the previous two live checks.

Recent metric windows:

| Step window | Loss mean | Loss max | Grad p95 | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1230-1254 | 0.0592 | 0.3638 | 1622.99 | 2858.94 | 0.5912 | 0.4392 | 0.5455 | 0.8594 | 0.0599 | 0.2416 | 5524 | 0.1167 | 89.21 |
| 1200-1254 | 0.0539 | 0.3638 | 811.06 | 2858.94 | 0.6219 | 0.4699 | 0.5504 | 0.8398 | 0.0713 | 0.2457 | 7961 | 0.1255 | 89.21 |
| 1150-1254 | 0.0632 | 1.6728 | 53.93 | 8544.73 | 0.6334 | 0.4797 | 0.5566 | 0.8481 | 0.0743 | 0.2344 | 7961 | 0.1180 | 89.21 |

Notable recent event:

- Step 1252 had a local spike: loss `0.15652245`, grad norm `1622.98510742`.
  It recovered by step 1253-1254: step 1254 loss is `0.04500557`, grad norm
  is `0.45747888`.

Current conclusion:

Continue the run and keep watching until `checkpoint-1500`. The newest spike is
not consecutive and has not produced NaN/OOM, but the run remains high-risk
because GPU4 has only about 2.8 GiB free in live checks.

## 2026-08-12 18:07 +0800 Check

Latest checked state:

- Latest log step: `1264/3000`
- Latest metrics: loss `0.03535068`, grad norm `0.53469509`, KL `0.80310814`
- Log memory high-water mark: `89.21 GiB`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Error scan: no Traceback, CUDA OOM, Killed, RuntimeError, HTTP 5xx, or
  numeric NaN/Inf found in non-completion run logs and recent logging rows.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 0% |
| 3 | train | 35489 / 97871 | 100% |
| 4 | train | 95017 / 97871 | 100% |
| 5 | train | 54087 / 97871 | 100% |
| 6 | train | 48853 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 87% |

GPU4 remains at the same high memory level as the previous live checks.

Recent metric windows:

| Step window | Loss mean | Loss max | Grad p95 | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1252-1264 | 0.0485 | 0.1565 | 1622.99 | 1622.99 | 0.6570 | 0.5047 | 0.5556 | 0.8229 | 0.0737 | 0.5889 | 4406 | 0.1500 | 89.21 |
| 1230-1264 | 0.0535 | 0.3638 | 1622.99 | 2858.94 | 0.6093 | 0.4585 | 0.5418 | 0.8500 | 0.0626 | 0.3655 | 5524 | 0.1250 | 89.21 |
| 1200-1264 | 0.0517 | 0.3638 | 493.94 | 2858.94 | 0.6268 | 0.4753 | 0.5477 | 0.8379 | 0.0717 | 0.3109 | 7961 | 0.1285 | 89.21 |

Notable recent events:

- Step 1252 had a local gradient spike: loss `0.15652245`, grad norm
  `1622.98510742`, followed by recovery in later steps.
- Step 1263 had a scoped answer teacher KL spike: answer KL `2.7549437`,
  max completion length `4406`, clipped ratio `0.25`. This did not coincide
  with loss or grad failure: loss was `0.03393109`, grad norm was
  `1.19373214`, and step 1264 stayed normal.

Current conclusion:

Continue the run. The newest risk signal is an isolated answer-teacher KL spike
plus persistent high GPU4 memory, but there is still no NaN/OOM or consecutive
optimization failure before `checkpoint-1500`.

## 2026-08-12 18:16 +0800 Check

Latest checked state:

- Latest log step: `1275/3000`
- Latest metrics: loss `0.03845213`, grad norm `0.74714327`, KL `0.86663339`
- Log memory high-water mark: `89.21 GiB`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Error scan: no Traceback, CUDA OOM, Killed, RuntimeError, HTTP 5xx, or
  numeric NaN/Inf found in non-completion run logs and recent logging rows.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 0% |
| 3 | train | 35489 / 97871 | 100% |
| 4 | train | 95017 / 97871 | 100% |
| 5 | train | 54087 / 97871 | 100% |
| 6 | train | 48853 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 86% |

GPU4 remains at the same high memory level as previous live checks; it has not
continued to rise.

Recent metric windows:

| Step window | Loss mean | Loss max | Grad p95 | Grad max | Reward mean | F1 reward | Relevance | Format | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1264-1275 | 0.0409 | 0.0446 | 1.58 | 1.58 | 0.6911 | 0.5376 | 0.5563 | 0.8438 | 0.2769 | 5339 | 0.1406 | 89.21 |
| 1252-1275 | 0.0459 | 0.1565 | 1.77 | 1622.99 | 0.6706 | 0.5179 | 0.5558 | 0.8313 | 0.4641 | 5339 | 0.1463 | 89.21 |
| 1230-1275 | 0.0513 | 0.3638 | 6.75 | 2858.94 | 0.6249 | 0.4735 | 0.5445 | 0.8488 | 0.3487 | 5524 | 0.1280 | 89.21 |

Notable recent events:

- No new optimization spike after the previously recorded step 1263 answer-KL
  event. In the 1264-1275 window, loss max is `0.04462031` and grad norm max
  is `1.57969546`.
- Long outputs are still present but below the earlier extreme range: max
  completion length in the newest window is `5339`, with clipped ratio max
  `0.1625`.

Current conclusion:

Continue the run. The newest window is stable, but the run remains high-risk
because GPU4 is still near 95 GiB and `checkpoint-1500` has not yet been
created.

## 2026-08-12 19:28 +0800 Check

Latest checked state:

- Latest log step: `1336/3000`
- Latest metrics: loss `0.03246848`, grad norm `0.71420097`, KL `0.69480687`
- Log memory high-water mark: `89.21 GiB`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Error scan: no Traceback, CUDA OOM, Killed, RuntimeError, HTTP 5xx, or
  numeric NaN/Inf found in non-completion run logs and recent logging rows.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 0% |
| 3 | train | 35491 / 97871 | 100% |
| 4 | train | 95017 / 97871 | 100% |
| 5 | train | 59119 / 97871 | 100% |
| 6 | train | 58361 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 98% |

GPU4 remains the memory risk, but it has not increased beyond the prior live
checks.

Notable recent events:

- Step 1333 had a local gradient lift: loss `0.06854999`, grad norm
  `226.52052307`; this is well below the earlier extreme gradient spikes and
  did not trigger NaN/OOM.
- Step 1334 had a large isolated loss spike: loss `42.12660217`, grad norm
  `46.53051376`, KL `1.00670241`. The following steps recovered:
  step 1335 loss `0.0423644`, grad norm `0.85484123`; step 1336 loss
  `0.03246848`, grad norm `0.71420097`.
- `checkpoint-1500` has not yet been created.

Current conclusion:

Continue the run, but treat step 1334 as a significant isolated instability
event to revisit during the `checkpoint-1500` audit. There is still no evidence
of consecutive optimization failure, NaN, or OOM.

## 2026-08-12 20:43 +0800 Check

Latest checked state:

- Latest log step: `1400/3000`
- Latest metrics: loss `0.04877957`, grad norm `1.13171065`, KL `1.1147145`
- Log memory high-water mark: `89.21 GiB`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`
- `checkpoint-1500`: not created yet
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Error scan on `logging.jsonl`: no Traceback, CUDA OOM, Killed,
  RuntimeError, HTTP 5xx, or explicit Exception found.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 0% |
| 3 | train | 54591 / 97871 | 100% |
| 4 | train | 65883 / 97871 | 100% |
| 5 | train | 59119 / 97871 | 100% |
| 6 | train | 58361 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 88% |

Recent metric windows:

| Step window | Loss mean | Loss max | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1330-1398 | 1.0302 | 42.1266 | 6404.63 | 0.6292 | 0.4839 | 0.5223 | 0.8173 | 0.0793 | 0.1985 | 7642 | 0.1588 | 89.21 |
| 1360-1398 | 0.7104 | 23.9177 | 6404.63 | 0.5991 | 0.4590 | 0.4981 | 0.8099 | 0.0843 | 0.2086 | 7642 | 0.1664 | 89.21 |
| 1379-1398 | 0.1262 | 1.5008 | 6404.63 | 0.5907 | 0.4560 | 0.4698 | 0.8138 | 0.1054 | 0.1863 | 7642 | 0.1575 | 89.21 |
| 1391-1398 | 0.2539 | 1.5008 | 6404.63 | 0.5976 | 0.4634 | 0.4703 | 0.8031 | 0.0214 | 0.2303 | 4323 | 0.1625 | 89.21 |

Notable recent events:

- Step 1365: loss `0.581557`, grad norm `1768.24`; recovered in the
  following steps.
- Step 1375 and 1377 had long completions (`7454` and `7528` tokens).
- Step 1378 had another large isolated loss spike: loss `23.917696`, grad norm
  `994.212`; steps 1379-1380 returned to normal loss and gradient range.
- Step 1385 had the longest recent completion (`7642` tokens), with low reward
  `0.27708334`, but normal loss and gradient.
- Step 1391 had a gradient spike: loss `1.50079966`, grad norm `6404.63037109`;
  steps 1392-1395 recovered to loss around `0.04`.
- Step 1396 had a smaller local gradient lift: loss `0.28735527`, grad norm
  `209.10591125`; steps 1397-1400 returned to normal loss and gradient range.

Current conclusion:

Continue the run. The instability pattern is still isolated spike followed by
recovery, not consecutive failure. GPU4 has dropped from the earlier near-95 GiB
pressure to about 66 GiB in the latest realtime check, so immediate memory risk
is lower than the 19:28 check. Revisit steps 1333-1341, 1365, 1375-1378, 1385,
1391, and 1396 during the `checkpoint-1500` audit.

## 2026-08-12 22:35 +0800 Checkpoint-1500 Audit

Checkpoint state:

- Latest checked log step: `1502/3000`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`
- `checkpoint-1500` size: `79M`
- `trainer_state.json`: present, `global_step=1500`, `max_steps=3000`,
  `epoch=0.02699492495410863`, `log_history_len=1500`
- LoRA adapter files present:
  - `adapter_model.safetensors` (`80792880` bytes)
  - `adapter_config.json` (`1055` bytes)
  - `additional_config.json`, `args.json`, `training_args.bin`, `README.md`
- `adapter_model.safetensors` was opened successfully with `safetensors`;
  tensor count: `392`
- Error scan on `logging.jsonl`: no Traceback, CUDA OOM, Killed,
  RuntimeError, HTTP 5xx, or explicit Exception found.
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 0% |
| 3 | train | 69019 / 97871 | 100% |
| 4 | train | 65883 / 97871 | 100% |
| 5 | train | 59119 / 97871 | 100% |
| 6 | train | 58361 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 83% |

Recent metric windows:

| Step window | Loss mean | Loss max | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1330-1500 | 0.4638 | 42.1266 | 6404.63 | 0.6357 | 0.4868 | 0.5398 | 0.8190 | 0.0727 | 0.2145 | 7752 | 0.1618 | 89.21 |
| 1440-1500 | 0.1080 | 3.4468 | 2264.69 | 0.6133 | 0.4610 | 0.5572 | 0.8179 | 0.0651 | 0.2736 | 7698 | 0.1633 | 89.21 |
| 1470-1500 | 0.1632 | 3.4468 | 2264.69 | 0.6067 | 0.4534 | 0.5613 | 0.8208 | 0.0748 | 0.2841 | 7631 | 0.1683 | 89.21 |
| 1490-1500 | 0.0429 | 0.0802 | 2.78 | 0.6930 | 0.5265 | 0.6225 | 0.8400 | 0.0372 | 0.2153 | 7631 | 0.1525 | 89.21 |

Notable events since the prior audit:

- Step 1421 had a long completion (`7752` tokens), but loss `0.029807` and
  grad norm `0.662` remained normal.
- Step 1433 and step 1435 had long completions (`7737` and `7639` tokens);
  both recovered without optimization instability.
- Step 1444 had a local gradient/loss spike: loss `0.352474`, grad norm
  `686.925`; subsequent steps returned to normal.
- Step 1471 had a local spike: loss `3.446822`, grad norm `281.115`,
  completion max length `4493`; steps 1472-1474 recovered.
- Step 1483 had a local gradient spike: loss `0.418516`, grad norm
  `2264.692`; steps 1484-1486 recovered.
- Step 1487 and step 1497 had long completions (`7529` and `7631` tokens);
  neither led to consecutive instability.
- Step 1500 itself is acceptable: loss `0.08015592`, grad norm `2.78289723`,
  KL `0.98401012`.

Current conclusion:

`checkpoint-1500` is structurally complete and readable. The training continues
normally after the save point. The run still contains isolated long-output and
gradient-spike events, but there is no evidence of consecutive failure, NaN, or
OOM. Continue monitoring toward `checkpoint-2000`.

## 2026-08-12 23:31 +0800 Post-1500 Patrol

Current state:

- Latest checked log step: `1554/3000`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`
- `checkpoint-2000` has not been created yet.
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Post-training orchestration process alive:
  `orchestrate_action_opsd_3src_after_train.sh`
- Error scan on `logging.jsonl`: no Traceback, CUDA OOM, Killed,
  RuntimeError, HTTP 5xx, explicit Exception, or NaN found.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 0% |
| 3 | train | 69019 / 97871 | 100% |
| 4 | train | 65883 / 97871 | 100% |
| 5 | train | 59119 / 97871 | 100% |
| 6 | train | 58361 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 88% |

Recent metric windows:

| Step window | Loss mean | Loss max | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1505-1554 | 0.0428 | 0.0924 | 776.16 | 0.6888 | 0.5291 | 0.5885 | 0.8385 | 0.0778 | 0.1980 | 7620 | 0.1455 | 89.21 |
| 1535-1554 | 0.0438 | 0.0924 | 418.94 | 0.7185 | 0.5520 | 0.6244 | 0.8325 | 0.0668 | 0.2301 | 7620 | 0.1488 | 89.21 |
| 1545-1554 | 0.0424 | 0.0497 | 1.61 | 0.7808 | 0.6117 | 0.6338 | 0.8475 | 0.1153 | 0.1522 | 5297 | 0.1300 | 89.21 |

Notable post-1500 events:

- Step 1501 had a long completion (`7457` tokens).
- Step 1524 had a local gradient spike: grad norm `776.162`; later steps
  recovered.
- Step 1539 had a local gradient spike plus long completion: grad norm
  `418.94`, completion max length `7620`; steps 1545-1554 returned to normal
  gradient range (`0.52` to `1.61`).

Current conclusion:

Training is still healthy after `checkpoint-1500`. The post-1500 spikes are
isolated and recovered. No new checkpoint audit is due until `checkpoint-2000`.

## 2026-08-12 23:39 +0800 Post-1500 Patrol

Current state:

- Latest checked log step: `1562/3000`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`
- `checkpoint-2000` has not been created yet.
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Post-training orchestration process alive:
  `orchestrate_action_opsd_3src_after_train.sh`
- Error scan on `logging.jsonl`: no Traceback, CUDA OOM, Killed,
  RuntimeError, HTTP 5xx, explicit Exception, or NaN found.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 100% |
| 3 | train | 69019 / 97871 | 100% |
| 4 | train | 65883 / 97871 | 100% |
| 5 | train | 59119 / 97871 | 100% |
| 6 | train | 58361 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 0% |

Recent metric windows:

| Step window | Loss mean | Loss max | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1511-1560 | 0.1540 | 5.4848 | 776.16 | 0.7109 | 0.5502 | 0.5933 | 0.8395 | 0.0778 | 0.1826 | 7630 | 0.1420 | 89.21 |
| 1541-1560 | 0.3197 | 5.4848 | 693.70 | 0.7487 | 0.5800 | 0.6325 | 0.8438 | 0.1153 | 0.1956 | 7630 | 0.1363 | 89.21 |
| 1551-1560 | 0.5964 | 5.4848 | 693.70 | 0.7499 | 0.5808 | 0.6388 | 0.8275 | 0.1274 | 0.1800 | 7630 | 0.1525 | 89.21 |

Notable post-1500 events added in this patrol:

- Step 1555 had a long completion and loss spike: loss `5.48483944`, grad norm
  `10.11456203`, max completion length `7630`, KL `0.96633948`.
- Step 1557 had a local gradient spike: loss `0.15349017`, grad norm
  `693.69769287`, max completion length `4242`, KL `0.95926847`.
- Recovery was observed by steps 1560-1562:
  - step 1560: loss `0.03591482`, grad norm `0.50760704`, KL `0.69611198`
  - step 1561: loss `0.04111473`, grad norm `4.63197374`, KL `0.80290408`
  - step 1562: loss `0.0597424`, grad norm `3.48724222`, KL `0.89481458`

Current conclusion:

Training remains valid to continue. The newest events are another isolated
long-output/loss spike and a local gradient spike, both followed by recovery.
There is still no evidence of NaN, OOM, or consecutive optimization failure.
Continue monitoring toward `checkpoint-2000`.

## 2026-08-12 23:45 +0800 Post-1500 Patrol

Current state:

- Latest checked log step: `1566/3000`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`
- `checkpoint-2000` has not been created yet.
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Post-training orchestration process alive:
  `orchestrate_action_opsd_3src_after_train.sh`
- Error scan on `logging.jsonl`: no Traceback, CUDA OOM, Killed,
  RuntimeError, HTTP 5xx, explicit Exception, or NaN found.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 0% |
| 3 | train | 69019 / 97871 | 100% |
| 4 | train | 65883 / 97871 | 100% |
| 5 | train | 59119 / 97871 | 100% |
| 6 | train | 58361 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 88% |

Recent metric windows:

| Step window | Loss mean | Loss max | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1517-1566 | 0.1548 | 5.4848 | 776.16 | 0.7030 | 0.5419 | 0.5968 | 0.8350 | 0.0811 | 0.1917 | 7630 | 0.1450 | 89.21 |
| 1547-1566 | 0.3215 | 5.4848 | 693.70 | 0.7478 | 0.5819 | 0.6181 | 0.8463 | 0.1041 | 0.1728 | 7630 | 0.1350 | 89.21 |
| 1557-1566 | 0.0559 | 0.1535 | 693.70 | 0.6978 | 0.5396 | 0.5800 | 0.8425 | 0.0925 | 0.1789 | 5266 | 0.1425 | 89.21 |

Newest recovery check:

- After the step 1555 long-output/loss spike and the step 1557 gradient spike,
  steps 1558-1566 did not show a new large loss spike or long completion above
  7400 tokens.
- Latest step 1566: loss `0.04300981`, grad norm `1.20916224`, KL
  `0.77028842`.
- Recent max completion after step 1557 was `5266` tokens at step 1561.
- `checkpoint-2000` is still roughly 434 optimizer steps away from the latest
  checked log step.

Current conclusion:

Training remains valid to continue. The newest evidence strengthens the recovery
claim after the step 1555 and 1557 events: no consecutive optimization failure,
no NaN/OOM, and no new extreme long-output event through step 1566. Continue
monitoring toward `checkpoint-2000`.

## 2026-08-12 23:50 +0800 Post-1500 Patrol

Current state:

- Latest checked log step: `1572/3000`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`
- `checkpoint-2000` has not been created yet.
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Post-training orchestration process alive:
  `orchestrate_action_opsd_3src_after_train.sh`
- Error scan on `logging.jsonl`: no Traceback, CUDA OOM, Killed,
  RuntimeError, HTTP 5xx, explicit Exception, or NaN found.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 100% |
| 3 | train | 69019 / 97871 | 100% |
| 4 | train | 65883 / 97871 | 100% |
| 5 | train | 59119 / 97871 | 100% |
| 6 | train | 58361 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 0% |

Recent metric windows:

| Step window | Loss mean | Loss max | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1521-1570 | 0.1547 | 5.4848 | 776.16 | 0.7003 | 0.5368 | 0.6079 | 0.8390 | 0.0811 | 0.1973 | 7630 | 0.1440 | 89.21 |
| 1551-1570 | 0.3202 | 5.4848 | 693.70 | 0.7005 | 0.5392 | 0.5966 | 0.8388 | 0.1016 | 0.1855 | 7630 | 0.1488 | 89.21 |
| 1561-1570 | 0.0441 | 0.0597 | 4.63 | 0.6510 | 0.4977 | 0.5544 | 0.8500 | 0.0930 | 0.1911 | 5266 | 0.1450 | 89.21 |

Newest recovery check:

- Steps 1561-1572 remained stable after the earlier 1555/1557 events.
- Latest steps:
  - step 1570: loss `0.03882753`, grad norm `0.61345595`, KL `0.84269477`
  - step 1571: loss `0.0394352`, grad norm `1.01030588`, KL `0.72125477`
  - step 1572: loss `0.04547619`, grad norm `0.88543421`, KL `0.95487028`
- No new post-1557 completion above `7400` tokens was observed in the checked
  window.
- `checkpoint-2000` is still roughly 428 optimizer steps away from the latest
  checked log step.

Current conclusion:

Training remains valid to continue. The post-1557 recovery now spans through
step 1572 with normal loss, gradient, and KL. No checkpoint audit is due until
`checkpoint-2000` appears.

## 2026-08-12 23:56 +0800 Post-1500 Patrol

Current state:

- Latest checked log step: `1578/3000`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`
- `checkpoint-2000` has not been created yet.
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Post-training orchestration process alive:
  `orchestrate_action_opsd_3src_after_train.sh`
- Error scan on `logging.jsonl`: no Traceback, CUDA OOM, Killed,
  RuntimeError, HTTP 5xx, explicit Exception, or NaN found.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 100% |
| 3 | train | 69019 / 97871 | 100% |
| 4 | train | 65883 / 97871 | 100% |
| 5 | train | 59119 / 97871 | 100% |
| 6 | train | 58361 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 0% |

Recent metric windows:

| Step window | Loss mean | Loss max | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1527-1576 | 0.1553 | 5.4848 | 693.70 | 0.6987 | 0.5351 | 0.6071 | 0.8430 | 0.0858 | 0.1913 | 7630 | 0.1425 | 89.21 |
| 1557-1576 | 0.0506 | 0.1535 | 693.70 | 0.6351 | 0.4807 | 0.5609 | 0.8438 | 0.1048 | 0.1846 | 5266 | 0.1438 | 89.21 |
| 1567-1576 | 0.0453 | 0.0803 | 336.92 | 0.5724 | 0.4218 | 0.5419 | 0.8450 | 0.1171 | 0.1902 | 4443 | 0.1450 | 89.21 |

Newest recovery check:

- Step 1574 had a local gradient lift: loss `0.08030923`, grad norm
  `336.91897583`, KL `0.55483723`; it recovered at steps 1575-1578.
- Latest steps:
  - step 1575: loss `0.04117112`, grad norm `0.57995945`, KL `0.91545975`
  - step 1576: loss `0.04585421`, grad norm `0.50249845`, KL `0.94847342`
  - step 1577: loss `0.03679916`, grad norm `0.69263661`, KL `0.77138393`
  - step 1578: loss `0.04302807`, grad norm `0.49891374`, KL `0.97538325`
- No new post-1557 completion above `7400` tokens was observed in the checked
  window.
- `checkpoint-2000` is still roughly 422 optimizer steps away from the latest
  checked log step.

Current conclusion:

Training remains valid to continue. The step 1574 gradient lift is isolated and
was followed by normal loss, gradient, and KL. No checkpoint audit is due until
`checkpoint-2000` appears.

## 2026-08-13 00:02 +0800 Post-1500 Patrol

Current state:

- Latest checked log step: `1582/3000`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`
- `checkpoint-2000` has not been created yet.
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Post-training orchestration process alive:
  `orchestrate_action_opsd_3src_after_train.sh`
- Error scan on `logging.jsonl`: no Traceback, CUDA OOM, Killed,
  RuntimeError, HTTP 5xx, explicit Exception, or NaN found.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 81% |
| 3 | train | 69019 / 97871 | 100% |
| 4 | train | 65883 / 97871 | 100% |
| 5 | train | 59119 / 97871 | 100% |
| 6 | train | 58361 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 0% |

Recent metric windows:

| Step window | Loss mean | Loss max | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1533-1582 | 0.1550 | 5.4848 | 693.70 | 0.7011 | 0.5414 | 0.5873 | 0.8445 | 0.0893 | 0.1905 | 7630 | 0.1400 | 89.21 |
| 1563-1582 | 0.0440 | 0.0803 | 336.92 | 0.6469 | 0.4979 | 0.5319 | 0.8525 | 0.1074 | 0.2007 | 4455 | 0.1363 | 89.21 |
| 1573-1582 | 0.0455 | 0.0803 | 336.92 | 0.6722 | 0.5293 | 0.5013 | 0.8525 | 0.1291 | 0.1871 | 4455 | 0.1300 | 89.21 |

Newest recovery check:

- No new large loss, high-KL, OOM, or extreme long-output event appeared through
  step 1582.
- The last notable local event remains step 1574: loss `0.08030923`, grad norm
  `336.91897583`; later steps stayed normal.
- Latest steps:
  - step 1579: loss `0.04477176`, grad norm `0.66915089`, KL `1.01208322`
  - step 1580: loss `0.04644791`, grad norm `0.36723384`, KL `1.06115307`
  - step 1581: loss `0.0437583`, grad norm `1.43791831`, KL `0.82005951`
  - step 1582: loss `0.0344053`, grad norm `1.25949836`, KL `0.54080966`
- `checkpoint-2000` is still roughly 418 optimizer steps away from the latest
  checked log step.

Current conclusion:

Training remains valid to continue. The run is stable after the step 1574 local
spike, with normal recent loss/grad/KL and no new extreme long-output event. No
checkpoint audit is due until `checkpoint-2000` appears.

## 2026-08-13 00:07 +0800 Post-1500 Patrol

Current state:

- Latest checked log step: `1588/3000`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`
- `checkpoint-2000` has not been created yet.
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Post-training orchestration process alive:
  `orchestrate_action_opsd_3src_after_train.sh`
- Error scan on `logging.jsonl`: no Traceback, CUDA OOM, Killed,
  RuntimeError, HTTP 5xx, explicit Exception, or NaN found.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 100% |
| 3 | train | 69019 / 97871 | 100% |
| 4 | train | 65883 / 97871 | 100% |
| 5 | train | 59119 / 97871 | 100% |
| 6 | train | 58361 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 0% |

Recent metric windows:

| Step window | Loss mean | Loss max | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1537-1586 | 0.1550 | 5.4848 | 693.70 | 0.6892 | 0.5326 | 0.5723 | 0.8420 | 0.0953 | 0.2578 | 7630 | 0.1435 | 89.21 |
| 1567-1586 | 0.0432 | 0.0803 | 336.92 | 0.6304 | 0.4848 | 0.5171 | 0.8438 | 0.1011 | 0.3158 | 4455 | 0.1450 | 89.21 |
| 1577-1586 | 0.0411 | 0.0483 | 1.78 | 0.6883 | 0.5477 | 0.4923 | 0.8425 | 0.0905 | 0.4414 | 4455 | 0.1450 | 89.21 |

Newest recovery check:

- The latest local spike remains step 1574; steps 1577-1588 stayed in a normal
  loss/gradient/KL range.
- Latest steps:
  - step 1586: loss `0.0362389`, grad norm `0.80878466`, KL `0.68414557`
  - step 1587: loss `0.04523862`, grad norm `2.12357426`, KL `0.70303874`
  - step 1588: loss `0.04213884`, grad norm `1.39239872`, KL `0.83890045`
- No new completion above `7400` tokens was observed in the checked window.
- `checkpoint-2000` is still roughly 412 optimizer steps away from the latest
  checked log step.

Current conclusion:

Training remains valid to continue. The recent window is stable and there is no
new evidence of consecutive optimization failure, NaN, OOM, or extreme long
output. No checkpoint audit is due until `checkpoint-2000` appears.

## 2026-08-13 00:13 +0800 Post-1500 Patrol

Current state:

- Latest checked log step: `1592/3000`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`
- `checkpoint-2000` has not been created yet.
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Post-training orchestration process alive:
  `orchestrate_action_opsd_3src_after_train.sh`
- Error scan on `logging.jsonl`: no Traceback, CUDA OOM, Killed,
  RuntimeError, HTTP 5xx, explicit Exception, or NaN found.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 43% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 0% |
| 3 | train | 69019 / 97871 | 100% |
| 4 | train | 65883 / 97871 | 100% |
| 5 | train | 59119 / 97871 | 100% |
| 6 | train | 58361 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 88% |

Recent metric windows:

| Step window | Loss mean | Loss max | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1543-1592 | 1.1093 | 47.8138 | 65603.20 | 0.6814 | 0.5287 | 0.5553 | 0.8335 | 0.1047 | 0.2534 | 7630 | 0.1515 | 89.21 |
| 1573-1592 | 2.4318 | 47.8138 | 65603.20 | 0.6375 | 0.4990 | 0.4877 | 0.8188 | 0.1070 | 0.3477 | 4455 | 0.1663 | 89.21 |
| 1583-1592 | 4.8180 | 47.8138 | 65603.20 | 0.6028 | 0.4687 | 0.4742 | 0.7850 | 0.0960 | 0.5084 | 4415 | 0.2025 | 89.21 |

Newest recovery check:

- Step 1590 had a significant isolated optimization spike: loss `47.81381226`,
  grad norm `65603.203125`, KL `0.63388205`. It did not coincide with an
  extreme logged completion length because step 1590 has no completion metrics.
- Recovery was observed immediately after the spike:
  - step 1591: loss `0.03803557`, grad norm `0.62595141`, KL `0.73190643`,
    max completion `4277`
  - step 1592: loss `0.0420973`, grad norm `0.70280594`, KL `0.87111439`
- No NaN/OOM/runtime error was found in the log scan, and the tmux session kept
  running after the spike.
- `checkpoint-2000` is still roughly 408 optimizer steps away from the latest
  checked log step.

Current conclusion:

Continue training, but carry step 1590 into the `checkpoint-2000` audit as a
major isolated spike. The immediate recovery at steps 1591-1592 means this is
not currently a consecutive failure. Memory remains below the emergency stop
threshold.

## 2026-08-13 00:19 +0800 Post-1500 Patrol

Current state:

- Latest checked log step: `1598/3000`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`
- `checkpoint-2000` has not been created yet.
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Post-training orchestration process alive:
  `orchestrate_action_opsd_3src_after_train.sh`
- Error scan on `logging.jsonl`: no Traceback, CUDA OOM, Killed,
  RuntimeError, HTTP 5xx, explicit Exception, or NaN found.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 0% |
| 3 | train | 69019 / 97871 | 100% |
| 4 | train | 65883 / 97871 | 100% |
| 5 | train | 59119 / 97871 | 100% |
| 6 | train | 58361 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 87% |

Recent metric windows:

| Step window | Loss mean | Loss max | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1547-1596 | 1.1096 | 47.8138 | 65603.20 | 0.6817 | 0.5271 | 0.5633 | 0.8370 | 0.1048 | 0.3614 | 7630 | 0.1480 | 89.21 |
| 1577-1596 | 2.4299 | 47.8138 | 65603.20 | 0.6701 | 0.5251 | 0.5193 | 0.8238 | 0.1003 | 0.6356 | 4455 | 0.1625 | 89.21 |
| 1591-1596 | 0.0413 | 0.0444 | 1.78 | 0.6728 | 0.5120 | 0.5938 | 0.8417 | 0.0878 | 1.1662 | 4368 | 0.1417 | 89.21 |

Newest recovery check:

- Step 1590 remains the latest major spike. Steps 1591-1598 stayed in normal
  loss/gradient/KL range.
- Latest steps:
  - step 1595: loss `0.04442401`, grad norm `0.82847518`, KL `0.9759248`
  - step 1596: loss `0.04434701`, grad norm `1.64747036`, KL `0.858197`
  - step 1597: loss `0.04332117`, grad norm `1.1202184`, KL `0.94631509`
  - step 1598: loss `0.0397889`, grad norm `0.94678354`, KL `0.9132636`
- No new completion above `7400` tokens was observed after step 1590.
- `checkpoint-2000` is still roughly 402 optimizer steps away from the latest
  checked log step.

Current conclusion:

Training remains valid to continue. The recovery after step 1590 now spans eight
checked steps, with no NaN/OOM/runtime error and no renewed optimization spike.
Step 1590 should still be reviewed during the `checkpoint-2000` audit.

## 2026-08-13 00:29 +0800 Post-1500 Patrol

Current state:

- Latest checked log step: `1607/3000`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`
- `checkpoint-2000` has not been created yet.
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Post-training orchestration process alive:
  `orchestrate_action_opsd_3src_after_train.sh`
- Error scan on `logging.jsonl`: no Traceback, CUDA OOM, Killed,
  RuntimeError, HTTP 5xx, explicit Exception, or NaN found.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 100% |
| 3 | train | 69019 / 97871 | 100% |
| 4 | train | 65883 / 97871 | 100% |
| 5 | train | 59119 / 97871 | 100% |
| 6 | train | 58361 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 0% |

Recent metric windows:

| Step window | Loss mean | Loss max | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1553-1602 | 1.1091 | 47.8138 | 65603.20 | 0.6598 | 0.5071 | 0.5562 | 0.8305 | 0.0939 | 0.4037 | 7630 | 0.1550 | 89.21 |
| 1583-1602 | 2.4298 | 47.8138 | 65603.20 | 0.6460 | 0.4928 | 0.5615 | 0.8188 | 0.0804 | 0.7312 | 4415 | 0.1688 | 89.21 |
| 1591-1602 | 0.0414 | 0.0467 | 1.78 | 0.6853 | 0.5173 | 0.6281 | 0.8479 | 0.0530 | 0.8274 | 4368 | 0.1396 | 89.21 |

Newest recovery check:

- Step 1604 had a local gradient/loss spike: loss `0.26920408`, grad norm
  `1299.89709473`, KL `1.10418087`.
- Recovery was observed at steps 1605-1607:
  - step 1605: loss `0.03411443`, grad norm `0.99392337`, KL `0.62302497`,
    max completion `6948`
  - step 1606: loss `0.04391265`, grad norm `2.62506413`, KL `0.89984885`
  - step 1607: loss `0.04239068`, grad norm `0.54113781`, KL `0.92339076`,
    max completion `7260`
- Step 1607 is a long-output case but still below the `7400` token threshold
  used for this patrol log.
- tmux briefly showed Hugging Face `config.json` HEAD request read-timeout
  retries around step 1606, but training continued to step 1607 and the JSON
  training log stayed clean.
- `checkpoint-2000` is still roughly 393 optimizer steps away from the latest
  checked log step.

Current conclusion:

Training remains valid to continue. Step 1604 is another isolated spike after
1590, but it recovered immediately. Keep both 1590 and 1604 in the
`checkpoint-2000` audit list; no stop condition is met because there is no
NaN/OOM/runtime error, no consecutive optimization failure, and memory remains
below the emergency threshold.

## 2026-08-13 00:43 +0800 Post-1500 Patrol

Current state:

- Latest checked log step: `1620/3000`
- Saved checkpoints: `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`
- `checkpoint-2000` has not been created yet.
- tmux sessions alive: `opsd_multi_s3000`, `opsd_multi_eval_after_train`
- Post-training orchestration process alive:
  `orchestrate_action_opsd_3src_after_train.sh`
- Error scan on `logging.jsonl`: no Traceback, CUDA OOM, Killed,
  RuntimeError, HTTP 5xx, explicit Exception, or NaN found.

Realtime GPU memory from `nvidia-smi`:

| GPU | Role | Used / Total MiB | Util |
|---:|---|---:|---:|
| 0 | retrieval | 67507 / 97871 | 0% |
| 1 | non-training reservation | 54923 / 97871 | 0% |
| 2 | train | 46769 / 97871 | 100% |
| 3 | train | 69019 / 97871 | 100% |
| 4 | train | 65883 / 97871 | 100% |
| 5 | train | 59119 / 97871 | 100% |
| 6 | train | 58361 / 97871 | 100% |
| 7 | rollout | 85835 / 97871 | 87% |

Recent metric windows:

| Step window | Loss mean | Loss max | Grad max | Reward mean | F1 reward | Relevance | Format | Query KL | Answer KL | Max completion | Clip mean | Memory max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1571-1620 | 1.0027 | 47.8138 | 65603.20 | 0.6537 | 0.5048 | 0.5352 | 0.8360 | 0.2449 | 0.4121 | 7629 | 0.1475 | 89.21 |
| 1601-1620 | 0.0540 | 0.2692 | 1299.90 | 0.6606 | 0.5125 | 0.5294 | 0.8450 | 0.4103 | 0.3077 | 7629 | 0.1363 | 89.21 |
| 1611-1620 | 0.0438 | 0.0606 | 66.04 | 0.6273 | 0.4918 | 0.4700 | 0.8300 | 0.6565 | 0.2628 | 7629 | 0.1425 | 89.21 |

Newest recovery check:

- Step 1604 remains an isolated spike. Steps 1605-1620 did not show NaN/OOM or
  consecutive optimization failure.
- Step 1613 had a long completion: max length `7629`, clipped ratio `0.2125`;
  loss `0.0375251`, grad norm `1.26905441`, KL `0.76627278` remained normal.
- Step 1615 and step 1617 had small gradient lifts (`66.0387` and `21.6551`),
  both far below the earlier major spikes and followed by normal steps.
- Latest step 1620: loss `0.04428687`, grad norm `1.32208097`, KL `0.98247233`.
- `checkpoint-2000` is still roughly 380 optimizer steps away from the latest
  checked log step.

Current conclusion:

Training remains valid to continue. The run is stable after 1604, with one long
output at 1613 and two minor gradient lifts, none causing consecutive failure.
Keep 1590, 1604, and 1613 in the `checkpoint-2000` audit list.

## 2026-08-13 00:47 +0800 Status Correction

Correction to the previous status check:

- Latest tmux/log tail shows training has advanced to `1624/3000`, not merely
  `1620/3000`.
- Saved checkpoints are still `checkpoint-500`, `checkpoint-1000`, and
  `checkpoint-1500`; `checkpoint-2000` has not been created yet.
- Latest step 1624: loss `0.03834005`, grad norm `0.5705992`, KL `0.69552252`.
- Step 1613 remains the latest extreme long-output event in this window
  (`7629` tokens), and it did not cause a loss/gradient failure.
- Worker state remains healthy: `opsd_multi_s3000` and
  `opsd_multi_eval_after_train` tmux sessions are alive; the orchestration
  process is alive; GPU memory remains below the emergency threshold.

Current conclusion:

Training is still running and healthy enough to continue. The next hard audit
point remains `checkpoint-2000`.

## Next Checks

1. Continue monitoring memory and latest loss/grad/KL until `checkpoint-2000`.
2. If realtime GPU memory approaches about 95 GiB on any training GPU, ask
   for approval before stopping and restarting from the latest complete
   checkpoint with more conservative generation length.
3. At `checkpoint-2000`, audit adapter integrity, recent metric window,
   long-output rate, repeated-query rate, and Evidence Agent none rate.
4. Let `opsd_multi_eval_after_train` continue waiting for training completion;
   it will run checkpoint sweep, full dev evaluation, and paired bootstrap.
