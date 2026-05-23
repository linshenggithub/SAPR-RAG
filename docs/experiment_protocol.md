# Experiment Protocol

Each meaningful experiment should have:

1. `run_config.yaml`
2. `raw_output.jsonl` or a link to the local raw output
3. `trajectories.jsonl` or a link to the local trajectory file
4. `metrics.json`
5. `badcases.jsonl`
6. `experiment_note.md`

Large raw files should not be committed. Use markdown summaries, metrics files, and local path references instead.

## Required Config Fields

```yaml
run_name:
date:
server:
gpu:
model:
adapter:
dataset:
retriever:
corpus:
top_k:
max_steps:
temperature:
num_candidates:
method:
baseline:
output_path:
log_path:
metric_path:
note:
```

## Result Labeling

Use clear labels:

- `real_result`: actually run and verified
- `expected_result`: planned or estimated
- `debug_result`: small-scale sanity check
- `failed_run`: failed experiment with useful diagnostics

