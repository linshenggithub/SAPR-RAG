# Coding Standard

## Principles

- Keep external baseline code separate unless a small stable module must be adapted.
- Put SAPR-RAG method code under `03_sapr_rag/`.
- Put experiment configs under `04_experiments/run_configs/`.
- Put reusable small scripts near the component they support.
- Do not commit generated outputs or large data files.

## Python

- Prefer explicit config files for experiment runs.
- Keep path assumptions configurable.
- Write small sanity checks for data builders and scoring modules.

