# ReasonRAG Baseline

ReasonRAG is used as a representative Agentic RAG baseline, a source of trajectories, and the first experimental platform for this project.

## External Repository

- Local working copy: `/home/mayi/RAG/ReasonRAG`
- Current local branch: `my_change`

This research repository does not vendor the full ReasonRAG source code by default. It stores configs, summaries, analysis files, small scripts, and links to local outputs.

## Role in This Project

1. Reproduce baseline results.
2. Generate trajectories for badcase analysis.
3. Expose common Agentic RAG failures.
4. Serve as the initial platform for SAPR-RAG V0.

## Current Baseline Metrics

| Dataset | F1 | EM | Acc | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | 45.60 | 34.95 | 0.40 | 0.46 | 0.47 |
| 2Wiki | 41.93 | 34.96 | 0.41 | 0.44 | 0.41 |
| PopQA | 34.85 | 28.94 | 0.35 | 0.37 | 0.34 |
| MuSiQue | 19.34 | 12.00 | 0.15 | 0.20 | 0.20 |
| Bamboogle | 39.64 | 30.40 | 0.31 | 0.39 | 0.41 |

