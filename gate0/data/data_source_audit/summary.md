# Gate 0 Data Source Audit

**Date**: 2026-06-03

**Scope**: audit `gate0/data/reasonrag_mcts/reward_data{0,1,2,3}.json`.

**Method**: local file inspection only. No API calls, no retrieval, no model inference.

## 1. Bottom Line

The current `gate0/data/reasonrag_mcts/reward_data*.json` files should **not** be treated as original ReasonRAG paper GPT-4o MCTS data.

The strongest conclusion is:

```text
These files are local reproduced / copied ReasonRAG MCTS-style data, and they contain strong evidence of Llama-format generation traces plus structural quality issues.
They can diagnose this local reward_data batch, but they cannot directly support claims about original ReasonRAG GPT-4o MCTS.
```

This matches the warning already recorded in `docs/history.md` and `docs/repo_overview.md`: the repository reward data is Llama-70B-int4 reproduction data, not paper GPT-4o data.

## 2. File Inventory

```text
gate0/data/reasonrag_mcts/reward_data0.json   421 trajectories
gate0/data/reasonrag_mcts/reward_data1.json   382 trajectories
gate0/data/reasonrag_mcts/reward_data2.json   774 trajectories
gate0/data/reasonrag_mcts/reward_data3.json   5000 trajectories
```

Total: **6577 trajectories**.

The files were copied or created under this repository at:

```text
2026-06-03 00:47-00:48 +0800
```

No matching source directory was found at:

```text
/home/mayi/RAG/ReasonRAG/output/hotpotqa
/home/mayi/ReasonRAG/output/hotpotqa
```

The only other local `reward_data0.json` found by filesystem search was:

```text
/home/mayi/ReasonRAG_modified/output/rolerag/reward_data0.json
```

It has a different size from the current audited file, so it is not a direct same-file match.

## 3. Evidence That This Is Not Paper GPT-4o MCTS Data

### 3.1 Llama Chat Template Appears in Prompts

`reward_data0/1/2` contain `input_prompt` strings with Llama-style chat template markers:

```text
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024
```

Full-sample counts:

| File | Nodes | Nodes with input_prompt | Llama-template prompts |
| --- | ---: | ---: | ---: |
| reward_data0.json | 7356 | 6981 | 6981 |
| reward_data1.json | 5038 | 4696 | 4696 |
| reward_data2.json | 10582 | 9877 | 9877 |
| reward_data3.json | 22660 | 20476 | 0 |

`reward_data3` uses plain prompt text rather than Llama chat markers, but it still is not enough to prove GPT-4o origin. It also has serious structural anomalies described below.

### 3.2 Repository Documentation Already Warns About Source

`docs/history.md` states:

```text
本仓库 reward_data 是用 Llama-70B-int4 复现的，不是论文配置的 GPT-4o。
```

`docs/repo_overview.md` states:

```text
本仓库 reward_data*.json 是 Llama-70B-int4 复现的，不是论文的 GPT-4o。
```

Therefore, all statistics based on these files must be labeled as local reproduction-data analysis, not original ReasonRAG paper-data analysis.

## 4. Structural Quality Issues

### 4.1 Many Trajectories Have No Root Node

Expected MCTS tree structure should normally have exactly one root node with `parent_id = -1`.

Observed root-count distribution:

| File | Trajectories | Root = 1 | Root = 0 |
| --- | ---: | ---: | ---: |
| reward_data0.json | 421 | 375 | 46 |
| reward_data1.json | 382 | 342 | 40 |
| reward_data2.json | 774 | 705 | 69 |
| reward_data3.json | 5000 | 2184 | 2816 |

This means many entries are not complete normal MCTS trees. `reward_data3.json` is especially problematic: **2816 / 5000** entries have no root node.

### 4.2 Q Values Are Often Outside [0, 1]

If Q is a normalized evaluator score, values should usually be in `[0,1]`. The audited files contain many Q values outside this range.

| File | Nodes | Q outside [0,1] | Extreme Q outside [-1,2] |
| --- | ---: | ---: | ---: |
| reward_data0.json | 7356 | 136 | 80 |
| reward_data1.json | 5038 | 92 | 68 |
| reward_data2.json | 10582 | 229 | 195 |
| reward_data3.json | 22660 | 2039 | 1548 |

Examples include:

```text
-19.96
20.13
329.5366666666667
988.61
```

This strongly suggests that the stored Q values are not clean normalized GPT-4o scalar scores, or that later aggregation / parsing / copying produced corrupted values.

## 5. Relation to Offline Branch Audit

The offline branch-quality audit found:

```text
total branch points: 20497
content-identical branch points: 20171
content-different branch points: 326
Llama-Q-same and content-different branch points: 55
```

Given the data-source findings above, the correct interpretation is:

```text
The current reward_data batch shows severe duplicate-sibling behavior and weak usable branch diversity.
But this should be attributed to this local reproduction batch unless original GPT-4o ReasonRAG MCTS data is obtained or regenerated.
```

It should **not** be written as:

```text
Original ReasonRAG MCTS has severe duplicate branches.
```

## 6. Relation to GPT-4o Sanity Check

The small GPT-4o root-node sanity check on 5 selected questions found:

```text
exact duplicate sibling generations: 0 / 5
near duplicate >= 0.95 similarity: 0 / 5
average pair similarity: 0.4658
```

This is only preliminary evidence, but it supports the data-source audit:

```text
The duplicate-sibling problem may be caused by the local reproduction model / sampling setup / expansion implementation, rather than being inherent to GPT-4o ReasonRAG MCTS.
```

## 7. Consequence for Gate 0

The current reward_data files are useful for:

1. diagnosing local reproduction quality;
2. showing that this batch is not suitable as direct evidence for scalar PRM branch blindness;
3. motivating a data-quality gate before any typed reward claim.

They are not sufficient for:

1. proving original ReasonRAG MCTS has duplicate-branch failure;
2. proving scalar PRM is intrinsically branch-blind;
3. justifying an expensive Gate0-B run without first fixing data provenance.

## 8. Recommended Next Step

Before running more paid experiments:

1. mark all current `reward_data`-based conclusions as local reproduction-data conclusions;
2. either locate the original GPT-4o ReasonRAG MCTS data, or regenerate a tiny GPT-4o MCTS subset with clean logging;
3. add data-quality checks before any branch-level evaluation:
   - root node count;
   - Q range sanity;
   - sibling duplicate rate;
   - prompt/model provenance;
   - content-different branch count.

Only after these checks pass should the project use branch-level statistics to support or reject the v4 typed transition idea.
