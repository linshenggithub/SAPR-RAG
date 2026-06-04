# ReasonRAG original GPT-4o MCTS sanity 分析

| run | id | status | sec | nodes | branches | root sim | root exact | root Q | top answers |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| 20260603_224136 | dev_0 | failed | 2.85 | 0 | 0 |  | None | None |  |
| 20260603_230639 | dev_0 | failed | 3.844 | 0 | 0 |  | None | None |  |
| 20260603_230757 | dev_0 | ok | 507.785 | 3 | 1 | 0.5413 | False | [0.6120930232558139, 0.39952380952380956] | yes, yes |
| 20260603_232625 | dev_1 | ok | 631.387 | 9 | 4 | 0.3372 | False | [0.0878125, 0.080625] | U.S. Ambassador |
| 20260604_111929 | dev_2 | failed | 18.285 | 0 | 0 |  | None | None |  |
| 20260604_111929 | dev_3 | failed | 19.357 | 0 | 0 |  | None | None |  |
| 20260604_111929 | dev_4 | failed | 15.974 | 0 | 0 |  | None | None |  |
| 20260604_113156 | dev_2 | ok | 835.502 | 65 | 31 | 0.1969 | False | [0.0584375, 0.053125] |  |
| 20260604_113156 | dev_3 | ok | 404.953 | 3 | 1 | 0.1444 | False | [0.04136363636363637, 0.21642857142857147] | No, no |
| 20260604_113156 | dev_4 | failed | 149.477 | 0 | 0 |  | None | None |  |
