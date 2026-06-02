# Coding Standard

> 给本项目所有代码（不论 AI 写还是人写）的硬约束。先看完再动手。

## 0. 核心原则

- **可复现 > 速度**：宁可慢一点跑正确的实验，也不要快但不可比的实验。
- **拒绝沉默降级**：脚本卡了/慢了/报错了，**先停下来报告**，再问下一步。详见 §5。
- **命名能自解释**：文件名、变量名、目录名打开之前就能猜出在做什么。详见 §3。

## 1. 仓库布局

- 外部 baseline 代码（如 ReasonRAG）单独放，不混入本仓库；只在必要时拷贝小模块进来并做少量改动。
- SAPR-RAG 方法主代码放 `03_sapr_rag/`。
- 实验配置放 `04_experiments/run_configs/`。
- 通用小工具脚本放它服务的组件目录附近。
- **不入库**：生成的实验产物、大数据文件（log 用 .gitignore 规则隔离）。

## 2. 路径管理

- **仓内路径**：用 `Path(__file__).resolve().parents[N]` 派生，不要写绝对路径。
- **仓外路径**（数据集、索引、模型等）：集中在 `config/paths.py`，用 `SAPR_*` 环境变量覆盖。
  ```python
  # bad
  WIKI_PATH = "/home/mayi/RAG/corpus/wiki18.jsonl"
  # good
  from config.paths import WIKI_CORPUS_PATH
  ```
- 跨机器跑实验只改环境变量，不改代码。

## 3. 脚本命名规范

### 3.1 命名格式

`<动词>_<对象>_<限定>.py`

| 动词前缀  | 含义                            | 例子 |
|----------|--------------------------------|------|
| `run_*`     | 跑完整实验，产 metrics             | `run_sapr_e_v0_e2e_eval.py` |
| `export_*`  | 从已有 pipeline 抽中间产物（不评估） | `export_evidence_decision_points.py` |
| `eval_*`    | 对已有结果做评估                  | `eval_branch_discrimination.py` |
| `analyze_*` | 分析数据，做诊断/case study       | `analyze_minimal_rerank_vs_baseline_cases.py` |
| `compare_*` | 对比多个 run/strategy 的结果      | `compare_3way_evidence_selectors.py` |
| `build_*`   | 构造索引/语料等基础设施           | `build_bge_index.py` |
| `fetch_*`   | 拉取数据                        | `fetch_reasonrag_corpus.py` |
| `relabel_*` | 重新标注                        | `relabel_q_with_gpt4o.py` |
| `sample_*`  | 采样                            | `sample_branch_points.py` |
| `compute_*` | 计算某个量                      | `compute_phi_q_typed.py` |
| `launch_*`  | bash launcher（包装上面任意脚本） | `launch_sapr_e_v0_e2e_200_queue.sh` |

### 3.2 禁止的命名

- `*_v1.py / *_v2.py / *_v3.py`：新版本**直接覆盖**旧文件，演化记录靠 git 历史和 `docs/history.md`。
- `*_debug.py / *_temp.py / *_test_xxx.py / mock_*.py`：debug 在命令行 verify，不入库（详见 §4）。
- `mcts_pilot.py` / `analyze_results.py` / `script1.py` 这种**单拎出来不知道在做什么**的命名。

### 3.3 launcher 与 Python 脚本同名

bash launcher 名字应当能直接对应它启动的 Python 脚本：

- `run_sapr_e_v0_e2e_eval.py`  ← `launch_sapr_e_v0_e2e_200_queue.sh`
- `export_evidence_decision_points.py`  ← `launch_export_evidence_decision_points.sh`

## 4. Debug 与"快速验证"的处理

### 4.1 debug 不入库

debug、sanity check、最小配置跑通**只在命令行 verify**，确认 OK 之后**直接跑正式版**，不在 git 里留 `*_debug.py`、`sanity_check_*.py`。

正确做法：

```bash
# 在命令行 verify 一下脚本能 import + 解析参数
python my_run.py --num_examples 3 --dry-run

# OK 了直接跑正式版
python my_run.py --num_examples 200 --mode treatment
```

错误做法：

- 写一个 `run_my_thing_debug.py` 跑 30 条提交进 git
- 同时保留 `run_my_thing.py`（跑 200 条），导致以后的人分不清谁是真的

### 4.2 同一份脚本支持多种规模

正式脚本应当通过 `--num_examples` / `--mode` 等参数统一支持小规模 verify 和正式跑，不需要 fork 出一个 debug 副本。

## 5. AI 执行行为约束（重要）

如果你（AI）被要求跑一个实验/pipeline，结果发现它**慢、卡、报错**：

### 5.1 必须做：先报告

立刻停下来，向用户汇报以下内容：

1. 跑了什么命令；
2. 哪一步慢/卡了（加载索引？跑模型？评估？）；
3. 大致耗时（如果能测就给数字，"FAISS 索引 21M 文档冷加载 ~12 分钟"）；
4. 已知或推测的根因。

### 5.2 禁止做：默默降级

**不要**在没有用户许可的情况下自作主张地：

- 把数据集切片缩小（200 → 30 → 3）
- 把真实检索换成 mock / dummy / 缓存假数据
- 把模型换小、context 截短
- 跳过检索、跳过评估，只跑能跑的部分
- 给文件加后缀 `_debug1` / `_debug6` / `_mock` 跑一遍存下来

历史上这种行为产生过 11 个配置不一致、不能横向比较的 results.json，最后只能全删（见 commit `cb867d1`）。

### 5.3 等用户决定再继续

汇报完之后**等用户的决定**再改实验配置。用户可能：

- 同意你换小规模先跑通；
- 让你解决根因（如换索引、修 bug）；
- 让你跳过这个实验改做别的。

无论选哪种，**先问再做**。

## 6. Bug 修复

- 修 bug 时**直接编辑原文件**，不要新建 `xxx_v2.py / xxx_fixed.py`。
- 在 commit message 里清楚写"修了什么 bug、根因是什么"。
- 如果旧实现还想留作对照，写在 `docs/history.md` 里说明，不在代码里留。

## 7. 中文优先

- 注释、commit message、docstring 主体用中文（user 母语）。
- 代码标识符（变量名、函数名、文件名）用英文 + 下划线。
- 论文/英文术语原文保留：`MCTS`、`scalar PRM`、`branch point` 等。
