# 仓库总览（Repo Overview）

> 写给下一个进来这个仓库的人 / AI：打开这一份文档就够获得当前研究状态、目录用途、最近改动、下一步动作、所有需要配置的路径。
>
> 角色定位：本文档是"地图"，不重复 `README.md` / `AGENTS.md` / `docs/proposal.md` 里已有的内容，只做指引。
>
> 上次更新：2026-06-02

---

## 1. 当前研究状态（一句话）

SAPR-RAG idea 已迭代到 **v4 FailureAttributedMCTS**（typed transition evaluation：φ_q / φ_c / φ_s），目前**正在做 Gate 0 验证**——用 GPT-4o 重打 ReasonRAG MCTS 的 Q 值，判断"标量 PRM 真的对分支盲吗"，结果决定 v4 是 Go / Pivot / Stop。

详细演化：[docs/history.md](./history.md)（v1 → v4）。
当前 idea 完整版：[docs/proposal.md](./proposal.md)。
Gate 0 现状（含本地资源、参数对齐审计、API key 位置）：[gate0/GATE0_STATUS.md](../gate0/GATE0_STATUS.md)。

---

## 2. 目录功能图

```
SAPR-RAG/
├── 00_project_management/   项目管理：milestones、周计划、会议纪要
├── 01_literature/           文献：survey csv、taxonomy、paper_notes/、related_work_drafts/
├── 02_baseline_reasonrag/   ReasonRAG baseline：scripts/、本地修改记录、badcase taxonomy
├── 03_sapr_rag/             SAPR-RAG 方法主代码（v0 evidence-only 已跑通；scripts/ 11 个脚本）
├── 04_experiments/          实验产物：metrics/、run_configs/、overnight_summary.md
├── 06_notes/                日常笔记：daily_notes/、idea_notes/
├── 07_assets/               figures、diagrams（写论文用）
│
├── config/                  仓内全局配置（path.py 集中仓外路径）
├── docs/                    全部规约 + 设计文档（见 §3 详表）
├── gate0/                   当前主线：Gate 0 typed vs scalar 验证
│   ├── data/                解析 ReasonRAG MCTS 树后的中间产物
│   ├── sample_branch_points.py        步骤 1：解析 reward_data*.json，提分支点
│   ├── compute_phi_q_typed.py         步骤 2：计算 φ_q（NER-based）
│   ├── relabel_q_with_gpt4o.py        步骤 3-A：GPT-4o 无偏重打 Q 值
│   ├── run_mcts_typed_vs_scalar_pilot.py  步骤 3-B：完整 MCTS pilot（baseline / treatment）
│   ├── typed_eval.py                  GPT-4o typed eval 调用封装
│   ├── test_typed_eval.py             typed_eval.py 的单元测试
│   ├── README.md                      Gate 0 步骤说明
│   └── GATE0_STATUS.md                Gate 0 现状（参数对齐、资源清单）
│
├── idea-stage/              v1-v3 时期的 idea 评审材料（保留作历史）
├── research-wiki/           研究 wiki（gap_map、query_pack 等，目前用得少）
│
├── README.md                项目目标 + 主线
├── AGENTS.md                项目操作指南（命名规则、git、服务器、§11.5 是 coding hygiene 摘要）
├── CLAUDE.md                Claude Code 入口（指向 AGENTS.md）
├── MANIFEST.md              历史产物清单（旧版本，已偏离当前布局）
├── ROADMAP.md               时间线：05-23 → 07 AAAI submission
├── TODO.md                  早期 todo 清单（部分已过时）
├── CHANGELOG.md             早期变更记录（已不维护）
└── RESEARCH_REVIEW_SAPR_RAG.md   2026-05-30 的项目阶段性审视
```

---

## 3. docs/ 文档索引

| 文件 | 用途 | 当前状态 |
|---|---|---|
| `proposal.md` | v4 当前 idea 完整版 | 活跃 |
| `history.md` | idea v1→v4 演化历史与教训 | 活跃 |
| `experiment_plan.md` | 实验计划 | 活跃 |
| `experiment_tracker.md` | 实验跟踪表 | 活跃 |
| `experiment_protocol.md` | 实验协议（指标、复现规则） | 活跃 |
| `pipeline.md` | v0 evidence-only pipeline | 活跃 |
| `coding_standard.md` | **代码硬约束**：命名/路径/debug/AI 行为 | 活跃，必读 |
| `setup.md` | 仓库基本路径与 git 规则 | 较简略 |
| `server_env.md` | 服务器信息（3090/5090） | 活跃 |
| `literature_note_standard.md` | 论文笔记格式规范 | 活跃 |
| `writing_standard.md` | 论文/报告写作规范 | 活跃 |
| `repo_overview.md` | **本文档** | 活跃 |

---

## 4. 最近改动（按时间倒序）

### 2026-06-02：命名规范 + 清理 debug + 写规约（4 commit，见下文 §6 后续）

| commit | 内容 |
|---|---|
| `c43df7e` | 修 `launch_export_evidence_decision_points.sh` 的旧文件名引用 + 更新 `docs/pipeline.md` 中已删的 refine-logs/ 引用 |
| `a87a507` | 全面重写 `docs/coding_standard.md`；`AGENTS.md` 新增 §11.5；`CLAUDE.md` 改指向新规约 |
| `6a61a82` | 13 个脚本按动词前缀规范重命名（命名能直接看出做什么），更新所有交叉引用，pilot 脚本接入 config.paths，新增 `HOTPOTQA_DEV_PATH` |
| `cb867d1` | 删除 15 个 debug/中间产物文件；`mcts_pilot.py` 重命名并从 `gate0/gpt4o_experiment/` 上提到 `gate0/run_mcts_typed_vs_scalar_pilot.py` |

### 2026-06-01：路径重构（不写死绝对路径）

| commit | 内容 |
|---|---|
| `58b0540` | `03_sapr_rag/scripts/` 全部脚本接入 `config/paths.py` |
| `ae0399a` | 集中仓外依赖路径到 `config/paths.py`，统一用 `SAPR_*` 环境变量覆盖 |
| `19f464d` | `gate0/` 仓内路径改用 `Path(__file__).resolve().parents[N]` 派生 |
| `5863f08` | 移除一次性清理脚本 `cleanup.sh` |

### 更早

- `35f9a3a` 新增语料合并、索引构建脚本与仓库清理工具
- `f688b06` 清理旧实验日志与空目录，重构项目文档
- 早期：03_sapr_rag/scripts/ 跑通 v0 evidence-only e2e（详见 [04_experiments/overnight_summary.md](../04_experiments/overnight_summary.md)）

---

## 5. 换机器三步走（仓库可移植性）

本仓库设计成"代码无机器假设、路径靠环境变量、每台机器一份 env 脚本"。在新机器上拉下来后，按这三步走：

### 步骤 1：填写本机的环境变量配置

仓库自带两份骨架：

- [config/env_3090.sh](../config/env_3090.sh) ：3090 服务器（4×RTX 3090），值已根据 [GATE0_STATUS.md §6.3](../gate0/GATE0_STATUS.md) 审计填好
- [config/env_5090.sh](../config/env_5090.sh) ：5090 服务器（rag-5090，3×RTX 5090），**含 TODO 占位**，第一次用前必须 ssh 上去 `ls` 确认路径再填回来

新机器（既不是 3090 也不是 5090）：复制一份命名为 `config/env_<host>.sh`，按字段一条条填路径。

### 步骤 2：复制 API key 模板

```bash
cp gate0/.env.example gate0/.env
# 编辑 gate0/.env，把 <YOUR_KEY> 换成真实 DMXAPI key
```

### 步骤 3：跑实验前 source

```bash
source config/env_3090.sh   # 或 env_5090.sh / env_<your_host>.sh
python gate0/run_mcts_typed_vs_scalar_pilot.py --mode sanity
```

如果忘记 source，脚本会在用到对应路径时**主动报错**，告诉你该 export 哪个 `SAPR_*` 变量，不会默默走错路径：

```
RuntimeError: SAPR_BGE_INDEX_PATH is not set. Purpose: BGE Flat FAISS index file.
Fix: source config/env_3090.sh (or env_5090.sh) before running this script.
```

### 配置变量一览（9 个 + 1 个 API key）

所有仓外路径集中在 [config/paths.py](../config/paths.py)，全部要求 `SAPR_*` 环境变量；不内置任何机器特定的默认值。

| 配置项 | 用途 | 环境变量 |
|---|---|---|
| `REASONRAG_OUTPUT_DIR` | ReasonRAG MCTS reward_data*.json 目录（gate0 输入） | `SAPR_REASONRAG_OUTPUT_DIR` |
| `REASONRAG_ROOT` | ReasonRAG 仓库根（用于 sys.path 注入） | `SAPR_REASONRAG_ROOT` |
| `FLASHRAG_ROOT` | FlashRAG 仓库根（pilot 脚本用） | `SAPR_FLASHRAG_ROOT` |
| `BGE_INDEX_PATH` | BGE Flat FAISS 索引 | `SAPR_BGE_INDEX_PATH` |
| `BGE_MODEL_PATH` | BGE encoder（bge-base-en-v1.5） | `SAPR_BGE_MODEL_PATH` |
| `WIKI_CORPUS_PATH` | 维基百科 corpus jsonl（与 index 配套） | `SAPR_WIKI_CORPUS_PATH` |
| `HOTPOTQA_DEV_PATH` | HotpotQA dev jsonl（gate0 pilot 输入） | `SAPR_HOTPOTQA_DEV_PATH` |
| `LORA_MODEL_PATH` | qwen2.5-7B-LoRA-DPO 合并模型 | `SAPR_LORA_MODEL_PATH` |
| `CONDA_BIN` | conda 可执行（仅 launch_*.sh 用） | `SAPR_CONDA_BIN` |
| DMXAPI key | GPT-4o 调用凭据 | `gate0/.env` 里的 `DMXAPI_API_KEY` |

仓内路径（不在此表）：用 `Path(__file__).resolve().parents[N]` 派生，跨机器无需配置。

---

## 6. 下一步要做什么

按优先级：

### P0 立即可做
1. **把 4 个 commit `cb867d1 / 6a61a82 / a87a507 / c43df7e` push 到 `origin/main`**（命名规范 + 清理 debug + 写规约的产物，本地已 commit，未推送）。
2. **跑 Gate 0 验证 A**：`gate0/relabel_q_with_gpt4o.py`（GPT-4o 重标 50 条 trajectory 的兄弟节点 Q 值，预算 ¥15-30，5-10 分钟）。
   - 看完结果回答关键问题：兄弟节点 Q 值用 GPT-4o 重打后，分布是否仍然集中？标量 PRM 是否真的对分支盲？
   - **结果决定 v4 走 Go / Pivot / Stop**，详见 [docs/history.md](./history.md) 末节。

### P1 视 Gate 0-A 结果决定
- 若结果显示标量 PRM 确实盲：跑 Gate 0-B（`gate0/run_mcts_typed_vs_scalar_pilot.py` 50 条无泄漏 inference-aligned MCTS pilot），对比 typed eval 和 scalar self-eval 的 EM/F1。
- 若结果显示 PRM 并不盲：暂停 v4，回到 idea 设计室找替代切入点（见 [docs/history.md §教训 1](./history.md)）。

### P2 写作类
- 完成 30 篇核心文献综述（[ROADMAP.md](../ROADMAP.md) 06-01 → 06-07 段）。
- 整理 SAPR-RAG v0 evidence-only 6-way ablation 的 case study，准备写到 midterm report。

### P3 工程清理（不阻塞研究）
- `MANIFEST.md` 已偏离当前布局（仍引用已删的 refine-logs/），下次清理时一起重写或废弃。
- `TODO.md` 部分项过时（refine-logs 阶段的），需要更新或合并到 ROADMAP。
- `CHANGELOG.md` 仅有 2026-05-23 一条，要么定期更新要么明确废弃改用 git log。
- `idea-stage/` 和 `research-wiki/` 多数文件已不活跃，可考虑归档到一个 `archive/` 目录。

---

## 7. 给后来 AI 的入口顺序

按这个顺序读，半小时内能上手：

1. **本文档** `docs/repo_overview.md`：地图。
2. `AGENTS.md`：项目硬规则（语言、布局、git、服务器、§11.5 命名/路径/AI 行为）。
3. `docs/coding_standard.md`：写代码前必看的硬约束。
4. `docs/proposal.md`：当前 v4 idea。
5. `docs/history.md`：v1→v4 演化与教训（不要再次踩坑）。
6. `gate0/GATE0_STATUS.md`：当前主线 Gate 0 的全部上下文。
7. `config/paths.py`：所有仓外路径与 `SAPR_*` 环境变量。

跑实验之前再读：
- `docs/experiment_plan.md` / `experiment_tracker.md`
- `docs/experiment_protocol.md`
- `docs/pipeline.md`（v0 evidence-only 是怎么跑通的）

---

## 8. 红线提示（来自历史教训）

1. **不要在 Gate 0 数据落地前迭代 idea 命名**——前 3 版 idea 都是 AI 主导的语言重构，没有产生新实证差异，反而留下 ~20 份带时间戳的快照。
2. **不要默默降级实验配置**——脚本慢/卡/报错时**先停下来报告**，不要自己把 200 条切成 30 条 / 加 `_debug` 后缀。违反过的代价是 11 个不能横向比较的 results.json 全删（commit `cb867d1`）。
3. **不要新建 `xxx_v2.py / xxx_fixed.py`**——bug 修复直接覆盖原文件，演化记录靠 git 历史和 `docs/history.md`。
4. **数据来源必须如实标注**——本仓库 `reward_data*.json` 是 Llama-70B-int4 复现的，**不是论文的 GPT-4o**。所有基于它的统计在写到论文前都需要 GPT-4o 无偏对照。
