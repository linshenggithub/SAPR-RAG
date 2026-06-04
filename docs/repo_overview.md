# 仓库总览（Repo Overview）

> 写给下一个进来这个仓库的人 / AI：打开这一份文档就够获得当前研究状态、目录用途、最近改动、下一步动作、所有需要配置的路径。
>
> 角色定位：本文档是"地图"，不重复 `README.md` / `AGENTS.md` / `docs/proposal.md` 里已有的内容，只做指引。
>
> 上次更新：2026-06-04

---

## 0. ⚠️ 主线切换（2026-06-04）—— 必读

**v4 FailureAttributedMCTS / Gate 0 验证已暂停。** 原因：

- Gate 0-A（GPT-4o 重打 50 条 trajectory）+ 离线全量审计（20K+ 分支点）+ 5 条 sanity + 2 条原版 GPT-4o 完整 MCTS 结果显示：**reward_data 里 98.4% 的 sibling 是字面重复**，"标量 PRM 对分支盲"不成立——重复来自 Llama-70B-int4 复现，**不是 ReasonRAG MCTS 本身的问题**。
- 在真正"内容不同 Q 相同"的子集（55 个）上，**typed eval 也只能区分 7.3%**，v4 立论被严重削弱。
- 详细数字与产物：[gate0/data/branch_quality_offline/summary.md](../gate0/data/branch_quality_offline/summary.md)、[gate0/data/data_source_audit/summary.md](../gate0/data/data_source_audit/summary.md)、[gate0/data/relabel_q_gpt4o_stats.json](../gate0/data/relabel_q_gpt4o_stats.json)、[gate0/data/reasonrag_original_gpt4o_mcts_sanity/analysis_summary.md](../gate0/data/reasonrag_original_gpt4o_mcts_sanity/analysis_summary.md)。

**当前主线**：**毕业设计中期答辩冲刺**（约 10 天，目标 2026-06-15 前后）——做出一个"有工作量 + 比 baseline 好一点"的方法即可，不再追求新颖性。

**主方法路线**（双线并行，根据 D1-D3 结果再决定哪个为主）：

| 路线 | 状态 | 做什么 |
|---|---|---|
| **SAPR-E v0** state-aware 文档选择 | **主线已收窄**：保持 ReasonRAG 原 pipeline 不变，只把 `retrieve top-3` 改成 `retrieve top-10 -> v0 score -> select top-3`；旧 LoRA e2e / inferred_subquery 诊断不能作为正式证据 | 见下文 §6 D1-D3 |
| **SAPR-R v1** 微调 reranker（DPA-RAG 风格 + state-aware） | 设想中，作为 v0 e2e 跑不出收益时的兜底 / 锦上添花 | 见下文 §6 D4-D8 |

**关键事实**：v0 evidence-only **不是"已经 work"**。已有 retrieval hit@3 信号只算弱证据；旧 e2e / inferred_subquery 诊断不能作为方法证据。正式 v0 只验证“同一 baseline generator + 同一 ReasonRAG pipeline 下，top-10 rerank top-3 是否优于原 top-3”。

> v4 / Gate 0 的资产（[gate0/](../gate0/) 全部）保留作为"调研深度"素材，下一个 AI 不要再投精力推进 v4 idea。

---

## 1. 当前研究状态（一句话）

毕业设计中期答辩冲刺，主方法是 **SAPR-E v0**（state-aware evidence selection）：保持 ReasonRAG 原 pipeline 不变，只替换检索文档选择；**SAPR-R v1**（DPA-RAG 启发的 trained reranker）作为升级 / 兜底。

历史 idea 演化：[docs/history.md](./history.md)（v1 → v4，已停 v4）。
v0 方法 pipeline：[docs/pipeline.md](./pipeline.md)。
Gate 0 阶段成果（已暂停）：[gate0/GATE0_STATUS.md](../gate0/GATE0_STATUS.md)。

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
| `LORA_MODEL_PATH` | baseline generator：LoRA 合并后的完整模型路径 | `SAPR_LORA_MODEL_PATH` |
| `CONDA_BIN` | conda 可执行（仅 launch_*.sh 用） | `SAPR_CONDA_BIN` |
| DMXAPI key | GPT-4o 调用凭据 | `gate0/.env` 里的 `DMXAPI_API_KEY` |

仓内路径（不在此表）：用 `Path(__file__).resolve().parents[N]` 派生，跨机器无需配置。

---

## 6. 下一步要做什么（中期答辩 10 天作战计划）

> 目标：2026-06-15 前后中期答辩，交付一个"有工作量 + 比 baseline 好一点"的完整方法。新颖性次要，**实验闭环 > 故事完整 > 增益大小**。

### P0 立刻做（D1，今天）

1. **固定 baseline generator**：使用用户 baseline 对齐的 LoRA 合并完整模型 `/home/mayi/LLaMA-Factory/examples/merge_lora/output/qwen2.5-7B-lora-dpo-RAG-ProGuide`。`/home/mayi/models/Qwen2.5-7B-Instruct-ReasonRAG-Lora` 只是 adapter 目录，不作为 vLLM 完整模型路径。
2. **按收窄定义修/用 v0 入口**：优先使用 [run_sapr_e_v0_minimal_rerank_ablation.py](../03_sapr_rag/scripts/run_sapr_e_v0_minimal_rerank_ablation.py)，它应保持 ReasonRAG 原 pipeline，只在检索边界做 `top-10 -> rerank -> top-3`。不要把 query-fix、inferred_subquery fallback 或解析逻辑变化混入 v0 主实验。
3. **重跑 baseline 30 条 e2e**：baseline 是原 ReasonRAG `retrieve top-3 -> prompt`，用于确认 generator + 当前数据/索引配置能正常工作。

### P1 视 P0 结果决定（D2-D3）

**跑收窄版 SAPR-E v0 e2e 30 条 + 200 条**，对比同一 generator 下的 baseline：

| 结果 | 应对 |
|---|---|
| EM/F1 涨（哪怕 +1pp） | ✅ v0 当主方法，按"v0 主线 + v1 锦上添花"走 |
| EM/F1 平 | ⚠️ pivot 到"evidence selection 提升 retrieval quality 但 LLM 鲁棒"的分析故事，**同时立刻启动 v1 trained reranker** |
| EM/F1 跌 | ❌ v0 退为 ablation 中的"heuristic baseline"，**全力做 v1 trained reranker** |

### P2 v1 trained reranker（D4-D8，根据 P1 结果决定优先级）

**SAPR-R**：State-Aware Process-Refined Reranker。受 [DPA-RAG (WWW 2025)](../../DPA-RAG/) 启发，微调 BGE 重排器。

**核心创新点**（用于答辩讲故事）：
> DPA-RAG 把"LLM 偏好"对齐到 reranker，但**它是 single-turn 的**；而 agentic RAG 是 multi-turn——同一个 query 在不同 trajectory state 下偏好的文档不同。我们提出 **state-aware reranker**：把 `[original_question, inferred_subquery, history_thoughts]` 拼接作为 query 编码，让 reranker 学到"在 state s 下，LLM 偏好哪些 doc"。

**预估工作量**：8 天（数据构造 2d / 模型适配 2d / 训练 1d / 评估接入 1d / 实验 + ablation 2d）。

**实现入口参考**：
- [DPA-RAG/train_bge_joined.py](../../DPA-RAG/train_bge_joined.py)：3-loss 训练（cls + rank + scl）
- [DPA-RAG/bge_joined_model.py](../../DPA-RAG/bge_joined_model.py)：模型结构
- [DPA-RAG/joined_dataset.py](../../DPA-RAG/joined_dataset.py)：数据格式

**数据来源候选**（按可行性排序）：
1. **首选**：从 ReasonRAG trajectory 抽 `(state, doc, label)`，label = 该 doc 是否在最终 hit 的 evidence 里（约 5K 三元组）
2. **进阶**（时间富裕时）：从 MCTS reward_data 蒸馏，用高 Q 子树的 evidence 作 positive、低 Q 子树作 negative——把 v4 调研资产用上

### P3 写作 + 答辩准备（D9-D10）

- 中期 PPT 骨架 + figure：方法图、主表、ablation、case study
- 中期报告：背景 / 方法 / 实验 / 总结 / 后续工作
- 现成资产可直接复用：
  - [overnight_summary.md](../04_experiments/overnight_summary.md) §5 的 4-way / §10 的 6-way ablation 数字
  - [04_experiments/metrics/](../04_experiments/metrics/) 现有数据
  - [analyze_minimal_rerank_vs_baseline_cases.py](../03_sapr_rag/scripts/analyze_minimal_rerank_vs_baseline_cases.py) 出 case study
- 第二个数据集泛化：2WikiMultihopQA 或 Musique 选 50 条小验证

### 历史 P0/P1（已弃）

- ~~跑 Gate 0 验证 A：`gate0/relabel_q_with_gpt4o.py`~~ —— **已完成（6/2-6/3），结论：v4 立论被削弱，已暂停**
- ~~Gate 0 验证 B：`gate0/run_mcts_typed_vs_scalar_pilot.py`~~ —— **已完成 5+2 条 sanity，重复分支不成立，停止扩样**
- ~~旧 SAPR-E e2e：max_tokens/query 诊断~~ —— **只能作为工程排错历史，不能作为 v0 方法证据**

### 工程清理（不阻塞研究）

- `MANIFEST.md` 已偏离当前布局
- `TODO.md` 部分项过时
- `CHANGELOG.md` 已不维护
- `idea-stage/` 和 `research-wiki/` 多数文件已不活跃

---

## 7. 给后来 AI 的入口顺序

按这个顺序读，半小时内能上手：

1. **本文档 §0 主线切换**：当前主线、为什么不再做 v4、要做什么。
2. `AGENTS.md`：项目硬规则（语言、布局、git、服务器、§11.5 命名/路径/AI 行为）。
3. `docs/coding_standard.md`：写代码前必看的硬约束。
4. `docs/pipeline.md`：v0 evidence-only 主方法 pipeline。
5. `04_experiments/overnight_summary.md`：v0 现有所有实验数字（**§11 是 e2e bug 根因，必读**）。
6. `config/paths.py` + `config/env_<host>.sh`：所有仓外路径与 `SAPR_*` 环境变量。
7. **历史背景（可选读）**：`docs/history.md`（v1→v4 演化）、`gate0/GATE0_STATUS.md`（v4 调研到哪一步）、`gate0/data/branch_quality_offline/summary.md`（为什么 v4 停了）。

跑实验之前再读：
- `docs/experiment_plan.md` / `experiment_tracker.md`
- `docs/experiment_protocol.md`

---

## 8. 红线提示（来自历史教训）

1. **不要在 Gate 0 数据落地前迭代 idea 命名**——前 3 版 idea 都是 AI 主导的语言重构，没有产生新实证差异，反而留下 ~20 份带时间戳的快照。
2. **不要默默降级实验配置**——脚本慢/卡/报错时**先停下来报告**，不要自己把 200 条切成 30 条 / 加 `_debug` 后缀。违反过的代价是 11 个不能横向比较的 results.json 全删（commit `cb867d1`）。
3. **不要新建 `xxx_v2.py / xxx_fixed.py`**——bug 修复直接覆盖原文件，演化记录靠 git 历史和 `docs/history.md`。
4. **数据来源必须如实标注**——本仓库 `reward_data*.json` 是 Llama-70B-int4 复现的，**不是论文的 GPT-4o**。所有基于它的统计在写到论文前都需要 GPT-4o 无偏对照。
5. **不要因为 retrieval 中间指标涨就声称"方法 work"**——v0 evidence-only 在 hit@3 上 +4.1pp，但该信号来自历史诊断数据；正式结论必须来自收窄版在线 e2e。中间指标涨 ≠ 端到端涨。
6. **不要混淆 adapter 和合并完整模型**——`/home/mayi/models/Qwen2.5-7B-Instruct-ReasonRAG-Lora` 是 adapter；正式 v0 generator 使用 `/home/mayi/LLaMA-Factory/examples/merge_lora/output/qwen2.5-7B-lora-dpo-RAG-ProGuide`。
