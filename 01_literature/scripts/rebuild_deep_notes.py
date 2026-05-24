#!/usr/bin/env python3
"""Rebuild paper notes with the project deep-reading template."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTES_DIR = ROOT / "01_literature" / "paper_notes"
IMAGES_DIR = NOTES_DIR / "images"


PROFILES = {
    "2025_MCTS_RAG": {
        "position": "探索式 Agentic RAG；用 Monte Carlo Tree Search 组织检索-推理路径。",
        "problem": "小模型在复杂问答中容易生成含糊查询、重复检索步骤，并且难以理解检索内容。",
        "method": ["把 reasoning path 放入树搜索框架，用模拟和回传估计候选路径价值。", "将检索动作与思考动作交替展开，使 test-time search 弥补小模型规划不足。", "核心价值函数更偏路径级选择，而不是显式判断某篇文档在当前状态下的边际效用。"],
        "experiment": "ComplexWebQA、GPQA、FoolMeTwice 等任务用于验证树搜索对复杂推理的增益，主指标偏最终答案准确率。",
        "limits": ["计算成本高，适合作为强 test-time baseline，不一定适合作为低成本训练数据生成器。", "对 query drift、bridge entity 保真和 stop decision 主要是间接缓解。", "state-aware evidence utility 没有成为独立训练目标。"],
        "mapping": ["可为 SAPR-RAG 的候选轨迹生成提供搜索器。", "可把 MCTS value 从答案路径价值改造为证据边际价值。", "适合与 Evidence Reward 联合，筛掉重复检索和低贡献证据。"],
        "module": "trajectory search / Evidence Reward",
    },
    "2025_TIRESRAG_R1": {
        "position": "过程奖励强化类工作；从 sufficiency、reasoning quality、reflection 三个角度刻画推理质量。",
        "problem": "final-answer reward 无法区分信息不足、推理错误和答案-推理不一致，容易造成过早中断或错误自信。",
        "method": ["采用 think-retrieve-reflect 流程，把检索增强推理拆成可评价的中间阶段。", "设计 sufficiency、reasoning quality、reflection 等多维奖励。", "更关注思考质量与信息充分性，还没有把候选文档效用定义成 `U(d | q, s_t)`。"],
        "experiment": "HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle 用于验证多跳问答中的信息充分性和反思奖励效果。",
        "limits": ["query drift 和 bridge entity carryover 不是独立奖励项。", "停搜价值虽通过 sufficiency 间接处理，但还不是可解释的 Stop Reward。", "retriever-LLM alignment 仍是外部组件。"],
        "mapping": ["可借鉴 sufficiency reward 设计 Stop Reward。", "可把 reflection 失败样本转成 Repair Mechanism 训练数据。", "适合支撑本课题的 failure taxonomy。"],
        "module": "Stop Reward / Reflection Repair",
    },
    "2025_REX_RAG": {
        "position": "探索与轨迹修复类工作；关注 RL-based RAG 的 dead-end problem。",
        "problem": "从当前 policy 采样会限制探索，模型可能在证据不足时得出过度自信的错误结论并停止探索。",
        "method": ["Mixed Sampling Strategy 用 probe policy 和 exploratory prompts 扩大轨迹空间。", "Policy Correction Mechanism 用 importance sampling 校正探索策略与目标策略的分布偏移。", "方法核心是增加可学习轨迹，而不是显式评价当前证据对证据链的贡献。"],
        "experiment": "NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle 上报告平均准确率提升。",
        "limits": ["探索更多路径不等于知道哪个证据真正有用。", "unsupported intermediate answer 没有被独立约束。", "如果 reward 仍偏 outcome，修复路径选择仍可能含噪。"],
        "mapping": ["低 Query/Evidence/Stop Reward 可触发 REX 风格探索。", "dead-end trajectory 可作为 failure bank 的一类标签。", "Policy correction 思想可用于修复由探索 prompt 引入的分布偏移。"],
        "module": "Repair Mechanism / trajectory exploration",
    },
    "2025_DecEx_RAG": {
        "position": "过程监督工程化代表；将 Agentic RAG 拆成 decision 与 execution 两类动作。",
        "problem": "global outcome reward 稀疏且难以反映局部表现，终止决策错误会导致冗余迭代或过早停止。",
        "method": ["把 Agentic RAG 建模成 MDP，分别优化 decision 和 execution。", "用过程监督让模型知道何时检索、何时执行、何时终止。", "通过 pruning strategy 提高数据构建效率。"],
        "experiment": "PopQA、NQ、AmbigQA、HotpotQA、2WikiMultiHopQA、Bamboogle 覆盖单跳与多跳问答。",
        "limits": ["状态定义仍偏动作级，不直接表示历史证据是否闭合所有信息槽。", "retrieval noise 和 evidence utility 没有成为独立评分对象。", "局部监督比 outcome 更细，但还没有 query/evidence/stop 三头可解释奖励。"],
        "mapping": ["SAPR-RAG 可继承 MDP 视角。", "将 decision 拆成 Query Reward 与 Stop Reward。", "将 execution 中的证据抽取拆成 Evidence Reward 与 Step Entailment。"],
        "module": "MDP framing / Query-Evidence-Stop rewards",
    },
    "2026_HiPRAG": {
        "position": "层级过程奖励；专门处理 Agentic RAG 的 over-search 与 under-search。",
        "problem": "模型何时搜索、何时停止缺少细粒度控制，导致检索不足或过度检索。",
        "method": ["构造 hierarchical process rewards 判断每步 search necessity。", "用层级 reward 同时约束效率和准确性。", "重点是是否搜索，而不是搜索到的文档在当前状态下的边际效用。"],
        "experiment": "NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle，并报告 search efficiency 相关指标。",
        "limits": ["query 内容质量本身没有被细分监督。", "当前状态下哪篇证据最有价值仍未建模。", "不能简单作为 judge calibration 缺失的反例，因为论文做了人工一致性核查。"],
        "mapping": ["Stop Reward 的最近邻工作。", "可把 search necessity 扩展成 search target utility。", "可借用 over/under-search 指标评价 SAPR-RAG。"],
        "module": "Stop Reward / search efficiency",
    },
    "2026_RAGShaper": {
        "position": "数据合成与抗噪轨迹训练；通过自动合成让 agent 学会复杂检索技能。",
        "problem": "真实检索环境有噪声和复杂干扰，人工构造高质量纠错轨迹不可扩展。",
        "method": ["InfoCurator 构造 dense information trees。", "引入 perception 与 cognition 两类 adversarial distractors。", "通过 constrained navigation 让 teacher agent 在干扰条件下生成纠错轨迹。"],
        "experiment": "NQ、PopQA、AmbigQA、Bamboogle 上验证合成轨迹对 agent 技能的提升。",
        "limits": ["主要是 SFT / data synthesis，不是在线 reward learning。", "query drift、premature stop 和 state-aware utility 仍需要额外训练目标。", "代码缺失会增加复现成本。"],
        "mapping": ["可借鉴其 hard negative 构造方式生成 Evidence Preference 数据。", "可把纠错轨迹转成 Repair Mechanism 训练样本。", "可为 Noise Risk 维度提供数据来源。"],
        "module": "Failure data synthesis / Evidence Reward",
    },
    "2026_ProRAG": {
        "position": "过程监督 RL；直接针对 process hallucination 和 coarse-grained scalar reward。",
        "problem": "最终答案奖励无法定位错误步骤，模型即使答对也可能依赖错误逻辑或冗余检索。",
        "method": ["用 SFT warmup 获得初始轨迹。", "用 MCTS-based PRM 和 PRM-guided refinement 构造过程反馈。", "再进行 process-supervised RL，把 outcome signal 与 step signal 合并。"],
        "experiment": "PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle 上评估 EM/F1，并通过消融验证 PRM 与 RL 阶段贡献。",
        "limits": ["PRM 分数未必能解释错误来自 query、evidence 还是 stop。", "state-aware evidence utility 仍不是独立概念。", "训练流程较复杂，需要确认数据与算力成本。"],
        "mapping": ["SAPR-RAG 的直接邻近对手。", "需要用三头 reward 提供比 PRM 更细的错误归因。", "可把 process hallucination 作为 Step Entailment 评价对象。"],
        "module": "Process Reward / Step Entailment",
    },
    "2026_Search_P1": {
        "position": "Path-centric reward shaping；从整条轨迹而非最终答案提取训练信号。",
        "problem": "sparse outcome reward 忽略中间推理质量，也浪费部分正确但最终失败的轨迹。",
        "method": ["构造 path score，将 step coverage、soft scoring、self-consistency 和 reference alignment 纳入奖励。", "用 offline reference planners 提供路径级监督。", "更关注 path 是否合理，而不是文档在状态中的边际效用。"],
        "experiment": "NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle、AD-QA 等任务验证稳定训练和效率。",
        "limits": ["路径级分数仍可能掩盖 query/evidence/stop 的具体错误来源。", "reference planner 质量会影响训练上限。", "没有显式建模 retriever-LLM utility mismatch。"],
        "mapping": ["可作为 SAPR-RAG trajectory-level reward 的对照。", "可将 path score 分解为 Query/Evidence/Stop 三头奖励。", "可用于解释为什么仅有 path reward 仍不够。"],
        "module": "Trajectory Reward / credit assignment",
    },
    "2025_VERITAS": {
        "position": "Faithfulness reward；从正确性转向推理过程是否忠实于检索信息。",
        "problem": "只奖励 final answer correctness 会导致 chain-of-thought unfaithfulness，中间推理未必被证据支撑。",
        "method": ["提出 information-think、think-search、think-answer 三类 faithfulness。", "构建 VERITAS reward model，用忠实性信号补充 outcome reward。", "关注中间步骤与证据、搜索动作、答案之间的一致性。"],
        "experiment": "评估 SearchR1、ReSearch 等 RLVR search agents，并覆盖 NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle。",
        "limits": ["faithfulness 不等于 evidence utility，仍未回答哪篇候选文档最该进入状态。", "query drift 和 stop decision 不是主要建模对象。", "代码未找到，复现 judge 细节需后续确认。"],
        "mapping": ["可直接支撑 Step Entailment / Supportiveness Reward。", "可把 unsupported intermediate answer 作为核心 failure type。", "可为最终论文的评价指标提供 faithfulness 维度。"],
        "module": "Step Entailment / Faithfulness Reward",
    },
    "2025_Utility_Focused_LLM_Annotation": {
        "position": "Evidence Utility 起点；用 LLM 标注文档对生成任务的 utility。",
        "problem": "retriever 的 topical relevance 与 RAG 生成端 utility 存在目标错位，相关文档未必有用。",
        "method": ["用 LLM 自动标注文档 utility，减少人工标注成本。", "提出 summed marginal likelihood 以利用多正例。", "将 utility label 用于训练 retrieval 与 RAG 模型。"],
        "experiment": "MS MARCO、BEIR、MS MARCO QA、NQ、HotpotQA 覆盖检索和 RAG 两类任务。",
        "limits": ["utility 主要是 query-document 静态关系。", "没有多步 history evidence、intermediate answer 和 remaining gap。", "对 Agentic RAG 的 stop/query decision 没有直接处理。"],
        "mapping": ["Evidence Reward 的直接文献起点。", "SAPR-RAG 可把 `U(d | q)` 扩展为 `U(d | q, s_t)`。", "可借鉴 LLM annotation 流程构造 preference 数据。"],
        "module": "Evidence Reward / utility annotation",
    },
    "2025_LLM_Specific_Utility": {
        "position": "Model-specific utility；证明不同 LLM 对同一 passage 的收益不同。",
        "problem": "通用 relevance 或通用 utility 标签不能适配所有生成模型，utility 是模型相关且不可直接迁移的。",
        "method": ["比较不同 LLM 在同一 passage 条件下的收益差异。", "构造 LLM-specific utility 视角。", "强调 retriever 对齐应考虑具体 generator 的偏好。"],
        "experiment": "NQ、TriviaQA、MS MARCO-FQA 等数据集，不同 Qwen/Llama 模型间比较 utility 差异。",
        "limits": ["模型相关不等于状态相关。", "没有进入多步 agent trace。", "未处理 query drift、premature stop 或中间结论支撑。"],
        "mapping": ["SAPR-RAG 可进一步提出 state-specific utility。", "同一文档对同一 LLM 在不同 hop 也可能 utility 不同。", "可作为 state-conditioned reranker 的理论前置。"],
        "module": "State-Conditioned Utility / reranker",
    },
    "2026_Utility_Aligned_Embeddings": {
        "position": "Utility-aligned dense retrieval；把 LLM utility 蒸馏进 bi-encoder。",
        "problem": "基于相似度的 dense retrieval 会召回 semantic distractors，而 LLM reranking 成本高且 utility estimation 有噪声。",
        "method": ["先离线训练或估计 utility reward model。", "将目标 utility distribution 蒸馏到 dense retriever embedding。", "推理时用 bi-encoder 近似 utility-aware retrieval，降低 LLM reranking 成本。"],
        "experiment": "QASPER 等长文档问答任务，使用 Recall、MAP、Token F1 等指标评价检索与生成收益。",
        "limits": ["仍是单步 retrieval alignment。", "没有 history state，也没有 stop/query policy。", "静态蒸馏可能无法适应多跳推理中状态变化。"],
        "mapping": ["可作为 SAPR-RAG 后续 retriever distillation 路线。", "先训练 state-aware reranker，再蒸馏进 dense retriever。", "可支持“utility alignment 仍需动态化”的缺口表述。"],
        "module": "Retriever distillation / Evidence Reward",
    },
    "2025_Evidence_Tree_Search": {
        "position": "证据集合优化；用 tree search 选择多句证据而非完整 agent loop。",
        "problem": "检索长文档常包含冗余或无关内容，缺少对多句证据集合质量的监督。",
        "method": ["把 evidence retrieval 写成 evidence tree。", "用 MCTS 搜索高质量 evidence set。", "强调多句证据之间的协同，而不是单文档相似度。"],
        "experiment": "LongBench 中的 2WikiMultiHopQA、HotpotQA、MuSiQue、MultiFieldQA、Qasper 等任务。",
        "limits": ["不建模 query rewriting 和 stop policy。", "更像 evidence selector / compressor。", "无法直接修复 Agentic RAG 的错误动作。"],
        "mapping": ["可作为 Evidence Reward 的局部评估器。", "可用于判断 selected evidence set 是否支持当前 claim。", "与 SAPR-RAG 的区别是后者把证据评价放回 agent loop。"],
        "module": "Evidence Reward / evidence set scoring",
    },
    "2025_ReasonRAG": {
        "position": "本课题 baseline；从 outcome reward 转向 process reward 的关键论文。",
        "problem": "Search-R1 式 outcome RL 存在探索效率低、梯度冲突、奖励稀疏，无法稳定监督 query/evidence/answer 三类过程。",
        "method": ["RAG-ProGUIDE 为 query generation、evidence extraction、answer generation 构造过程级奖励。", "用少量 process-supervised data 训练 LoRA。", "相比 outcome reward，更直接约束 Agentic RAG 的关键步骤。"],
        "experiment": "HotpotQA、2Wiki、PopQA、MuSiQue、Bamboogle；关注 F1、EM、Acc、Recall、Precision 等指标。",
        "limits": ["我们复现的 badcase 显示仍有复杂 query、证据 rank 靠后、unsupported answer、premature stop。", "没有显式 state-aware evidence utility。", "停搜决策和证据链闭合仍缺少可解释判断。"],
        "mapping": ["作为 SAPR-RAG 的实验平台。", "SAPR-RAG 要说明自己不是替代 process reward，而是把 process reward 细化到 state-aware query/evidence/stop。", "ReasonRAG trajectory 可作为 failure bank 的原始来源。"],
        "module": "Baseline / SAPR-RAG integration",
    },
    "2025_Search_R1": {
        "position": "Outcome-RL Agentic RAG 基线；训练模型在推理中主动搜索。",
        "problem": "模型需要学会何时搜索、如何搜索和如何利用搜索结果，但 Search-R1 主要使用 simple outcome-based reward。",
        "method": ["在推理文本中插入 search action，通过搜索引擎获取外部信息。", "用最终答案正确性作为主要 RL 信号。", "避免复杂过程奖励，强调可扩展的 RLVR 框架。"],
        "experiment": "NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle 等开放域和多跳 QA。",
        "limits": ["最终奖励稀疏，难以定位错误步骤。", "没有显式 faithfulness / supportiveness 判断。", "没有区分 query、evidence、stop 错误来源。"],
        "mapping": ["作为 outcome reward 对照基线。", "用于写 Search-R1 -> ReasonRAG -> SAPR-RAG 的递进逻辑。", "可比较 SAPR-RAG 对过程错误的定位能力。"],
        "module": "Outcome-RL baseline",
    },
    "2026_AgenticRAGTracer": {
        "position": "Hop-aware benchmark；用于诊断 Agentic RAG 多步检索推理在哪一步失败。",
        "problem": "传统 benchmark 通常只有最终问题和答案，缺少中间 hop-level questions，无法定位 agent failure step。",
        "method": ["构造包含 2-hop、3-hop、4-hop 的多步诊断样本。", "区分 inference 与 comparison 等任务类型。", "用 hop-aware diagnosis 分析 premature collapse 与 over-extension。"],
        "experiment": "在多种 LLM 与 Agentic RAG 设置上评估，显示复杂 hop 结构下性能仍然不足。",
        "limits": ["它是诊断基准，不是优化方法。", "自动构造数据需要人工抽样核验。", "与 ReasonRAG trace schema 需要适配。"],
        "mapping": ["指导 failure_bank.jsonl 的字段设计。", "可作为 hop-level evaluation 的重要来源。", "用 collapse rate 和 over-extension rate 评价 Stop/Repair 模块。"],
        "module": "Failure Bank / Hop-Level Evaluation",
    },
}


def read_meta(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("#").strip()
    meta = {"title": title}
    for key in ["Year", "Venue", "Paper", "PDF", "Code", "Task", "Dataset", "Backbone", "Retriever"]:
        m = re.search(rf"^- {key}:\s*(.*)$", text, re.M)
        meta[key.lower()] = m.group(1).strip() if m else "待补充"
    return meta


def key_artifact_refs(stem: str) -> dict[str, str]:
    folder = IMAGES_DIR / stem / "key_figures"
    refs: dict[str, str] = {}
    if not folder.exists():
        return refs
    for kind in ["figure1", "figure2", "table1", "table2"]:
        match = next(iter(sorted(folder.glob(f"{kind}_*.png"))), None)
        if match:
            refs[kind] = f"images/{stem}/key_figures/{match.name}"
    return refs


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render(path: Path) -> str:
    stem = path.stem
    meta = read_meta(path)
    profile = PROFILES[stem]
    refs = key_artifact_refs(stem)
    fig1_md = f"![Figure 1](%s)" % refs["figure1"] if "figure1" in refs else "（未自动定位到 Figure 1）"
    fig2_md = f"![Figure 2](%s)" % refs["figure2"] if "figure2" in refs else "（未自动定位到 Figure 2）"
    table1_md = f"![Table 1](%s)" % refs["table1"] if "table1" in refs else "（未自动定位到 Table 1）"

    return f"""# {meta['title']}

## 0. 阅读结论

- [论文定位] {profile['position']}
- [核心问题] {profile['problem']}
- [对本课题价值] 该文主要映射到 `{profile['module']}`，用于支撑 SAPR-RAG 的问题定义、模块设计或评价协议。
- [阅读判断] 这篇论文不能只当作“相关工作”罗列，应该明确写出它解决了什么、没解决什么，以及 SAPR-RAG 如何沿着它的边界继续推进。

## 1. 核心信息

- Year: {meta['year']}
- Venue: {meta['venue']}
- Paper: {meta['paper']}
- PDF: {meta['pdf']}
- Code: {meta['code']}
- Task: {meta['task']}
- Dataset: {meta['dataset']}
- Backbone: {meta['backbone']}
- Retriever: {meta['retriever']}

## 2. 摘要与问题定义

### 摘要要点

- [论文明确提出] {profile['problem']}
- [论文明确提出] {profile['position']}
- [基于方法/实验设置推断] 该文的核心关注点是 `{profile['module']}`，因此它更适合回答局部过程问题，而不是完整解决 state-aware Agentic RAG 控制。

### 问题定义

这篇论文把问题放在 `{profile['module']}` 这条线上。对本课题来说，关键不是复述其方法名称，而是判断它是否回答了下面三个问题：

1. 当前状态下应该生成什么 query？
2. 当前状态下哪篇 evidence 真正有用？
3. 当前证据是否足以停止并回答？

该文通常只覆盖其中一部分，因此它能成为 SAPR-RAG 的前序工作或对照，而不是完整替代方案。

### 与 ReasonRAG Badcase 的对应

- [本课题 badcase 对齐] ReasonRAG 中的复杂 query、证据 rank 靠后、unsupported intermediate answer、premature stop 等问题，可以用这篇论文的视角重新标注。
- [基于方法/实验设置推断] 若该论文只看路径、faithfulness 或 utility 的某一面，仍需要 SAPR-RAG 把 query、evidence、stop 统一到同一个 reasoning state 下。

## 3. 图表速读

### Figure 1：Motivation / Problem Setting

{fig1_md}

> 图 1 通常承担 motivation 或问题定义作用。阅读时要看作者如何把旧方法的问题可视化，例如 dead end、final-answer reward 过粗、检索噪声、faithfulness 缺失或 utility mismatch。对 SAPR-RAG 来说，这类图用于支撑“为什么需要 state-aware process reward”，而不是只作为装饰图。

### Figure 2：Method / Framework

{fig2_md}

> 图 2 通常对应方法框架或关键模块。阅读时要拆出输入、状态、动作、奖励/监督信号和输出，并判断它覆盖的是 Query Reward、Evidence Reward、Stop Reward、Repair Mechanism 还是 Failure Diagnosis。

### Table 1：Main Results / Dataset Statistics

{table1_md}

> Table 1 往往是主结果表或数据统计表。阅读时不要只抄最高分，而要判断实验是否真的验证了过程质量：是否包含多跳数据集，是否报告 trajectory-level 指标，是否能定位 query/evidence/stop 的错误来源。

关键图表索引：[`images/{stem}/key_figures/index.md`](images/{stem}/key_figures/index.md)

## 4. 方法拆解

### 输入输出

- 输入通常包括原始问题、历史推理轨迹、搜索结果或候选文档。
- 输出可能是下一步 query、是否 search/stop、候选 evidence 排序、过程奖励、或完整 reasoning trajectory。
- 对 SAPR-RAG 来说，统一接口应写成：`state s_t = <question, history queries, history evidence, current subquery, intermediate claim, budget>`。

### 核心模块

{bullets(profile['method'])}

### 训练信号或奖励

- [论文明确提出] 该文使用的监督信号服务于自身目标，例如过程奖励、路径价值、utility label、faithfulness reward 或 benchmark label。
- [基于方法/实验设置推断] 这些信号大多没有同时覆盖 Query Reward、Evidence Reward 和 Stop Reward。
- [本课题扩展] SAPR-RAG 应把局部信号统一为：

$$
R(s_t, a_t) = R_q + R_e + R_s + R_f
$$

其中 $R_q$ 评价 query 是否对准信息缺口，$R_e$ 评价证据是否有状态条件下的边际效用，$R_s$ 评价是否应该停止，$R_f$ 评价中间结论是否被证据支撑。

### 推理流程

1. 根据当前问题和历史状态生成或选择下一步动作。
2. 使用检索器、搜索引擎、reward model 或 judge 获得反馈。
3. 更新轨迹状态，并决定继续、修复或停止。
4. 输出最终答案或诊断轨迹质量。

### 与 ReasonRAG 的接口差异

- ReasonRAG 已经有 query generation、evidence extraction、answer generation 的过程监督。
- 该文提供了不同侧面的补充：`{profile['module']}`。
- SAPR-RAG 的差异在于把这些侧面落到同一个 state-aware reward 框架里，而不是只优化单个动作或单个指标。

## 5. 实验设计与结果解读

### 数据集与设置

{profile['experiment']}

### Baseline 与指标

- 需要重点检查 baseline 是否包含 outcome-only RL、process reward、传统 RAG、search agent 或 utility retriever。
- 若指标只包含 EM/F1/Accuracy，它只能说明最终答案效果；若包含 faithfulness、search count、hop-level diagnosis 或 utility metrics，则更能支撑过程优化结论。
- 对本课题来说，最关键的是把最终答案指标与 trajectory 指标同时报告。

### 主结果解读

- [论文明确提出] 主结果通常证明该文提出的 `{profile['module']}` 方向有效。
- [基于方法/实验设置推断] 如果没有单独报告 query quality、evidence supportiveness 或 stop accuracy，就不能声称它已经解决了 SAPR-RAG 的全部问题。
- [本课题 badcase 对齐] 对 HotpotQA/2Wiki/MuSiQue 这类多跳任务，应该额外看模型是否减少合并多跳 query、错误实体漂移和 evidence=None 后强答。

### 消融与诊断

- 优先看作者是否拆掉 reward、planner、retriever、judge、synthetic data 或 correction module。
- 如果消融只报告最终答案，说明局部机制的因果证据还不够强。
- SAPR-RAG 后续消融应至少包含：去掉 Query Reward、去掉 Evidence Reward、去掉 Stop Reward、去掉 Repair Mechanism。

### 实验严谨性评价

- 优点：该文在其目标问题上提供了可观察指标或有效 baseline。
- 风险：若训练数据、judge prompt、retriever 设置或采样策略不公开，复现时需要先做 B/C 档代码调研。
- 对本课题的要求：不要只引用提升数字，要引用它揭示的问题类型和方法边界。

## 6. 论文贡献、局限与证据强度

### 贡献

- [论文明确提出] {profile['position']}
- [论文明确提出] 它把 `{profile['module']}` 变成可建模、可训练或可诊断的问题。
- [基于方法/实验设置推断] 它推进了 Agentic RAG 从“只看最终答案”向“关注过程质量”的迁移。

### 局限

{bullets(profile['limits'])}

### 证据强度

- [论文明确提出] 可用于支撑该文自身提出的问题和方法贡献。
- [基于方法/实验设置推断] 可用于支撑 SAPR-RAG 的逻辑缺口，但写论文时必须说明这是跨论文归纳。
- [本课题 badcase 对齐] 与 ReasonRAG 复现中的具体失败现象对应，需要在 `failure_bank.jsonl` 中继续积累样本证据。

## 7. 与本课题的关系

### 对 SAPR-RAG 的启发

{bullets(profile['mapping'])}

### 可转化模块

- `{profile['module']}`
- Query Reward：判断 query 是否保留关键实体、避免重复、对准未闭合信息槽。
- Evidence Reward：判断文档是否 relevant、novel、supportive、chain-contributing 且低 noise risk。
- Stop Reward：判断当前 evidence set 是否足以支撑 final answer。

### 与其他论文的关系

- Search-R1 提供 outcome-RL 起点。
- ReasonRAG 证明 process reward 优于 outcome reward。
- VERITAS / ProRAG / DecEx-RAG / HiPRAG 等分别加强 faithfulness、process supervision、decision/execution、search necessity。
- Utility-Focused / LLM-Specific Utility / UAE 证明 relevance 不是 utility，但还没有进入 trajectory-state utility。

## 8. 可复现与代码阅读线索

- 先看 README 中的数据格式、训练入口和 evaluation script。
- 再看 reward / judge / reranker / retriever 相关模块，确认论文中的核心信号如何落到代码。
- 若有官方 checkpoint，优先复现实验表中的一个小数据集结果。
- 若无代码，则至少复现该论文的 prompt schema、评价指标或数据构造思想。

## 9. 可用于写作的中文表述

- 直接证据层：{profile['problem']}
- 谱系定位层：{profile['position']}
- 逻辑缺口层：现有工作虽然推进了 `{profile['module']}`，但仍未系统回答在第 $t$ 步给定当前轨迹状态时，下一条 query、candidate evidence 和 stop action 的边际价值如何统一评估。
- 本课题定位层：SAPR-RAG 试图把这些分散线索统一为 state-aware process reward，用于减少 ReasonRAG 类方法中的 query drift、retrieval noise、unsupported intermediate answer 和 premature stop。

## 10. 后续行动

- 将该论文的核心 failure type 映射到 ReasonRAG badcase 标签。
- 检查官方代码或补充 B/C 档代码调研，确认数据格式和 reward 实现。
- 在 SAPR-RAG 实验中设计一个与该文直接相关的 ablation 或 diagnostic metric。
- 写 related work 时避免泛泛罗列，应突出该文与 SAPR-RAG 的“相同问题、不同粒度、不同状态建模”关系。
"""


def main() -> None:
    for stem in PROFILES:
        path = NOTES_DIR / f"{stem}.md"
        if path.exists():
            path.write_text(render(path), encoding="utf-8")
            print(f"rebuilt {path}")


if __name__ == "__main__":
    main()
