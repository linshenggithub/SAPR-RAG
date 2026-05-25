# SAPR-RAG Idea Draft

> 本文档是待审核和多轮打磨的 idea 草案，不代表最终方法设计。

## 1. 方法定位

SAPR-RAG 暂定为：

> State-Aware Process Repair for Agentic RAG

它不是从零重写一个 Agentic RAG 系统，而是在 ReasonRAG 复现基础上，对多步检索推理轨迹进行状态感知的诊断、打分和修复。

本方案针对两类核心问题：

1. **多步推理过程控制能力不足**
   包括 query drift、query redundancy、bridge entity 丢失、unsupported intermediate answer、premature stop、over-search / under-search。

2. **状态感知证据利用不足**
   包括检索噪声、gold evidence rank 靠后、正确证据被召回但未被使用、证据是否推进当前证据链缺少显式判断。

核心思想是：在 ReasonRAG 每一步推理中，将当前轨迹状态 `s_t` 显式输入到 query、evidence 和 stop 三类 evaluator 中，得到可解释的过程反馈；当某一类反馈较差时，触发 query rewrite、document rerank、continue search 或 reject unsupported claim 等修复动作。

## 2. 状态定义

在第 `t` 步，SAPR-RAG 使用如下状态：

```json
{
  "question": "original complex question",
  "gold_answer": "optional, only for training/eval",
  "step_id": 2,
  "history_subqueries": ["..."],
  "history_evidence": ["..."],
  "current_subquery": "...",
  "retrieved_docs": [
    {
      "doc_id": "...",
      "rank": 1,
      "title": "...",
      "text": "...",
      "score": 0.0
    }
  ],
  "selected_evidence": ["..."],
  "intermediate_claim": "...",
  "remaining_gap": "what is still missing",
  "predicted_answer": "optional",
  "stop_decision": false
}
```

其中 `remaining_gap` 可以先由 LLM judge 根据 question、history evidence 和 intermediate claim 自动生成，后续再用规则或小模型替代。

## 3. 模块设计

### 3.1 Failure Bank

Failure Bank 是所有模块的训练和评测基础。它把 ReasonRAG badcase 从自然语言分析转为 step-level 结构化数据。

建议字段：

```json
{
  "sample_id": "hotpotqa_dev_xxx",
  "dataset": "hotpotqa",
  "question": "...",
  "gold_answer": "...",
  "prediction": "...",
  "step_id": 1,
  "subquery": "...",
  "retrieved_docs": [...],
  "selected_evidence": "...",
  "intermediate_claim": "...",
  "failure_type": "over_complex_query",
  "failure_subtype": "merged_two_hops",
  "repair_action": "rewrite_query",
  "oracle_next_query": "optional",
  "gold_supporting_facts": ["optional"],
  "judge_reason": "..."
}
```

初始 failure type：

| Failure Type | 含义 | 对应修复 |
| --- | --- | --- |
| `over_complex_query` | query 合并多个 hop，太宽或不可检索 | rewrite / decompose query |
| `repeated_query` | query 与历史 query 重复 | rewrite query |
| `missing_bridge_entity` | 下一跳 query 丢失上一跳关键实体 | insert bridge entity |
| `retrieval_noise` | top 文档被同名实体或相似事件污染 | rerank / filter docs |
| `gold_rank_low` | gold evidence 被召回但 rank 靠后 | state-aware rerank |
| `unsupported_claim` | 中间结论没有证据支持 | reject claim / continue search |
| `premature_stop` | 证据不足却停止 | continue search |
| `over_search` | 证据已足够仍继续搜索 | stop |

### 3.2 Query Reward and Query Repair

目标：判断当前 query 是否对准当前未解决的信息缺口。

输入：

```text
original question
history subqueries
history evidence
intermediate claim
current subquery
remaining gap
```

输出：

```json
{
  "query_score": 0.0,
  "is_repeated": false,
  "is_over_complex": true,
  "keeps_bridge_entity": false,
  "targets_remaining_gap": false,
  "failure_type": "over_complex_query",
  "rewrite_query": "Who portrayed Corliss Archer in Kiss and Tell?"
}
```

可先用 LLM judge 实现，再沉淀为训练数据。后续可训练小模型或 7B evaluator。

评分维度：

- Repetition：是否重复历史 query；
- Atomicity：是否只针对一个合理 hop；
- Bridge Preservation：是否保留上一跳关键实体；
- Gap Targeting：是否指向当前未解决的信息槽；
- Retrieval Friendliness：是否适合送入检索器。

修复动作：

```text
bad query -> query evaluator -> rewrite query -> retrieve again
```

### 3.3 State-Aware Evidence Reranker

目标：判断候选文档在当前状态下是否真正有用。

传统 reranker 多输入：

```text
query + document
```

SAPR-RAG reranker 输入：

```text
original question
history subqueries
history evidence
current subquery
intermediate claim
remaining gap
candidate document
```

输出：

```json
{
  "doc_id": "...",
  "evidence_score": 0.0,
  "relevance": 0.0,
  "novelty": 0.0,
  "supportiveness": 0.0,
  "chain_contribution": 0.0,
  "noise_risk": 0.0,
  "label": "positive"
}
```

核心形式：

```text
U(d | q, s_t)
```

其中 `s_t` 包含历史 query、历史 evidence、中间结论、当前未解决缺口和剩余预算。

训练数据构造：

1. 从 ReasonRAG trajectories 中抽取每一步 top-k retrieved docs。
2. 使用 HotpotQA supporting facts 判断文档是否包含 gold evidence。
3. 对没有明确 gold label 的样本，用 LLM judge 辅助判断：
   - 是否回答当前 subquery；
   - 是否提供新信息；
   - 是否支撑 intermediate claim；
   - 是否推进 final evidence chain；
   - 是否是噪声或同名实体干扰。
4. 构造 pairwise preference：
   - positive：gold / useful / chain-contributing doc；
   - negative：irrelevant / noisy / redundant / misleading doc。

训练形式：

- V0：不训练，直接用 LLM judge rerank top-k；
- V1：训练 cross-encoder reranker；
- V2：训练 7B reward model 或 reranker；
- V3：将 reranker 蒸馏到轻量 bi-encoder 或 cross-encoder。

### 3.4 Stop Reward / Evidence Sufficiency Verifier

目标：判断当前 evidence set 是否足以支持最终答案，避免 premature stop 和 unsupported answer。

输入：

```text
original question
history subqueries
selected evidence
intermediate claim
predicted answer
remaining gap
```

输出：

```json
{
  "stop_score": 0.0,
  "is_sufficient": false,
  "has_unsupported_claim": true,
  "missing_information": "the government position held by Shirley Temple",
  "suggested_next_query": "What government position did Shirley Temple hold?"
}
```

判断维度：

- Evidence Coverage：当前 evidence 是否覆盖全部 hop；
- Answer Entailment：final answer 是否被 evidence set 支撑；
- Missing Slot：是否仍有未闭合实体、关系、时间、地点槽；
- Continue Value：继续检索是否可能有收益；
- Unsupported Claim：中间结论是否无证据支持。

修复动作：

```text
if stop_score < threshold:
    continue search with suggested_next_query
else:
    output answer
```

### 3.5 Repair Controller

Repair Controller 根据三个 evaluator 的输出决定下一步动作。

规则版 V0：

```text
if query_score < tq:
    rewrite query
elif evidence_score < te:
    rerank docs or increase top-k
elif stop_score < ts:
    continue search
elif unsupported_claim:
    reject claim and continue search
else:
    proceed
```

后续版本可以把 controller 训练成 policy model，但硕士论文阶段优先做可解释规则控制，降低实现风险。

## 4. 推理流程

V0 pipeline：

```text
Question
  -> ReasonRAG generates subquery
  -> Query Reward evaluates subquery
  -> if bad: rewrite subquery
  -> Retriever returns top-k docs
  -> State-Aware Evidence Reranker reranks docs
  -> ReasonRAG extracts evidence / generates intermediate claim
  -> Stop Reward checks evidence sufficiency
  -> if insufficient: continue search
  -> else: final answer
```

V0 尽量不改 ReasonRAG 主体，只在外部包一层 evaluator / reranker / verifier，便于调试和做消融。

## 5. 需要做的实验

### 5.1 实验 0：ReasonRAG Baseline 复现与轨迹标准化

目的：得到稳定 baseline 和可分析 trajectories。

数据集：

- HotpotQA dev；
- 之后扩展到 2Wiki、MuSiQue、Bamboogle。

产物：

- `trajectories.jsonl`
- `metrics.json`
- `badcases.jsonl`

指标：

- EM / F1 / Acc；
- Recall / Precision；
- average steps；
- retrieval top-k evidence recall。

### 5.2 实验 1：Failure Bank 与错误分布

目的：证明 ReasonRAG 的失败主要集中在两类问题上。

步骤：

1. 从 baseline badcases 中抽取 HotpotQA 错误样本。
2. 将每个错误拆成 step-level 记录。
3. 用 LLM judge 初标 failure_type。
4. 人工抽查并修正一部分样本。
5. 统计 failure distribution。

推荐规模：

- V0：100 个错误样本；
- V1：300-500 个错误样本；
- V2：1000+ step-level records。

指标：

- failure type 占比；
- inter-judge agreement，若有人审；
- 每类错误的 answer EM/F1 损失贡献；
- rerankable failure 占比。

### 5.3 实验 2：State-Aware Evidence Reranker

目的：验证 `U(d | q, s_t)` 是否比普通 query-doc rerank 更适合多跳 Agentic RAG。

对比方法：

```text
ReasonRAG original retrieval order
ReasonRAG + query-doc reranker
ReasonRAG + state-aware reranker
ReasonRAG + oracle reranker upper bound
```

实验设置：

- 对每一步 retrieved top-20 / top-50 文档重排；
- top-k evidence 输入 ReasonRAG 后续 evidence extraction；
- 先在 HotpotQA badcase 上做，再扩展全 dev。

指标：

- gold evidence Recall@5 / Recall@10；
- gold evidence MRR；
- noise@5；
- final EM / F1；
- badcase repair rate；
- average steps。

预期结论：

> 仅用 query-doc relevance 的 reranker 无法充分利用历史 evidence 和中间状态；state-aware reranker 能更好地把当前需要的下一跳证据排到前面。

### 5.4 实验 3：Query Reward / Query Repair

目的：验证 query evaluator 能否减少过宽、重复和丢失 bridge entity 的查询。

对比方法：

```text
ReasonRAG
ReasonRAG + query evaluator only
ReasonRAG + query evaluator + query rewrite
```

指标：

- query repetition rate；
- average query length；
- over-complex query rate；
- bridge entity preservation rate；
- second-hop retrieval success rate；
- final EM / F1。

重点 badcase：

- Corliss Archer / Shirley Temple 类需要先定位实体再查属性的问题。

预期结论：

> Query Repair 能减少合并多跳的大 query，并提升第二跳检索成功率。

### 5.5 实验 4：Stop Reward / Sufficiency Verifier

目的：验证停止充分性判断是否能减少 premature stop 和 unsupported answer。

对比方法：

```text
ReasonRAG
ReasonRAG + stop verifier
ReasonRAG + stop verifier + suggested next query
```

指标：

- premature stop rate；
- unsupported answer rate；
- answer entailment rate；
- average steps；
- over-search rate；
- final EM / F1。

预期结论：

> Stop verifier 能阻止模型在证据不足时输出答案，并通过 suggested next query 修复部分缺失证据链。

### 5.6 实验 5：组合消融

最终组合：

```text
ReasonRAG
+ Query Reward / Repair
+ State-Aware Evidence Reranker
+ Stop Reward / Sufficiency Verifier
```

消融矩阵：

| Method | Query Repair | Evidence Rerank | Stop Verifier |
| --- | --- | --- | --- |
| ReasonRAG | No | No | No |
| +Q | Yes | No | No |
| +E | No | Yes | No |
| +S | No | No | Yes |
| +Q+E | Yes | Yes | No |
| +E+S | No | Yes | Yes |
| Full SAPR-RAG | Yes | Yes | Yes |

主指标：

- EM；
- F1；
- Acc；
- evidence Recall@5 / Recall@10；
- unsupported answer rate；
- premature stop rate；
- badcase repair rate；
- average retrieval steps。

## 6. 数据与文件落点

建议文件组织：

```text
02_baseline_reasonrag/
  trajectories/
    hotpotqa_dev_reasonrag.jsonl
  badcases/
    hotpotqa_badcases.jsonl
  analysis/
    failure_distribution_hotpotqa.md

03_sapr_rag/
  query_reward/
    query_reward_prompt.md
    query_repair_prompt.md
  evidence_reward/
    evidence_reward_prompt.md
    reranker_training_schema.md
  stop_reward/
    stop_reward_prompt.md
    sufficiency_verifier_prompt.md
  reward_prompts/
    judge_common_instructions.md
  scripts/
    build_failure_bank.py
    build_reranker_data.py
    run_sapr_v0.py

04_experiments/
  run_configs/
    hotpotqa_sapr_v0.yaml
  metrics/
    hotpotqa_sapr_v0_metrics.json
  tables/
    hotpotqa_ablation_table.md
```

大文件不进 git，只提交 schema、config、metrics summary 和实验笔记。

## 7. 最小可行版本

V0 目标：先不训练任何模型，只用 prompt-based evaluator 验证方向。

V0 包含：

1. Failure Bank：100 个 HotpotQA badcase；
2. Query Reward：LLM judge + query rewrite prompt；
3. Evidence Reranker：LLM judge rerank top-20；
4. Stop Verifier：LLM judge 判断 sufficient / insufficient；
5. 小规模 badcase repair evaluation。

V0 成功标准：

- 能修复一批典型 badcase；
- 能证明 state-aware rerank 提升 gold evidence rank；
- 能减少 premature stop 和 unsupported answer；
- 有清晰的消融结果。

## 8. 后续版本

V1：训练 evidence reranker。

- 用 Failure Bank + supporting facts 构造 preference pairs；
- 训练 cross-encoder 或 7B reward model；
- 评估 evidence Recall@k 和 final EM/F1。

V2：训练 query / stop evaluator。

- query evaluator 学习判断 query 是否 atomic、是否保留 bridge entity；
- stop evaluator 学习判断 evidence sufficiency。

V3：端到端组合。

- 将 query/evidence/stop 三个 evaluator 接入 ReasonRAG 推理；
- 在 HotpotQA、2Wiki、MuSiQue、Bamboogle 上做完整实验。

## 9. 论文可写贡献

如果实验跑通，可以形成三点贡献：

1. **Failure Diagnosis**
   系统分析 ReasonRAG 在多跳问答中的 step-level failure，归纳出过程控制不足和状态感知证据利用不足两类核心问题。

2. **State-Aware Evidence Utility**
   提出 `U(d | q, s_t)`，将文档效用从 query-document relevance 推进到当前轨迹状态下的证据边际贡献。

3. **Process Repair Framework**
   通过 Query Reward、Evidence Reward、Stop Reward 和 Repair Controller，对 ReasonRAG 类方法进行细粒度过程修复。

## 10. 当前优先级

建议按以下顺序推进：

1. 标准化 ReasonRAG trajectories；
2. 构造 HotpotQA Failure Bank；
3. 做 State-Aware Evidence Reranker V0；
4. 做 Stop Verifier V0；
5. 做 Query Repair V0；
6. 做三模块组合消融；
7. 再决定是否训练 7B reward model / reranker。
