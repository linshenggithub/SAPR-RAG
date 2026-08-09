
---
document_type: ai_implementation_handoff
status: ready_for_implementation
priority: P0
repository: linshenggithub/SAPR-RAG
baseline_commit: fd1a1bdd85131aeff1bc7b1b3394b34bbe85db14
created_at: 2026-08-09
target_environment: Linux + NVIDIA GPU + ms-swift + vLLM
---

# SAPR-RAG 下一轮改造与 GPU 实验交接

> 本文面向在另一台 GPU 机器上继续工作的 AI。目标是把当前诊断转化为可验证的代码改动和最小实验，不再仅依据在线 reward 或不同 pipeline 的指标判断模型是否变好。
>
> 执行原则：先消除数据、reward、pipeline 和评测口径中的混杂，再训练；先 LoRA 小实验，再考虑全参数或 OPSD。

## 0. 任务目标

本轮只解决以下四个问题：

1. 抑制完全重复和语义重复查询，避免 6 轮循环。
2. 将 relevance 从“最终文档并集得分”改为有时序信息的增量证据得分。
3. 让训练、零步基线和 checkpoint 评测使用同一套 raw-document Swift scheduler。
4. 让公开 demo 不再依赖不受约束的生成式 evidence，或至少对 evidence 做原文一致性校验。

Kimi K3 等最新知识问题属于语料时效性问题。本轮只保留接口边界，不得把 Web 搜索结果混入固定 wiki18 benchmark。

## 1. 当前事实

### 1.1 已完成

最新代码已经包含：

- SaprTurnCostORM
- SaprRepeatQueryORM
- SaprMaxTurnORM
- 2Wiki context.content 读取修复
- unique title 与 supporting sentence 对齐
- relevance reward 删除 gold-answer text fallback
- Reward-v2 LoRA 启动脚本
- reward mock sanity check
- agent_infer.py 中的反重复 prompt

关键文件：

- 03_sapr_rag/scripts/grpo/plugin.py
- 03_sapr_rag/scripts/grpo/build_grpo_dataset_mixed.py
- 03_sapr_rag/scripts/grpo/build_grpo_dataset_mixed_opsd.py
- 03_sapr_rag/scripts/grpo/sanity_check.py
- 03_sapr_rag/scripts/grpo/launch_grpo_reward_v2_lora.sh
- 03_sapr_rag/scripts/grpo/run_grpo_opsd_colocate.sh
- 03_sapr_rag/scripts/eval/run_direct_rollout_eval.py
- demo/backend/agent.py
- demo/backend/prompts.py

### 1.2 尚未解决

- SaprRepeatQueryORM 只检测规范化后完全相同的 query，无法识别同义改写。
- SaprRelevanceORM 仍然先合并所有轮次文档，再计算最终 coverage；它不是逐轮增量 reward。
- sequence-level reward 仍广播到整条轨迹，无法直接指出哪轮 query 有效。
- demo 仍使用独立 evidence agent；evidence 是同一个 SFT LoRA 的自由生成文本。
- evidence 不要求是文档原文 span，也没有 source/citation 校验。
- Reward-v2 训练使用 Swift raw-document observation，主 SFT 表和 demo 使用 evidence-extraction，口径不一致。
- wiki18 没有实时信息，任何 reward 都无法召回不存在的 Kimi K3 内容。
- 当前 OPSD teacher 已知 gold evidence，却评价 student 的搜索动作；本轮不得继续扩大 OPSD。

### 1.3 已知实验结论

- 严格 LoRA GRPO-control checkpoint-1000 与 SFT 基本持平。
- Full GRPO 提高 gold title coverage，但回答率降低、重复查询和 max-turn 显著升高。
- OPSD 改坏了 comparison question 的子问题分解。
- Reward-v2 的代码已经提交，但仓库中尚无足以证明其有效的固定集完整结果。

## 2. 开始前检查

执行前先记录，不得直接覆盖本地修改：

~~~bash
git status --short
git branch --show-current
git rev-parse HEAD
git pull --ff-only
~~~

要求：

- 基于不早于 fd1a1bdd85131aeff1bc7b1b3394b34bbe85db14 的 commit。
- 若工作区已有其他人的修改，保留并绕开，不得使用 git reset --hard。
- checkpoint、完整数据集、日志和模型权重不得提交到 Git。
- 所有新实验必须记录 git SHA、数据 SHA256、模型起点、scheduler、prompt version 和随机种子。

## 3. P0-A：验证实际 Reward-v2 数据

### 3.1 增加数据审计脚本

新增：

~~~text
03_sapr_rag/scripts/grpo/audit_reward_v2_dataset.py
~~~

输入一个 JSONL，至少检查并输出 JSON 报告：

- 总样本数与 source 分布。
- train/dev ID 是否重叠。
- gold_titles 是否非空。
- len(gold_titles) == len(gold_sup_sents)。
- HotpotQA/2Wiki 各自 supporting sentence 非空率。
- 重复 question/ID 数。
- 每条 gold title 在 corpus 中的可达性。
- teacher_prompt 是否存在，防止 plain GRPO 意外读到 OPSD 数据。
- 数据文件 SHA256。

必须对实际传给 launcher 的文件运行，而不是只测 mock：

~~~text
data/grpo/hotpotqa_2wiki_train_reward_v2.jsonl
~~~

### 3.2 数据 Gate

满足以下条件才允许训练：

~~~text
train/dev overlap = 0
gold_titles empty = 0
title/sentence length mismatch = 0
2Wiki all-sentence-empty = 0
teacher_prompt rows = 0
source count 与预期一致
~~~

若数据没有通过，不得启动 GPU 训练。

## 4. P0-B：实现运行时重复查询检测

### 4.1 不只依赖 prompt 和 reward

修改 SaprRagScheduler，在每条 trajectory 中保存：

~~~python
{
    "turn": int,
    "query": str,
    "normalized_query": str,
    "docs": list,
    "doc_keys": list,
    "exact_duplicate": bool,
    "semantic_duplicate": bool,
    "max_doc_overlap": float,
    "search_executed": bool,
}
~~~

### 4.2 完全重复

在请求 retriever 之前判断 normalized query 是否已出现。

若完全重复：

- 不再调用 retriever。
- 向下一轮注入明确 observation：

~~~text
This query duplicates a previous query and would return the same documents.
Use a different entity/relation/missing fact, or answer if the evidence is sufficient.
~~~

- 将 search_executed=false 和 exact_duplicate=true 写入 rollout info。
- 不得静默结束 trajectory，模型仍应有一次纠正机会。

### 4.3 语义/结果重复

第一版采用确定性的检索结果重叠，不额外加载大模型：

~~~text
doc_key = normalized title + stable text prefix
overlap = |current_doc_keys ∩ previous_doc_keys| / |current_doc_keys ∪ previous_doc_keys|
semantic_duplicate = max_previous_overlap >= 0.8
~~~

说明：

- 它检测的是“操作上没有新增信息”，不要求 query 文本严格同义。
- 若语义重复已经执行了检索，仍保留本轮 docs，但必须标记并进入 reward。
- 后续可增加 query embedding cosine，相同实验中不要同时改两种判据。

### 4.4 状态清理

确认 scheduler 在 trajectory 完成后清理 self._traj[uuid]，防止长时间训练内存增长或 UUID 状态串扰。增加至少一个多 trajectory 单元测试。

## 5. P0-C：将 relevance 改为增量证据 reward

不要覆盖旧实现。注册新 ORM 名称，例如：

~~~text
sapr_marginal_relevance
~~~

保留 sapr_relevance 以便复现实验。

### 5.1 推荐定义

对第 t 轮首次命中的 gold fact：

~~~text
gain_t = new_gold_hits_t / num_gold
discounted_gain_t = gamma^(t-1) * gain_t
~~~

默认：

~~~text
gamma = 0.9
~~~

总分建议：

~~~text
R_marginal =
    sum(discounted_gain_t)
    - 0.05 * semantic_duplicate_count
    - 0.10 * queries_after_full_coverage
~~~

要求：

- 一个 gold fact 只能首次命中一次。
- answer-text 不得作为 evidence 命中捷径。
- title 与 supporting sentence 继续使用 OR 判定，但必须按同一 gold item 对齐。
- 一旦全部 gold facts 已覆盖，后续 query 计入 queries_after_full_coverage。
- 输出诊断字段：first-hit turn、每轮新增 hit、最终 coverage、full-coverage turn。

### 5.2 Reward-v3 暂定组合

先通过离线 replay 再最终定权重。首个候选：

~~~text
sapr_f1                 1.00
sapr_marginal_relevance 0.15
sapr_format             0.05
sapr_turn_cost          0.02
sapr_repeat_query       0.15
sapr_max_turn           0.50
~~~

不要同时改变学习率、LoRA rank、batch、generation 数和 reward；否则无法归因。

### 5.3 离线 reward replay

新增：

~~~text
03_sapr_rag/scripts/grpo/replay_reward_v2.py
~~~

输入历史保存的 completions/rollout_infos，比较旧 reward、Reward-v2 和候选 Reward-v3。

必须包含成对断言：

1. 正确且及时回答 > 正确但检索六轮。
2. 获取全部证据后回答 > 获取全部证据后继续检索。
3. 两个不同实体的有效查询 > 六次同义查询。
4. 错误短答案 < 经过必要检索的正确答案。
5. 偶然包含 gold answer 的错误文档不能获得满 relevance。

报告：

- 各 reward 与 Cover-EM、answered、max-turn、repeat 的 Spearman 相关。
- group reward 零方差率。
- old/new reward 排名反转的代表样本。
- reward 各分量均值、方差和绝对贡献。

若固定训练数据上的 group 零方差率没有明显下降，不要立即训练。

## 6. P0-D：统一零步基线与 checkpoint 评测

### 6.1 论文主实验的 canonical pipeline

本轮 canonical pipeline 定为：

~~~text
Swift SaprRagScheduler + raw-document user observation
~~~

原因：

- GRPO 实际在该 pipeline 中训练。
- strict HTTP 可以保留完整 rollout info。
- 避免 evidence agent 引入额外生成误差。

不得将以下结果直接放在同一增益表中：

~~~text
agent_infer.py evidence-extraction baseline
vs
Swift raw-document Reward-v2 checkpoint
~~~

### 6.2 评测脚本改造

扩展 run_direct_rollout_eval.py 输出：

- answered
- num_turns
- num_queries
- exact_duplicate_count
- semantic_duplicate_count
- max_doc_overlap
- gold_coverage
- full_coverage_turn
- queries_after_full_coverage
- finish_reason
- 完整 retrieved_steps

不要只把 Top-1 文档转成伪 evidence 后再分析。

新增行为分析脚本：

~~~text
03_sapr_rag/scripts/eval/analyze_rollout_behavior.py
~~~

输出统一 metrics JSON 和按 ID 的 paired comparison。

### 6.3 固定评测集合

从未参与训练的 HotpotQA dev 固定 500 题，固定 seed，并分层覆盖：

- bridge
- comparison
- yes/no
- 旧 badcase 中 repeated query
- 旧 badcase 中 gold evidence 已齐但未回答
- 旧 badcase 中 hallucinated hop

保存固定 ID 文件并提交 Git，不提交大体积模型输出。

### 6.4 必跑矩阵

使用完全相同的 scheduler、prompt、temperature、Top-K 和最大轮次评测：

~~~text
SFT merged zero-step
Reward-v2/v3 checkpoint-100
checkpoint-200
checkpoint-300
checkpoint-500
~~~

每 100 step 评测，不要只看最终 checkpoint。

### 6.5 扩大训练的 Gate

只有同时满足以下条件，才能继续完整 epoch：

~~~text
Cover-EM >= 同 pipeline SFT baseline
LLM-judge accuracy 不下降
answer rate 下降 <= 2 percentage points
max-turn rate 不上升
semantic-repeat rate 相对下降 >= 30%
exact-repeat rate 相对下降 >= 50%
gold coverage 不因过度早停下降 > 2 points
~~~

若只降低轮数但准确率和 coverage 同时下降，判定为 premature-stop reward hacking。

## 7. P0-E：修正 demo 的 evidence 链路

### 7.1 推荐方案：demo 对齐 raw-document scheduler

优先将 demo 改成：

~~~text
reasoning -> query -> retrieve Top-K raw docs -> 下一轮 reasoning
~~~

不再额外调用同一个 LoRA 自由生成 evidence。

前端仍可展示：

- query
- Top-K 文档标题与分数
- 每篇文档的原始 snippet
- 最终答案

不要把 LLM 生成的 evidence 当成检索事实展示。

同步修改：

- demo/backend/agent.py
- demo/backend/prompts.py
- demo/tests/test_app.py
- 必要时前端 trajectory schema

demo 也必须复用与 scheduler 相同的反重复 observation 文案和 query normalization helper，避免三份 prompt 漂移。

### 7.2 兼容方案：保留 evidence agent 时的最低要求

若暂时不能移除 evidence agent，则必须：

1. 输出 source index/title。
2. evidence 必须能在对应 doc text 中做归一化连续子串匹配。
3. 校验失败时强制设为 None，不得进入下一轮 history。
4. 对无关 Top-K 建立负例测试，统计 false-evidence rate。
5. history 同时保留 source 标识，不能只保留一条无来源摘要。

至少新增 100 条 evidence micro-eval：

- 直接相关
- 部分相关
- 同名错误实体
- 完全无关

报告：

~~~text
None precision/recall
exact-span rate
false-evidence rate
source-attribution accuracy
~~~

## 8. P1：最新信息路由，仅用于公开 demo

wiki18 不可能回答最新发布模型。研究 benchmark 与实时 demo 必须分开。

新增独立接口边界，例如：

~~~text
demo/backend/freshness.py
~~~

第一版只做路由，不强制接入具体搜索供应商：

- 检测 latest、recent、released、年份、中文“最新/发布/今年”等时效词。
- 低检索分数、实体冲突或连续两轮无新增文档时标记 needs_fresh_source=true。
- 未配置 Web provider 时明确返回“固定知识库不包含足够的新信息”，不得编造。
- 配置 Web provider 后，回答必须带 URL、标题和发布日期。

固定 wiki18 benchmark 中必须关闭该路由，避免评测污染。

## 9. OPSD 暂停条件

在 Reward-v2/v3 通过固定 500 题 Gate 前：

- 不继续 full-parameter GRPO。
- 不继续完整 OPSD。
- 不调大 teacher KL。
- 不使用 gold-evidence teacher 直接塑造 query token。

以后恢复 OPSD 时，至少满足一项：

1. teacher 只作用于 answer token，不作用于 query token。
2. teacher 只看到 student 已经检索到的 observation。
3. teacher log-ratio 做裁剪且保持 environment advantage 符号。
4. SFT+DPO 起点和 OPSD checkpoint 使用完全相同 scheduler 评测。

## 10. 测试要求

### 10.1 单元测试

至少覆盖：

- query normalization。
- 完全重复在检索前被拦截。
- 文本不同但 Top-K 高重叠被标记 semantic duplicate。
- 新 gold fact 只奖励一次。
- full coverage 后继续查询被惩罚。
- answer fallback 不产生 relevance。
- max-turn answered/unanswered 分支。
- scheduler trajectory 结束后状态清理。
- 两个并发 UUID 不串轨迹。
- 2Wiki title/sentence 全量对齐。

### 10.2 集成测试

按顺序执行：

~~~bash
python 03_sapr_rag/scripts/grpo/sanity_check.py --skip_daemon
# 启动 retrieval daemon 后
python 03_sapr_rag/scripts/grpo/sanity_check.py
# 对 Reward-v2/v3 launcher 执行 DRY_RUN
# 运行 10 条 strict rollout
# 运行固定 500 条 gate
~~~

实际命令参数应以仓库脚本 --help 为准。若命令与本文冲突，以代码的最新 CLI 为准，并在 commit 中同步修正文档。

## 11. 提交顺序

不要把所有改动压成一个不可审查的大 commit。建议顺序：

1. fix(data): audit and validate reward-v2 dataset
2. feat(reward): add marginal evidence and semantic repeat signals
3. feat(eval): add strict rollout behavior gates
4. fix(demo): align demo with grounded raw-document observations
5. docs(results): record fixed-500 reward-v2/v3 results

每个 commit：

- 只包含该阶段代码与小型测试。
- commit message 写明行为变化。
- 不提交模型、完整输出、缓存、私有路径或 API key。
- push 后记录 commit SHA。

## 12. 实验记录模板

每次 run 必须生成一份可提交的小型 Markdown/JSON 汇总：

~~~yaml
run_name:
git_sha:
base_model:
adapter:
dataset_sha256:
train_sources:
scheduler:
prompt_version:
reward_names:
reward_weights:
seed:
max_steps:
checkpoint:
eval_ids_sha256:
eval_pipeline:
top_k:
max_turns:
temperature:
cover_em:
llm_acc:
answer_rate:
max_turn_rate:
exact_repeat_rate:
semantic_repeat_rate:
gold_coverage:
queries_after_full_coverage:
group_zero_std_rate:
result:
~~~

result 只能是：

~~~text
PASS_GATE
FAIL_ACCURACY
FAIL_TERMINATION
FAIL_REPETITION
FAIL_RETRIEVAL
INVALID_PIPELINE
INVALID_DATA
~~~

## 13. Definition of Done

只有以下全部完成，本轮才算结束：

- [ ] 实际 Reward-v2 数据通过全量审计。
- [ ] exact duplicate 有运行时拦截。
- [ ] semantic/document-overlap duplicate 有记录和 reward。
- [ ] marginal evidence reward 有离线 replay 结果。
- [ ] SFT zero-step 与所有 checkpoint 使用同一 strict scheduler。
- [ ] 固定 500 题结果包含准确率、回答率、max-turn 和重复查询指标。
- [ ] 至少一个 checkpoint 通过联合 Gate，或得到清晰的失败归因。
- [ ] demo 不再展示未经原文校验的生成式 evidence。
- [ ] Kimi K3 类问题在无实时源时明确拒绝编造。
- [ ] 所有实现、测试和小型结果汇总已分 commit push。
- [ ] 未泄露私有路径、密钥、个人信息或未公开模型权重。

## 14. 最终决策规则

- Reward-v2/v3 若只减少查询但降低 gold coverage：减弱 turn penalty，检查过早停止。
- gold coverage 上升但 answer rate 下降：加强 termination/answer reward，不扩大 relevance。
- repeat 下降但 Cover-EM 不升：检查 evidence/pipeline，而不是继续加重复惩罚。
- raw-document strict 有增益、evidence demo 无增益：问题在 evidence agent，优先移除或做 span 校验。
- 固定 wiki18 对最新问题失败：属于 corpus freshness，不得归因给 GRPO。
- 500 题 Gate 不通过：停止 full epoch、full parameter 和 OPSD，先修信号。
