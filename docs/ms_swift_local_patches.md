# ms-swift 本地修改与迁移手册

## 1. 文档目的

SAPR-RAG 依赖同级目录中的 `ms-swift` 源码。当前训练链路不是只依赖官方
ms-swift，而是包含一组尚未提交到上游仓库的本地修改。

如果只迁移 SAPR-RAG 而没有恢复这些修改，可能出现以下问题：

- 日志显示加载了 SFT/GRPO LoRA，但 rollout 实际使用基础模型；
- 多轮工具交互后的最终 Answer token 没有完整进入训练；
- teacher 与 student 的 completion token 数不一致；
- Query/Answer 分动作 OPSD 退化为全轨迹 teacher；
- failed-only external-teacher OPD gate 失效；
- 纯 OPSD 仍混入 GRPO advantage；
- 多轮 teacher API logprob 错位。

本文是这些外部仓库修改的权威迁移入口。完整补丁保存在：

```text
patches/ms-swift/sapr-rag-ms-swift-full-1dbd1bf64.patch
```

## 2. 固定上游基线

完整补丁基于以下 ms-swift 状态生成：

| 项目 | 值 |
|---|---|
| 上游仓库 | `https://github.com/modelscope/ms-swift.git` |
| 基线 commit | `1dbd1bf64a46bd6bb710d9ace05d529ff071cd1f` |
| commit 标题 | `fix(template): keep MiMo dependencies optional (#9886)` |
| commit 日期 | `2026-08-11T16:08:41+08:00` |
| ms-swift 版本 | `4.5.0.dev0` |
| 完整补丁 SHA256 | `710ec851f05ce05a5d684c784bfde083e310b91ac88fa8072a104a58c35dc7dd` |

补丁统计：

```text
16 files changed
2390 insertions
60 deletions
```

补丁包含的 16 个文件：

```text
swift/dataset/preprocessor/core.py
swift/infer_engine/grpo_vllm_engine.py
swift/pipelines/infer/rollout.py
swift/rl_core/action_credit.py          # 新增：Evidence-Attributed GRPO 数值核心
swift/rl_core/advantage.py
swift/rl_core/data.py
swift/rlhf_trainers/args_mixin.py
swift/rlhf_trainers/gkd_helpers.py
swift/rlhf_trainers/grpo_trainer.py
swift/rollout/multi_turn.py
tests/utils/test_action_credit.py        # 新增：动作级信用单元测试
tests/utils/test_action_causal.py        # 新增：完整动作因果优势单元测试
tests/utils/test_signed_query_reweight.py # 新增：符号保持 Query 重加权单元测试
tests/utils/test_causal_return.py       # 新增：因果 return-to-go 与精确 Query 标签段测试
tests/utils/test_multi_teacher.py
tests/utils/test_teacher_advantage.py
```

以下本地调试备份明确不属于补丁：

```text
swift/rlhf_trainers/utils.py.bak_20260803
```

## 3. 一键迁移

### 3.1 准备干净的 ms-swift

假设 SAPR-RAG 与 ms-swift 是同级目录：

```text
workspace/
├── SAPR-RAG/
└── ms-swift/
```

进入 SAPR-RAG 根目录后执行：

```bash
git -C ../ms-swift fetch origin
git -C ../ms-swift checkout 1dbd1bf64a46bd6bb710d9ace05d529ff071cd1f
git -C ../ms-swift status --short
```

应用前必须确认 ms-swift 工作区干净。不要在存在其他未提交修改时直接应用。

### 3.2 校验并应用完整补丁

```bash
sha256sum patches/ms-swift/sapr-rag-ms-swift-full-1dbd1bf64.patch

git -C ../ms-swift apply --check \
  "$(pwd)/patches/ms-swift/sapr-rag-ms-swift-full-1dbd1bf64.patch"

git -C ../ms-swift apply \
  "$(pwd)/patches/ms-swift/sapr-rag-ms-swift-full-1dbd1bf64.patch"
```

预期 SHA256：

```text
710ec851f05ce05a5d684c784bfde083e310b91ac88fa8072a104a58c35dc7dd
```

应用后确认：

```bash
git -C ../ms-swift diff --check
git -C ../ms-swift diff --stat
```

### 3.3 确认 Python 实际导入该 checkout

推荐 editable install：

```bash
python -m pip install -e ../ms-swift
```

然后检查：

```bash
python - <<'PY'
import swift
print(swift.__file__)
PY
```

输出必须指向当前 `../ms-swift/swift/`，不能指向其他 site-packages 副本。

### 3.4 旧最小补丁的关系

仓库还保留：

```text
patches/ms-swift/0001-opsd-teacher-response-loss-mask.patch
```

它只包含早期 `response_loss_mask` 修复，作为历史最小补丁保留。

**应用完整补丁时不要先应用这个旧补丁**，因为完整补丁已经包含相同修改，
重复应用会冲突。

## 4. 修改分组总览

| 分组 | 目的 | 普通 GRPO | OPSD | 外部 OPD |
|---|---|---:|---:|---:|
| A. rollout LoRA 正确加载 | 确保采样模型真的是指定 adapter | 必需 | 必需 | 必需 |
| B. 多轮最终 turn 收集 | 保存最后 Answer 的 token/mask/logprob | 必需 | 必需 | 必需 |
| C. scoped teacher 数据契约 | 保留三类 teacher prompt | 不使用 | 必需 | 可选 |
| D. teacher/student token 对齐 | 多轮 observation 与 padding 正确映射 | 不使用 | 必需 | 必需 |
| E. 动作级 teacher mask | 只监督 Query/Evidence/Answer 指定动作 | 不使用 | 必需 | 可选 |
| F. teacher sequence gate | 只纠正失败轨迹 | 不使用 | 可选 | 必需 |
| G. 纯 OPD/OPSD | 可关闭基础 GRPO advantage | 不使用 | 必需 | 必需 |
| H. RLSD/SDAR scope | 替代蒸馏目标也遵守动作 mask | 不使用 | 可选 | 可选 |
| I. Evidence-Attributed GRPO | 逐轮证据增益优势只加到对应 Query token | 可选 | 可选 | 可选 |

注意：

> B（GRPO-only）仍使用标准 sequence-level GRPO 数学目标。它依赖 A/B 两组
> 正确性修复，但不会进入 teacher、动作 mask 或纯 OPSD 分支。

> I（Evidence-Attributed GRPO）由 `action_credit_mode` 门控，默认 `off`。默认关闭时
> 与标准 GRPO **逐位一致（bit-identical）**：不读取 rollout_infos、不 gather、不构建
> turn-index map、不改动任何 advantage。只有显式 `--action_credit_mode query_evidence`
> 才会进入该分支。

## 5. 逐文件修改说明

### 5.1 `swift/pipelines/infer/rollout.py`

#### 问题

`swift rollout --adapters <checkpoint>` 虽然解析并打印 adapter 参数，但
`SwiftRolloutDeploy.get_infer_engine()` 创建 `GRPOVllmEngine` 时没有传入
`args.adapters`。

结果可能是：

```text
日志显示 SFT/GRPO checkpoint
实际 rollout 使用未加载 LoRA 的基础模型
```

这会破坏 on-policy 假设，也会使 checkpoint 评测无效。

#### 修改

构造 inference engine 时增加：

```python
'adapters': args.adapters,
```

#### 影响

- 普通 GRPO、OPSD、OPD 的 rollout 都依赖；
- 不改变 GRPO loss，只修复实际采样策略；
- 迁移后必须通过“同 prompt 的 rollout 输出与直接 LoRA 推理一致”验证。

---

### 5.2 `swift/infer_engine/grpo_vllm_engine.py`

#### 问题

不同 vLLM engine 版本并不都暴露 `list_loras()`。原实现无条件调用该方法，
可能在查询 LoRA 是否加载时直接报错。

#### 修改

通过 `getattr(self.engine, 'list_loras', None)` 检查接口：

- 接口存在：保持原来的同步/异步兼容逻辑；
- 接口不存在：返回已有 `adapter_request`，不在这里崩溃。

#### 影响

这是版本兼容修复，不改变 LoRA 权重或训练目标。

---

### 5.3 `swift/rollout/multi_turn.py`

#### 问题

在多轮工具/环境交互后，如果模型生成新的最终 assistant turn，部分路径没有把
最后一轮 token、loss mask 和 rollout logprob 追加到累计结果。

典型轨迹：

```text
Query
-> retrieval observation
-> Query
-> retrieval observation
-> final Answer
```

最后 Answer 可能未完整进入训练 frame。

#### 修改

当已有前序多轮响应且出现新的 final assistant turn 时，追加：

```python
total_response_ids.append(final_token_ids)
total_response_loss_mask.append([1] * len(final_token_ids))
total_rollout_logprobs.append(current_logprobs)
```

#### 影响

- 普通多轮 GRPO 也依赖；
- 保证最终 Answer 真正参与 loss；
- 保证后续动作 mask 能找到最后 Answer；
- observation 仍由 `response_loss_mask=0` 排除。

---

### 5.4 `swift/dataset/preprocessor/core.py`

#### 问题

数据预处理器只认识旧的 `teacher_prompt`。新的动作级字段可能在预处理阶段被
视为额外列而丢失。

#### 修改

将以下字段加入标准保留字段：

```text
teacher_query_prompt
teacher_evidence_prompt
teacher_answer_prompt
```

#### 影响

支持 JSONL 数据携带三类 teacher view。普通 GRPO 不读取这些字段。

---

### 5.5 `swift/rl_core/data.py`

该文件扩展 on-policy sample 与 batch 的数据契约。

#### 新增 scoped teacher 字段

`OnPolicySample` 新增：

```python
teacher_query_prompt
teacher_evidence_prompt
teacher_answer_prompt
teacher_messages_by_scope
```

#### 新增 teacher view 选择

`get_teacher_prompt(scope)` 根据 `query/evidence/answer` 返回对应 prompt。

`build_teacher_view(scope, fallback_to_student)` 支持：

- Query privilege 追加到第一个 user turn；
- Evidence/Answer privilege 追加到最后一个 user turn；
- 旧 unscoped `teacher_prompt` 继续替换最后一个 user message；
- 混合 batch 中缺少某类 prompt 时，可构造 identity fallback；
- 不原地修改 student messages。

#### 修复 `response_loss_mask` 透传

teacher view 除了共享 `response_token_ids`，还共享：

```python
d['response_loss_mask'] = self.response_loss_mask
```

否则 teacher 会把 retrieval observation 误认为 completion token，导致：

```text
OPSD response length mismatch
```

#### 新增训练状态

`GRPOSample` 新增：

```text
teacher_sequence_gate
```

`GRPOBatch` 新增：

```text
teacher_action_mask
teacher_action_coef_mask
```

它们分别表示：

- 哪些序列允许 teacher；
- 哪些 token 属于 teacher 目标动作；
- 每个 token 应使用哪个 Query/Evidence/Answer 系数。

---

### 5.6 `swift/rlhf_trainers/args_mixin.py`

`GRPOArgumentsMixin` 新增以下参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `teacher_sequence_gate` | `none` | `none/failed_em/failed_f1` |
| `teacher_sequence_gate_threshold` | `1.0` | failed-F1 gate 阈值 |
| `opd_use_grpo_advantage` | `true` | 是否保留基础 GRPO advantage |
| `teacher_action_scope` | `all` | `all/query/evidence/answer/multi` |
| `teacher_query_kl_coef` | `0.0` | Query teacher 系数 |
| `teacher_evidence_kl_coef` | `0.0` | Evidence teacher 系数 |
| `teacher_answer_kl_coef` | `0.0` | Answer teacher 系数 |
| `action_credit_mode` | `off` | `off/query_evidence`，动作级证据信用总开关 |
| `action_credit_coef` | `0.0` | 逐轮证据增益优势的系数 |
| `action_credit_scale` | `group_turn` | `group_turn/group_turn_center/none`，逐 turn-slot 组归一化方式 |
| `action_credit_infos_key` | `query_evidence_gain` | rollout_infos 中承载逐轮原始向量的 key |
| `action_credit_gate` | `all` | `all/zero_outcome`；后者只救援 sequence advantage 整组为零的 prompt |
| `advantage_mode` | `sequence` | `sequence/action_causal/signed_query_reweight/causal_return`；最后一项启用 E21 剩余 evidence return 路由 |
| `action_query_outcome_coef` | `0.25` | action-causal 模式下 Query 保留的长程 Answer F1 advantage 系数 |
| `action_credit_clip` | `2.0` | signed-query 模式下逐轮信用裁剪上限；要求 `action_credit_coef * action_credit_clip < 1` |

默认值保持旧路径：

```text
sequence advantage 模式
+ 无 gate
+ 保留 GRPO advantage
+ teacher 作用全 completion
+ 分动作系数关闭
```

因此不显式传新参数时，普通 GRPO 行为不变。

---

### 5.7 `swift/rl_core/advantage.py`

#### 新增失败轨迹 gate

`compute_teacher_sequence_gate()` 根据原始 reward 列选择 teacher 轨迹：

```text
none       -> 全部序列
failed_em  -> EM < threshold 或非有限值
failed_f1  -> F1 < threshold 或非有限值
```

它同时兼容注册名：

```text
sapr_em / SaprEMORM
sapr_f1 / SaprF1ORM
```

缺少 gate 所需 reward 时直接报错，避免静默失效。

#### 扩展 per-token advantage

`expand_advantage_to_per_token()` 由原来的：

```text
A_t = A_GRPO + beta * teacher_logratio_t
```

扩展为：

```text
A_t =
  use_base_advantage * A_GRPO
  + sequence_gate
    * action_mask
    * action_coef
    * (logp_teacher_t - logp_student_t)
```

支持：

- 只在目标动作施加 teacher；
- Query/Evidence/Answer 使用不同系数；
- 只纠正失败序列；
- `use_base_advantage=false` 时实现纯 OPD/OPSD。

#### 限制 RLSD 作用范围

`apply_rlsd_reweight()` 新增 `teacher_action_mask`：

- 目标动作按 teacher/student gap 重加权；
- 非目标 token 的 reweight 固定为 1；
- 防止 action-scoped 配置仍影响整条 completion。

---

### 5.8 `swift/rlhf_trainers/gkd_helpers.py`

该文件负责 teacher view 编码、API logprob 对齐和动作 token 定位。

#### scoped teacher view 编码

`encode_teacher_view()` 接收：

```text
scope
fallback_to_student
```

并调用对应的 Query/Evidence/Answer teacher view。

#### 多轮 full-frame logprob 对齐

旧逻辑假设 response token 位于 full sequence 尾部且连续。多轮 RAG 中 assistant
turn 之间夹着 user/environment observation，因此该假设不成立。

修改后 `assemble_teacher_completion_logprobs()` 使用：

```text
input_ids
attention_mask
completion_mask
```

进行绝对位置映射，并处理：

- vLLM 缺失首个 prompt token logprob；
- 左 padding；
- `logits_to_keep` 只保留 suffix frame；
- 多个 assistant span 被 observation 分隔。

旧的 `response_token_ids` 尾部匹配仍作为单轮兼容 fallback。

#### 动作 token mask

新增 `build_teacher_action_mask()`：

1. 按 `response_token_ids` 逐 turn 处理；
2. 从 cursor 开始在完整 encoded frame 中精确匹配 token ID 子序列；
3. 使用 `response_loss_mask` 排除 observation；
4. 根据协议标签分类：

```text
answer > query > evidence
```

5. 只将对应动作 token 标记为 teacher scope；
6. chat template 边界 token 不自动获得 teacher 信号；
7. 对 token 数、batch 数和对齐失败做 fail-fast 校验。

cursor 顺序匹配保证两轮完全相同的 Query 也不会对齐到同一位置。

---

### 5.9 `swift/rlhf_trainers/grpo_trainer.py`

这是所有新增参数进入实际训练的总控位置。

#### 参数接线与配置校验

Trainer 保存并验证：

- sequence gate；
- action scope；
- 三类动作系数；
- 是否保留 GRPO advantage；
- pure OPD 与 RLSD/SDAR 的互斥关系；
- multi scope 暂不支持 teacher API；
- multi scope 至少有一个非零动作系数。

#### 序列 gate 写入 sample

在原始 reward 完成、group normalization 之前读取 EM/F1 列，为每条 sample
构造 `teacher_sequence_gate`，随后分发到各训练 rank。

#### multi-scope teacher forward

`teacher_action_scope=multi` 时：

1. 为每个非零动作系数构造独立 action mask；
2. 为该 scope 编码独立 teacher prompt；
3. 分别运行 teacher forward；
4. 将各 scope 的 teacher logprob scatter 回统一 student frame；
5. 检查不同 scope mask 不重叠；
6. 缺少某类 teacher prompt 的样本使用 student identity view，不产生伪信号。

#### advantage 融合

将以下量传给 `expand_advantage_to_per_token()`：

```text
base GRPO advantage
teacher per-token logprob
action mask
per-action coefficient map
sequence gate
use_base_advantage
```

因此支持三种主要模式：

```text
普通 GRPO：
  A_t = A_GRPO

GRPO + 分动作 OPSD：
  A_t = A_GRPO + beta_action * teacher_logratio_t

纯分动作 OPSD：
  A_t = beta_action * teacher_logratio_t
```

#### 指标

新增：

```text
teacher_kl_scoped
teacher_action_scope_ratio
teacher_sequence_gate_ratio
teacher_kl_scoped_query
teacher_kl_scoped_evidence
teacher_kl_scoped_answer
```

用于确认 teacher 信号是否实际生效及其覆盖比例。

#### RLSD/SDAR 兼容

- RLSD reweight 只作用于 teacher action mask；
- SDAR loss 使用 `completion_mask & teacher_action_mask`；
- 避免配置为 action-scoped 后替代蒸馏目标仍污染非目标动作。

---

### 5.10 `tests/utils/test_multi_teacher.py`

此文件原本包含 multi-teacher routing 测试，本地修改在同一文件中新增以下验证：

- scalar teacher coefficient 仍保持兼容；
- action mask 只改变目标 token；
- per-action coefficient map 正确作用；
- Query privilege 追加到第一个 user turn；
- Answer privilege 追加到最后一个 user turn；
- 缺失 scope 时 identity fallback；
- 多轮 observation 间隔下的 full-frame logprob 对齐；
- 左 padding 对齐；
- `logits_to_keep` suffix frame 对齐；
- Answer 对 Query 的分类优先级；
- environment loss mask；
- chat template boundary token 排除。

该文件名不表示本补丁新增了 multi-teacher routing；routing 主体是上游已有能力，
这里只扩展了相关回归测试。

---

### 5.11 `tests/utils/test_teacher_advantage.py`

新增以下 advantage 回归测试：

- `failed_em` 只选择错误序列；
- `failed_f1` 遵守阈值；
- reward 注册类名与短名均可识别；
- gate 依赖的 reward 缺失时 fail-fast；
- 纯 OPD/OPSD 清零 base GRPO advantage；
- sequence gate 只允许目标序列接收 teacher log-ratio。

### 5.12 `swift/rl_core/action_credit.py`（新增文件）

Evidence-Attributed GRPO（动作级证据信用分配）的**任务无关数值核心**，纯 tensor/list
函数，不依赖任何 trainer 状态，便于单测。解决的问题：标准 GRPO 只有一个 per-sequence
标量 advantage 广播给整条轨迹的每个 token；一旦最终答案错误，找到关键证据的那一轮
`<query>` 也会被一起惩罚——这就是长程任务的信用分配问题。

在保留原有 outcome advantage 的前提下，新增一个**逐轮**优势项，只 scatter 到产生它的
那一轮 token 上：

```text
A_token(t) = A_outcome(seq)            # 现有 GRPO 广播
           + coef * A_action(turn(t))  # 新增：只加到该轮 token
```

核心函数：

- `gate_action_credit_by_outcome(credit_by_sample, prompt_ids, outcome_advantages, mode)`
  在 `mode='zero_outcome'` 时，仅保留 sequence-level GRPO advantage 整组为零的
  prompt 的动作信用；标准 GRPO 已能排序的组全部置零，避免动作信号与 outcome
  advantage 重复或冲突。默认 `mode='all'` 保持 E17 行为。
- `normalize_action_credit_by_group(credit_by_sample, prompt_ids, num_generations, scale)`
  按 prompt 分组，再**逐 turn-slot** 归一化：某 prompt 下第 `k` 个 query 轮跨 `K` 条
  rollout 组成一个归一化组（GiGPO step-grouped micro-advantage 的多轮类比）。轨迹 query
  数不同，slot `k` 只归一化真正拥有第 `k` 轮的那些轨迹。`scale='group_turn'` 做
  center+scale（除以 `std+1e-4`，unbiased），`'group_turn_center'` 仅 center，`'none'`
  原样返回。单成员/零成员 slot 无组内对比 → 置零。
- `scatter_turn_credit_to_tokens(turn_credit, turn_index_map, completion_mask)`
  按 `turn_index_map[b,t]` 里存的 query-turn 序号，把 `turn_credit[b][k]` 铺到
  `(idx==k) & completion_mask` 的 token 上，返回 `[B,T]` 附加项（未乘系数）。

- `normalize_sequence_credit_by_group(values, num_generations)`
  对单个序列 reward 分量按 prompt 的 rollout 组执行标准 GRPO 归一化。E19 用它
  从 `SaprF1ORM` 单独构造 Answer advantage，避免 relevance/format 混入 Answer。
- `compose_action_causal_advantages(answer_advantages, turn_credit, query_turn_index_map,
  answer_mask, completion_mask, query_outcome_coef, query_credit_coef)`
  将独立分量按动作 token 路由：Query 接收 `alpha*A_F1 + lambda*A_query_gain(k)`，
  Answer 接收 `A_F1`，其他 completion token、observation 和 padding 为 0；Query 与
  Answer mask 重叠时 fail-fast。
- `compose_outcome_signed_query_advantages(sequence_advantages, turn_credit,
  query_turn_index_map, completion_mask, credit_coef, credit_clip)`
  完整保留标准 GRPO advantage，只在 Query token 上增加
  `lambda*abs(A_seq)*clip(A_query_gain)`。强制 `lambda*clip<1`，从数值上保证
  非零 advantage 不会翻转符号。
- `compute_causal_query_return_advantages(...)`
  从 B 的原始复合 reward 中移除 terminal relevance，再加入第 k 轮之后仍可取得的
  evidence gain 总和并按 prompt/turn-slot 归一化。第一轮以及无有效局部对比的槽位
  回退到标准 sequence advantage；逐轮 gain 与 terminal relevance 不一致时 fail-fast。
- `compose_causal_return_advantages(...)`
  只在精确 Query 标签段写入上述 turn advantage，其他 completion token 保持标准 GRPO。

原始逐轮信号（“本轮新覆盖 gold evidence 数 / gold 总数”）由**任务侧** SAPR reward plugin
产出（见下方数据通道），本文件只负责归一化与 token scatter。

### 5.8 补充：`build_query_turn_index_map`（gkd_helpers.py）

`gkd_helpers.py` 除动作 mask 外，新增 `build_query_turn_index_map(samples,
completion_mask, tokenizer, input_ids=None, payload_only=False)`，返回 `[B,T]` long
张量：每个 response token 存它所属 **query 轮的序号**（0-based，仅数 `<query>` 轮），
非 query 轮/模板/observation 为 `-1`。E21 设置 `payload_only=True`，基于原始 token 的逐 token 解码字符区间
的 offset mapping 只标记 `<query>...</query>` 标签段，排除同一 assistant turn 中的
前置推理文本。它**完全复用** `build_teacher_action_mask` 的多轮 token-子序列对齐与
`classify_action`（answer > query > evidence 优先级）逻辑，保证与动作 mask 同一套对齐语
义。序号 `k` 与任务侧逐轮向量下标一一对应（第 `k` 个 query 轮 ↔ 向量第 `k` 项）。

### 5.9 补充：`grpo_trainer.py` 的动作级信用装配

全部逻辑严格 gated 在 `self.action_credit_mode != 'off'`，默认关闭时零开销、bit-identical：

- `__init__`：读取 5 个 `action_credit_*` 参数并做前置校验——`mode!=off` 时要求
  `coef!=0`、要求 `opd_use_grpo_advantage=true`（本项是 advantage 的加项）、不兼容
  `advantage_reweight=rlsd`。
- `_compute_action_turn_credit(samples)`：从 `rollout_infos[action_credit_infos_key]` 读
  本地逐轮原始向量 → `gather_object` 跨进程汇总 → `normalize_action_credit_by_group` 按
  prompt 组逐 turn-slot 归一化 → 按 `action_credit_gate` 使用全局 outcome advantage
  门控 → `get_even_process_data` 切回本地。监控 `action_credit_gate_group_ratio`。
- `_postprocess_batch`：`mode!=off` 时把归一化后的逐轮向量写入
  `samples[i].action_turn_credit`（随 SP all-gather 存活到 mini-batch 循环）。
- mini-batch 循环：`build_query_turn_index_map` + `scatter_turn_credit_to_tokens` 得到
  per-token 附加项，`grpo_batch.advantages += action_credit_coef * credit_term`；并记录
  `action_credit_token_ratio`、`action_credit_abs_mean` 两个监控指标。

**完整 Action-Causal 模式**：`advantage_mode=action_causal` 时，trainer 额外从
`SaprF1ORM` 计算并保存 `answer_advantage`，构造 Query turn map 与 Answer mask 后调用
`compose_action_causal_advantages()`。旧 additive 分支只在 `advantage_mode=sequence`
时执行，防止同一信号重复叠加。该模式要求 `action_credit_mode=query_evidence`、
`action_credit_gate=all`、GRPO group scaling，并明确禁止 teacher、RLSD、SDAR 和
dynamic sampling 的未验证组合。

新增监控：`action_answer_zero_std`、`action_answer_advantage_abs_mean`、
`action_credit_nonzero_group_ratio`、`action_query_token_ratio`、
`action_answer_token_ratio`、`action_unrouted_token_ratio`、
`action_query_advantage_abs_mean`、`action_answer_token_advantage_abs_mean`。

`GRPOSample`（data.py，分组 I）相应新增 `action_turn_credit: Optional[List[float]]`、
`answer_advantage: Optional[torch.Tensor]` 和
`query_turn_advantage: Optional[List[float]]`。

**Causal Return-to-Go 模式**：`advantage_mode=causal_return` 时，trainer 读取现有
`SaprRelevanceORM` 与逐轮 `query_evidence_gain`，验证二者 telescoping 一致后重构每轮
Query return。第一轮 Query、无方差后续槽位、Answer 和所有非 Query token 均回退标准
GRPO；只有后续精确 Query 标签段允许改变。该模式不使用 `action_credit_coef`。

**数据通道**：reward ORM 收到的 `rollout_infos` 与 sample 上的是同一对象引用，且 reward
计算发生在 advantage 装配**之前**，因此 SAPR plugin 可把逐轮向量以**副作用**写入
`rollout_infos`，ms-swift core 只按通用 key 读取，二者解耦。

### 5.13 `tests/utils/test_action_credit.py`（新增文件）

- `TestGateActionCredit`：默认全开、只保留 zero-outcome 组、浮点容差、长度不匹配报错。
- `TestNormalizeActionCredit`：center/scale、变长 turn、跨 prompt 不混、`scale=none`、单
  位 std、长度不匹配报错、单成员 slot 置零。
- `TestScatterTurnCredit`：按 ordinal scatter、masked token 无 credit、空 credit 为 0、
  shape 不匹配报错。

`test_multi_teacher.py` 另新增 `TestQueryTurnIndexMap`（两 query 不同 ordinal、
answer/evidence 排除、遵守 loss_mask、input_ids-frame 对齐）。

> 任务侧配套：SAPR-RAG 仓库 `03_sapr_rag/scripts/grpo/plugin.py` 的
> `SaprQueryEvidenceGainORM`（注册名 `sapr_query_evidence_gain`）产出逐轮向量并写入
> `rollout_infos`，reward 本身恒返回 0.0；训练入口 `run_grpo_opsd.sh` 通过
> `ACTION_CREDIT_MODE=query_evidence` 打开并把 `--action_credit_*` 透传给 `swift rlhf`。
> E17 launcher 为 `run_canonical_sft_action_credit_s1000.sh`（B 的单变量增量）。

### 5.14 `tests/utils/test_action_causal.py`（新增文件）

覆盖单 reward component 的 GRPO 组归一化、Query/Answer token 路由、Answer 错误时
正 query gain 的保留、padding/未路由 token 清零，以及 Query/Answer mask 重叠
fail-fast。与其他动作信用测试共同覆盖数值核心与 trainer 路由。

### 5.15 `tests/utils/test_signed_query_reweight.py`（新增文件）

覆盖正负 outcome 的对称重加权、零 advantage 保持、局部 credit 裁剪、非 Query
token 完全保持标准 GRPO，以及 `credit_coef*credit_clip<1` 的符号保持约束。

### 5.16 `tests/utils/test_causal_return.py`（新增文件）

覆盖逐轮 gain 的 telescoping 校验、第一轮 Query 与 B 完全一致、延迟 evidence 的
return-to-go、单成员槽位回退、仅 Query 标签段路由，以及非 Query/padding 保持标准优势。
同时覆盖 vLLM stop 移除 `</query>` 时将标签段延伸到当前 turn 末尾。

## 6. 数据契约

### 6.1 普通 GRPO

至少需要：

```text
messages
golden_answers
gold_titles
gold_sup_sents
source
```

不应包含有效的 teacher prompt 字段。

### 6.2 旧单 scope OPSD

```text
teacher_prompt
```

配合：

```text
teacher_action_scope=all/query/evidence/answer
teacher_kl_coef=<scalar>
```

### 6.3 multi-scope OPSD

数据可包含：

```text
teacher_query_prompt
teacher_evidence_prompt
teacher_answer_prompt
```

训练参数：

```text
teacher_action_scope=multi
teacher_query_kl_coef=<float>
teacher_evidence_kl_coef=<float>
teacher_answer_kl_coef=<float>
```

当前正式 Query/Answer OPSD 配置：

```text
teacher_query_kl_coef=0.01
teacher_evidence_kl_coef=0.00
teacher_answer_kl_coef=0.03
```

### 6.4 纯 OPSD

```text
opd_use_grpo_advantage=false
```

SAPR-RAG wrapper 还应关闭全部 reward：

```text
ENABLE_REWARD=false
```

这样最终 token advantage 只保留 teacher/student log-ratio。

## 7. 运行模式矩阵

| 模式 | reward | teacher | 关键参数 |
|---|---:|---:|---|
| 普通 GRPO | 有 | 无 | `ENABLE_OPSD=false` |
| GRPO + 全动作 OPSD | 有 | 有 | `teacher_action_scope=all` |
| GRPO + 单动作 OPSD | 有 | 有 | `teacher_action_scope=query/answer/...` |
| GRPO + 多动作 OPSD | 有 | 有 | `teacher_action_scope=multi` + 分动作系数 |
| 纯分动作 OPSD | 无 | 有 | `opd_use_grpo_advantage=false` |
| failed-only 外部 OPD | 有或无 | 外部 teacher | `teacher_sequence_gate=failed_em/failed_f1` |
| Evidence-Attributed GRPO | 有 | 可无 | `action_credit_mode=query_evidence` + `action_credit_coef>0` |
| Hybrid Action-Causal GRPO | 有 | 无 | `advantage_mode=action_causal` + 独立 F1/query credit 路由 |
| Outcome-Signed Query Reweighting | 有 | 无 | `advantage_mode=signed_query_reweight` + 有界、符号保持的 Query 梯度重加权 |
| Causal Return-to-Go GRPO | 有 | 无 | `advantage_mode=causal_return` + 剩余 evidence return + 精确 Query 标签段 |

## 8. 验证流程

### 8.1 静态检查

在 ms-swift 根目录：

```bash
python -m py_compile \
  swift/dataset/preprocessor/core.py \
  swift/infer_engine/grpo_vllm_engine.py \
  swift/pipelines/infer/rollout.py \
  swift/rl_core/advantage.py \
  swift/rl_core/action_credit.py \
  swift/rl_core/data.py \
  swift/rlhf_trainers/args_mixin.py \
  swift/rlhf_trainers/gkd_helpers.py \
  swift/rlhf_trainers/grpo_trainer.py \
  swift/rollout/multi_turn.py
```

### 8.2 定向单元测试

```bash
PYTHONPATH=. python -m pytest -q \
  tests/utils/test_teacher_advantage.py \
  tests/utils/test_multi_teacher.py \
  tests/utils/test_action_credit.py \
  tests/utils/test_action_causal.py \
  tests/utils/test_signed_query_reweight.py \
  tests/utils/test_causal_return.py
```

### 8.3 普通 GRPO smoke

确认：

- rollout 日志中的 adapter 路径正确；
- rollout 输出与直接加载同一 LoRA 的输出一致；
- 多轮 final Answer 出现在 `response_token_ids`；
- `teacher_kl*` 指标不应出现；
- reward/advantage 有限。

### 8.4 分动作 OPSD smoke

确认：

- `teacher_action_scope=multi`；
- `teacher_action_scope_ratio_query > 0`；
- `teacher_action_scope_ratio_answer > 0`；
- `teacher_kl_scoped_query` 和 `teacher_kl_scoped_answer` 有限；
- Evidence 系数为 0 时不产生 Evidence teacher signal；
- 不再出现 `OPSD response length mismatch`。

### 8.5 纯 OPSD smoke

确认命令中：

```text
reward_funcs=<none>
opd_use_grpo_advantage=false
```

同时 `teacher_kl_scoped_*` 非零且 loss/grad norm 有限。

### 8.6 Hybrid Action-Causal GRPO smoke

确认：

- `advantage_mode=action_causal`；
- `action_answer_advantage_abs_mean` 与 `action_query_advantage_abs_mean` 有限且非零；
- `action_credit_nonzero_group_ratio` 显著高于 0；
- Query/Answer mask 不重叠，`action_unrouted_token_ratio` 仅对应模板边界；
- teacher/RLSD/SDAR 均关闭；
- `advantage_mode=sequence` 的历史路径回归测试通过。

### 8.7 Outcome-Signed Query Reweighting smoke

确认：

- `advantage_mode=signed_query_reweight`；
- `action_credit_nonzero_group_ratio` 显著高于 0；
- `action_credit_clipped_ratio` 有限且不过高；
- `action_query_reweight_delta_abs_mean` 有限且非零；
- `action_query_sign_flip_ratio` 恒为 0；
- 非 Query token 的 advantage 与标准 GRPO 完全一致。

### 8.8 Causal Return-to-Go GRPO smoke

确认：

- `advantage_mode=causal_return`；
- `causal_return_first_query_delta_max` 恒为 0；
- `causal_return_query_payload_ratio` 明显小于旧 Query-turn mask 比例；
- `causal_return_delta_abs_mean` 有限且只统计后续 Query；
- gain 总和与 terminal relevance 不一致时训练立即报错；
- Answer、reasoning 与非 Query token 的 advantage 与标准 GRPO 完全一致。

### 8.9 当前快照的实际验证记录

2026-09-09 首次验证、2026-09-17 提交前复验完整补丁：

```text
干净基线 worktree 前向 git apply：通过
反向 git apply 校验：通过
10 个修改源码文件 py_compile：通过
六个定向测试模块：81 tests passed
```

当前默认 Python 未安装 pytest，因此实际使用：

```bash
PYTHONPATH=. python -m unittest \
  tests.utils.test_teacher_advantage \
  tests.utils.test_multi_teacher \
  tests.utils.test_action_credit \
  tests.utils.test_action_causal \
  tests.utils.test_signed_query_reweight \
  tests.utils.test_causal_return
```

输出：

```text
Ran 81 tests
OK
```

## 9. 已验证实验

这些修改已支持完成：

- E09/E10：Answer-only OPSD；
- E11/E12：Query/Answer 分动作 OPSD；
- E13：failed-EM external-teacher OPD；
- E16：canonical SFT + GRPO + 分动作 OPSD；
- B：canonical SFT + GRPO-only；
- D：canonical SFT + 纯分动作 OPSD；
- E17：Evidence-Attributed GRPO additive prototype；
- E18：Zero-Outcome Rescue GRPO；
- E19：Hybrid Action-Causal GRPO（250-step pilot 正向、1000-step 全量失败）；
- E20：Outcome-Signed Query Reweighting（250-step pilot 失败）；
- E21：Causal Return-to-Go GRPO（实现和离线验证完成，待 pilot）；

对应配置和结果以：

```text
docs/experiment_tracker.md
docs/post_training_experiment_overview.md
```

为准。

## 10. 升级 ms-swift 时的处理

不要把完整补丁盲目应用到未知版本。

推荐流程：

1. 记录目标上游 commit；
2. 在干净分支执行 `git apply --check`；
3. 如果失败，按本文第 5 节逐功能迁移，而不是按行号机械搬运；
4. 先确认上游是否已合入 LoRA 或 multi-turn 修复；
5. 所有 teacher/action 功能重新跑定向单测；
6. 再做 2-step GRPO、OPSD、纯 OPSD smoke；
7. 最后运行正式训练。

新版本中若上游已经包含某个修复，应删除本地对应 hunk，避免重复逻辑。

## 11. 回滚

仅当 ms-swift 没有叠加其他修改时，才可执行：

```bash
git -C ../ms-swift apply --check --reverse \
  "$(pwd)/patches/ms-swift/sapr-rag-ms-swift-full-1dbd1bf64.patch"

git -C ../ms-swift apply --reverse \
  "$(pwd)/patches/ms-swift/sapr-rag-ms-swift-full-1dbd1bf64.patch"
```

如果 ms-swift 已有其他本地修改，不要强制回滚；应先保存 diff，再按文件处理。

## 12. 当前补丁不包含什么

完整补丁不包含：

- SAPR-RAG 自己的 scheduler、reward plugin 和训练 wrapper；
- BGE/FAISS 检索服务；
- 模型 checkpoint 或训练数据；
- `utils.py.bak_20260803` 等调试备份；
- SAPR plugin 侧的逐轮证据增益产出（`SaprQueryEvidenceGainORM`，属 SAPR-RAG 仓库，不在 ms-swift 补丁内）；
- 未来对新 ms-swift 版本的兼容修改。

因此迁移时还需要同时保留 SAPR-RAG 仓库，并按
`docs/retrieval_service_gpu_runbook.md` 启动检索服务。
