# GRPO / OPSD Pipeline Overview

> **2026-09-05 运行配置更新**：当前 H20 正式训练使用单进程常驻
> **BGE GPU + FAISS GPU** 检索服务。本文第 3 节和后文关于
> “FAISS index 仍在 CPU mmap”的描述是历史实现说明，不再是当前推荐部署。
> 新 Worker 的环境、启动命令、健康检查和训练/评测接入统一以
> [`retrieval_service_gpu_runbook.md`](retrieval_service_gpu_runbook.md)
> 为准。

本文用于从代码层面梳理 SAPR-RAG 当前 GRPO 与 OPSD 训练链路。目标不是复述实验日志，而是回答：

- 训练数据从哪里来；
- rollout 如何生成多轮 RAG 轨迹；
- 检索结果如何进入下一轮；
- reward 如何拿到检索证据并打分；
- OPSD 在普通 GRPO 上额外加了什么；
- 哪些脚本是入口，哪些代码是关键改动点。

## 0. 一句话总览

当前实现可以理解为：

```text
数据行(messages + gold fields + optional teacher_prompt)
  -> swift rlhf(GRPO trainer)
  -> HTTP 调 swift rollout(vLLM server)
  -> SaprRagScheduler 多轮生成 <query>/<answer>
  -> scheduler 调 retrieval daemon 取 reference
  -> reference 作为下一轮 user message 注入
  -> rollout 返回 response_token_ids / response_loss_mask / rollout_infos
  -> GRPO trainer 调 reward ORM 算 F1 / relevance / format
  -> 若 ENABLE_OPSD=true，再用 teacher_prompt 构造 teacher view
  -> teacher 对同一串 on-policy response 计算 logp
  -> teacher logp remap 回 student frame，作为 teacher_kl 注入 GRPO loss
```

普通 GRPO 只用规则 reward；OPSD 是在普通 GRPO 的 on-policy 轨迹上额外加一个 privileged teacher view。

## 1. 关键入口文件

| 作用 | 文件 |
|---|---|
| 训练入口 | [`03_sapr_rag/scripts/grpo/run_grpo_opsd.sh`](../03_sapr_rag/scripts/grpo/run_grpo_opsd.sh) |
| rollout/vLLM server 入口 | [`03_sapr_rag/scripts/grpo/run_rollout_opsd.sh`](../03_sapr_rag/scripts/grpo/run_rollout_opsd.sh) |
| 检索服务入口 | [`03_sapr_rag/scripts/grpo/retrieval_service.sh`](../03_sapr_rag/scripts/grpo/retrieval_service.sh), [`run_retrieval_daemon_flexible.sh`](../03_sapr_rag/scripts/grpo/run_retrieval_daemon_flexible.sh) |
| 检索 daemon 实现 | [`03_sapr_rag/scripts/grpo/retrieval_daemon.py`](../03_sapr_rag/scripts/grpo/retrieval_daemon.py) |
| 检索 HTTP client | [`03_sapr_rag/scripts/grpo/retrieval_client.py`](../03_sapr_rag/scripts/grpo/retrieval_client.py) |
| scheduler + reward plugin | [`03_sapr_rag/scripts/grpo/plugin.py`](../03_sapr_rag/scripts/grpo/plugin.py) |
| GRPO/OPSD 数据构造 | [`03_sapr_rag/scripts/grpo/build_grpo_dataset_mixed_opsd.py`](../03_sapr_rag/scripts/grpo/build_grpo_dataset_mixed_opsd.py) |
| OPSD teacher sample 数据结构 | [`ms-swift/swift/rl_core/data.py`](../../ms-swift/swift/rl_core/data.py) |
| OPSD teacher encoding / remap | [`ms-swift/swift/rlhf_trainers/gkd_helpers.py`](../../ms-swift/swift/rlhf_trainers/gkd_helpers.py) |
| GRPO trainer reward / teacher KL | [`ms-swift/swift/rlhf_trainers/grpo_trainer.py`](../../ms-swift/swift/rlhf_trainers/grpo_trainer.py) |

注意：[`run_opsd_smoke.sh`](../03_sapr_rag/scripts/grpo/run_opsd_smoke.sh) 是 smoke 入口，不应作为所有机器的固定资源布局模板。正式运行前需要根据当前可见设备做 preflight，并显式传入训练、rollout 和检索设备。

## 2. 数据构造阶段

### 2.1 普通 GRPO 数据

构造脚本：

```text
03_sapr_rag/scripts/grpo/build_grpo_dataset_mixed_opsd.py
```

每行核心字段：

```json
{
  "messages": [
    {"role": "system", "content": "...reasoning system prompt..."},
    {"role": "user", "content": "Question: ..."}
  ],
  "golden_answers": ["..."],
  "gold_titles": ["..."],
  "gold_sup_sents": ["..."],
  "source": "hotpotqa | 2wiki"
}
```

这些字段在训练时有两个用途：

- `messages`：作为 rollout 初始 prompt；
- `golden_answers` / `gold_titles` / `gold_sup_sents`：作为 reward 函数的监督信号。

### 2.2 OPSD 数据额外字段

当构造脚本使用：

```bash
--teacher_prompt_mode gold
```

每行会额外带上：

```json
{
  "teacher_prompt": "...privileged gold evidence prompt...",
  "teacher_prompt_version": "sapr-gold-v1",
  "teacher_prompt_source": "gold_supporting_facts",
  "teacher_prompt_truncated": false,
  "teacher_prompt_tokens": 1234
}
```

`teacher_prompt` 的含义：

- student 仍只看到普通问题和在线检索结果；
- teacher 在同一条 student rollout response 上，用替换后的 privileged prompt 计算 logp；
- teacher_prompt 通常包含 gold supporting facts 和 gold answer；
- teacher 不生成新答案，只对 student 已经采样出来的 token 逐 token 打分。

### 2.3 数据与训练模式的 fail-fast

[`run_grpo_opsd.sh`](../03_sapr_rag/scripts/grpo/run_grpo_opsd.sh) 会检查：

```bash
ENABLE_OPSD=true  <=>  dataset 第一行存在 teacher_prompt
ENABLE_OPSD=false <=>  dataset 第一行不存在 teacher_prompt
```

这是为了避免两种静默错误：

- 用普通数据跑 OPSD，teacher KL 实际无效；
- 用 OPSD 数据跑 plain control，control 被 teacher 字段污染。

## 3. 检索服务阶段

### 3.1 daemon 为什么存在

GRPO 是 online rollout。每个训练 step 会对每个 prompt 采样多条轨迹，轨迹内部又会多轮检索。如果每个训练进程都加载一份 FAISS Flat index，会重复加载约 68GB 索引，内存和带宽都会崩。

所以当前使用独立 daemon：

```text
rollout scheduler -> HTTP -> retrieval daemon -> BGE encode + FAISS search
```

### 3.2 当前检索实现

关键代码在 [`retrieval_daemon.py`](../03_sapr_rag/scripts/grpo/retrieval_daemon.py)：

```python
self.model = AutoModel.from_pretrained(bge_path).to(device).eval()
self.index = faiss.read_index(
    str(index_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
)
```

当前含义：

```text
BGE encoder: 按 --device 放到 cuda/npu/cpu
FAISS index: CPU mmap 只读
corpus fetch: CPU datasets
```

因此如果检索服务跑在某张 GPU 上，该 GPU 默认只负责 BGE query embedding，FAISS search 仍在 CPU。GPU 利用率低是预期现象，不代表 FAISS 已上 GPU。

### 3.3 HTTP 接口

daemon 暴露：

```text
GET  /health
POST /search
POST /search_batch
```

rollout 端通过 [`retrieval_client.py`](../03_sapr_rag/scripts/grpo/retrieval_client.py) 调：

```python
client.search(query, top_k=3)
```

## 4. Rollout server 阶段

入口：

```text
03_sapr_rag/scripts/grpo/run_rollout_opsd.sh
```

核心命令：

```bash
swift rollout \
  --model "$BASE_MODEL" \
  --adapters "$ADAPTER_PATH" \
  --vllm_enable_lora true \
  --vllm_use_async_engine true \
  --multi_turn_scheduler sapr_rag_scheduler \
  --external_plugins "$PLUGIN" \
  --max_turns 6 \
  --host 127.0.0.1 \
  --port "$PORT"
```

关键点：

- rollout server 是单独的 vLLM 服务；
- 训练进程通过 HTTP 连接它；
- `--external_plugins plugin.py` 注册 scheduler；
- `--multi_turn_scheduler sapr_rag_scheduler` 指定 SAPR-RAG 多轮调度器；
- `--vllm_enable_lora true` 必须显式打开，否则 `--adapters` 可能不会以 LoRA serving 方式生效。

## 5. SaprRagScheduler 多轮协议

实现位置：

```text
03_sapr_rag/scripts/grpo/plugin.py
```

### 5.1 模型输出协议

每一轮 assistant 生成必须以以下两种动作之一结束：

```text
So the next query is <query>...</query>
So the answer is <answer>...</answer>
```

### 5.2 check_finished

如果本轮输出包含 `<answer>...</answer>`，scheduler 停止：

```python
if RE_ANSWER.search(response_choice.message.content or ""):
    return True
```

否则交给 ms-swift 母类处理 `max_turns` 等停止条件。

### 5.3 step

如果本轮输出包含 `<query>...</query>`：

```python
query = m.group(1).strip()
docs = self.client.search(query, top_k=self.top_k)
obs = self._format_observation(docs)
infer_request.messages.append({"role": "user", "content": obs})
steps.append({"turn": current_turn, "query": query, "docs": docs})
```

当前协议是：

```text
assistant: ... <query>...</query>
user: Reference: <reference>...</reference> ...
assistant: 继续生成下一轮 query 或 answer
```

这对齐 ms-swift 官方 `VisualToolBoxScheduler` 的工具返回方式。

### 5.4 为什么不是把 reference 拼到 assistant 后面

旧实现曾经这样做：

```python
infer_request.messages[-1]["content"] += obs
```

这会把环境返回拼进 assistant completion 里，模型容易把 `Reference` 误认为自己已经完成输出，然后直接 EOS。实测旧协议下 `checkpoint-100` 严格 `<answer>` 只有 `5/200`；改为 user observation 后提升到 `174/200`。

### 5.5 response_token_ids 与 response_loss_mask

当前 scheduler 返回：

```python
token_ids = list(response_choice.token_ids)
loss_mask = [1] * len(token_ids)
```

因为 reference 不再拼入 assistant completion，所以 response token 只包含模型自己生成的 assistant token。reference 是下一轮 user message，不需要在 response 里追加 `loss_mask=0`。

### 5.6 rollout_infos

scheduler 每轮把检索记录写入：

```python
"rollout_infos": {
  "retrieved_steps": [
    {"turn": 0, "query": "...", "docs": [...]}
  ],
  "uuid": "..."
}
```

`rollout_infos` 是 reward 端读取检索证据的唯一通道。ms-swift 对同名 key 是覆盖语义，所以 scheduler 每次必须返回完整列表，而不是只返回本轮增量。

## 6. Reward 计算阶段

三个 reward 都在 [`plugin.py`](../03_sapr_rag/scripts/grpo/plugin.py) 注册：

```python
orms["sapr_f1"] = SaprF1ORM
orms["sapr_relevance"] = SaprRelevanceORM
orms["sapr_format"] = SaprFormatORM
```

训练入口中启用：

```bash
--reward_funcs sapr_f1 sapr_relevance sapr_format
--reward_weights 1.0 0.2 0.05
```

### 6.1 SaprF1ORM

从 completion 里解析最终答案：

```python
pred = parse_final_answer(comp)
f1_score(pred, golden_answers)
```

这是主 reward，连续值 `[0,1]`，比 EM / cover_em 更适合 GRPO 的 group baseline。

### 6.2 SaprRelevanceORM

从 `kwargs["rollout_infos"]` 里取 scheduler 存下来的检索 docs，然后和 gold supporting facts 做三级 OR：

```text
1. retrieved title == gold title
2. gold supporting sentence 是 retrieved doc text 的子串
3. gold answer 是 retrieved doc text 的子串
```

返回：

```text
hit_gold_count / num_gold_titles
```

这个 reward 的目标是鼓励模型发出更好的 `<query>`，让检索结果覆盖 gold evidence。

### 6.3 SaprFormatORM

检查最终协议标签：

```text
允许前面多轮 <query>
最后一个事件必须是非空 <answer>
```

它权重很小，只用于兜底格式，不应主导训练。

## 7. ms-swift GRPO 主流程

训练入口：

```text
03_sapr_rag/scripts/grpo/run_grpo_opsd.sh
```

核心命令：

```bash
swift rlhf \
  --rlhf_type grpo \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port "$VLLM_PORT" \
  --vllm_server_group_port "$VLLM_GROUP_PORT" \
  --vllm_server_pass_dataset true \
  --reward_funcs sapr_f1 sapr_relevance sapr_format \
  --teacher_kl_coef "$TEACHER_KL_COEF"
```

重要超参：

```text
num_generations=8
steps_per_generation=8
gradient_accumulation_steps=4
per_device_train_batch_size=1
learning_rate=1e-6
save_steps=25
```

概念上：

```text
一个原始 prompt -> 采样 8 条 completion/trajectory -> 组成一个 group
同组内根据 reward 做归一化 advantage
用 GRPO loss 更新 LoRA
```

## 8. rollout_infos 如何进入 reward

ms-swift 里关键链路：

```text
SaprRagScheduler.step()
  -> RolloutOutput.rollout_infos
  -> rollout_mixin.py 写回 input_data["rollout_infos"]
  -> OnPolicySample.to_reward_row()
  -> score_completions(..., kwargs)
  -> SaprRelevanceORM.__call__(..., rollout_infos=...)
```

[`grpo_trainer.py`](../../ms-swift/swift/rlhf_trainers/grpo_trainer.py) 的 `_compute_rewards_per_func` 最终调用 reward 函数。reward 函数拿到的是 batched kwargs，所以 `golden_answers[i]`、`gold_titles[i]`、`rollout_infos[i]` 要按样本索引对齐。

## 9. OPSD 在 GRPO 上加了什么

普通 GRPO 的 advantage 只来自规则 reward。OPSD 多了 teacher 对同一条 student response 的逐 token logp。

当前 OPSD 是 dynamic self-distillation：

```text
student prompt: 普通问题 + 在线检索 reference
teacher prompt: teacher_prompt 替换最后一个 user message，包含 gold evidence
response: 两者共享同一串 student on-policy response token
teacher signal: log p_teacher(response_token | teacher_prompt)
student signal: log p_student(response_token | student_prompt)
```

直观理解：

```text
如果 teacher 在 privileged prompt 下也更认可某些 token，
这些 token 对应的 student 采样会被额外鼓励。
```

### 9.1 teacher_prompt 如何进入 sample

[`OnPolicySample`](../../ms-swift/swift/rl_core/data.py) 有字段：

```python
teacher_prompt: Optional[Any]
teacher_messages: Optional[Messages]
```

当样本存在 `teacher_prompt` 时：

```python
build_teacher_view()
```

会复制 student messages，并用 `teacher_prompt` 替换最后一个 user message。

### 9.2 teacher 与 student 共享 response_token_ids

[`to_teacher_template_dict`](../../ms-swift/swift/rl_core/data.py) 中：

```python
d["messages"] = self.teacher_messages
d["response_token_ids"] = self.response_token_ids
d["response_loss_mask"] = self.response_loss_mask
```

这里的核心是：

- teacher prompt 变了；
- response token 不变；
- response loss mask 也必须不变。

这保证 teacher 和 student 比较的是同一串 on-policy token。

### 9.3 为什么 response_loss_mask 必须透传

早期 OPSD smoke 曾失败：

```text
OPSD response length mismatch: student=82 teacher=382
```

原因是 teacher view 没拿到 `response_loss_mask`，导致 teacher 认为更多 token 属于 completion。现在 `to_teacher_template_dict` 已透传 `response_loss_mask`，`encode_teacher_view` 会读取它并替换 assistant response。

这部分对应：

```text
ms-swift/swift/rl_core/data.py
ms-swift/swift/rlhf_trainers/gkd_helpers.py
```

### 9.4 teacher logp remap

teacher 和 student prompt 长度不同，所以 response token 在序列里的绝对位置不同。OPSD 需要把 teacher completion 区域的 logp remap 到 student completion frame。

关键函数：

```python
remap_teacher_logps_to_student_frame(
    teacher_logps,
    teacher_completion_mask,
    student_completion_mask,
)
```

它要求：

```text
teacher valid completion token count == student valid completion token count
```

否则直接 assert 失败，因为逐 token KL 已经无意义。

## 10. GRPO loss 与 teacher KL 的关系

在 `run_grpo_opsd.sh` 里：

```bash
ENABLE_OPSD=true  -> --teacher_kl_coef "$TEACHER_KL_COEF"
ENABLE_OPSD=false -> --teacher_kl_coef 0
```

也就是说：

```text
plain GRPO:
  loss = GRPO reward advantage + policy KL 等常规项

OPSD:
  loss = GRPO reward advantage + teacher_kl_coef * teacher/student token-level signal
```

实验里常用：

```text
TEACHER_KL_COEF=0.1
```

如果 `logging.jsonl` 里出现 `teacher_kl`，说明 OPSD teacher 路径确实生效。

## 11. 当前资源布局建议

典型布局是：

```text
多卡: training
单卡: rollout/vLLM
CPU 或低负载设备: retrieval daemon
```

在任何机器上启动前都应先确认目标设备空闲。如果某张设备存在无法解释的大显存占用，应直接视为不可用，改用其他设备或重启运行环境。

```text
目标设备 memory.used 接近 0
无无法解释的 compute app
训练 / rollout / retrieval 设备互不冲突
```

资源紧张时可使用：

```text
多卡: training
单卡: rollout/vLLM
retrieval: CPU daemon
```

如果要追求训练吞吐，应该优先把低利用率的 GPU retrieval 改为 CPU retrieval，把 GPU 留给 rollout 或训练。完整 FAISS Flat index 当前没有上 GPU。

## 12. 当前已知坑

### 12.1 `plugin.py` 曾出现方案注释与实现不一致

历史上文件头曾写：

```text
方案 A：只训 reason，检索文档作 observation 以 loss_mask=0 注入
```

但当前实际代码已经是：

```text
方案 B：reference 作为下一轮 user message
```

当前已把文件头说明同步为方案 B。后续看实现时仍应以 `SaprRagScheduler.step()` 当前代码为准。

### 12.2 `run_opsd_smoke.sh` 的默认设备布局不能直接照抄

这个脚本仍默认：

```text
多卡 train / 单卡 rollout / 单独 retrieval
```

这是 smoke 时代的默认布局。换机器或换资源状态时，必须通过环境变量显式覆盖设备，并先检查显存和端口。

### 12.3 `--vllm_enable_lora true` 必须显式打开

后续推理验证发现，仅传 `--adapters` 不足以确保 vLLM 以 LoRA serving 方式工作。当前 `run_rollout_opsd.sh` 已加：

```bash
--vllm_enable_lora "$VLLM_ENABLE_LORA"
```

默认 `true`。

### 12.4 retrieval GPU 利用率低是设计结果

当前 GPU retrieval 只把 BGE encoder 放 GPU，FAISS index 仍是 CPU mmap，所以 GPU 利用率低。不要误以为整份 FAISS index 已在 GPU 上。

### 12.5 训练 step 时间不能按单个优化 step 估

GRPO 的 `steps_per_generation=8` 会导致：

```text
生成/rollout step 很慢
后续若干优化 step 很快
```

看速度时应按一整个 generation cycle 或多个 step 的平均时间判断。

## 13. 如何确认 OPSD/GRPO 路径有效

训练日志位置通常在：

```text
<output_dir>/v*/logging.jsonl
```

关键字段：

```text
reward
reward_std
rewards/SaprF1ORM/mean
rewards/SaprRelevanceORM/mean
rewards/SaprFormatORM/mean
teacher_kl
kl
num_turns
completions/mean_length
global_step/max_steps
```

判断标准：

- `reward_std` 不能长期为 0，否则 GRPO group advantage 没有有效区分；
- `teacher_kl` 出现且为有限值，说明 OPSD teacher 路径生效；
- `num_turns` 应明显大于 1，说明多轮检索真的在发生；
- `completions/clipped_ratio` 应尽量接近 0，避免训练被长度截断污染；
- `checkpoint-*` 每 `save_steps` 落盘一次，默认 25 step。

## 14. 推荐阅读顺序

如果要从零理解代码，建议按这个顺序看：

1. [`build_grpo_dataset_mixed_opsd.py`](../03_sapr_rag/scripts/grpo/build_grpo_dataset_mixed_opsd.py)：先理解一行训练样本长什么样。
2. [`retrieval_daemon.py`](../03_sapr_rag/scripts/grpo/retrieval_daemon.py)：理解检索服务输出什么 docs。
3. [`plugin.py`](../03_sapr_rag/scripts/grpo/plugin.py) 的 `SaprRagScheduler`：理解一轮 `<query>` 如何变成下一轮 user reference。
4. [`plugin.py`](../03_sapr_rag/scripts/grpo/plugin.py) 的三个 ORM：理解 reward 怎么从 answer 和 rollout_infos 来。
5. [`run_rollout_opsd.sh`](../03_sapr_rag/scripts/grpo/run_rollout_opsd.sh)：理解 rollout server 如何启动。
6. [`run_grpo_opsd.sh`](../03_sapr_rag/scripts/grpo/run_grpo_opsd.sh)：理解训练进程如何连接 rollout，并启用 reward / OPSD。
7. [`ms-swift/swift/rl_core/data.py`](../../ms-swift/swift/rl_core/data.py)：理解 `teacher_prompt` 如何替换 student prompt。
8. [`ms-swift/swift/rlhf_trainers/gkd_helpers.py`](../../ms-swift/swift/rlhf_trainers/gkd_helpers.py)：理解 teacher view encode 和 logp remap。
9. [`ms-swift/swift/rlhf_trainers/grpo_trainer.py`](../../ms-swift/swift/rlhf_trainers/grpo_trainer.py)：理解 reward 和 teacher KL 如何进入 GRPO trainer。

## 15. 最小心智模型

最后用一句话压缩：

```text
GRPO 负责让模型在自己的多轮 RAG 轨迹上，根据 answer/retrieval/format reward 更新；
OPSD 负责给同一条轨迹加一个“看过 gold evidence 的 teacher 对这些 token 是否更认可”的逐 token 信号。
```

因此：

- scheduler 决定“模型如何和检索环境互动”；
- reward 决定“这条轨迹好不好”；
- OPSD teacher 决定“在 privileged context 下，同一串 response token 是否更像正确行为”。
