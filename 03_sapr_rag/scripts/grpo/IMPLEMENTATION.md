# SAPR-RAG × ms-swift GRPO 实施说明（施工图）

本文档把 `docs/grpo_plan.md` 的战略方案落到**代码级蓝图**：每个要写的文件、每个类/函数签名、对接 ms-swift 的哪个接口、复用现有哪段代码、启动命令、怎么 sanity 验证。执行者读完应能直接动手，无需再做架构决策。

> 战略依据见 `docs/grpo_plan.md`（reward 三分量、三级 OR 命中、λ 初值、实证 a~d、里程碑 P0~P4）。本文档不重复论证，只讲"怎么搭"。

---

## §0 已就位 vs 待建

**已就位**（`03_sapr_rag/scripts/grpo/`）：
- `retrieval_daemon.py` — FAISS+BGE 检索 daemon，`/health` `/search` `/search_batch`，worker 已验证 top-3 正常（n_vectors=22352695, dim=768）。
- `retrieval_client.py` — `RetrievalClient(base_url)`，含 `wait_until_ready(max_wait, interval)` / `search(query, top_k)` / `search_batch(queries, top_k)`。
- `run_retrieval_daemon.sh` — `GPU= PORT=` 启动脚本，已加端口预检。

**待建**（本文档定义其规格）：
1. `plugin.py` — 同时承载 scheduler 注册 + 三个 reward 注册（ms-swift `--external_plugins` 约定指向同一文件）。
2. `build_grpo_dataset.py` — dev.jsonl → ms-swift 训练 jsonl。
3. `run_grpo.sh` — 三进程启动（daemon / rollout / train）。
4. `sanity_check.py` — 不启真训练，验证三个 ORM 值域与方向。

**ms-swift 版本**：`/mlx_devbox/users/mayi.summer/playground/ms-swift`，v4.4.0.dev0。

---

## §1 总体架构（三进程分离）

```
┌─────────────────────┐   HTTP :8100   ┌──────────────────────────┐
│ 进程A 检索 daemon     │◀──────────────│ 进程B swift rollout        │
│ FAISS+BGE, GPU 7      │   /search     │ async server, GPU 6        │
│ 索引只加载一份(mmap)   │──────────────▶│ 加载 SFT LoRA policy        │
└─────────────────────┘   docs        │ 跑 SaprRagScheduler         │
                                       │ scheduler 内调 RetrievalClient│
                                       └────────────┬─────────────┘
                                          vllm server :8000
                                                    │
                                       ┌────────────▼─────────────┐
                                       │ 进程C swift rlhf grpo      │
                                       │ 训练, GPU 0-5              │
                                       │ --vllm_mode server 连B     │
                                       │ reward 在此进程算           │
                                       └──────────────────────────┘
```

数据流：dataset → rollout(B) 产出多轮轨迹 + `rollout_infos` → 训练(C) 取 `completions` + `rollout_infos` + dataset 各列算 reward → GRPO 更新 LoRA → 同步回 B。

架构决策沿用 **1A+2B+3B**：1A=vllm 暂不升级（用现有 0.10.0 先试跑）；2B=server 模式 rollout（更成熟）；3B=检索 daemon 独立进程（索引只加载一份，避免 OOM）。

---

## §2 核心难点：三阶段 agent loop → ms-swift 单段 step()

SAPR-RAG 一"轮"含 reason + evidence 两次生成（见 `agent_infer.py` 状态机 L330-395），而 ms-swift 每个 turn 只生成一段 assistant。**已确认采用方案 A（只训 reason）**。

### 2.1 loss_mask=0 的含义

一条多轮轨迹的 token 序列由两种来源拼成：
1. **模型生成的**：`<reason>...<query>X</query>` 或 `...<answer>Y</answer>`
2. **环境塞进去的**：FAISS 检索回的原始文档原文（title + text）

GRPO 优化的是"模型的生成策略"，所以只能对 (1) 算梯度。`loss_mask` 是与 token 等长的 0/1 数组，1=参与 loss，0=跳过。检索文档 token 标 0，模型就不会被迫"拟合/背诵"文档原文，只学"看到这些证据后如何 reason、如何出下一个 query / 给最终 answer"。

范式直接照搬 `ToolCallScheduler.step()`（`ms-swift/examples/train/grpo/plugin/plugin.py:1196-1207`）：
```
token_ids = response_choice.token_ids        # 模型本轮生成
loss_mask = [1] * len(token_ids)             # 模型部分全 1
result_tokens = tokenizer.encode(obs, add_special_tokens=False)  # 检索结果
token_ids.extend(result_tokens)
loss_mask.extend([0] * len(result_tokens))   # 环境部分全 0
```

### 2.2 SaprRagScheduler 规格

继承 `MultiTurnScheduler`（`ms-swift/swift/rollout/multi_turn.py:180`）。母类 `run()`（L222-399）已封装多轮循环 + token_ids/loss_mask/logprobs 累积；子类只实现 `step()`（L401）与 `check_finished()`（L425）。

```python
class SaprRagScheduler(MultiTurnScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = RetrievalClient(base_url="http://127.0.0.1:8100")
        self.client.wait_until_ready()
        self.top_k = 3
        self._traj = {}      # per-uuid: {'steps': [{turn, query, docs}]}

    def check_finished(self, infer_request, response_choice, current_turn) -> bool:
        text = response_choice.message.content
        if RE_ANSWER.search(text):          # 出现 <answer> 即停
            return True
        return super().check_finished(...)  # 或达 max_turns

    def step(self, infer_request, response_choice, current_turn) -> Dict:
        text = response_choice.message.content
        token_ids = response_choice.token_ids
        loss_mask = [1] * len(token_ids)
        uuid = infer_request.uuid            # per-uuid 状态键
        m = RE_QUERY.search(text)
        if m:
            query = m.group(1).strip()
            docs = self.client.search(query, top_k=self.top_k)
            obs = self._format_observation(docs)      # " ".join(f"{title}. {text}")
            infer_request.messages[-1]['content'] += obs   # 供下一轮 reason 看到
            result_tokens = self.tokenizer.encode(obs, add_special_tokens=False)
            token_ids.extend(result_tokens)
            loss_mask.extend([0] * len(result_tokens))     # 环境 token 不训
            self._traj.setdefault(uuid, {'steps': []})['steps'].append(
                {'turn': current_turn, 'query': query, 'docs': docs})
        return {
            'infer_request': infer_request,
            'response_token_ids': token_ids,
            'response_loss_mask': loss_mask,
            # 覆盖语义陷阱：rollout_infos 同名 key 覆盖不追加，每次写完整列表
            'rollout_infos': {'retrieved_steps': self._traj[uuid]['steps']},
        }

    def _format_observation(self, docs) -> str:
        # 与 agent_infer.build_evidence_prompt 同款拼法，doc.text 已在 daemon 截 [:500]
        ref = " ".join(f"{d['title']}. {d['text']}" for d in docs)
        return f" Reference: <reference>{ref}</reference>"
```

- **stop**：模型每轮以 `<query>`/`<answer>` 收尾，stop=`["</query>","</answer>"]`（与 `agent_infer.py` L339 一致；vllm 吃掉 stop 串本身，解析前需补回，见 L343-346）。
- **per-uuid 状态清理**：轨迹结束（check_finished=True）后从 `self._traj` 删除该 uuid，参考 `GYMScheduler`（multi_turn.py:725）的 `_close_and_remove` 模式，避免内存泄漏。
- **evidence 不单独生成、不单独受训**（方案 A）：检索文档直接作 observation，证据抽取交由模型在下一轮 reason 隐式完成。这是**待 sanity 后可升级的开放项**（升 B = evidence 也受训）。
- 注册：`multi_turns['sapr_rag_scheduler'] = SaprRagScheduler`（multi_turn.py:828 注册表）。

复用 `agent_infer.py` 的：`REASONING_SYSTEM`(L40)、`RE_QUERY/RE_ANSWER`(L168-170)、`build_evidence_prompt` 拼法(L159-164)。

---

## §3 Reward 函数规格（三个 ORM）

reward 函数签名：`__call__(self, completions, **kwargs) -> List[float]`。
- `completions`：每条轨迹末轮 assistant content（`grpo_trainer.py:342` 取 `messages[-1]['content']`）。
- `kwargs`：含 `rollout_infos`（List[Dict]，每条轨迹一个，来自 scheduler）+ dataset 透传列（`golden_answers`/`gold_titles`/`gold_sup_sents`，经 `rows_to_batched` 展开，`grpo_trainer.py:354`）。

注册：`orms['name'] = Class`（plugin.py:91/180 样板）。

### 3.1 SaprF1ORM（注册 `sapr_f1`，主信号，权重 1.0）
- 从 `completions[i]` 用 `RE_ANSWER` 解析 `<answer>`；解析不到 → pred=""。
- `f1_score(pred, golden_answers[i])`（复用 `score.py:57`，token 级 best over golds）。
- 值域 [0,1]。

### 3.2 SaprRelevanceORM（注册 `sapr_relevance`，权重 0.2）
- 从 `kwargs['rollout_infos'][i]['retrieved_steps']` 收集所有检索 doc 的 title/text（跨 turn 去重，参考 `retrieval_recall.py:collect_retrieved` L59）。
- 对透传的 `gold_titles[i]` / `gold_sup_sents[i]` / `golden_answers[i]` 做**三级 OR 命中**（复用 `retrieval_recall.py` 的 `norm_title` L21 / `norm_text` L27 + 命中逻辑 L103-110）：
  1. doc.title 与某 gold title 归一化精确匹配；
  2. 某 gold supporting 句子文本出现在某 doc 正文；
  3. gold answer 文本出现在某 doc 正文。
- 返回 `hit / num_gold_supporting`（连续命中比例，**不用**全覆盖硬阈值——17.7% 题天然缺 ≥1 篇 gold，见 grpo_plan §2.0c）。值域 [0,1]。
- **边界**：`num_gold_supporting == 0`（gold 全不可达）→ 返回 0.0（这些题已在 §4 预过滤剔除，正常不会出现；保留兜底）。

### 3.3 SaprFormatORM（注册 `sapr_format`，权重 0.05）
- 轨迹是否严格符合 `<query>`/`<answer>` 协议（每个 assistant 段恰好命中一个 RE_QUERY 或 RE_ANSWER，末轮必须是 `<answer>`）。
- 合法 1.0 / 非法 0.0。

### 3.4 权重
`--reward_weights 1.0 0.2 0.05`（对应 `sapr_f1 sapr_relevance sapr_format`，来自 grpo_plan §2.2）。trainer 自动加权求和：`R_total = 1.0·F1 + 0.2·Relevance + 0.05·Format`。

---

## §4 数据打包脚本规格：build_grpo_dataset.py

输入 `data/eval/hotpotqa/dev.jsonl`（每行 `{id, question, golden_answers, metadata:{supporting_facts:{title,sent_id}, context:{title,sentences}}}`）。

每行输出（ms-swift jsonl）：
- `messages`: `[{role:"system", content:REASONING_SYSTEM}, {role:"user", content:"Question: {question}"}]`（首轮 prompt，复用 `agent_infer.REASONING_SYSTEM` + `build_reasoning_prompt` L152 无 history 分支）。
- 透传列（供 reward，均为该行独立 list，满足 `rows_to_batched`）：
  - `golden_answers`: `List[str]`
  - `gold_titles`: `List[str]`（从 `metadata.supporting_facts.title` 去重）
  - `gold_sup_sents`: `List[str]`（按 `supporting_facts.(title,sent_id)` 去 `metadata.context.sentences` 取句，复用 `retrieval_recall.load_gold` L34 抽取逻辑）

**预过滤**：剔除 grpo_plan §2.0(c) 的 78 题"gold 全部不可达"——复用 `retrieval_recall.py` 的 corpus 可达性扫描产出名单，或脚本内现扫。输出 `data/grpo/hotpotqa_train.jsonl`。

---

## §5 启动脚本规格：run_grpo.sh（server 模式，8 卡）

```bash
# 进程A：检索 daemon（或复用已起的）
GPU=7 PORT=8100 bash run_retrieval_daemon.sh &

# 进程B：swift rollout async server
CUDA_VISIBLE_DEVICES=6 swift rollout \
  --model {base=03_sapr_rag/models/Qwen2.5-7B-Instruct} \
  --adapters {sft_lora=03_sapr_rag/saves/qwen2_5_7b/lora/sft/checkpoint-1650} \
  --vllm_use_async_engine true \
  --multi_turn_scheduler sapr_rag_scheduler \
  --external_plugins plugin.py \
  --max_turns 6 \
  --vllm_max_model_len 8192 \
  --vllm_server_port 8000 &

# 进程C：swift rlhf grpo（连 B）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 swift rlhf \
  --rlhf_type grpo \
  --model {base} --adapters {sft_lora} --train_type lora \
  --vllm_mode server --vllm_server_port 8000 \
  --reward_funcs sapr_f1 sapr_relevance sapr_format \
  --reward_weights 1.0 0.2 0.05 \
  --external_plugins plugin.py \
  --num_generations 8 \
  --dataset data/grpo/hotpotqa_train.jsonl \
  ...（lr/batch/save 等沿用 examples/train/grpo/external/vllm_multi_turn.sh 默认）
```

- 卡分配总数 ≤ 8：daemon(7) + rollout(6) + train(0-5)，互不冲突。
- `plugin.py` 同文件承载 scheduler + 三 reward 注册（ms-swift 约定 `--external_plugins` 单文件）。
- 启动样板参照 `ms-swift/examples/train/grpo/external/vllm_multi_turn.sh`。

---

## §6 Sanity 验证（不启真训练）

1. **daemon**：`curl /health` + 单查询返回 top-3。（已通过：query "who founded Apple Inc" 返回 Steve Wozniak）
2. **mock reward**（`sanity_check.py`）：手构一条 `rollout_infos` + dataset 行，分别调三个 ORM，断言：值域 [0,1]；检到 gold → relevance 高；answer 含 gold → f1 高；格式合法 → format=1.0。
3. **100 条小数据**：跑通 `swift rollout` + `swift rlhf` 不崩，且 reward 有非零方差（GRPO group baseline 前提）。
4. **里程碑**（grpo_plan §4）：P2 通过=不退步；P3 通过=cover_em 比 SFT-only +2~3 点。

---

## §7 文件清单与依赖顺序

| 文件 | 对接 ms-swift | 复用源 | 依赖 |
|---|---|---|---|
| `plugin.py` | `MultiTurnScheduler.step` / `multi_turns[]` / `orms[]` | `agent_infer`(prompt/正则)、`score`(f1)、`retrieval_recall`(命中) | `retrieval_client.py` |
| `build_grpo_dataset.py` | dataset jsonl 格式 | `agent_infer.REASONING_SYSTEM`、`retrieval_recall.load_gold` | dev.jsonl |
| `run_grpo.sh` | `swift rollout`/`swift rlhf` CLI | `run_retrieval_daemon.sh` | daemon + plugin + dataset |
| `sanity_check.py` | — | plugin 的三 ORM | plugin.py |

施工顺序：build_grpo_dataset.py → plugin.py → sanity_check.py → run_grpo.sh。

---

## §8 开放项与风险

- **evidence 是否单独受训**：首版方案 A 不训（对齐评估口径风险最低），sanity 后视效果升方案 B。
- **vllm 版本**：当前 0.10.0，ms-swift 4.4 推荐更高 → 1A 决策"先试跑踩坑再升"。
- **relevance num_gold=0 / 全不可达**：已在 §4 预过滤 78 题；reward 内保留返回 0.0 兜底。
- **rollout_infos 覆盖语义**：同名 key 覆盖不追加，scheduler 每次必须写完整 `retrieved_steps` 列表。
- **per-uuid 状态泄漏**：轨迹结束须清理 `self._traj[uuid]`。
