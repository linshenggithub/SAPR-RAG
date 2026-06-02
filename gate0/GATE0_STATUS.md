# Gate 0 实验现状文档

**Date**: 2026-05-31
**Goal**: 用 GPT-4o 跑 v4 typed eval prompt 版，与 ReasonRAG baseline 对比 EM/F1

---

## 1) 原始 ReasonRAG Benchmark 已知信息

### 1.1 数据生成配置（data_generation.py）

```python
"generator_model": "gpt-4o-2024-05-13",   # GPT-4o via OpenAI API
"framework": "openai",                      # OpenAI API 调用
"retrieval_method": "bge",                  # BGE-base-en-v1.5
"retrieval_topk": 3,                        # 每步检索 top-3 文档
"save_note": args.model + "_MCTS",
```

Pipeline 参数：
```python
ReasonRAGPipeline(config, max_iter=7, max_children=2, max_rollouts=64)
```

- `max_iter=7`：最大推理步数
- `max_children=2`：每个节点最多 2 个子节点
- `max_rollouts=64`：每条问题做 64 次 MCTS rollout
- `beta=0.95`（pipeline 默认值，非 data_generation.py 设置的）
- `c=1.414`（UCT 探索常数，√2）

### 1.2 MCTS 流程

```
search(item):
  for _ in range(max_rollouts):    # 64 次
    leaf = select(root)             # UCT 选择
    child = expand(leaf)            # 生成一个新 child
    Q = evaluate_thoughts(child)    # GPT-4o 自评 0-100 → 归一化到 [0,1]
    reward = get_reward(child)      # 仅 terminal 节点：F1 × β^step
    backpropagate(child, Q, reward) # 向上回传

  return best_child(root)           # 选 N 最大的 child
```

### 1.3 Prompt 格式（原始 reasonrag_pipeline.py）

4 种 prompt，全部用 `<query>/<answer>/<evidence>` 标签格式：

**BEGIN_REASONING_PROMPT**：
- 输入：Question
- 输出：`So the answer is <answer>...</answer>` 或 `So the next query is <query>...</query>`
- 用途：初始分析，决定直接答还是先搜

**DOCUMENT_ANALYSIS_PROMPT**：
- 输入：Question + Reference（检索结果）
- 输出：`Based on the query, the relevant evidence is <evidence>...</evidence>.` 或 `<evidence>None</evidence>`
- 用途：从检索结果中抽取 evidence

**REASONING_PROMPT**：
- 输入：Question + Previous Thoughts
- 输出：同 BEGIN_REASONING（决定继续搜还是答）
- 用途：反思 + 决策

**ANSWER_GENERATION_PROMPT**：
- 输入：Question + Previous Thoughts
- 输出：`So the answer is <answer>...</answer>`
- 用途：强制生成最终答案（达到 max_iter 时触发）

**EVALUATION_PROMPT**（用于 simulate 阶段的 Q 值）：
- 输入：Question + Golden Answer + Agent Reasoning Process
- 输出：`So the score is [Score].`（0-100）
- ⚠️ **需要 golden answer**，训练数据生成时可用，推理时也用

### 1.4 Action 状态机（next_state）

```
begin_reasoning → 如果有 <answer> → terminal
                → 如果有 <query>  → document_analysis

document_analysis → reasoning（无条件）

reasoning → 如果有 <answer> → terminal
           → 如果有 <query>  → document_analysis

answer_generation → terminal（max_iter 达到上限时强制调用）
```

即：`begin_reasoning → document_analysis → reasoning → document_analysis → reasoning → ... → answer_generation`

### 1.5 Reward 计算

**Terminal 节点**：
```python
reward = F1(prediction, golden_answers) × β^step   # β=0.95
```

**非 Terminal 节点**（bottom-up 计算，preference_data_generatoin.py）：
```python
reward = Σ(child_reward × child_N) / Σ(child_N)   # visit-count 加权平均
```

**Q 值**：
```python
Q = evaluate_thoughts(node)   # GPT-4o 自评 0-100，归一化到 [0,1]
Q = mean(Q_list)              # 多次 rollout 的平均值
```

### 1.6 DPO 数据（RAG_ProGuide.json）

从 MCTS 树中提取 chosen/rejected 对：
- 过滤：`|reward_a - reward_b| < 0.01` 的跳过
- 过滤：chosen_content == rejected_content 的跳过
- 13,289 条 DPO 对，13,068 个不同问题
- 81% 同 action type（质量对比），19% 跨 action type（策略选择）

---

## 2) 本地已有资源

### 2.1 硬件

- 本地：4×RTX 3090（24GB each），全部空闲
- 远程：3×RTX 5090（rag-5090），暂不使用

### 2.2 模型

| 模型 | 路径 | 大小 | 状态 |
|------|------|------|------|
| Qwen2.5-7B-Instruct | `/home/mayi/RAG/models/Qwen2.5-7B-Instruct/` | 15G | ✅ 完整 |
| Qwen2.5-7B-Instruct-ReasonRAG | `/home/mayi/RAG/models/Qwen2.5-7B-Instruct-ReasonRAG/` | 3.2G | ❌ 只有 config，无权重文件 |
| Llama-3.1-8B-Instruct | `/home/mayi/RAG/models/llama3.1-8B-Instruct/` | 30G | ✅ 完整 |
| BGE-base-en-v1.5 | `/home/mayi/RAG/retriever/bge-base-en-v1.5/` | - | ✅ 完整 |
| E5-base-v2 | `/home/mayi/RAG/retriever/e5-base-v2/` | - | ✅ 完整 |

ReasonRAG 训练模型需从 HuggingFace 下载（~15GB）：
`reasonrag/Qwen2.5-7B-Instruct-ReasonRAG`

### 2.3 数据

| 数据 | 路径 | 条数 |
|------|------|------|
| HotpotQA dev | `/home/mayi/RAG/ReasonRAG/dataset/hotpotqa/dev.jsonl` | 7,405 |
| HotpotQA train | `/home/mayi/RAG/ReasonRAG/dataset/hotpotqa/train.jsonl` | 90,447（FlashRAG 格式） |
| DPO 训练数据 | `/home/mayi/RAG/ReasonRAG/training_data/RAG_ProGuide.json` | 13,289 |
| LLaMA-70B MCTS 中间数据 | `/home/mayi/RAG/ReasonRAG/output/hotpotqa_*MCTS*/chunk_*/` | 2,878 条轨迹 |

### 2.4 检索索引

| 索引 | 路径 | 大小 |
|------|------|------|
| BGE Flat | `/home/mayi/RAG/retriever/bgeindex/bge_Flat.index` | 64G |
| E5 Flat | `/home/mayi/RAG/retriever/flashrag_index/e5_flat_inner.index` | 61G |
| Wiki corpus | `/home/mayi/RAG/corpus/wiki18_100w.jsonl` | 27G |

⚠️ 原始 ReasonRAG 用 `bge` + `bge_Flat.index`，但你的推理脚本（inference.py）用的是 `e5` + `e5_flat_inner.index`。**需要确认用哪个**。

### 2.5 API

- DMXAPI：已验证可用
  - base_url: `https://www.dmxapi.cn/v1`
  - model: `gpt-4o`
  - API key: 已配置在 `gate0/.env`

### 2.6 代码仓库

| 仓库 | 路径 | 说明 |
|------|------|------|
| 原始 ReasonRAG | `/home/mayi/RAG/ReasonRAG_original/` | 7 个文件，干净版本 |
| 你的 ReasonRAG | `/home/mayi/RAG/ReasonRAG/` | 大量自定义修改 |
| FlashRAG | `/home/mayi/RAG/FlashRAG/` | 底层框架 |

---

## 3) 尚未确认 / 缺失的信息

### 3.1 需要对齐的关键配置

| 配置项 | 原始 ReasonRAG | 你的仓库 | 需要对齐？ |
|--------|---------------|---------|-----------|
| generator_model | gpt-4o-2024-05-13 | Qwen2.5-7B | ⚠️ **Gate 0 用 GPT-4o** |
| retrieval_method | bge | e5 | ⚠️ **需确认原始用的是哪个** |
| retrieval_topk | 3 | 5 | ⚠️ **不一致** |
| max_iter | 7 | 8 | ⚠️ **不一致** |
| max_children | 2 | 3 | ⚠️ **不一致** |
| max_rollouts | 64 | 64 | ✅ |
| beta | 0.95（pipeline 默认） | 0.9（data_generation.py BETA） | ⚠️ **不一致** |
| max_tokens | 256 | 256 | ✅ |
| do_sample | False | False | ✅ |
| framework | openai | vllm | ⚠️ **Gate 0 用 openai** |

### 3.2 原始 ReasonRAG 的推理配置 vs 数据生成配置

**推理时**（inference.py，用训练好的 Qwen2.5-7B-ReasonRAG）：
```python
# 推理不走 MCTS search()，走的是 batch mode（run_batch）
# batch mode = 线性推理，不做树搜索
# 所以推理时的参数和数据生成不同
```

**数据生成时**（data_generation.py，用 GPT-4o）：
```python
# 走 MCTS search()，完整树搜索
pipeline = ReasonRAGPipeline(config, max_iter=7, max_children=2, max_rollouts=64)
pipeline.search(item)
```

### 3.3 不确定的问题

1. **原始 ReasonRAG 的检索到底用 bge 还是 e5？**
   - `data_generation.py` 里写了 `"retrieval_method": "bge"`
   - 但 `my_config.yaml` 默认也是 bge
   - 你的 `inference.py` 用的是 e5
   - 原始论文需要确认

2. **beta 值不一致**：
   - pipeline 默认 `beta=0.95`（影响 MCTS 的 Q 值计算）
   - `preference_data_generatoin.py` 里 `BETA = 0.9`（影响 reward bottom-up 计算）
   - 这是两个不同的 beta，用在不同阶段

3. **retrieval_topk 不一致**：
   - 原始数据生成用 3
   - 你的推理用 5
   - 论文里写的是多少？

4. **原始 ReasonRAG 论文报告的 HotpotQA EM/F1 是多少？**
   - 需要确认 baseline 数字

5. **evaluate_thoughts 在推理时是否也调用？**
   - 数据生成时：调用（需要 golden answer 来打分）
   - 推理时（batch mode）：不调用（没有树搜索）
   - **Gate 0 跑 MCTS 模式**：会调用，但没有 golden answer → 需要处理

---

## 4) Gate 0 实验计划（待确认）

### 4.1 目标

验证 v4 typed transition evaluation + failure-attributed expansion 是否比标量 PRM 产生更好的最终答案。

### 4.2 方案

**Baseline**：原始 ReasonRAG MCTS pipeline（GPT-4o + 标量 PRM evaluate_thoughts）
**Treatment**：v4 typed eval 版（GPT-4o + typed φ_q/φ_c/φ_s + failure-attributed expansion）

### 4.3 对齐参数（待确认）

```yaml
# 与原始 ReasonRAG 数据生成对齐
generator_model: gpt-4o-2024-05-13
framework: openai
retrieval_method: bge          # 需确认
retrieval_topk: 3              # 原始数据生成配置
max_iter: 7
max_children: 2
max_rollouts: 64
beta: 0.95
max_tokens: 256
do_sample: False
```

### 4.4 数据

- HotpotQA dev 50 条（随机采样，含 bridge + comparison）
- 固定 random seed 保证可复现

### 4.5 指标

- EM / F1（主指标）
- avg steps
- premature_stop rate
- latency

### 4.6 成本估算

50 条 × ~300 次 LLM 调用/条 × ~500 tokens/次 ≈ 750 万 tokens
GPT-4o via DMXAPI ≈ ¥100-200

---

## 5) 需要你确认的事项

1. **检索器**：原始 ReasonRAG 论文里 HotpotQA 用的是 bge 还是 e5？
2. **retrieval_topk**：论文里是 3 还是 5？
3. **max_children**：论文里是 2 还是 3？
4. **原始 ReasonRAG 在 HotpotQA 上的 benchmark 数字**（EM/F1）是多少？
5. **Gate 0 是否需要完全对齐原始数据生成的参数**，还是可以对齐推理参数？
6. **evaluate_thoughts 在没有 golden answer 的情况下怎么处理**？推理时不用，但 MCTS 模式下 Q 值需要这个
