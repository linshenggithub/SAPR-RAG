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

### 4.7 跨服务器执行与同步计划

采用"3090 控制/开发，5090 运行实验"的方式，避免在两台机器上同时改代码，也避免搬运 64G 级别索引文件。

角色分工：

```text
Server B / 3090:
  - 主开发节点
  - 修改 Gate0 / SAPR-RAG 代码
  - 写实验配置、计划、总结和指标分析脚本
  - 通过 GitHub push 同步代码
  - 通过 ssh 调度 rag-5090 实验

Server A / rag-5090:
  - 主实验节点
  - pull GitHub 最新代码
  - 使用本机已有 extended BGE index / corpus
  - 只运行实验和生成结果
  - 不在远程直接手改核心代码，除非是临时 debug，并且必须回传到 3090 后入库
```

标准执行流程：

```bash
# 1. 在 3090 修改代码和配置
cd /home/mayi/RAG/agentic-rag-process-optimization
git status
git add <changed files>
git commit -m "[exp] align gate0 execution plan"
git push

# 2. 在 5090 拉取最新代码
ssh rag-5090 "cd /home/mayi/RAG/agentic-rag-process-optimization && git pull"

# 3. 从 3090 调度 5090 运行实验
ssh rag-5090 "cd /home/mayi/RAG/agentic-rag-process-optimization && bash <launch_script>"

# 4. 实验结束后只同步小文件结果
rsync -av rag-5090:/home/mayi/RAG/agentic-rag-process-optimization/04_experiments/metrics/<run_id>/ \
  /home/mayi/RAG/agentic-rag-process-optimization/04_experiments/metrics/<run_id>/
```

约束：

- 不从 5090 拉取或复制 64G FAISS index、corpus、checkpoint。
- 不覆盖 `/home/mayi/ReasonRAG/output/`、`indexes/`、`corpus/`、`dataset/`、`training_dataset/` 等已有结果/数据目录。
- 5090 上的新实验输出必须写到：

```text
/home/mayi/RAG/agentic-rag-process-optimization/04_experiments/logs/<run_id>/
/home/mayi/RAG/agentic-rag-process-optimization/04_experiments/metrics/<run_id>/
```

- 每个跨服务器实验都要记录：

```yaml
server: rag-5090
control_node: 3090
sync_method: github_pull
index_path:
corpus_path:
output_path:
metric_path:
```

---

## 5) 需要你确认的事项

1. **检索器**：原始 ReasonRAG 论文里 HotpotQA 用的是 bge 还是 e5？
2. **retrieval_topk**：论文里是 3 还是 5？
3. **max_children**：论文里是 2 还是 3？
4. **原始 ReasonRAG 在 HotpotQA 上的 benchmark 数字**（EM/F1）是多少？
5. **Gate 0 是否需要完全对齐原始数据生成的参数**，还是可以对齐推理参数？
6. **evaluate_thoughts 在没有 golden answer 的情况下怎么处理**？推理时不用，但 MCTS 模式下 Q 值需要这个

---

## 6) Codex Benchmark 对齐审计补充（2026-06-01）

### 6.1 `/nas/mayi/RAG/retrievers/bge_extended_index` 是否就是论文索引？

**结论：不是一个可由论文直接确认的路径，而且当前该目录为空；不能把它当作可用 benchmark 索引。**

论文不会记录本机绝对路径。论文中能确认的是逻辑配置：

- retriever 使用 **BGE**；
- knowledge source 使用 **Wikidump 2018**；
- 为保证检索质量，corpus 额外加入 PopQA、HotpotQA、2WikiMultiHopQA 的相关内容；
- 推理/实验中 **consistently retrieving the top 3 documents**。

对应论文原文位置：

- Appendix F.1: "we employ BGE as our retriever, consistently retrieving the top 3 documents"
- Table 5: BGE 资源链接为 `https://huggingface.co/BAAI/bge-base-en-v1.5`
- Appendix F.1: "augment our corpus by incorporating relevant content from the PopQA, HotpotQA, and 2WikiMultiHopQA datasets"

本机检查结果：

```text
/nas/mayi/RAG/retrievers/bge_extended_index/
  当前为空目录，没有 .index 文件
```

因此它最多可能是过去某次预期存放 extended BGE index 的目录名，但**不是现在可直接用于 Gate 0 的索引文件路径**。

### 6.2 原始 ReasonRAG 仓库里真正对应 extended corpus 的路径

原始仓库 `/home/mayi/RAG/ReasonRAG_original/data_generation.py` 明确写的是：

```python
"index_path": "indexes/bge_Flat_wiki_extend.index",
"corpus_path": "indexes/wiki18_100w_extend.jsonl",
"retrieval_method": "bge",
"retrieval_topk": 3,
```

这和论文中的描述一致：**BGE + extended corpus + top-3**。

注意：`ReasonRAG_original/inference.py` 写的是：

```python
"index_path": "indexes/bge_Flat.index",
"corpus_path": "indexes/wiki18_100w.jsonl",
"retrieval_method": "bge",
parser.add_argument("--retrieval_top_k", default=3, type=int)
```

所以：

- **MCTS 数据生成 / RAG-ProGuide 构造**：对齐 `bge_Flat_wiki_extend.index` + `wiki18_100w_extend.jsonl`
- **训练后模型 inference benchmark**：对齐 `bge_Flat.index` + `wiki18_100w.jsonl`

如果 Gate 0 是 "GPT-4o MCTS baseline vs typed MCTS treatment"，应优先对齐 **data_generation.py**，不是 inference.py。

### 6.3 当前机器上可用资源的判断

当前已发现的可用索引/语料：

```text
本机 3090:
  /home/mayi/RAG/retriever/bgeindex/bge_Flat.index          64G
  /home/mayi/RAG/corpus/wiki18_100w.jsonl                  27G

远程 rag-5090:
  /home/mayi/ReasonRAG/indexes/bge_extended/bge_Flat.index  64G
  /nas/mayi/RAG/corpus/wiki18_extended.jsonl                15G
  /nas/mayi/RAG/corpus/wiki18_100w.jsonl                    14G

NAS:
  /nas/mayi/RAG/retrievers/bge_extended_index/              empty directory
```

系统判断：

1. `/nas/mayi/RAG/retrievers/bge_extended_index` 当前不可用，因为没有索引文件。
2. 若要复现论文 **inference benchmark**，最稳对齐是：

```yaml
retrieval_method: bge
retrieval_topk: 3
index_path: /home/mayi/RAG/retriever/bgeindex/bge_Flat.index
corpus_path: /home/mayi/RAG/corpus/wiki18_100w.jsonl
```

3. 若要复现论文 **MCTS 数据生成 / Gate 0**，逻辑上应使用 extended corpus/index：

```yaml
retrieval_method: bge
retrieval_topk: 3
index_path: /home/mayi/ReasonRAG/indexes/bge_extended/bge_Flat.index
corpus_path: /nas/mayi/RAG/corpus/wiki18_extended.jsonl
```

这一路径目前只在 `rag-5090` 上确认存在。它文件名不是原始仓库里的 `bge_Flat_wiki_extend.index`，但语义上最接近：BGE extended index + extended corpus。

### 6.4 回答"论文里面是不是这个"

严格回答：

> 论文没有、也不可能写 `/nas/mayi/RAG/retrievers/bge_extended_index` 这个本机路径。论文确认的是 BGE retriever、top-3 retrieval、Wikidump 2018 + PopQA/HotpotQA/2Wiki augmented corpus。原始代码的数据生成配置对应 `indexes/bge_Flat_wiki_extend.index` 和 `indexes/wiki18_100w_extend.jsonl`。

工程判断：

> `/nas/mayi/RAG/retrievers/bge_extended_index` 当前为空，不是可用索引。若 Gate 0 要对齐原始 ReasonRAG 的 MCTS 数据生成，应使用 `rag-5090:/home/mayi/ReasonRAG/indexes/bge_extended/bge_Flat.index` + `/nas/mayi/RAG/corpus/wiki18_extended.jsonl`，并记录这是"语义对齐 extended BGE index"，不是论文原文路径。

### 6.5 `evaluate_thoughts` 的 golden answer 泄漏问题

**结论：必须区分"ReasonRAG 数据生成模式"和"推理 benchmark 模式"。**

原始 ReasonRAG 的 MCTS `search()` 不是论文最终 inference benchmark 的运行方式，而是用于构造 RAG-ProGuide 过程偏好数据。它在 `evaluate_thoughts()` 和 terminal reward 中都使用 `golden_answers`：

```python
question_thoughts = (
    node.question
    + "\nGolden Answer: " + " or ".join(node.golden_answers)
    + "\nAgent Reasoning Process: " + " ".join(node.thoughts)
)
Q = GPT-4o(question_thoughts) / 100

reward = F1(pred, golden_answers) * beta ** step
```

因此：

1. **如果 Gate 0 目标是复现 / 对齐 ReasonRAG 的 MCTS 数据生成流程**
   可以使用 golden answer，因为原始代码就是这样做的。这类实验应标注为：

```text
MCTS annotation / process-data-generation benchmark
```

它验证的是：typed transition evaluation 是否能生成更好的过程轨迹/偏好数据，而不是一个无泄漏的 test-time inference 方法。

2. **如果 Gate 0 目标是和论文 Table 2 的 ReasonRAG inference benchmark 对齐**
   不能在 tree search 的分支选择中使用 golden answer。否则这是 answer leakage，不是合法推理 benchmark。论文最终 inference 用的是：

```text
inference.py -> trained Qwen2.5-7B-Instruct-ReasonRAG -> run_batch()
```

不是 GPT-4o + MCTS + golden-answer evaluator。

所以 Gate 0 当前的 "Baseline = 原始 ReasonRAG MCTS pipeline（GPT-4o + 标量 PRM evaluate_thoughts）" 只能作为 **oracle/annotation-style MCTS baseline**，不能直接声称对齐 ReasonRAG Table 2 inference benchmark。

**推荐 Gate 0 拆成两个可解释版本：**

```text
Gate0-A: Annotation-aligned MCTS
  baseline: 原始 GPT-4o MCTS + golden-answer scalar evaluate_thoughts
  treatment: GPT-4o MCTS + typed evaluation，可选择同样允许 final answer label 参与打分
  用途: 比较过程数据生成质量 / branch attribution 是否更好
  标签: oracle_result 或 annotation_result，不作为 test-time inference

Gate0-B: Inference-aligned MCTS
  baseline: 无 golden answer 的 test-time tree/batch baseline
  treatment: 无 golden answer 的 typed transition evaluation
  用途: 比较真实推理 EM/F1
  约束: selection / expansion / stop 不能访问 golden answer
```

如果只做一个首要计划，建议先做 **Gate0-B 的 30-50 条无泄漏小样本**。原因是它更接近论文可发表的主张；Gate0-A 虽然更贴近原始数据生成，但 reviewer 会立刻指出它不能证明推理时有效。

### 6.6 本地普通 BGE index 是否可接受，还是必须拉 extended index？

**结论：取决于 Gate 0 对齐目标。**

#### 情况 1：要严格对齐原始 ReasonRAG 的 MCTS 数据生成

应该用 extended index/corpus。原始 `data_generation.py` 写的是：

```python
index_path = "indexes/bge_Flat_wiki_extend.index"
corpus_path = "indexes/wiki18_100w_extend.jsonl"
retrieval_method = "bge"
retrieval_topk = 3
```

论文 Appendix F.1 也说明 corpus 被 PopQA/HotpotQA/2WikiMultiHopQA 相关内容增强。因此严格对齐 MCTS 数据生成时，推荐在 5090 上直接用：

```yaml
index_path: /home/mayi/ReasonRAG/indexes/bge_extended/bge_Flat.index
corpus_path: /nas/mayi/RAG/corpus/wiki18_extended.jsonl
retrieval_method: bge
retrieval_topk: 3
```

不要从 5090 拉 64G index 到本地，除非确实需要本地跑。直接在 `rag-5090` 跑更少移动数据，也避免索引/语料版本不一致。

#### 情况 2：要对齐论文最终 inference benchmark

普通 BGE index 是可接受甚至更贴近 `inference.py` 的。原始 `inference.py` 写的是：

```python
index_path = "indexes/bge_Flat.index"
corpus_path = "indexes/wiki18_100w.jsonl"
retrieval_method = "bge"
retrieval_top_k = 3
```

本机已有：

```yaml
index_path: /home/mayi/RAG/retriever/bgeindex/bge_Flat.index
corpus_path: /home/mayi/RAG/corpus/wiki18_100w.jsonl
retrieval_method: bge
retrieval_topk: 3
```

这组可以用于 inference-aligned 小实验。但必须在报告中明确：

```text
This run uses the non-extended BGE wiki18_100w index, aligned with ReasonRAG inference.py rather than data_generation.py.
```

#### 最终建议

为了避免 benchmark 混乱：

```text
Gate0-B / test-time inference: 先用本地普通 BGE index 跑 30-50 条，无 golden answer，快速判断方向。
Gate0-A / original MCTS data-generation alignment: 若需要复现原始 MCTS 数据生成，再在 rag-5090 用 extended BGE index 跑。
```

不要在同一张表里混用普通 index 和 extended index；如果两者都跑，必须分成：

```text
Index setting A: bge_Flat + wiki18_100w
Index setting B: bge_extended + wiki18_extended
```

否则 SAPR/typed evaluation 的收益会和 corpus/index 差异纠缠在一起。

---

## 7) 实现代码（2026-06-01）

### 7.1 实验命名

**Inference-style no-label MCTS pilot**（不对齐 ReasonRAG Table 2 benchmark）

### 7.2 代码位置

`gate0/gpt4o_experiment/mcts_pilot.py`

### 7.3 设计要点

1. **golden_answers 隔离**：
   - `MCTSNode` 不存储 golden_answers
   - `pipeline.search(question)` 只接收 question 字符串
   - golden_answers 仅在实验循环外用于 post-hoc 计算 EM/F1
   - 代码中有 assert 检查评估 prompt 不包含 "golden"

2. **Baseline（scalar self-eval）**：
   - 替换原始 `evaluate_thoughts`（需 golden answer）
   - 用 `SELF_EVAL_SYSTEM` prompt 让 GPT-4o 在无 golden answer 下评估推理质量
   - 输出 0-100 分数，归一化到 [0,1]

3. **Treatment（typed eval）**：
   - 用 `TYPED_EVAL_SYSTEM` prompt 让 GPT-4o 输出 query_quality/claim_quality/stop_quality 三维评分
   - 组合为 typed score = φ_q × φ_c × φ_s_norm

4. **参数对齐**（MCTS_CONFIG）：
   ```yaml
   max_iter: 7
   max_children: 2
   max_rollouts: 64
   c: 1.414
   retrieval_topk: 3
   max_tokens: 256
   ```

5. **索引**：本地普通 BGE（`bge_Flat.index` + `wiki18_100w.jsonl`）

### 7.4 运行流程

```bash
# Step 1: Sanity check（1 条，验证格式+无泄漏+指标计算）
python mcts_pilot.py --mode sanity

# Step 2: Baseline（确认 sanity 通过后）
python mcts_pilot.py --mode baseline --n_samples 50

# Step 3: Treatment
python mcts_pilot.py --mode treatment --n_samples 50

# Step 4: 对比
python mcts_pilot.py --mode eval
```

### 7.5 成本控制

- API 额度：¥100
- Sanity check: ~5-10 次 LLM 调用（< ¥0.5）
- 50 条 baseline: ~15,000 次 LLM 调用（估算 ¥30-50）
- 50 条 treatment: ~15,000 次 LLM 调用（估算 ¥30-50）
- 总计: ~¥60-100（在额度内）
