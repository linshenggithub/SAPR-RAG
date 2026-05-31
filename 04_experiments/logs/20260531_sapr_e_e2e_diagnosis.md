# SAPR-E End-to-End 无效诊断报告

**日期**: 2026-05-30 → 2026-05-31
**状态**: ✅ 根因定位完成
**标签**: `debug_result`

---

## 0. 结论

SAPR-E v0 evidence re-ranking 端到端 EM/F1 无提升（EM=0.2333 不变，F1 降 1.1pp），**根因是实验配置错误（`max_tokens=32`），而非方法本身无效。**

实验配置将 ReasonRAG 原始 `max_tokens=256` 改为了 `32`，导致模型每步生成被物理截断，pipeline 路由从未到达文档注入阶段，模型全程纯参数知识盲答，SAPR-E 重排的文档从未被模型看到。

---

## 1. 现象

### 1.1 端到端结果

| 实验 | EM | F1 | 检索方式 | 文档选择 |
|------|----|----|---------|---------|
| Baseline (v1) | 0.2333 | 0.3283 | 空 query, top-3 | retriever top-3 |
| SAPR-E v1 | 0.2333 | 0.3171 | 空 query, top-10 | SAPR-E rerank top-3 |
| SAPR-E v2 (修复 subquery) | 0.2333 | 0.3171 | inferred_subquery, top-10 | SAPR-E rerank top-3 |

三组实验 EM 完全相同，F1 无改善。

### 1.2 逐条轨迹对比

对比 Baseline 和 SAPR-E 的 `intermediate_data.json`（30 条）：

| 指标 | 数值 |
|------|------|
| Thought 序列完全一致（byte-level） | **30/30 (100%)** |
| 答案发生变化 | 9/30（均为格式/解析差异） |
| EM 变化 | 0/30（baseline right → sapr wrong = 0，反向 = 0） |
| F1 改善/恶化 | 0 改善, 2 恶化, 28 不变 |

**关键发现：30/30 条的完整 thought 序列在 baseline 和 SAPR-E 之间完全一致。** 模型没有因为文档不同而产生任何推理差异。

---

## 2. 根因分析

### 2.1 直接原因：模型每步输出被 max_tokens=32 截断

实验配置文件 `04_experiments/logs/20260529_evidence_debug_30samples_v2/.../config.yaml`:

```yaml
generation_params:
    max_tokens: 32          # ← 实验配置（错误）
```

ReasonRAG 原始配置 `my_config.yaml`:

```yaml
generation_params:
  do_sample: False
  max_tokens: 256           # ← 原始配置
```

**32 tokens 不足以让模型完成"分析 + 结论标记"。**

截断证据（dev_0 的 5 步 CoT）：

| Step | 末尾内容 | 是否截断 |
|------|---------|---------|
| 0 | `"...\n2. Determine Ed Wood's nationality.\n\nNow"` | ✅ "Now" 后截断 |
| 1 | `"...born on September 26, 1967,"` | ✅ 逗号后截断 |
| 2 | `"...an American filmmaker.\n"` | ✅ 换行后截断 |
| 3 | `"...Ed Wood were A"` | ✅ 单词中间截断 |
| 4 | `"So the answer is <answer>yes</answer>"` | ❌ 正常结束 |

只有最后一步（`answer_generation_prompt` 更短）才能在 32 token 内塞下 `"So the answer is <answer>..."` 闭合标签。

### 2.2 级联后果

```
max_tokens=32
  → 模型每步只能写 ~30 token 的分析片段
    → 来不及输出 "So the next query is <query>..."
      → get_flags() 返回 "None"
        → batch pipeline 走 answer_generation_prompt（无文档 slot）
          → 模型纯靠参数知识盲答
            → SAPR-E 换什么文档都不影响输出
```

### 2.3 Flag 统计

30 条 item、143 个 reasoning steps 的 flag 分布：

| Flag | 次数 | 比例 | 含义 |
|------|------|------|------|
| `"None"` | 114 | 79.7% | 模型未生成任何结论标记 |
| `"answer"` | 29 | 20.3% | 模型输出 "So the answer is..." |
| `"query"` | **0** | **0%** | 模型**从未**请求检索 |
| `"evidence"` | **0** | **0%** | 模型**从未**生成 evidence 标记 |

模型从不生成 `"So the next query is..."`，因此 pipeline 的 `run_batch` 方法中：

```python
# reasonrag_pipeline.py line ~575-583
if "query" in item.flag:       # ← 从未满足
    → document_analysis_prompt   # (有文档) ← 从未触发
else:
    → answer_generation_prompt   # (无文档) ← 100% 走这里
```

### 2.4 文档注入证据

| 指标 | 数值 |
|------|------|
| 包含文档/retrieval/Wikipedia 引用的 step | **0/143** |
| intermediate_node 含 `retrieval_result` 的 item | **0/30** |
| Step 0 flag 为 `"None"` 的 item | **30/30** |

---

## 3. Tree Mode vs Batch Mode 架构对比

这个问题只在 batch mode 中出现。ReasonRAG 的 tree mode (MCTS) 设计了不同的路由：

### 3.1 Tree mode（原始，正确）

固定状态机，pipeline 驱动：

```python
# next_state() 转换表
action_transitions = {
    "begin_reasoning":  → 没出 <answer> 就走 "document_analysis"   # 强制注入文档
    "reasoning":        → 没出 <answer> 就走 "document_analysis"   # 强制注入文档
    "document_analysis": → 走 "reasoning"                          # 固定循环
    "answer_generation": → 结束                                     # 终止
}
```

→ **只要模型没直接输出最终答案，pipeline 就强制进入 `document_analysis`（注入文档）。**

循环模式：`begin_reasoning → document_analysis → reasoning → document_analysis → ... → answer`

### 3.2 Batch mode（我们使用的，设计依赖模型配合）

模型驱动路由：

```python
# run_batch() 路由逻辑
if "query" in item.flag:       # 需要模型主动说 "So the next query is..."
    → document_analysis_prompt  # 注入文档
else:
    → answer_generation_prompt  # 不注入文档
```

→ **只有模型主动生成 `"So the next query is..."` 才注入文档，否则不注入。**

### 3.3 设计差异

| | Tree mode | Batch mode |
|---|---|---|
| 驱动方式 | 状态机驱动（确定性） | 模型输出驱动（条件性） |
| 默认行为 | 默认注入文档 | 默认不注入文档 |
| query 来源 | `handle_document_analysis` 从 thoughts 提取 | 依赖模型 `<query>` 标记 |
| 检索调用 | 每节点单独 `retriever.search()` | 循环开头统一 `batch_search()` |
| 对模型要求 | 低（pipeline 安排何时看文档） | 高（模型需主动请求检索） |

**在 `max_tokens=256` 下，batch mode 设计是合理的**——模型有足够空间生成结论标记，能正确路由。`max_tokens=32` 的配置错误破坏了这个前提。

---

## 4. 对 SAPR-E 方向的影响

### 4.1 之前判定的无效结论需要推翻

之前的判定：
> "SAPR-E v0 启发式 evidence re-ranking 在端到端上无效"

**这个判定不成立。** 端到端无效是因为 `max_tokens=32` 配置错误导致文档从未注入，不是 SAPR-E 方法本身的问题。

### 4.2 Evidence-only 信号仍然有效

在正确的 evidence-only 评估框架下（后置重检索 + heuristic scorer 对比），SAPR-E v0 的信号在 200-sample 上稳定：

| Selector | ItemHit% | vs retriever |
|----------|---------:|:------------:|
| retriever_top3 | 47.7% | — |
| **sapr_e_v0** | **51.8%** | **+4.1pp** |

### 4.3 需要重做的实验

修正 `max_tokens=256` 后重跑端到端：
1. Baseline 30-sample（max_tokens=256）
2. SAPR-E e2e 30-sample（max_tokens=256）
3. 对比轨迹，确认文档被注入、模型输出有差异
4. 如果有差异，再评估 EM/F1 变化

---

## 5. 配置错误来源

实验配置是通过 `experiment-bridge` skill 在 `2026-05-29` 创建的（`04_experiments/run_configs/run_20260529_evidence_debug.yaml`），`max_tokens=32` 不是来自 ReasonRAG 原始配置（原始为 256）。

可能原因：
1. experiment-bridge 创建配置时沿用了某个最小测试模板
2. 或手动修改时误改

**建议**：后续实验配置创建时，自动对照 ReasonRAG 原始 `my_config.yaml` 的关键参数做 diff 校验。

---

## 6. CoT 样例

### dev_1（答错，模型纯参数知识）

```
Question: What government position was held by the woman who portrayed
          Corliss Archer in the film Kiss and Tell?
Gold:     Chief of Protocol (Shirley Temple)
Pred:     none

Step 0 [NONE]: "To address the question, I'll break it down into manageable parts:
               1. Identify the woman who portrayed Corliss Archer..."     ← 截断

Step 1 [NONE]: "Actress Glenn Close played the character Corliss Archer
               in the film 'Kiss and Tell..."                            ← 事实错误 + 截断

Step 2 [NONE]: "Glenn Close is known for her acting roles but does not
               hold any government position..."                          ← 截断

Step 3 [ANSWER]: "So the answer is <answer>none</answer>"                ← 盲答
```

模型在第 1 步把 Corliss Archer 的扮演者从 **Shirley Temple** 记成了 **Glenn Close**，然后基于错误前提推理。如果文档被注入，模型能看到 Wikipedia 页面 "Shirley Temple" 和 "Kiss and Tell (1945 film)"，可能纠正这个错误。

---

## 7. 文件索引

| 文件 | 路径 |
|------|------|
| 本诊断报告 | `04_experiments/logs/20260531_sapr_e_e2e_diagnosis.md` |
| Overnight summary | `04_experiments/overnight_summary.md` |
| Baseline config (max_tokens=32) | `04_experiments/logs/20260529_evidence_debug_30samples_v2/.../config.yaml` |
| ReasonRAG 原始 config (max_tokens=256) | `my_config.yaml` (rag-5090) |
| SAPR-E e2e script | `03_sapr_rag/scripts/run_sapr_e_e2e.py` |
| SAPR-E e2e 输出 | `04_experiments/logs/20260530_sapr_e_e2e_sapr_e/` (rag-5090) |
| Baseline 输出 | `04_experiments/logs/20260530_sapr_e_e2e_baseline/` (rag-5090) |
| ReasonRAG pipeline | `pipeline/reasonrag_pipeline.py` (rag-5090) |
| 200-sample evidence metrics | `04_experiments/metrics/20260530_sapr_evidence_v0_200_reretrieved/metrics.json` |
