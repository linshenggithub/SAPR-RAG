# SAPR-R v1 数据构造管线 Handoff

> 写给下一个 AI / 协作者：本仓库当前主线是 **SAPR-R v1 trained reranker** 的离线训练数据构造。本文档是单一信源，包含整体方案、已完成代码清单、跑法、下一步任务。
>
> 更高层背景（为什么做 v1、为什么否决 v0/v4）请先读 [repo_overview.md §0–§0bis](./repo_overview.md)。本文档不重复那部分。
>
> 上次更新：2026-06-04（step5 + launcher 全步骤集成完毕；管线代码全部就位）

---

## 1. 整体方案速览

### 1.1 目标
为 SAPR-R v1 trained reranker 构造 ~5k question 规模的离线训练数据，每条样本是 `(state, doc, cls_label, retriever_score, step_gold)` 五元组。

### 1.2 架构定位（必读）
- **小替换路线**：保留 ReasonRAG 推理 pipeline + Qwen LoRA generator，**仅替换** reranker 模块。详见 [repo_overview.md §0bis.1](./repo_overview.md)。
- **训-推 thought 接口翻译**：训练用 DeepSeek 干净 thought；推理时在 reranker 入口插 `clean_thought()` 适配层把 ReasonRAG-Qwen 脏 thought 规则化。其他链路不动。

### 1.3 离线管线五步骤

```
HotpotQA train.jsonl  (输入)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ step2: DeepSeek 拆解 question 为 reasoning_steps        │ ✅ 已落地
│   每行: {qid, question, gt_answer, supporting_titles,   │
│          reasoning_steps:[{subquery, subject_entity,    │
│                            thought, step_gold} × ≤4]}   │
└─────────────────────────────────────────────────────────┘
        │ reasoning_steps.jsonl
        ▼
┌─────────────────────────────────────────────────────────┐
│ step3: BGE+FAISS 给每个 subquery 检索 wiki18 top-K      │ ✅ 已落地
│   每行: {qid, step_idx, …, subquery, prior_thoughts,    │
│          candidates:[{doc_id, title, text, score} × K]} │
└─────────────────────────────────────────────────────────┘
        │ candidates.jsonl
        ▼
┌─────────────────────────────────────────────────────────┐
│ step4: DeepSeek 给每个 (state, doc) 打 {label, evidence}│ ✅ 已落地
│   每行: {qid, step_idx, doc_id, cls_label, evidence,    │
│          raw_response, ok}                              │
└─────────────────────────────────────────────────────────┘
        │ cls_labels.jsonl
        ▼
┌─────────────────────────────────────────────────────────┐
│ step5: 拼装训练 jsonl（合并 step3 + step4）             │ ✅ 已落地
│   每行: {qid, step_idx, split, state, candidates:[      │
│           {doc, retriever_score, cls_label,             │
│            rank_target} × K_kept], meta}                │
│   输出: train.jsonl + dev.jsonl                         │
└─────────────────────────────────────────────────────────┘
```

> **step1 故意省略**：脏元素的来源是 ReasonRAG 显式 prompt schema（`<query>` / `<answer>` / `Error Reflection:` 等），无需 dump 训练 trajectory 自审计。详见 [repo_overview.md §0bis.2](./repo_overview.md)。

### 1.4 训练目标（数据构造决定的）
- **cls_label ∈ {0, 1}**：DeepSeek 在 step4 给出，"该 doc 是否显式陈述 step_gold"。
- **listwise rank target**：每组 K 个 doc，target = `softmax(α·rationale_score + (1-α)·retriever_score)`，其中 `rationale_score = cls_label`，α=0.7。
- **loss**：`L = 1.0·L_cls(BCE) + 0.5·L_rank(KL)`。
- 详见 [repo_overview.md §0bis.5](./repo_overview.md)。

---

## 2. 已完成代码清单

所有路径相对仓库根 `SAPR-RAG/`。

### 2.1 工具层 `03_sapr_rag/utils/`

| 文件 | 行数 | 作用 | 备注 |
|---|---|---|---|
| [thought_cleaner.py](../03_sapr_rag/utils/thought_cleaner.py) | ~205 | `clean_thought(raw, max_words=25, fallback)` / `clean_subquery(raw, fallback)` / `extract_evidence(text)` 三个纯字符串规则函数 | **推理时**在 reranker 入口插入；训练数据本身不经过它（DeepSeek 直出已干净）。黑名单覆盖 XML 标签 / 固定句式 / meta 段头 / 前导样板。 |
| [test_thought_cleaner.py](../03_sapr_rag/utils/test_thought_cleaner.py) | ~190 | 7 个 unittest 用例 | 跑法：`python 03_sapr_rag/utils/test_thought_cleaner.py`，全 OK。 |
| [deepseek_client.py](../03_sapr_rag/utils/deepseek_client.py) | ~380 | DeepSeek API 统一入口；支持 `from_env(prefer="deepseek"|"dmxapi")` / `chat` / `chat_json` / `chat_batch` / `chat_json_batch` | 指数退避（base 1.5s, max 30s, 5 retry） + jitter；`ThreadPoolExecutor` 并发；累计 prompt/completion tokens 统计。**所有 DeepSeek 调用必须从这里出**，不要散到各脚本。 |

### 2.2 数据构造管线 `03_sapr_rag/data/build_v1/`

| 文件 | 行数 | 作用 | 状态 |
|---|---|---|---|
| [prompts.py](../03_sapr_rag/data/build_v1/prompts.py) | ~370 | `build_step2_messages(question, gt_answer, supporting_titles)` 与 `build_step4_messages(question, prior_thoughts, subquery, subject_entity, step_gold, doc_title, doc_text)` 两个 message builder | 定稿。step2 输出 4 字段 schema、最多 4 步、含 2 个 few-shot；step4 含 A1/A2/A3 yes 规则 + B1/B2/B3/B4 no 规则 + 3 个 few-shot。 |
| [step2_generate_reasoning_steps.py](../03_sapr_rag/data/build_v1/step2_generate_reasoning_steps.py) | ~330 | 读 HotpotQA train.jsonl → DeepSeek 拆解为 `reasoning_steps[≤4]` → 流式落 jsonl | ✅ AST + --help 通过；未实跑。 |
| [step3_retrieve_candidates.py](../03_sapr_rag/data/build_v1/step3_retrieve_candidates.py) | ~440 | 读 step2 输出 → flatten 成 (qid, step_idx) 单元 → BGE batch encode subquery → FAISS top-K → corpus 一次扫描抽 title/text | ✅ AST + --help 通过；需在 5090 实跑。 |
| [step4_label_cls.py](../03_sapr_rag/data/build_v1/step4_label_cls.py) | ~370 | 读 step3 candidates.jsonl → flatten 成 (qid, step_idx, doc_id) tasks → 分 chunk 调 DeepSeek `build_step4_messages` 给每对 (state, doc) 打 `{label, evidence}` → 流式落 jsonl | ✅ AST + --help 通过；需在 5090 实跑。 |
| [test_step4_smoke.py](../03_sapr_rag/data/build_v1/test_step4_smoke.py) | ~280 | step4 本地 smoke test：5 个 mock task 覆盖 yes/no/schema-fail/json-fail/api-fail 五条路径 + 多 chunk + 断点续跑；monkey-patch `DeepSeekClient.from_env` 注入 FakeClient，无网络/无 API 即可跑通 | ✅ 本地通过（`[test] ALL OK ✓`）。跑法：`python 03_sapr_rag/data/build_v1/test_step4_smoke.py` |
| [step5_assemble_train_jsonl.py](../03_sapr_rag/data/build_v1/step5_assemble_train_jsonl.py) | ~440 | 读 step3 candidates.jsonl + step4 cls_labels.jsonl → 按 (qid, step_idx, doc_id) join → 各类过滤（missing label / ok=False / 全负 group / k_kept 不足）→ 组内算 listwise rank_target = `softmax(α·cls_label + (1-α)·norm(retriever_score))` → 按 qid hash 切 train/dev → 写 train.jsonl + dev.jsonl + run_meta_step5.json | ✅ AST + --help + smoke 通过。 |
| [test_step5_smoke.py](../03_sapr_rag/data/build_v1/test_step5_smoke.py) | ~250 | step5 本地 smoke test：4 个 mock unit + 15 个 mock label 覆盖 全负丢 / missing-label / ok=False / 全正保留 / dev_ratio 极值 / `--keep-all-negative` 六类分支；纯 IO 无外部依赖 | ✅ 本地通过（`[test] ALL OK ✓`）。跑法：`python 03_sapr_rag/data/build_v1/test_step5_smoke.py` |
| [launch_build_v1_data.sh](../03_sapr_rag/data/build_v1/launch_build_v1_data.sh) | ~210 | 一键 launcher，串联 step2 → step3 → step4 → step5；`set -euo pipefail`、必需环境变量自检、每步独立 log（tee）、空产物报错、SKIP_STEP{2,3,4,5} 跳过开关、LIMIT_DEBUG 透传 smoke 模式 | ✅ `bash -n` 通过。 |

### 2.3 路径配置 `config/`

| 文件 | 改动 |
|---|---|
| [paths.py](../config/paths.py) | 新增 `HOTPOTQA_TRAIN_PATH = _RequiredPath("SAPR_HOTPOTQA_TRAIN_PATH", ...)` |
| [__init__.py](../config/__init__.py) | 同步导出 `HOTPOTQA_TRAIN_PATH` |
| [env_3090.sh](../config/env_3090.sh) | 加 `export SAPR_HOTPOTQA_TRAIN_PATH="/home/mayi/RAG/ReasonRAG/dataset/hotpotqa/train.jsonl"` |
| [env_5090.sh](../config/env_5090.sh) | 加 `export SAPR_HOTPOTQA_TRAIN_PATH="/home/mayi/ReasonRAG_modified/dataset/hotpotqa/train.jsonl"` |

> 仓内路径一律用 `Path(__file__).resolve().parents[N]` 派生（[coding_standard.md](./coding_standard.md) §2）；仓外路径走 `_RequiredPath` + 环境变量，**不内置机器特定默认值**。

---

## 3. 关键设计决策（不要回炉）

下表汇总已对齐的决策，新 AI 不要重新讨论：

| # | 决策点 | 选择 | 来源 |
|---|---|---|---|
| D1 | 数据底库 | HotpotQA train ~90k，先抽 5k | reward_data 是 Llama 复现版不能用 |
| D2 | 推理打标 LLM | **DeepSeek-V3 API**（备用 DMXAPI） | ¥45 / 30k 调用 |
| D3 | candidate 来源 | **BGE wiki18 top-10**（不是 supporting_facts） | 语料不同源 |
| D4 | step2 schema | **4 字段** `{subquery, subject_entity, thought, step_gold}` | subject_entity 专门给 step4 wrong-entity 检测用 |
| D5 | step2 reasoning_steps 上限 | **≤4 步** | HotpotQA 99% 是 2-hop |
| D6 | step2 输入是否带 doc | **只喂 supporting_titles** | 不喂 sentence body 防止泄题，又能锚定主题 |
| D7 | step4 prior_reasoning 来源 | **前 k-1 个完整 thought 句**（不是 step_gold 片段） | 信息更完整 |
| D8 | step4 subject_entity 衔接 | **修法 A**：step2 直接输出 subject_entity 字段，step4 不再自抽 | 避免 step4 LLM 抽错 |
| D9 | history_thoughts 黑/白名单 | **只装 evidence**（document_analysis 输出）；丢弃 begin_reasoning 元-计划 + reasoning 元-元话语 | 训-推分布对齐 |
| D10 | clean_thought 作用域 | **仅在 reranker 输入处插入**；generator 链路 / trajectory log / reward 计算保持脏 | [repo_overview.md §0bis.2](./repo_overview.md) |
| D11 | cls 标注方式 | **方案 I**：answer-aware verify，喂七元组，binary `{label, evidence}` | LLM judge 噪声更低 |
| D12 | retriever 实现 | **路径 B**：纯 BGE encoder + FAISS（不引 FlashRAG DenseRetriever） | batch encode 比 DenseRetriever for-loop 快得多 |
| D13 | step3 跑机器 | **5090** | 需 ≥80GB 内存装 60GB FAISS Flat 索引 |

---

## 4. 跑法（5090）

### 4.1 一次性环境准备
```bash
cd /home/mayi/.../SAPR-RAG
source config/env_5090.sh        # 设置所有 SAPR_*_PATH 环境变量
conda activate reasonrag         # 已有的 reasonrag conda env
echo "DEEPSEEK_API_KEY=sk-..." > 03_sapr_rag/.env   # 或 DMXAPI_API_KEY
```

### 4.2 一键 launcher（推荐）

把 step2 → step3 → step4 → step5 串联成一条命令：

```bash
# 全量 5k：
bash 03_sapr_rag/data/build_v1/launch_build_v1_data.sh

# 小批 smoke（先跑这个验证一切正常）：
LIMIT_DEBUG=10 RUN_NAME=v1_smoke bash 03_sapr_rag/data/build_v1/launch_build_v1_data.sh
```

可调环境变量（全部可选，有默认值）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `RUN_NAME` | `v1_5k` | 输出子目录名（`out/<RUN_NAME>/`） |
| `N_SAMPLES` | `5000` | step2 抽样数 |
| `SEED` | `42` | step2 随机种子 + step5 split 种子 |
| `TOP_K` | `10` | step3 检索 top-K |
| `MAX_WORKERS_S2` | `30` | step2 DeepSeek 并发 |
| `MAX_WORKERS_S4` | `50` | step4 DeepSeek 并发 |
| `CHUNK_SIZE_S4` | `2000` | step4 每块 chat_batch + flush 大小 |
| `ALPHA_S5` | `0.7` | step5 rank_target 中 cls_label 的权重 |
| `NORM_MODE_S5` | `minmax` | step5 retriever_score 组内归一方式（`minmax`/`zscore`/`none`） |
| `DEV_RATIO_S5` | `0.1` | step5 dev 切分比例 |
| `LIMIT_DEBUG` | （未设） | 若设，所有 step 共用此 `--limit-debug` 值 |
| `SKIP_STEP2` / `SKIP_STEP3` / `SKIP_STEP4` / `SKIP_STEP5` | （未设） | 非空跳过对应步骤（用前次产物） |

行为：
- 跑前自检 `SAPR_HOTPOTQA_TRAIN_PATH` / `SAPR_BGE_INDEX_PATH` / `SAPR_BGE_MODEL_PATH` / `SAPR_WIKI_CORPUS_PATH` 是否设置（没设就让你 source env_*.sh），并检查 `DEEPSEEK_API_KEY` / `DMXAPI_API_KEY` / `03_sapr_rag/.env` 至少存在其一
- 每步 log 落 `out/<RUN_NAME>/logs/step{2,3,4,5}.log`（tee 同时打印 + 落盘）
- 每步结束后检查产物非空（`[[ -s ... ]]`），空文件直接报错停止
- 任意一步 fail → `set -e` 立刻停（断点续跑安全：step2-4 重跑同命令会自动跳过已完成 key；step5 是纯 join 重跑覆盖）
- 结束打印每个 jsonl 的 `wc -l`

典型工作流：
1. 先 `LIMIT_DEBUG=10 RUN_NAME=v1_smoke ...` 跑 smoke（约 5 分钟，~¥0.5）
2. 检查 `out/v1_smoke/{reasoning_steps,candidates,cls_labels,train,dev}.jsonl` 非空 + 字段齐全
3. 再无 LIMIT_DEBUG 跑全量 5k（~3-4 小时，~¥200-250）

如需手动按步执行（debug 单步、重跑某步），见 4.3-4.6。

### 4.3 step2 — 生成 reasoning_steps
```bash
python 03_sapr_rag/data/build_v1/step2_generate_reasoning_steps.py \
    --n-samples 5000 \
    --seed 42 \
    --max-workers 30 \
    --out-dir 03_sapr_rag/data/build_v1/out/v1_5k
```
- 产物：`out/v1_5k/reasoning_steps.jsonl` + `run_meta.json`
- 预估：30000 调用 ~30-40 分钟，~¥45（DeepSeek-V3）
- 断点续跑：直接重跑同一命令，已完成 qid 自动跳过
- debug：加 `--limit-debug 10` 只跑前 10 条

### 4.4 step3 — BGE 检索 top-K
```bash
CUDA_VISIBLE_DEVICES=0 python 03_sapr_rag/data/build_v1/step3_retrieve_candidates.py \
    --in-dir 03_sapr_rag/data/build_v1/out/v1_5k \
    --top-k 10
```
- 产物：`out/v1_5k/candidates.jsonl` + `run_meta_step3.json`
- 资源：≥80GB 内存（FAISS Flat 60GB 全装内存），~2GB GPU（encoder，编码完即释放）
- 预估：5k × 平均 2.5 step ≈ 12.5k subquery，encode ~3 分钟，FAISS search ~5 分钟，corpus 抽取 ~5 分钟
- 自检：meta 中 `gold_recall_top_k` 字段（top-K 命中 supporting_titles 的比例），健康基线 ≥70%
- 断点续跑：复合 key `(qid, step_idx)`

### 4.5 step4 — cls 打标
```bash
python 03_sapr_rag/data/build_v1/step4_label_cls.py \
    --in-dir 03_sapr_rag/data/build_v1/out/v1_5k \
    --max-workers 50 \
    --chunk-size 2000
```
- 输入：`candidates.jsonl`
- 产物：`out/v1_5k/cls_labels.jsonl` + `run_meta_step4.json`
- 调用次数：12.5k × 10 doc ≈ 125k 次 DeepSeek 调用
- 预估：~¥150-200，~2-3 小时（50 并发）
- 分块：`--chunk-size 2000` 每 2k pair 一次 chat_batch + flush，崩溃最多丢一个 chunk
- 健康基线：meta 中 `pos_ratio` 应在 30-50% 之间；过低（<20%）说明 prompt 太严，过高（>60%）说明太松
- 断点续跑：复合 key `(qid, step_idx, doc_id)`

### 4.6 step5 — 拼装训练 jsonl
```bash
python 03_sapr_rag/data/build_v1/step5_assemble_train_jsonl.py \
    --in-dir 03_sapr_rag/data/build_v1/out/v1_5k \
    --alpha 0.7 --norm-mode minmax --dev-ratio 0.1 --seed 42
```
- 输入：`candidates.jsonl` + `cls_labels.jsonl`
- 产物：`train.jsonl` + `dev.jsonl` + `run_meta_step5.json`
- 默认参数：α=0.7（cls 权重） / `minmax` 归一 / dev 10% / `min-k-kept=2` / 全负 group 丢
- 用 `--keep-all-negative` 保留全负 group（rank_target 退化为 retriever-only 排序）
- 健康基线：meta `pos_ratio` 应跟 step4 接近；`avg_k_kept` 接近 K=10；`drop_*` 各分支占比合理
- step5 是纯 join，无外部调用；重跑直接覆盖 train/dev jsonl，需保留旧产物请改 `--out-dir`

### 4.7 step4 本地 smoke（无网络/无 API 也能跑）
```bash
python 03_sapr_rag/data/build_v1/test_step4_smoke.py
```
- 不依赖任何外部服务（monkey-patch `DeepSeekClient.from_env` 注入 FakeClient）
- 5 个 mock task 覆盖 yes / no / schema-fail / json-fail / api-fail 五条路径，并验证多 chunk + 断点续跑
- 改了 step4 主逻辑后必跑；秒级出结果

### 4.8 step5 本地 smoke（无任何外部依赖）
```bash
python 03_sapr_rag/data/build_v1/test_step5_smoke.py
```
- 纯 IO，无网络无 API 无 GPU
- 4 个 mock unit + 15 个 mock label 覆盖：全负 group 丢 / missing-label / ok=False / 全正 group 保留 / dev_ratio 极值（0.0 / 1.0） / `--keep-all-negative` 行为
- 验证 rank_target 计算：cls=1 严格压制 cls=0；retriever_score 单调时 rank_target 单调
- 改了 step5 主逻辑后必跑；秒级出结果

---

## 5. 关键 schema 速查

### 5.1 step2 输出（reasoning_steps.jsonl 每行）
```json
{
  "qid": "5a8b57f25542995d1e6f1371",
  "question": "...",
  "gt_answer": "...",
  "supporting_titles": ["Oberoi Group", "Oberoi family"],
  "reasoning_steps": [
    {"subquery": "...", "subject_entity": "Oberoi Group", "thought": "...", "step_gold": "..."},
    ...
  ],
  "raw_response": "...DeepSeek 原始 JSON...",
  "ok": true,
  "error": null
}
```

### 5.2 step3 输出（candidates.jsonl 每行）
```json
{
  "qid": "5a8b57f25542995d1e6f1371",
  "step_idx": 0,
  "question": "...",
  "gt_answer": "...",
  "supporting_titles": ["...", "..."],
  "subquery": "...",
  "subject_entity": "Oberoi Group",
  "step_gold": "...",
  "prior_thoughts": [],          // step_idx==0 时为 []
  "candidates": [
    {"doc_id": 12345, "title": "...", "text": "...", "retriever_score": 0.92},
    ...×K
  ]
}
```
> 粒度刻意选 `(qid, step_idx)` 单元而非整 qid，因为 step4 prompt 是逐 (state, doc) 调一次，扁平化后 step4/step5 处理更直接。

### 5.3 state 三元组（训-推统一）
| 字段 | 训练时来源 | 推理时来源 |
|---|---|---|
| `question` | HotpotQA `question` | ReasonRAG pipeline 输入 |
| `history_thoughts` | DeepSeek 前 k-1 个 `thought` 句（完整 SVO 陈述） | 前 document_analysis 步的 `extract_evidence()` 结果 |
| `subquery` | DeepSeek 第 k 个 `subquery` | reasoning / begin_reasoning 步的 `clean_subquery()` 结果 |

详见 [repo_overview.md §0bis.4](./repo_overview.md)。

---

## 6. 工程约定

### 6.1 包名以数字开头的导入
`03_sapr_rag/` 包名以数字开头，无法 `import 03_sapr_rag.xxx`。所有 step 脚本统一用 `importlib.util.spec_from_file_location` 加载兄弟模块，并 **必须** 注册到 `sys.modules` 否则 `@dataclass` 会崩：
```python
mod = importlib.util.module_from_spec(spec)
sys.modules[name] = mod   # ← 必须这一行
spec.loader.exec_module(mod)
```
[step2](../03_sapr_rag/data/build_v1/step2_generate_reasoning_steps.py) / [step3](../03_sapr_rag/data/build_v1/step3_retrieve_candidates.py) / [step4](../03_sapr_rag/data/build_v1/step4_label_cls.py) 都用同一模板。step5 不依赖兄弟模块所以无需 importlib。

### 6.2 jsonl 流式落盘 + 断点续跑
所有 step 脚本统一：
1. `load_completed_keys(out_jsonl)` 读已存在 jsonl 收集主键
2. 在 `_build_inputs(...)` 时跳过已完成
3. 每写一行立即 `flush()`
4. 失败样本也写入（带 `ok: false, error: "..."`），不中断

> **例外**：step5 是纯 join，无外部调用，每次重跑覆盖 train/dev jsonl。

### 6.3 输出目录约定
```
03_sapr_rag/data/build_v1/out/v1_5k/
├── reasoning_steps.jsonl   # step2 产物
├── run_meta.json           # step2 元信息
├── candidates.jsonl        # step3 产物
├── run_meta_step3.json     # step3 元信息
├── cls_labels.jsonl        # step4 产物
├── run_meta_step4.json     # step4 元信息
├── train.jsonl             # step5 产物
├── dev.jsonl               # step5 产物
├── run_meta_step5.json     # step5 元信息
└── logs/                   # launcher 落 step{2,3,4,5}.log
```

---

## 7. 下一步任务（按优先级）

| # | 任务 | 输入 | 输出 | 备注 |
|---|---|---|---|---|
| **N1** | **5090 上跑 launcher smoke**（`LIMIT_DEBUG=10 RUN_NAME=v1_smoke ...`） | — | 全套 jsonl 各 ~10 行 | 验证整条管线在真实环境跑通 |
| N2 | 5090 上跑全量 5k | — | reasoning_steps / candidates / cls_labels / train / dev | ~3-4 小时，~¥200-250 |
| N3 | 自检：`gold_recall_top_k` ≥70%、`pos_ratio` 30-50%、`avg_k_kept` ≈ 10 | meta json | — | 不达标先调 prompt/参数 |
| N4 | 写训练模块（dataset.py / model.py / loss.py / train.py + configs YAML） | train.jsonl | LoRA ckpt | 见 [repo_overview.md §0bis.5](./repo_overview.md) |
| N5 | 写 e2e 推理接入 (run_sapr_r_v1_e2e.py) | LoRA ckpt | EM/F1 | 在 v0 脚本基础上接 trained reranker + clean_thought 适配层 |
| N6 | 写评估 (eval_v1_vs_baseline.py + audit_clean_thought_coverage.py) | — | 答辩用表格 | — |
| N7 | 更新 README | — | — | 项目级 README，对外可读 |

---

## 8. 红线（不要踩）

1. **不要** 用 [gate0/data/reasonrag_mcts/](../gate0/data/reasonrag_mcts/) 下的 reward_data*.json 当 v1 训练数据来源——那是 Llama-70B-int4 复现版，sibling 重复 98.4%，已被多次确认不能用。
2. **不要** 用 HotpotQA 自带 `supporting_facts` 做 cls label——语料与 BGE 检索不同源，启发式不可信；只能作为 step3 retrieval recall 的弱自检指标。
3. **不要** 在 generator 链路 / trajectory log / reward 计算里调 `clean_thought()`——那会破坏 generator 训练分布。`clean_thought()` 仅在 reranker 入口出现一次。
4. **不要** 在 step4 让 LLM 自己抽 subject_entity——已采用修法 A（step2 直接输出该字段）。
5. **不要** 在脚本里写绝对路径——仓内用 `Path(__file__).resolve().parents[N]` 派生，仓外走 `_RequiredPath` + env_*.sh。

---

## 9. 单点信源指引

| 想知道什么 | 看哪里 |
|---|---|
| 项目主线 / v0 / v1 高层方案 | [repo_overview.md](./repo_overview.md) §0–§0bis |
| v1 数据构造管线全貌 / 跑法 / 下一步（本档） | **本文档** |
| 跨机环境配置 | [setup.md](./setup.md) + [server_env.md](./server_env.md) + `config/env_*.sh` |
| 编码规范 | [coding_standard.md](./coding_standard.md) |
| 历史决策（v4 暂停 / Gate 0 结果等） | [history.md](./history.md) |
| 实验记录 | [experiment_tracker.md](./experiment_tracker.md) |
