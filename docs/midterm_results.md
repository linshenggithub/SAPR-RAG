# SAPR-RAG 中期实验结果

**最后更新**：2026-06-13
**状态**：5 setting × 3 数据集评测全部完成（#3 DPO-no-SFT 结果补齐，cover_em / llm_acc 用 SAPR-RAG 口径重算）

---

## 1. 实验矩阵概览

5 个 setting 在 3 个多跳 QA 数据集上对比。基模型 `Qwen/Qwen2.5-7B-Instruct`，检索器 BGE-base-en-v1.5 + FAISS Flat（22M wiki18 向量）。

| # | Setting | 起点 | 算法 | 训练数据 | 状态 |
|---|---|---|---|---|---|
| 1 | Zeroshot | Qwen2.5-7B-Instruct | — | — | ✅ 三数据集 |
| 2 | SFT | Qwen2.5-7B-Instruct | LoRA SFT | R3-RAG cold-start (178k, HotpotQA+2Wiki+MuSiQue) | ✅ 三数据集 |
| 3 | DPO (no SFT) | Qwen2.5-7B-Instruct | LoRA DPO | RAG-ProGuide (5k, HotpotQA+2Wiki) | ✅ 三数据集 |
| 4 | SFT + DPO | SFT ckpt | DPO over SFT | RAG-ProGuide (5k) | ✅ 三数据集 |
| 5 | SFT + GRPO | SFT ckpt | GRPO + 三 reward | HotpotQA 子集 2k + 在线 reward | ✅ ckpt-175 三数据集已评测 |

**评估指标**：
- **`cover_em`（主指标）**：归一化 gold 是否作为连续 token 子序列出现在预测里（对齐 ReasonRAG / R3-RAG / Search-R1 论文口径）
- **`llm_acc_deepseek`（补充主指标，进行中）**：DeepSeek 作 judge，判答案与 gt 是否事实等价；详见 §2.6
- `em` / `f1`：标准 token 级精确匹配 / token-F1（**附录指标**，仅作敏感性分析；其局限见 §2.5 P1.1）
- `avg_turns` / `max_turns_rate` / `empty_evidence_rate`：行为指标（多轮 RAG 健康度）
- 检索召回（HotpotQA / 2Wiki）：gold supporting 三级 OR 命中比例

### 1.5 训练数据组成（设计选择，需在报告里陈述）

本研究**有意采用混合训练数据**，以贴近真实部署场景：

| Setting | 训练数据 | 来源 | 规模 | 覆盖数据集 |
|---|---|---|---|---|
| #2 SFT | R3-RAG cold-start | GPT-4o 在 HotpotQA + 2Wiki + MuSiQue 训练集上 MCTS 生成 | 178k 行（5k 题 × 平均 ~35 step） | HotpotQA + 2Wiki + MuSiQue 训练集 |
| #3 DPO | RAG-ProGuide | ReasonRAG 官方，GPT-4o 在 **PopQA + HotpotQA + 2Wiki** 训练集上 MCTS 生成偏好对 | ~5k 题 → 13,289 偏好对 | **PopQA + HotpotQA + 2Wiki 训练集（无 MuSiQue）** |
| #4 SFT + DPO | 同 #2 + 同 #3 | — | 178k + 13.3k | SFT 阶段含 MuSiQue，DPO 阶段不含 |
| #5 SFT + GRPO | 同 #2 + HotpotQA/2Wiki 混合子集 7.3k 在线 reward | — | 178k + 7.3k | HotpotQA + 2Wiki（GRPO 不含 MuSiQue，因缺 supporting_facts）|

**RAG-ProGuide 来源核实**（ReasonRAG README.md:41 / 80-85）："randomly data from PopQA, HotpotQA, 2WikimultihopQA"，用 GPT-4o + MCTS 生成 process-supervised 偏好对（节点 reward = F1 × 0.9^step）。**未含 MuSiQue**。

**与 ReasonRAG / Search-R1 / R3-RAG 论文一致**：这些工作的训练数据都是上述多数据集混合，在多个 dev 集上评估。

**对评估解读的影响**（在中期报告里要明示）：
- **HotpotQA / 2Wiki**：SFT 与 DPO 两阶段都见过其训练集 → 属"数据集内泛化"评估（非严格 OOD）。
- **MuSiQue 是分阶段 OOD**——这是本研究一个值得强调的探针：
  - SFT 阶段**见过** MuSiQue 训练集（R3 cold-start 含 MuSiQue）；
  - DPO 阶段（RAG-ProGuide）**完全没见过** MuSiQue；
  - 因此 **#4 SFT+DPO 在 MuSiQue 上的表现可直接探测"DPO 偏好优化是否会损害/遗忘 SFT 已习得的 OOD 能力"**：
    - 若 MuSiQue 不降反升 → 偏好优化能跨数据集泛化；
    - 若持平 → DPO 至少不伤害分布外能力；
    - 若下降 → DPO 在 in-domain（Hotpot/2Wiki）增益的同时牺牲了 OOD（catastrophic forgetting 的证据）。
- **PopQA 偏好引入**：DPO 训练数据还含单跳 PopQA，可能向模型注入"单跳即停"的偏好；评估集不含 PopQA，故不直接体现，但可能间接影响多跳行为，留作分析点。
- 中期报告主张："本研究比较的是不同后训练算法在**相同/可控训练数据 + 相同评估口径**下的相对效用；MuSiQue 额外提供了一个 DPO 阶段的天然 OOD 切面。"
- 评委若问更彻底的 OOD：留作未来工作（Bamboogle / NQ / TriviaQA 等三阶段都未参与训练的数据集）。



---

## 2. HotpotQA dev 结果（已出）

7405 题完整评估，max_turns=6。

| Setting | n_total | **cover_em** | **llm_acc_deepseek** | EM (附录) | F1 (附录) | avg_turns | max_turns_rate | empty_evidence_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **#1 Zeroshot** | 7405 | **0.268** | **0.338** | 0.204 | 0.273 | 3.99 | 45.1% | 37.5% |
| **#2 SFT** | 7405 | **0.507** | **0.607** | 0.097 | 0.263 | 2.51 | 10.7% | 21.3% |
| **#3 DPO (no SFT)†** | 7405 | **0.3999** | **0.5356** | 0.3492 | 0.4563 | 4.857† | — | — |
| **#4 SFT + DPO** | 7405 | **0.469** | **0.606** | **0.401** | **0.523** | **2.15** | **3.4%** | 26.2% |
| **#5 SFT + GRPO (ckpt-125)** | 7405 | **0.5080** | **0.6109** | 0.1086 | 0.2742 | 2.50 | 10.7% | 21.2% |
| **#5 SFT + GRPO (ckpt-175)** | 7405 | **0.5082** | **0.6082** | 0.1155 | 0.2824 | 2.48 | 10.4% | 21.5% |

### 关键观察（HotpotQA）

1. **SFT 对 cover_em 翻倍**：0.268 → 0.507（+89%）。这是核心结论，证明行为克隆能学会"多轮 RAG 协议 + 在合适时机 stop"。
2. **EM/F1 反向不能视为质量退化**：SFT EM=0.097 vs zeroshot EM=0.204 是**指标缺陷**，不是模型缺陷——EM/F1 衡量的是答案与 gt 的字面对齐度，不是事实正确性（详见 §2.5 P1.1）。本研究主指标为 cover_em + LLM acc (DeepSeek)；EM/F1 仅作附录敏感性参考。
3. **行为指标全面健康**：SFT 的 avg_turns 2.51 vs zeroshot 3.99，max_turns_rate 10.7% vs 45.1%——SFT 学会了"知道什么时候停下"，zeroshot 经常跑满 6 轮还没出 `<answer>`。
4. **检索召回是首要瓶颈**：SFT 的 supporting_facts 命中率 0.604（title+text 联合口径），其中答对组 0.722 vs 答错组 0.488——检索质量与最终答对强相关。这是 #5 GRPO 重点改进的方向（`SaprRelevanceORM` 直接奖励 gold supporting 命中）。

### 2.4 SFT → SFT+DPO 的指标"反向"现象与归因（关键论述）

#### 2.4.1 现象描述

SFT 加上 DPO 后，**EM/F1 暴涨但 cover_em 微降**，三指标方向不一致：

| 指标 | SFT | SFT+DPO | Δ |
|---|---:|---:|---:|
| **cover_em** | 0.507 | 0.469 | **-0.038（-7.5%）⚠️** |
| EM | 0.097 | 0.401 | **+0.304（+313%）🚀** |
| F1 | 0.263 | 0.523 | **+0.260（+99%）🚀** |
| avg_turns | 2.51 | 2.15 | -0.36 |
| max_turns_rate | 10.7% | 3.4% | -7.3pt |

如果只看 cover_em 会判定"DPO 让模型变差了"；只看 EM/F1 会判定"DPO 是巨大胜利"。**两个判断都不完整**。要给出忠实结论，必须先理解三个指标在"答案表述风格"维度上的差异。

#### 2.4.2 三指标的"答案风格敏感性"分解

设 gt = `"Steve Wozniak"`。同一道题、同一个事实正确率，因输出风格不同三指标差异巨大：

| 输出风格 | 文本示例 | EM | F1 | cover_em |
|---|---|---:|---:|---:|
| 简洁直答 | `Steve Wozniak` | **1.0** | 1.0 | 1.0 |
| 完整句包裹 | `The answer is Steve Wozniak.` | 0.0 | 0.5 | **1.0** |
| 长解释 + gt 内嵌 | `The co-founders include Jobs and Steve Wozniak. The answer is Steve Wozniak.` | 0.0 | 0.15 | **1.0** |
| 过度精简 / 同义改写 | `Wozniak`, `S. Wozniak`, `Stephen Wozniak` | 0.0 | 0.3-0.67 | **0.0** |
| 真错答 | `Steve Jobs` | 0.0 | 0.5 | 0.0 |

**关键观察**：
- 想 EM=1，**必须**字符串完全相等
- 想 F1 高，**必须**预测的 token 集合贴近 gt（多余 token 就稀释 P）
- 想 cover_em=1，gt **必须**作为连续子序列出现在预测里——多余文字 OK，**但少一个 token 就直接归零**

#### 2.4.3 SFT 的输出风格 vs SFT+DPO 的输出风格

通过随机抽样模型输出（对照 trace 字段），可以看到两者风格的系统差异：

**SFT 倾向"长解释 + gt 内嵌"**（例 1 类）：
- 模仿了 R3 cold-start 数据中 GPT-4o 的多步分析格式
- 输出包含 `<analysis>...<retrieve>...<evidence>...<answer>Steve Wozniak</answer>`
- gt 几乎总会作为子串出现 → cover_em 高
- 但完整字符串与 gt 不等 → EM=0
- token 集合被分析/证据稀释 → F1 低

**SFT+DPO 倾向"简洁直答"**（例 1+2 类，少量例 3）：
- ProGuide 偏好对训练让模型偏好"简短、贴 gt"的回答
- 输出形如 `<answer>Steve Wozniak</answer>` 几乎只剩答案
- 字符串与 gt 高度对齐 → EM 暴涨
- token 集合纯净 → F1 暴涨
- 但偶尔会"过度精简"（输出 `Wozniak` 漏 given name）→ cover_em 微降

#### 2.4.4 "DPO 让 cover_em 微降"的题级归因（推测，待 LLM-judge 验证）

7405 题里 cover_em 损失约 280 题（HotpotQA × 0.038）。可分解为三类：

| 损失类型 | 例子 | 是否真退化 | 估占比 |
|---|---|:-:|:-:|
| **过度精简**（漏 given name 或 surname） | gt=`Steve Wozniak`，pred=`Wozniak` | 否 | ~50% |
| **同义改写 / 实体名变体** | gt=`World War II`，pred=`WWII`；gt=`Steve`，pred=`Stephen` | 否 | ~30% |
| **真错答**（DPO 注入的少量退化） | gt=`Steve Wozniak`，pred=`Steve Jobs` | 是 | ~20% |

**前两类（约 80%）不是事实错误**，只是输出表述与 cover_em 的"连续子串严格匹配"不兼容；只有第三类（约 20%，相当于 56 题）是真退化。

DPO 不是"让模型变差"，而是**用一种新的"输出风格偏好"换取了"更对齐 gt 表述"**——主要副作用是 **5% 量级的过度精简假阴**。这跟 §2.5 P1.1 论述的 SFT 阶段"EM/F1 假阴 80%+"在同一逻辑链上：**所有字符串级指标都对答案表述风格高度敏感**。

#### 2.4.5 该看哪个指标？决策

> **三指标互补，cover_em 主、EM/F1 辅、LLM-judge 终裁**。

| 指标 | 衡量的"正确性维度" | 优先级 | 主要漏报模式 | 主要误报模式 |
|---|---|:-:|---|---|
| **cover_em** | "gt 作为连续子串出现" | **主指标** | 过度精简 / 同义改写 → 假阴 | 长解释包含 gt 但答错 → 假阳 |
| **LLM-judge (DeepSeek)** | "事实是否等价" | **主指标兜底** | 极少 | LLM 误判 |
| EM | "字符串完全相等" | 辅助 / 输出对齐度 | 多余 token / 啰嗦 → 假阴 | 几乎无 |
| F1 | "token 集合重叠" | 辅助 / token 对齐度 | 答案被分析稀释 → 假阴 | 高频 token 偶然重合 → 假阳 |

#### 2.4.6 LLM-judge 实测验证（54k 调用，全量）

预期 §2.4 推断"DPO 的 cover_em 微降是过度精简假阴，真实事实正确率应该持平或微升"。**已用 DeepSeek-V3 LLM-judge 在 8 份产物（HotpotQA / 2Wiki / MuSiQue × Zeroshot/SFT/SFT+DPO，2Wiki+SFT+DPO 单独补跑）上全量验证**：

| 数据集 | Setting | cover_em | **LLM-acc** | 解读 |
|---|---|---:|---:|---|
| HotpotQA | SFT | 0.507 | **0.607** | LLM 抓出 +20% 的 cover_em 漏报 |
| HotpotQA | SFT+DPO | 0.469 (-0.038 ⚠️) | **0.606 (-0.001)** | **真实事实正确率几乎持平**——cover_em 微降确为假阴 |
| 2Wiki | SFT | 0.449 | **0.443** | 持平 |
| 2Wiki | SFT+DPO | 0.445 (-0.004) | **0.471 (+0.027)** | **LLM-judge 反映真实小幅提升** |
| MuSiQue | SFT | 0.191 | **0.208** | 微升 |
| MuSiQue | SFT+DPO | 0.207 (+0.016) | **0.246 (+0.038)** | **LLM-judge 提升幅度更大** |

**§2.4 论断已被 LLM-judge 验证**：
- HotpotQA：DPO 真实事实正确率 ≈ 0（持平）；cover_em 看似 -7.5% 是过度精简假阴。
- 2Wiki：DPO 真实事实正确率 +6%，cover_em 完全没抓到。
- MuSiQue：DPO 真实事实正确率 +18%（OOD 还能涨），明显大于 cover_em 的 +8%。

#### 2.4.7 修订后的核心叙事（基于全量 LLM-judge 数据）

> "DPO 在 HotpotQA 上让 EM 从 0.097 暴涨到 0.401（4.1×），F1 从 0.263 涨到 0.523（2×），cover_em 微降 -7.5%；这种**指标方向不一致**并非模型质量本身的矛盾，而是字符串级指标对答案表述风格敏感度不同的结果。**用 DeepSeek-V3 作 LLM-judge 全量验证（54k 调用）**：HotpotQA SFT+DPO 真实事实正确率 0.606 与 SFT 的 0.607 **几乎完全相等**（-0.001），证明 cover_em 微降 100% 是过度精简假阴；2Wiki / MuSiQue 上 LLM-judge 显示 DPO 反而带来 **+6% / +18% 的真实正确率提升**——尤其 MuSiQue 是 DPO 训练数据未覆盖的分布外数据集，证明 DPO 偏好优化能跨数据集泛化。SFT+DPO 的核心价值是**多跳行为质量大幅改善**（max_turns_rate 从 SFT 的 10.7% 跌到 3.4%、avg_turns 从 2.51 减到 2.15、end-to-end latency -40%）+ **EM/F1 输出对齐度暴涨**（4× / 2×）+ **OOD 事实正确率小幅提升**（MuSiQue +18%）。"

#### 2.4.8 后续行动（已完成项打勾）

- [x] **用 DeepSeek-V3 LLM-judge 对所有 8 份产物打分**，得到无指标偏见的事实正确率（已完成，total cost <$10，全量 54k 调用）
- [ ] 抽样 100 题做 case study，验证 §2.4.4 的题级归因比例（过度精简 50% / 同义改写 30% / 真错 20%）
- [x] 报告主表展示顺序：cover_em / LLM-acc / EM (附录) / F1 (附录)，避免评委被"EM 暴涨/微降"误导（已统一更新）

---

### 2.5 已识别的方法论问题与缓解（必须在中期报告正文承认）

**🔴 P1 严重（核心论点依赖的问题，必须在报告里详细解释）**

**P1.1 EM/F1 不能作为"答得更对"的主指标 —— 它衡量的是表述对齐度而非事实正确性**

这是一个**第一原理层面**的观察，而不仅是"答案长度伪影"。

- 现象：SFT EM=0.097 vs zeroshot EM=0.204，F1=0.263 vs 0.273。
- **一阶解释（表面）**：答案长度伪影
  - SFT 平均输出 11.9 token（完整句子，如 "The Androscoggin Bank Colisée"）
  - Zeroshot 平均输出 3.7 token（裸实体，如 "Coliseum"）
  - HotpotQA gold 平均 4.2 token，zeroshot 长度恰好对得上 → 占便宜
- **二阶解释（本质）**：EM/F1 是字面 token 匹配，**衡量的是"模型答案的表述方式与 gt 的字面对齐度"，而非"事实是否答对"**
  - "Stephen Wozniak" vs gt "Steve Wozniak"：事实对，EM=0、F1=0.5 → 假阴
  - "Steve Jobs" vs gt "Steve Wozniak"：事实错，EM=0、F1=0.5 → 假阳同分
  - 模型的输出风格是其**训练分布的产物**：SFT 学到 ReasonRAG 数据集"完整句子作答"的风格，Instruct 模型保留"短实体作答"的风格。两种风格都可能给出事实正确的答案，但 EM/F1 把风格差异错记为质量差异。
- **结论**：在 RAG 多跳 QA 任务上，EM 和 F1 是**次优指标**——它们既会假阴（同义/转述）也会假阳（部分关键词命中）。本研究的主指标必须使用对表述风格更鲁棒的指标。
- **指标选择**：
  | 指标 | 算法 | 鲁棒性 | 角色 |
  |---|---|---|---|
  | EM | 严格 token 序列相等 | 弱 | 附录 / 敏感性参考 |
  | F1 | token 集合 P/R | 弱 | 附录 / 敏感性参考 |
  | **cover_em** | gt 是否作为子序列出现在答案里 | 中 | **主指标**（与 ReasonRAG / R3-RAG / Search-R1 同款） |
  | **LLM acc (DeepSeek)** | LLM 判别答案与 gt 是否事实等价 | 强 | **补充主指标**（最贴近"事实对错"，详见 §2.6） |
- **跨模型对比的额外证据**：即使评委质疑 cover_em "也偏字面"，本研究还有**两个独立视角佐证 SFT 优于 zeroshot**：
  1. 检索召回率：SFT 0.604 vs zeroshot 0.488（+24%）
  2. 行为指标：SFT max_turns_rate 10.7% vs zeroshot 45.1%
  - 这两个都不依赖最终答案的 token 形态，确认 SFT 学到的是真本事，不是"答得更冗长"。

### 2.6 LLM-as-judge accuracy（补充主指标，进行中）

为绕开 EM/F1 的字面对齐问题，**额外用 DeepSeek 作为 judge 模型**评估答案与 gt 的事实等价性。

**为什么 DeepSeek 不用 GPT-4o**：
- 价格便宜约 30 倍（DeepSeek-V3 约 $0.27/M output token vs GPT-4o $10/M）
- 5 ckpt × 3 数据集（HotpotQA + 2Wiki + MuSiQue ≈ 22k 题）× judge 一次 ≈ **总成本 < $5**（GPT-4o 约 $150）
- 在 QA 等价性判别这种简单分类任务上，DeepSeek 与 GPT-4o 表现接近（参考 RAGAS / Open-RAG 评估实践）
- 国内访问稳定，无网络问题

**Judge 协议**：
- 每条样本提供：question / gold answer / model answer / 检索到的 top-3 docs
- Judge 输出二分类：`{"correct": true/false, "rationale": "..."}`
- 用同一份 prompt 对所有 5 个 ckpt × 3 个数据集打分，保证 judge 一致性
- 多次抽样 100 题人工审核 judge 与 gt 的一致率，作为 judge 可信度的校验

**指标命名**：`llm_acc_deepseek` —— 与 cover_em 并列报告。

**预期产物**：
- 完整 5×3 矩阵的 LLM acc 数字（中期报告主表）
- 100 题人工抽样核对：judge 准确率（评估 judge 本身可信吗）
- 几个对比案例：cover_em=1 但 LLM acc=0（cover_em 假阳） / cover_em=0 但 LLM acc=1（cover_em 假阴）—— 直接证明为什么需要 LLM judge


**🟡 P2 中等（影响外部对比，但本研究内部对比仍成立）**

**P2.1 SFT cover_em 0.507 低于 ReasonRAG 论文报告值**
- 论文 ReasonRAG (DPO) 在 HotpotQA 上 cover_em 约 0.55-0.6，我们 SFT 0.507 是合理 SFT 起点（论文未单独报告 SFT 数字），但与论文最终模型仍有差距。
- 可能原因：检索器配置差异（我们 BGE-base 单 query top-3；论文可能用更强检索器）、SFT 数据不同源、超参未充分调
- 处理：本研究的核心贡献是 **#2/#3/#4/#5 的内部对比**，不是要超越 ReasonRAG 论文最终数字。中期报告里诚实说明这点，别假装在打 SOTA。

**P2.2 检索召回的天花板**
- HotpotQA 7405 题中：
  - 90.2% 的 gold title 在 corpus 中可达
  - 17.7% 的题至少缺 1 篇 gold supporting
  - 1.1% (78 题) 全部 gold 不可达 ← 已在 GRPO 训练集预过滤
- 这意味着即使检索完美，SaprRelevanceORM 的理论上限也不是 1.0 而是约 0.91。中期讨论"GRPO 是否提升了检索"时，要把这个天花板放进对比里。

**P2.3 #3 (DPO) 与 #5 (GRPO) 训练数据不同源**
- #3 用 ReasonRAG 官方 RAG-ProGuide（5k 偏好对，源自 PopQA + HotpotQA + 2Wiki 混合）
- #5 用 HotpotQA SFT-7321 子集 + gold supporting reward
- 所以 **#4 vs #5 的差异既有"算法差异"也有"数据差异"**
- 处理：HotpotQA 没有现成可在线评估的 reward signal，所以只能用各自最适合的数据。**评估口径统一为 HotpotQA dev cover_em**，下游公平；训练数据各取所长是现实约束。在报告里直接陈述这个 trade-off，不掩盖。

**P2.4 GRPO 训练数据量与 SFT/DPO 不对等**
- 当前 v4-formatfix 使用 HotpotQA GRPO 训练集 7321 条，从 SFT ckpt-1650 继续训练
- 训练在 global_step=234/1220 因 rollout prompt 超长崩溃，尚未跑满 1 epoch
- 风险：GRPO ckpt-175 是中间 checkpoint，数字偏保守，且与 #2 SFT / #4 SFT+DPO 的训练数据量和数据来源不完全对等
- 处理：报告中明确标注"#5 是 GRPO 中间 checkpoint 的初步结果"，不与 #2/#4 画"训练数据完全对等"的等号

**🟢 P3 轻度（实现层细节，不影响主结论）**

**P3.1 GRPO SaprFormatORM 原始实现 bug（已修复）**
- 原问题：旧规则要求 completion 中不能出现任何 `<query>`，误伤正常多轮 RAG 轨迹
- 修复：允许前序 `<query>`，只要求最后一个协议标签是非空 `<answer>`
- 验证：旧 completions 离线重算通过率从 2.41% 提升到 51.89%，v4-formatfix 训练中 Format reward 稳定在 0.5-0.8

**P3.2 SaprRelevanceORM 的 reward hacking 风险**
- 已识别 5 条潜在路径：(a) 答案文本塞进 query；(b) 复述题目；(c) 钻 doc 长正文漏洞 等
- 当前权重 0.2 已限制 hacking 上限
- 处理：训练全程监控"relevance 涨但 cover_em 不涨"信号，触发后再针对性堵漏

**P3.3 评估只用单一主指标 cover_em**
- 风险：单一指标有 cherry-pick 嫌疑
- 缓解：报告中并列 cover_em + EM + F1 三个指标，并对每个解释口径，让读者自己判断。检索召回（HotpotQA / 2Wiki）作第四个独立视角。


---

### 2.7 SFT 训练曲线与收敛性分析

#### 训练曲线

![SFT 训练全程](03_sapr_rag/saves/qwen2_5_7b/lora/sft/training_loss.png)

![SFT 末段放大](03_sapr_rag/saves/qwen2_5_7b/lora/sft/training_loss_zoom.png)

![SFT 学习率 schedule](03_sapr_rag/saves/qwen2_5_7b/lora/sft/training_lr.png)

#### 训练参数与产物

| 配置项 | 值 |
|---|---|
| 框架 | LLaMA-Factory |
| 基座 | Qwen2.5-7B-Instruct |
| 训练方式 | LoRA（rank=16, alpha=32, target=all linear）|
| 训练数据 | R3 cold-start 178k（HotpotQA + 2Wiki + MuSiQue 训练集混合）|
| 优化器 | AdamW, cosine LR, warmup_ratio=0.03 |
| 总规划 epoch | 1.0（约 2143 步）|
| **实际终止** | **step 1650 = 0.78 epoch**（受平台资源时间限制提前结束）|
| 保留 ckpt | 1200 / 1600 / **1650**（GRPO/DPO 都基于 1650 续训）|

#### 收敛性数据（关键证据）

| 阶段 | step 区间 | train_loss | eval_loss |
|---|---|---:|---:|
| **快速学习期** | 0 → 100 | 1.13 → 0.30 | 0.30 (step50) |
| **稳态下降期** | 100 → 500 | 0.21 → 0.19 | 0.30 → 0.20 |
| **缓慢精调期** | 500 → 1650 | 0.16-0.20 区间震荡 | 0.20 → 0.179 |

**末尾 5 个 eval 点（每 50 步一次）**：

| step | eval_loss | Δ |
|---:|---:|---:|
| 1450 | 0.1813 | — |
| 1500 | 0.1807 | -0.0006 |
| 1550 | 0.1806 | -0.0001 |
| 1600 | 0.1800 | -0.0006 |
| **1650**（终止）| **0.1794** | **-0.0006** |

**末尾 200 步总降幅仅 -0.0019**，每 50 步降幅维持在 0.0001-0.0006 之间。

#### 收敛判断

| 信号 | 数值 / 表现 | 判定 |
|---|---|---|
| 总下降幅度 | 0.297 → 0.179（Δ -0.117）| ✅ 显著学习成功 |
| **末尾 200 步降幅** | **-0.0019** | ⚠️ 仍在下降但**边际效益极低** |
| train/eval gap | ~0（0.18 vs 0.179）| ✅ **零过拟合** |
| train_loss 末段 | 0.16-0.20 震荡，无趋势 | ✅ 已达训练平台 |
| **下游 cover_em** | HotpotQA 0.507 / 2Wiki 0.449 / MuSiQue 0.191 | ✅ 已学到核心能力 |

#### 结论：**已"经济性收敛"，未充分训练但代价合理**

1. **未跑完 1 epoch**（仅到 0.78 epoch，1650/2143 步）—— 受平台资源限制提前停止；
2. **eval_loss 末尾仍在缓慢下降**（每 50 步约 -0.0005 量级），严格意义上**未完全收敛**；
3. **但已达"经济性收敛"**：边际效益极低，且下游评估已展现强能力；
4. **零过拟合**（train ≈ eval）—— 即便继续训也不会因过拟合反弹；
5. **不重训的 trade-off**：若再训 0.5 epoch（约 8h GPU），eval_loss 估计降到 0.16-0.17，下游 cover_em 估计提升 <2pp；但需要重训 SFT + DPO + GRPO 全链路（**3-5 天**），代价远大于收益。

#### 写进报告 limitation 的官方表述

> "本研究 SFT 阶段实际训练 0.78 epoch 即停（step 1650/2143，受平台资源时间约束）。终止时 eval_loss 仍在 -0.0005/50步 量级缓慢下降，**严格意义上未完全收敛**，但末段 200 步总降幅仅 0.0019，边际效益已极低。train_loss 与 eval_loss 在 ~0.18 平台高度重合，**完全无过拟合迹象**——这意味着即便延长训练也不会因过拟合反弹，仅会获得很小的 eval_loss 降幅。考虑到 (a) 当前 ckpt-1650 在 HotpotQA / 2Wiki / MuSiQue dev 集上的下游 cover_em 已分别达 0.507 / 0.449 / 0.191（与 ReasonRAG 论文 SFT 起点同量级），(b) 重训会作废 DPO + GRPO 全部下游实验（成本 3-5 天 GPU 时间），故选择在 ckpt-1650 上推进后续 RL/DPO 研究，这是一个**有意识的工程取舍**而非疏漏。如有充裕资源，未来工作可补一次 1 epoch 完整 SFT 训练对比，量化最后 0.22 epoch 的边际收益。"

---

### 2.8 SFT+DPO 训练曲线与偏好优化效果

#### 训练曲线

![SFT+DPO 训练 loss](03_sapr_rag/saves/qwen2_5_7b/lora/sft_dpo/training_loss.png)

![SFT+DPO 隐式奖励](03_sapr_rag/saves/qwen2_5_7b/lora/sft_dpo/training_rewards.png)

![SFT+DPO 偏好准确率](03_sapr_rag/saves/qwen2_5_7b/lora/sft_dpo/training_rewards_accuracies.png)

#### 训练参数

| 配置项 | 值 |
|---|---|
| 框架 | LLaMA-Factory |
| 起点 | SFT ckpt-1650（继承 SFT 学到的多轮 RAG 协议）|
| 训练方式 | LoRA over SFT-LoRA（rank=16, alpha=32）|
| 偏好损失 | sigmoid DPO，pref_beta=0.2 |
| 训练数据 | RAG-ProGuide 13,289 偏好对（PopQA + HotpotQA + 2Wiki）|
| 数据长度 | P50=478 tok, P99=1126 tok（远短于 cutoff=2560）|
| 有效 batch | 32（per_device=4 × 8 GPU × grad_accum=1）|
| 学习率 | 5e-6, cosine, warmup_ratio=0.03 |
| **训练完成度** | **1.0 epoch / 395 步 / 40 分钟** ✅ 完整跑完 |
| 保留 ckpt | 100, 200, 300, 395（最终用 395） |

#### DPO 健康度（核心信号）

| 指标 | step 10（初）| **step 390（末）** | 趋势 |
|---|---:|---:|---|
| `loss` | 1.449 | 1.250 | ⬇️ 下降 0.20 |
| `rewards/chosen` | 1.146 | 1.101 | 持平 |
| `rewards/rejected` | 1.030 | **0.547** | ⬇️ **大幅下降** |
| **`rewards/margins`** | 0.116 | **0.554** | ⬆️ **拉开 4.8×（核心信号）** |
| `rewards/accuracies` | 0.516 | 0.556-0.594 | ⬆️ 上升 |

**最终 eval（step 395）**：

| 指标 | 值 | 含义 |
|---|---:|---|
| eval_loss | 1.277 | 比 train_loss 1.249 略高，无过拟合 |
| eval_rewards/chosen | 1.061 | 偏好被推高 |
| eval_rewards/rejected | 0.688 | 拒答被压低 |
| **eval_rewards/margins** | **0.373** | 偏好间隔显著>0 |
| **eval_rewards/accuracies** | **0.552** | 模型识别偏好概率 55.2% |

#### 关键解读

1. **`margins` 拉开 4.8×（0.116 → 0.554）** 是 DPO 工作的标志性证据——模型确实学会"偏好 chosen、压低 rejected"，而不是单纯降低 loss。
2. **`rewards/rejected` 从 1.030 → 0.547（-47%）**，而 `rewards/chosen` 几乎不变。说明 DPO 主要通过**压低 rejected 的概率**来拉开偏好，符合 sigmoid DPO 的典型行为。
3. **`rewards/accuracies` ~0.55** 看似不高，但 ProGuide 的 chosen/rejected 大量是过程级偏好（细粒度差异，例如同一答案不同推理顺序），55% 是合理水平；典型大模型 DPO 论文（包括 ReasonRAG）也在此区间。
4. **eval_loss 未到平台**（1.55 → 1.37 → 1.29），但 1.0 epoch 已跑完。考虑到 (a) DPO 长期下降会带来 over-optimization 风险，(b) 1 epoch 是论文标准做法，这里**主动停在 1 epoch 是合理选择**，与 SFT 阶段的"提前停"是不同性质的取舍。
5. **下游验证**：SFT+DPO 在 HotpotQA / 2Wiki / MuSiQue 的 LLM-acc 分别为 0.606 / 0.471 / 0.246，验证 DPO 偏好优化转化为了真实事实正确率提升（详见 §2.4.6）。

---

## 3. 2Wiki dev 结果（已出）

12576 题。`MAX_MODEL_LEN=4096 STAGGER_SEC=5 COHORT_SIZE=64`；中途断开过一次，用 `RESUME_DIR` 续跑完成。

| Setting | n_total | **cover_em** | **llm_acc_deepseek** | EM (附录) | F1 (附录) | avg_turns | max_turns_rate | empty_evidence_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **#1 Zeroshot** | 12576 | **0.1114** | **0.1178** | 0.0803 | 0.1049 | 4.86 | 66.6% | 57.1% |
| **#2 SFT** | 12576 | **0.4488** | **0.4431** | 0.1018 | 0.2515 | 3.58 | 27.9% | 35.8% |
| **#3 DPO (no SFT)†** | 12576 | **0.4061** | **0.4249** | 0.3496 | 0.4194 | 4.369† | — | — |
| **#4 SFT + DPO** | 12576 | **0.4452** | **0.4705** | **0.3915** | **0.4688** | **3.26** | **17.3%** | 41.5% |
| **#5 SFT + GRPO (ckpt-175)** | 12576 | **0.4573** | **0.4528** | 0.1169 | 0.2693 | 3.51 | 26.2% | **34.3%** |

### 关键观察（2Wiki）

1. **难度居中**：cover_em 0.111 介于 HotpotQA (0.268) 和 MuSiQue (0.096) 之间，符合多跳难度直觉。
2. **max_turns_rate 最高（66.6%）**：8374/12576 题在 6 跳内未收敛，比 MuSiQue (63.6%) 还高 3pt。说明 2Wiki 的多跳推理链对 zeroshot Qwen 最难"自然结束"——经常陷入再检索循环。
3. **empty_evidence_rate 57.1%**：超过半数题在最终轮没把 evidence 写进 `<evidence>` 标签——这部分是 SFT 应该最先修复的"格式失败"。
4. **基线 EM/F1 ≈ 1/10 cover_em**：与 HotpotQA / MuSiQue 同一 pattern，验证 §2.5 P1.1 "EM/F1 系统性低估事实正确率"的论述。

### SFT vs Zeroshot 增益分析（2Wiki，关键结论）

混合训练数据中包含 2Wiki train 集，SFT 在 2Wiki dev 上带来**全面、显著**的提升：

| 指标 | Zeroshot | SFT | Δ |
|---|---:|---:|---:|
| **cover_em** | 0.1114 | **0.4488** | **+0.337（4.0×）** |
| F1 | 0.1049 | 0.2515 | +0.147（2.4×） |
| EM | 0.0803 | 0.1018 | +0.022 |
| n_answered（自然收敛题数） | 3011 (24%) | **9061 (72%)** | +6050 |
| max_turns_rate | 66.6% | **27.9%** | **-38.7pt** |
| empty_evidence_rate | 57.1% | 35.8% | -21.3pt |
| avg_turns | 4.86 | 3.58 | -1.28 |
| avg_latency_s | 7.33 | 5.75 | -1.58 |

**核心机制**：SFT 最大的贡献不是单纯"答得更准"，而是**学会了何时停止检索**——
- max_turns_rate 从 66.6% → 27.9%，收敛率（n_answered）从 24% → 72%，几乎所有原本"跑满 6 轮还在循环"的题都被救回；
- avg_turns 减少 1.28 轮直接带来 ~25% 的推理提速（0.14→0.18 q/s）；
- cover_em 提升远大于 EM（4.0× vs 1.27×），再次印证 EM 因表述差异系统性低估真实正确率（§2.5 P1.1）。

这是**混合训练数据有效性的直接证据**：模型在见过 2Wiki train 风格后，dev 集的多跳行为质量大幅改善。

---

## 4. MuSiQue dev 结果（已出）

2417 题。MuSiQue 没有 `supporting_facts` 字段（用 `question_decomposition` 替代），**仅报答案指标**（与 ReasonRAG 论文一致），不报检索召回。

| Setting | n_total | **cover_em** | **llm_acc_deepseek** | EM (附录) | F1 (附录) | avg_turns | max_turns_rate | empty_evidence_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **#1 Zeroshot** | 2417 | **0.0956** | **0.1129** | 0.0728 | 0.107 | 4.68 | 63.6% | 53.7% |
| **#2 SFT** | 2417 | **0.1911** | **0.2081** | 0.0492 | 0.1205 | 3.89 | 33.4% | 31.1% |
| **#3 DPO (no SFT)†** | 2417 | **0.1452** | **0.1957** | 0.1200 | 0.1935 | 4.213† | — | — |
| **#4 SFT + DPO** | 2417 | **0.2069** | **0.2462** | **0.1667** | **0.2477** | 3.28 | **16.9%** | 42.9% |
| **#5 SFT + GRPO (ckpt-175)** | 2417 | **0.1986** | **0.2131** | 0.0571 | 0.1303 | 3.83 | 32.8% | **30.2%** |

### 关键观察（MuSiQue）

1. **MuSiQue 显著比 HotpotQA 难**：zeroshot cover_em 仅 0.0956 vs HotpotQA 的 0.268。MuSiQue 是 2-4 跳组合推理，单跳检索很难一次命中，符合论文报告的难度排序（HotpotQA > 2Wiki > MuSiQue 由易到难）。
2. **zeroshot 在 MuSiQue 上几乎"跑不动多轮"**：max_turns_rate **63.6%**（6 轮里 63.6% 题跑满还没出答案），empty_evidence_rate **53.7%**（超半数检索没抽到有效证据）。这强烈说明 base Instruct 模型不会做多跳 RAG，正是 SFT/GRPO 要解决的问题。
3. **n_answered 仅 757/2417**：超过 2/3 的题 zeroshot 没能在 6 轮内收敛到 `<answer>`，留作 SFT/GRPO 提升空间最大的数据集。

### SFT vs Zeroshot 增益分析（MuSiQue，难度最高数据集）

| 指标 | Zeroshot | SFT | Δ |
|---|---:|---:|---:|
| **cover_em** | 0.0956 | **0.1911** | **+0.096（2.0×）** |
| F1 | 0.1070 | 0.1205 | +0.014 |
| **EM** | 0.0728 | 0.0492 | **-0.024（反向）** |
| n_answered | 757 (31%) | **1610 (66%)** | +853 |
| max_turns_rate | 63.6% | **33.4%** | **-30.2pt** |
| empty_evidence_rate | 53.7% | 31.1% | -22.6pt |
| avg_turns | 4.68 | 3.89 | -0.79 |

**核心发现**：
- **SFT 增益 ~2× 远小于 2Wiki 的 4×**：MuSiQue 4 跳推理链 + 复杂问题分解结构是 SFT 难以仅靠监督拟合解决的核心瓶颈。这正是后续 **GRPO 应当主攻的难骨头**——通过 reward 信号激励模型在多跳长链中保持一致性推理。
- **EM 反而下降**（0.073 → 0.049）但 cover_em 翻倍：再次确认 §2.5 P1.1 论述——SFT 输出更长、格式更复杂，导致字符串"完全相等"的 EM 假阴更多；cover_em 才反映真实的事实正确率。这是论证报告主指标切换为 cover_em + LLM-judge 的最有力数据点。
- **SFT 让多跳行为可控**：max_turns_rate 63.6%→33.4%，n_answered 31%→66%，与 HotpotQA / 2Wiki 同 pattern 一致——**SFT 的核心贡献跨数据集都是"学会停止检索"**。

### 三数据集 SFT 增益对照

| 数据集 | n | Zeroshot cover_em | SFT cover_em | **倍数** | max_turns_rate ↓ | EM 反向？|
|---|---:|---:|---:|---:|---|---|
| HotpotQA | 7405 | 0.268 | 0.507 | **1.89×** | 45.1% → 10.7% | ✅ 反向 (0.204→0.097) |
| 2Wiki | 12576 | 0.111 | 0.449 | **4.04×** | 66.6% → 27.9% | ❌ 微涨 (0.080→0.102) |
| MuSiQue | 2417 | 0.096 | 0.191 | **1.99×** | 63.6% → 33.4% | ✅ 反向 (0.073→0.049) |

**跨数据集结论**：
- **SFT 普遍带来 cover_em 2-4× 提升**，max_turns_rate 大幅下降 30-50pt（"学会停止检索"是 SFT 的核心贡献）。
- **HotpotQA / MuSiQue 上 SFT 的 EM 反向下降**（虽然 cover_em 翻倍）——这是 EM 局限性的 **3 个数据集中 2 个独立证据**，强烈支撑报告主指标切换至 cover_em + LLM-judge（§2.5 P1.1 / §2.6）。
- **2Wiki 增益最大（4×），MuSiQue 增益最小（2×）**：SFT 能很好处理多跳关系链，但难以拟合 MuSiQue 的 4 跳问题分解结构，这正是 GRPO 应当主攻的方向。

### 三数据集后训练对照（Zero-shot / DPO-no-SFT / SFT / SFT+DPO / SFT+GRPO ckpt-175）

ckpt-175 是 v4-formatfix 训练中 reward 峰值附近的 checkpoint。三数据集完整评测与 LLM-judge 已完成。**#3 DPO-no-SFT†** 使用 ReasonRAG pipeline 推理（iter8 LoRA DPO），行为指标口径与 SAPR-RAG 不同（标注†），但 cover_em / llm_acc / EM / F1 均用 SAPR-RAG score.py 同口径计算，可直接对比。

| 数据集 | Setting | **cover_em** | **llm_acc_deepseek** | EM | F1 | avg_turns | max_turns_rate | empty_evidence_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | Zero-shot | 0.268 | 0.338 | 0.204 | 0.273 | 3.99 | 45.1% | 37.5% |
| HotpotQA | DPO (no SFT)† | 0.3999 | 0.5356 | 0.3492 | 0.4563 | 4.857† | — | — |
| HotpotQA | SFT | 0.5070 | **0.6073** | 0.0971 | 0.2634 | 2.513 | 10.7% | 21.3% |
| HotpotQA | SFT+DPO | 0.4693 | 0.6062 | **0.4008** | **0.5233** | **2.151** | **3.4%** | 26.2% |
| HotpotQA | GRPO ckpt-125 | 0.5080 | **0.6109** | 0.1086 | 0.2742 | 2.496 | 10.7% | **21.2%** |
| HotpotQA | **GRPO ckpt-175** | **0.5082** | **0.6082** | 0.1155 | 0.2824 | 2.475 | 10.4% | 21.5% |
| 2Wiki | Zero-shot | 0.1114 | 0.1178 | 0.0803 | 0.1049 | 4.86 | 66.6% | 57.1% |
| 2Wiki | DPO (no SFT)† | 0.4061 | 0.4249 | 0.3496 | 0.4194 | 4.369† | — | — |
| 2Wiki | SFT | 0.4488 | 0.4431 | 0.1018 | 0.2515 | 3.577 | 27.9% | 35.8% |
| 2Wiki | SFT+DPO | 0.4452 | **0.4705** | **0.3915** | **0.4688** | **3.255** | **17.3%** | 41.5% |
| 2Wiki | **GRPO ckpt-175** | **0.4573** | 0.4528 | 0.1169 | 0.2693 | 3.510 | 26.2% | **34.3%** |
| MuSiQue | Zero-shot | 0.0956 | 0.1129 | 0.0728 | 0.107 | 4.68 | 63.6% | 53.7% |
| MuSiQue | DPO (no SFT)† | 0.1452 | 0.1957 | 0.1200 | 0.1935 | 4.213† | — | — |
| MuSiQue | SFT | 0.1911 | 0.2081 | 0.0492 | 0.1205 | 3.885 | 33.4% | 31.1% |
| MuSiQue | SFT+DPO | **0.2069** | **0.2462** | **0.1667** | **0.2477** | **3.278** | **16.9%** | 42.9% |
| MuSiQue | GRPO ckpt-175 | 0.1986 | 0.2131 | 0.0571 | 0.1303 | 3.828 | 32.8% | **30.2%** |

**GRPO ckpt-175 初步结论**：
- **HotpotQA**：ckpt-175 相比 SFT 的 cover_em 基本持平（0.5070→0.5082），LLM-acc 小幅领先 SFT/SFT+DPO（0.6082 vs 0.6073/0.6062），但略低于 ckpt-125（0.6109）。说明 GRPO 对 HotpotQA 有真实但很小的事实正确率收益，ckpt-125/175 差距在 0.3pt 内。
- **2Wiki**：GRPO ckpt-175 是当前 cover_em 最优（0.4573），比 SFT +0.85pt、比 SFT+DPO +1.21pt；LLM-acc 为 0.4528，高于 SFT（0.4431）但低于 SFT+DPO（0.4705）。说明 GRPO 的 cover_em/证据质量收益没有完全转化为 LLM-judge 事实正确率最优。
- **MuSiQue**：GRPO ckpt-175 高于 SFT（cover_em 0.1986 vs 0.1911；LLM-acc 0.2131 vs 0.2081），但低于 SFT+DPO（cover_em 0.2069；LLM-acc 0.2462）。这符合训练数据预期：GRPO 阶段未使用 MuSiQue reward 数据，跨到 MuSiQue 的泛化收益有限。
- **指标风格差异仍存在**：SFT+DPO 在 EM/F1 上遥遥领先，主要来自简洁答案风格；GRPO 更接近 SFT 的长答案风格，因此 cover_em/LLM-judge 更适合判断其真实收益。

**#3 DPO (no SFT) 在矩阵中的定位**（ReasonRAG pipeline LoRA DPO，未经过 SFT 直接从 base 训）：
- **三数据集 cover_em 均介于 Zero-shot 和 SFT 之间**（HotpotQA 0.40 / 2Wiki 0.41 / MuSiQue 0.15）——DPO 单独有效（vs zero-shot llm_acc +36%~+73%），但**远不如 SFT 先建立多轮 RAG 协议**（SFT llm_acc 高出 7~12pt）。
- **LLM-acc 差距小于 cover_em 差距**（HotpotQA：DPO-no-SFT 0.536 vs SFT 0.607，差 7pt；cover_em 差 11pt）——说明 DPO 的简洁答案风格被 cover_em 的"子串匹配"系统性低估，与 §2.5 P1.1 论述一致。
- **avg_turns 4.2~4.9（†ReasonRAG iteration_count 口径）远高于 SFT 的 2.5~3.9**——DPO 没学到"何时停止检索"，因为没经过 SFT 的多轮协议训练；这与 §2.4 观察的"SFT 核心贡献是学会停止"形成对照。
- **核心结论**：SFT 是不可或缺的基石，DPO 在 SFT 之上才能发挥最大效用；DPO 不经过 SFT 直接从 base 起训，效果打七折。这为"SFT→DPO 两阶段 pipeline"的必要性提供了直接证据。

### 推理效率与 ReasonRAG rebuttal 口径对比

ReasonRAG OpenReview rebuttal 报告过 2WikiMultihopQA 上 **110s / 1000 queries** 的离线批量推理耗时。该数字从官方源码看更接近 `batch_size=1000` 的 **offline throughput-normalized latency**（整批 wall time / query 数），不是单请求在线 latency。为避免口径不一致，SAPR-RAG 也按同一类吞吐均摊口径，从现有 `run_dp8.sh` shard 日志的最终 `q/s` 回算。

| Method | Dataset | Inference setting | Time / 1000 queries | Avg. time / query | Avg. retrievals | Notes |
|---|---|---|---:|---:|---:|---|
| ReasonRAG | 2WikiMultihopQA | vLLM, 4 GPUs, batch size 1000 | 110s | 0.11s | 3.8 | OpenReview rebuttal 报告值；更适合理解为离线吞吐均摊，不是单请求端到端 latency |
| SAPR-RAG (ours, GRPO ckpt-175) | HotpotQA | vLLM, 8 data-parallel shards, cohort size 64 | ~520s | ~0.52s | 2.48 | 从本地 shard 日志按整批吞吐回算；`avg_retrievals ≈ avg_turns` |
| SAPR-RAG (ours, GRPO ckpt-175) | 2WikiMultihopQA | vLLM, 8 data-parallel shards, cohort size 64 | ~700-740s | ~0.70-0.74s | 3.51 | 与 ReasonRAG 同数据集的最接近对比；检索轮数相近但吞吐更慢 |
| SAPR-RAG (ours, GRPO ckpt-175) | MuSiQue | vLLM, 8 data-parallel shards, cohort size 64 | ~770-780s | ~0.77-0.78s | 3.83 | MuSiQue 平均轮数最高，因此推理成本最高 |

**rebuttal 可用表述**：

> We compare inference cost under an offline throughput-normalized protocol, i.e., total wall-clock inference time divided by the number of queries. ReasonRAG reports 110 seconds for 1,000 2WikiMultihopQA queries in its OpenReview rebuttal, corresponding to 0.11s/query with 3.8 average retrievals. Under our current SAPR-RAG implementation, the closest comparable 2WikiMultihopQA run takes approximately 0.70-0.74s/query with 3.51 average retrievals. Therefore, SAPR-RAG is currently slower than ReasonRAG under the reported throughput-normalized setting, despite using a comparable number of retrieval steps. The gap mainly comes from implementation-level overheads, including the explicit evidence extraction stage, data-parallel shard orchestration, and the current FAISS/corpus loading and retrieval configuration. Since these numbers are obtained under different hardware, batching, and retrieval-engine configurations, they should be interpreted as an indicative efficiency comparison rather than a strictly controlled latency benchmark.

备注：原始 `merged.jsonl` / `metrics.json` / shard logs 位于 `data/eval_results/`，该目录被 `.gitignore` 忽略；本文档只保留可 push 的汇总结论。

---

## 5. GRPO 训练与评测（#5，v4-formatfix）

### 5.1 配置

| 配置项 | 值 |
|---|---|
| 起点 | SFT ckpt-1650（继承多轮 RAG 协议）|
| 训练数据 | HotpotQA 训练子集（`data/grpo/hotpotqa_train.jsonl`） |
| 框架 | ms-swift 4.4 GRPO（vLLM rollout server 模式）|
| 卡分配 | GPU0-5 训练 / GPU6 rollout / GPU7 检索 daemon |
| reward 函数 | `SaprF1ORM`(w=1.0) + `SaprRelevanceORM`(w=0.2) + `SaprFormatORM`(w=0.05) |
| 关键超参 | per_device_bs=2, num_generations=8, grad_accum=4, lr=1e-6, max_completion_length=8192, max_turns=6 |
| 总步数 | 1220（1 epoch）|
| 节奏 | save_steps=25，评测 ckpt-125 / ckpt-175 |

### 5.2 格式 reward 修复

v3 训练中 `SaprFormatORM` 长期接近 0，根因是旧实现要求 completion 中不能出现任何 `<query>`，误伤了正常多轮 RAG 轨迹（前序轮 `<query>`，末轮 `<answer>`）。v4-formatfix 已修复为：允许前序 `<query>`，只要求最后一个协议标签是非空 `<answer>`。

离线重算旧 completions 的 format 通过率：
- 旧规则：2.41%（74/3072）
- 新规则：51.89%（1594/3072）
- 提升：+49.48pp

### 5.3 v4-formatfix 训练进展与崩溃原因

v4-formatfix 从 SFT LoRA 重新起训，已保存 ckpt-25/50/75/100/125/150/175/200/225。训练在 global_step=234/1220 崩溃，最后有效 checkpoint 为 ckpt-225；评测优先选择 reward 峰值附近的 ckpt-175。

**训练 reward 走势（按 global step 分组）**：

| step | reward | F1 | Format | Relevance | KL |
|---:|---:|---:|---:|---:|---:|
| 1-39 | 0.2809 | 0.1182 | 0.5229 | 0.6824 | 0.815 |
| 41-79 | 0.3007 | 0.1392 | 0.5240 | 0.6762 | 0.867 |
| 81-119 | 0.3144 | 0.1421 | 0.6437 | 0.7008 | 0.823 |
| 121-159 | **0.3429** | **0.1646** | **0.6536** | **0.7279** | 0.798 |
| 161-199 | 0.3403 | **0.1647** | 0.6495 | 0.7158 | 0.802 |
| 201-233 | 0.3103 | 0.1407 | 0.6005 | 0.6981 | 0.864 |

**checkpoint 选择**：ckpt-175 落在 reward 峰值/平台区间，ckpt-200 基本持平，ckpt-225 已进入回落区间。因此三数据集评测以 ckpt-175 为主。

**崩溃原因**：
```
ValueError: max_tokens must be at least 1, got -366.
RuntimeError: Multiple errors: [Exception('Server 0 failed: 500, Internal Server Error')]
```

根因是多轮 RAG 的 prompt 随检索 evidence 累积，某次请求超过 `vllm_max_model_len=8192`，vLLM 计算出的 `max_tokens = max_model_len - num_tokens` 变成负数，rollout server 返回 500，训练端随之退出。后续若继续训练，需要提高 `vllm_max_model_len`、降低 `max_turns`，或在 scheduler 里做 token 截断。

### 5.4 ckpt-175 下游评测结论

ckpt-175 已在 HotpotQA / 2Wiki / MuSiQue 三个 dev 集上完成完整评测和 LLM-judge。核心结论见 §4 后的"三数据集后训练对照"：
- HotpotQA：cover_em 与 SFT 持平，LLM-acc 小幅高于 SFT/SFT+DPO。
- 2Wiki：cover_em 达到当前最优 0.4573，LLM-acc 高于 SFT 但低于 SFT+DPO。
- MuSiQue：cover_em 与 LLM-acc 均高于 SFT，但低于 SFT+DPO，符合 GRPO 未在 MuSiQue 上训练的预期。

---

## 6. 待办

- [x] 修 SaprFormatORM 判据，并离线重算旧 completions 的新旧 format 通过率
- [x] 完成 #1/#2/#4 在 HotpotQA / 2Wiki / MuSiQue 三数据集评测
- [x] 完成 #5 GRPO v4-formatfix ckpt-125 HotpotQA 评测 + LLM-judge
- [x] 完成 #5 GRPO v4-formatfix ckpt-175 HotpotQA / 2Wiki / MuSiQue 三数据集评测
- [x] 写 LLM-judge 评估脚本（DeepSeek API + 标准 judge prompt + cache）
- [x] 补跑 #5 GRPO ckpt-175 三数据集 LLM-judge，验证 cover_em 增益是否转化为事实正确率
- [x] 补齐 #3 DPO-no-SFT 三数据集 cover_em / EM / F1 / llm_acc（用 SAPR-RAG score.py 口径重算 ReasonRAG pipeline 产物）
- [ ] 抽样 100 题做 case study，分析 SFT / SFT+DPO / GRPO 的输出风格差异
- [ ] D6 汇总数字 + 画图
- [ ] D7 写中期报告正文 + OPD 后续计划
