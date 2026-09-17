# SAPR-RAG 六天实验交接：证据状态感知的特权教师

日期：2026-09-18。状态：**待诊断、待实现、待验证；不是已成立的方法或实验结论**。

目标读者：可以访问 SAPR-RAG、其配套 ms-swift checkout、训练数据及 GPU 的执行 AI。
本次提交只提供计划，不启动训练、不改变生产推理、不修改已有实验结果。
核对基线：`dd86086d2f66fea9539b1238fa3a33742591f028`（main）。执行时先检查更新，不覆盖更新后的事实。

## 0. 执行摘要与边界

用户只剩六天，希望提出并验证一个有依据的方法改进。主线不是继续叠加 reward，而是检验：

> 特权教师是否在学生尚未获得足够证据时偏好回答，或提供与当前证据缺口不匹配的查询监督？若存在，逐轮、因果地构造教师上下文能否改善训练？

先诊断，后改动；诊断不成立则停止这个方向。不得承诺正结果或录用，不得把普通 prompt 修改、门控、OPSD 或训练顺序声称为已验证的原创算法。

- 不修改 GRPO reward、PPO clip、reference KL、全词表损失、检索器和部署前端。
- 不同时实验树搜索、全词表蒸馏、大模型教师及多种奖励。
- 不覆盖已有 checkpoints、metrics 或用户未提交改动；不执行 reset/clean/force-push。
- 只在用户授权的项目及相关依赖内工作，不修改其他用户目录、不占用未获准 GPU、不停止别人的任务。
- 不提交密钥、内部 IP、绝对个人路径、原始私有日志、checkpoint 或完整数据集。配置使用环境变量。
- 本文所有建议新增的参数、脚本和产物名都不是现成接口；实现前检查冲突，禁止直接运行虚构命令。

## 1. 已有证据：避免重复失败路线

先读 `docs/experiment_tracker.md`、`docs/opsd_objective_and_implementation.md`、
`docs/ms_swift_local_patches.md`、`docs/grpo_opsd_pipeline_overview.md`。
旧 overview 含历史配置，实际代码、checkpoint args 和 metrics 优先于文字总结。

以下为 tracker 中 ckpt1000 的 EM/F1/Cover-EM，数值为 0 到 1：

| 路线 | HotpotQA | 2Wiki | MuSiQue |
|---|---|---|---|
| B：canonical SFT -> GRPO | .4629/.5837/.5026 | .5161/.5654/.5314 | .1808/.2794/.2056 |
| C/E16：canonical SFT -> GRPO+OPSD | .4636/.5816/.5025 | .5154/.5659/.5307 | .1837/.2786/.2089 |
| D：canonical SFT -> 纯 OPSD | .4462/.5703/.5030 | .4948/.5548/.5270 | .1758/.2717/.2085 |

C-B 没有稳健增益。不能据此证明梯度冲突、信号重叠或梯度被淹没；这些是解释假设。
记录里的 C-D 不是先做 D 再接 GRPO，不应误读为串行实验。
E17 至 E23 已尝试附加证据信用、零优势组救援、动作分解、查询重加权、return-to-go、动态采样、长度归一化；没有形成稳健独立收益。不要重新包装这些方案。

## 2. 必须核实的现有实现

已核对 `03_sapr_rag/scripts/grpo/build_grpo_dataset_action_opsd.py`：

- `build_query_prompt` 使用 R3 查询计划，且已经明确要求适应已观察文档。因此“原教师完全不看状态”不是事实。
- `build_answer_prompt` 使用 gold answer 和支持证据，提示证据充分时回答。是否真正遵从必须测量。
- `gold_sup_sents` 当前是按 title 聚合的文本，不能未经解析便当作独立原子事实。MuSiQue 可能是支持段落，粒度不同。
- 已有 Query 教师只在匹配 R3 计划的样本上可用。对照必须控制覆盖率，不能把新增监督样本造成的收益归因为状态感知。

现有 ms-swift 路径（相对于配套 checkout）：

| 位置 | 检查内容 |
|---|---|
| `swift/rl_core/data.py` | `OnPolicySample`、`build_teacher_view`；是否保留逐轮前缀 |
| `swift/rlhf_trainers/gkd_helpers.py` | teacher encoding、logprob remap、`build_teacher_action_mask` |
| `swift/rlhf_trainers/grpo_trainer.py` | teacher 前向与 advantage 写回 |
| `swift/rl_core/advantage.py` | detached log-ratio、GRPO advantage、监控 KL |
| SAPR `03_sapr_rag/scripts/grpo/plugin.py` | 实际 student observation、检索轨迹、Evidence Agent、rollout_infos |

不要仅根据文档猜接口。记录实际导入的 swift 路径、git SHA、补丁 SHA256、adapter SHA/路径哈希、检索配置。
已有 teacher mask 按整个 assistant turn 分类，不只是 query/answer 标签内部；主实验保持 mask 不变。
本地 ms-swift 修改须提供可迁移补丁和校验值，但不可覆盖别人未提交的依赖改动。

## 3. 第一天诊断：先证明问题存在

从 training split 分层抽取约 100 个问题的轨迹，覆盖三个数据集、成功/失败、重复检索、提前结束和 max-turn。
保留随机抽样分母；故障定向补充案例单独统计，不用于估计总体发生率。
可复用 train rollout；没有则用选定基线生成。不要用最终评测集构造训练监督。

逐个 decision turn 构造 `state_before_action`，只包含：原问题、过去动作、截至该动作前实际传给 student 的 observation。
同时保存检索原文覆盖率和 student 可见覆盖率，两者不能混淆：若 student 只见摘要，未被摘要保留的事实不算已知。
若上下文被截断，以模型实际看到的窗口为准，不能假设被截去的内容仍可见。

### 3.1 证据状态标注

采用 `sufficient / insufficient / unknown` 三态：

- sufficient：有可审核证据支持回答所需关系链；不是仅出现答案字符串或 gold title。
- insufficient：存在明确尚未支撑的必要关系；支持事实缺失是线索，不是绝对判决。
- unknown：数据标注不全、语料版本不匹配、同义表述无法可靠判定、存在替代支持路径。

gold supporting facts 只是标注的一种证明路径：未覆盖全部 gold 不等于一定不可回答；覆盖全部 gold 也须检查其是否真在可见上下文中。
先做规范化文本匹配作为高精度覆盖代理，匹配失败记 unknown 或人工复核，不用“title 命中 OR 答案命中”作为充分性判断。
有条件用已有模型辅助标注，但保留引用原文、版本和抽样人工复核；不得把 LLM 判断当无误真值。

### 3.2 教师诊断

在相同前缀上，用原教师和候选教师分别评估固定候选动作：
1. 补齐可解释缺口的 query；2. 重复或无关 query；3. 提前 answer。
候选仅用于诊断，不直接变成新的训练标签。评 query 质量不能只看相似度，应核实实际检索返回及证据变化。
记录各动作长度归一化 logprob、排序、标签分支的概率，以及已有学生动作的 logp_T-logp_S。
不同长度动作的均值 logprob 只作代理，不当作严格跨动作概率或真实 Q 值。
特别注意：现有按动作选择 teacher context 的打分不可直接作为同上下文的 query/answer 决策概率比较。
若要测决策偏好，额外用同一个 teacher context 对候选动作评分，或用相同采样设置生成下一步动作；与实际训练信号分别报告。

输出诊断表：错误指导发生率、可判定状态比例、覆盖率、各数据集分布、原/新教师排序差异，以及至少 10 个可审核案例（含反例）。
决策门：若未发现稳定错配、充分性标签不可靠，或候选教师未改善方向，不进入长训练。记录失败原因，不人为挑选支持故事的样本。

## 4. 候选方法：逐决策状态构造教师视图

这是待实现方案，不是已完成算法。训练时以问题 gold 元信息和 student 历史构造指导；部署时完全不用 gold。

令 H_k 为动作 k 开始前的实际可见历史，G 为训练 gold 信息：

```text
z_k = diagnose_evidence_state(H_k, G)   # sufficient / insufficient / unknown
c_T,k = build_guidance(H_k, G, z_k)
d_k,t = stop_gradient(log p_T(y_k,t | c_T,k, y_k,<t)
                     - log p_S(y_k,t | H_k, y_k,<t))
A_k,t = A_GRPO + beta_action * mask_k,t * d_k,t
```

保留现有 PPO surrogate、loss reduction、reference KL 和 old-policy 定义；实际采用何时缓存 student logp 以当前实现为准并记录。
这不是新的 KL 公式或严格的新 policy-gradient 推导，不对其做原创声明。

### 4.1 指导内容

- insufficient：指出仍缺的关系或检索目标，要求补充查询；不能把尚未检索的事实写成“已经观察到”。
- sufficient：要求利用已有可见证据回答，避免重复查询。
- unknown：回退原教师，不凭不可靠标签强制继续检索或停止。

Query 指导优先使用原问题及历史已出现实体描述下一步关系；避免直接给出未来桥接实体或最终答案，鼓励学生发出当时可执行的查询。
若实现使用 gold 派生实体/事实，必须明确为额外特权信息，并在静态对照中匹配信息量；不能声称“无特权信息”。
不要自动把 gold 文档加入 student 检索结果，也不要把诊断标签写入 student messages。
主版本只改 teacher context，不另增硬惩罚、强制动作规则、额外 SFT 或改 mask。

### 4.2 最危险的实现错误：未来信息回流

不能把最终轨迹总结出的缺口追加到第一个 user message，然后给整条轨迹打分。这会让早期动作教师看到未来 observation。
每个 assistant turn 必须使用独立的动作前历史构造 teacher view，只评分当前 turn，映射回原 student token 索引。
同一 turn 内每个 token 使用该 turn 开始时固定的指导及已生成前缀，不能根据本 turn 完整输出反推指导内容。
不同 turn 长度、跨进程 gather、截断、padding、observation mask 必须保持严格对齐。
逐轮重算前缀可能显著增加开销：先测吞吐，再按机器资源分批；不以牺牲因果边界换速度。
如果第 2 天内做不对 turn-level scoring，停止此方案，不将一次性静态 prompt 冒充动态方法。

## 5. 最小代码任务与验收

建议在 `03_sapr_rag/scripts/grpo/` 下新增独立、可单测的 evidence-state helper 和诊断入口。
数据构造只保留结构化 gold 元信息；动态状态在 rollout 结束后的 teacher 打分阶段按动作前缀计算，不预先写成整题静态状态。
若需改配套 ms-swift，尽量限制在 teacher view 构造/逐轮打分和 metadata 传递；不重写 loss。
建议新增默认关闭的配置 `teacher_state_mode=off|static_matched|dynamic`，这是建议名，不是现有参数。

必须通过：

- off 与旧 teacher 路径在确定性测试下输出一致；普通 GRPO 不访问新增 gold 元信息。
- 修改未来 observation 不改变过去 turn 的 teacher context 或 logprob。
- 两条同题不同检索历史应得到不同动态状态；缓存 key 不能只有 question id。
- teacher context 构建前后 student messages 和 response ids 不变；用特权哨兵文本检测 student 泄漏。
- gold 丢失、部分缺失、替代证据、截断、零检索、最终 answer、格式异常、空 turn 均有测试。
- teacher/student 所评分 token ids 严格相同；只 observation/padding 的位置无 teacher loss。
- teacher 无梯度；学生有非零有限梯度；optimizer step 后权重变化；LoRA 真实加载并同步到 rollout。
- unknown 回退可复现；错误日志不输出隐私或完整 gold 数据。
- 先做 1-2 optimizer step smoke，再开 pilot。测试失败不能拿 loss 下降替代正确性验证。

## 6. 有限预算实验：先固定设计，再看结果

主实验三组：

| 组 | 教师 | 作用 |
|---|---|---|
| P0 | 原动作教师 | 原方法匹配对照 |
| P1 | 信息量匹配的静态教师 | 控制额外 gold 信息、模板和新增 Query 覆盖 |
| P2 | 动态证据状态教师 | 候选方法 |

P1 与 P2 使用同一批监督样本、相同可用特权事实、动作系数、模板预算和 turn-level scorer。P1 不提供动态覆盖/缺口标注；P2 提供。保留相同 student 历史。
两者都采用因果逐 turn 评分，避免把 scorer 改动当方法收益。记录实际 teacher token 数和 GPU 小时，不虚称算力完全相同。
若 P1/P2 信息量或监督覆盖无法匹配，明确多因素混杂，不得归因于动态状态。

优先三组从同一 C/E16 checkpoint-1000 启动新的匹配续训，每组先 250 step；这回答“在已有联合训练上改教师是否有用”，不回答从头训练优越性。
模型/adapter 权重起点一致，optimizer/scheduler 要么三组都完整恢复、要么都重新初始化，不能混用。续训 reference model 选择和 step 计数显式一致。
如果 C checkpoint 不存在，三组统一使用可用 canonical SFT checkpoint，禁止每组自选起点。
历史 B/C/D 是背景参考，不是匹配续训对照；不能只报告新 P2 超过旧 C。

固定数据、seed、采样温度、GRPO group size、top-k、max-turn、检索索引、Evidence Agent、reward、lr、batch、KL、mask、保存频率。
第一轮不调 query/answer 系数，不改现有查询监督覆盖规则之外的因素。新覆盖若不可避免，必须在 P1 同步启用。
总预算以 GPU 小时、wall time、rollout/teacher tokens 记录，不只记录 step。

验证集从训练源划出并排除训练，固定问题 ID；已有 dev 若长期用于选配置就如实称 validation，不包装为未见测试。
锁定主要选择指标为三数据集 macro F1；EM、Cover-EM 和行为指标为次要。pilot 只允许一个预定 250-step checkpoint，避免反复挑选最优。
预先采用工程继续门槛：P2 对 P0/P1 的 macro F1 都至少 +0.5 个百分点，且无单集下降超过 1 个百分点，并有诊断指标同向变化，才考虑统一延长到 500 step。
这只是预算决策阈值，不代表统计显著；必须报告置信区间。若样本不足，称证据不足，不判定机制已被证伪。
若预算无法负担三组，优先 P1/P2，明确缺少与原方法的匹配新对照；不得用旧 C 冒充 P0。

## 7. 正式评测与论文证据

固定 checkpoint 后再做三个数据集全量评测。沿用同一 strict HTTP/evidence-extraction 配置，不能与另一 pipeline 的指标直接归因比较。
报告 EM、F1、Cover-EM、三集宏平均、样本数、失败请求比例；服务失败显式重试/单列，不悄悄删题。
按 question id 做 paired bootstrap（建议 10000 次），三集分别及分层宏平均给差值区间；单 seed 显著不等于训练稳定性。
有余量优先补 P1/P2 第二 seed，而不是多扫超参。

行为指标定义必须固定：

- 重复 query 率：规范化后与过去 query 完全相同的次数 / query 总次数；语义重复另报。
- 可见支持事实覆盖：可验证已见事实数 / 有效 gold 事实数；不同数据集粒度分别报告，unknown 不当零。
- 证据不足回答率：在可靠标注为 insufficient 的可判定状态中回答的比例，必须同时报告 unknown 比例。
- 证据充分后继续检索率、max-turn 率、平均检索轮数和答案长度。
- 训练信号：按 query/answer 和充分性状态报告 teacher log-ratio 符号/幅度、非零覆盖率、clip 比例及实际教师成本。

即使行为指标改善，也不能替代 EM/F1；即使最终指标改善，也不能单凭相关性证明因果机制。
展示至少一例改善、一例无改善、一例退化，不只展示漂亮案例。

## 8. 六天时间盒与停止规则

| 时间 | 交付与决策 |
|---|---|
| 第1天 | 环境清点、100题诊断、相关工作近邻核对；不成立就停止主线 |
| 第2天 | 实现、测试、1-2步 smoke、吞吐测量；因果/对齐无法保证则停止 |
| 第3天 | 匹配250-step pilot；仅通过预定门槛才延长，不无限调参 |
| 第4天 | 锁定 checkpoint，完成全量评测；资源允许补第二 seed |
| 第5天 | 统计、行为分析、主表、消融、失败案例 |
| 第6天 | 方法与限制写作、复现命令、最终文档整理 |

只在主线未启动大训练且用户仍希望探索时，备选 D->GRPO 与 B->继续GRPO 的匹配续训；不要同时抢资源运行两个方向。
备选只有训练顺序价值，不能自动成为原创贡献。不得通过新增推理 oracle、放宽评测或选择性报数制造收益。
若到第3天无可信信号，停止开发新算法，整理现有结果与失败分析，明确论文定位与证据限制。

## 9. 新颖性核对与可写结论

下列是需要阅读比较的近邻，不是已完成的新颖性排查：

- Tree-GRPO: https://arxiv.org/abs/2509.21240 ，同状态树分支/过程优势，不是本计划创新。
- OPSD: https://arxiv.org/abs/2601.18734 ，特权自蒸馏本身不是创新。
- Distilling Realizable Students from Unrealizable Teachers: https://arxiv.org/abs/2505.09546 ，信息不对称已有研究。
- Privileged Information Distillation for Language Models: https://arxiv.org/abs/2602.04942 。
- Verify Before You Distill: https://arxiv.org/abs/2609.02998 ，需比较教师可靠性门控与本方案的边界。
- 同时检索 state-conditioned teacher、evidence sufficiency、retrieval stopping、privileged agent distillation，不能只对比 Tree-GRPO。

若成功，只能按真实证据表述：在固定学生轨迹和匹配监督条件下，基于动作前可见证据构造特权指导改善了哪些数据集和行为。
不得宣称首次、彻底解决、理论最优，或把 sampled-token PG、三态分类本身当成原创。
若已有论文实质覆盖，定位为复现/领域适配，不继续制造新缩写。

## 10. 执行 AI 的交付契约

第一条反馈先给出：当前 git SHA/dirty 状态、可用 GPU/预算、依赖补丁是否一致、B/C/D checkpoint 是否存在、实际推理 pipeline，以及与本文不符之处。
将结果追加到新报告 `docs/evidence_state_teacher_experiment_report.md`（建议路径），不要篡改旧实验结论。
保存诊断 split IDs/哈希、配置 manifest、代码和补丁 SHA、运行命令、日志位置、metrics 和统计脚本。私有大文件只记录本机路径，不 push。
建议实验产物放到独立 `data/eval_results/evidence_state_teacher_<run_id>/`；是否入库遵循现有 .gitignore，只提交允许公开的小型汇总。
每次实验更新表格：run_id、起点、训练步数、GPU小时、教师token量、数据/语料版本、pipeline、EM/F1/Cover、行为指标、可信度及下一步决定。
最终交接必须区分：已证实事实、待验证解释、实现完成但未验证部分、失败结果；没有 GPU 实测就明确写未运行。

建议下一台机器收到的启动指令：

> 阅读本文及引用的仓库文档，先完成第1至3节的核验与诊断。不要立即启动长训练。只有发现可复现的教师状态错配、并通过第5节的因果与token对齐测试后，才执行匹配pilot。严格遵守六天预算，不引入额外研究方向，不为正结果改变评测口径。

