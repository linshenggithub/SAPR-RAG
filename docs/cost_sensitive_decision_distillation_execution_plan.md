# SAPR-RAG：成本敏感 Search／Answer 决策蒸馏执行计划

日期：2026-09-20。状态：**待实现、待 GPU 验证的研究计划，不是已完成算法或实验结论**。

目标读者：能够访问 SAPR-RAG、配套 ms-swift 源码、训练数据、模型与 GPU 服务器的执行 AI。
本次提交仅增加此文档，不修改训练代码、不启动服务或训练、不宣称方法优于现有基线。
仓库核对版本：`a124bfad63fff8fade540f48d85a0dcfa8767bdc`，分支 `main`。执行时重新核对 HEAD 和未提交改动。

## 0. 一句话目标与执行边界

在学生实际走到的同一个状态上，分别试验“现在回答”和“至少再检索一次后回答”，用**答案质量提升减去额外检索成本**形成 Search／Answer 软标签；把软标签直接蒸馏进学生的二元动作决策，再与原 GRPO 联合训练。

主线要回答：与匹配的 GRPO 相比，这种局部决策监督能否改善**答案质量—检索成本**权衡，而不是只让模型更早停止。

- 这是对上一轮讨论的明确实现化：新增可训练的二元决策接口是实现选择，不是仓库已有能力。必须有协议匹配对照，不能把协议改动的收益归因于软标签。
- 本计划与 [证据状态感知教师计划](evidence_state_teacher_six_day_plan.md) 是两个独立研究方向；不覆盖该文件、不自动同时运行。本文不采用三态证据教师，不混入它的 teacher prompt。
- 第一版关闭旧 OPSD、动作证据信用、动态采样等附加项。不是把若干未验证方法一起堆叠。
- 文中新增脚本、配置字段、测试名均是**开发契约，尚未实现**。只能先实现并验证 `--help`／dry-run，再执行相应命令。
- 执行 AI 首次回复先提交环境核验、预算估算、风险与阶段计划。用户未确认 GPU 分配、最大 GPU 小时、wall time 和实验范围前，只允许只读检查与不占 GPU 的开发测试。
- 不重置工作区、不强推、不覆盖旧模型或数据、不杀其他任务、不擅自租赁或使用付费 API。OOM、超时或资源不够时停下来报告，不偷偷缩小样本、模型、检索语料或上下文。
- 现有 `AGENTS.md` 优先约束工程操作；方法细节以本计划和用户后续明确选择为准。历史文档中的旧研究优先级不代表授权同时运行旧实验。

## 1. 为什么尝试这一方向，以及不能预设什么

### 1.1 已有事实

先读 [实验总记录](experiment_tracker.md)、[既有 OPSD 目标](opsd_objective_and_implementation.md)、[ms-swift 迁移手册](ms_swift_local_patches.md)、[检索服务手册](retrieval_service_gpu_runbook.md)，再看实际代码、checkpoint 参数和原始 metrics。

- tracker 的 B（canonical SFT→GRPO）与 C/E16（canonical SFT→GRPO+OPSD）没有稳定独立差距。不能写成“融合已经有效”，也不能仅凭这个结果断言梯度冲突。
- E19 小规模收益没有扩展到全量；E20、E21、E22、E23 也未形成稳健收益。不要重新包装局部 evidence credit、return-to-go、动态采样、长度归一化作为本计划核心。
- 当前累计 relevance 奖励可能无法区分“这一次检索值不值得”。但模型是否确实因此冗余检索，必须由同状态分支诊断验证，不能把猜测当既成事实。
- F1、EM、检索轮数必须按实际 pipeline 比较。原始文档直塞、Evidence Agent 压缩、canonical 重建历史及线性多轮聊天不能混作同一实验。

### 1.2 可检验假设

H1：存在足够多的状态，继续检索的答案增益小于额外成本，但当前策略仍搜索；同时也存在尚需搜索而提前回答的状态。

H2：该局部效用差比“累计命中多少 gold 文档”更直接反映 Search／Answer 决策价值，且少量分支采样的噪声可以接受。

H3：在相同协议、相同数据和预算下，把局部效用转成软目标，比单纯长度惩罚或在相同分支数据上做局部 RL 更有效。

H1/H2 不成立就停止长训练；只有 H3 得到实证支持，才考虑把决策目标作为方法贡献。交叉熵、sigmoid、GRPO、分支采样本身都不是新发明。

## 2. 必须先完成的服务器与代码核验

### 2.1 只读预检

在 Linux 服务器的 SAPR-RAG 根目录检查以下信息。变量须由实际环境提供，不把文中示例当作真实路径或 GPU 编号。

```bash
git remote -v
git rev-parse HEAD
git status --short
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
python -c "import torch, swift; print(torch.__version__, torch.version.cuda); print(swift.__file__)"
sha256sum patches/ms-swift/sapr-rag-ms-swift-full-1dbd1bf64.patch
```

必须记录：SAPR 与 ms-swift SHA、dirty diff、实际导入路径、CUDA/PyTorch/vLLM/FAISS/ms-swift 版本、GPU 显存与占用归属、模型/tokenizer/adapter 标识、训练数据及 split ID 哈希、语料与索引指纹、检索服务 health、可写剩余磁盘、端口归属。

配套 ms-swift 文档基线为 `1dbd1bf64a46bd6bb710d9ace05d529ff071cd1f`；现有完整补丁 SHA256 为 `710ec851f05ce05a5d684c784bfde083e310b91ac88fa8072a104a58c35dc7dd`。如果服务器已更新，不盲目回退或重复打补丁；先比较实际差异。

完整补丁应用仅限用户授权且干净的依赖 checkout，先 `git apply --check`；不与旧最小补丁重复叠加。新实现的依赖改动另存增量补丁并注明应用顺序和 SHA256，不能只留在某台服务器。

检索器必须复用现有健康服务。当前 GPU 检索手册涉及约 2235 万向量，不能假定普通 24GB 单卡装得下。没有原配置所需资源时，先报清 CPU/远程服务等可选方案及吞吐、精度差异，取得确认后形成新的匹配配置。

### 2.2 现有入口与陷阱

| 已有位置 | 需要核验的内容 |
|---|---|
| `03_sapr_rag/scripts/grpo/plugin.py` | `SaprRagScheduler`、`SaprCanonicalScheduler`、reward、实际可见 observation |
| `03_sapr_rag/scripts/grpo/run_canonical_sft_grpo_control_s1000.sh` | B 的历史起点和开关；不要直接运行其中固定 GPU/端口 |
| `03_sapr_rag/scripts/grpo/launch_action_scoped_opsd_worker.sh` | rollout/训练组织、权重同步、scheduler 选择 |
| `03_sapr_rag/scripts/grpo/run_grpo_opsd.sh` | GRPO reward、loss reduction、上下文与采样参数 |
| `03_sapr_rag/scripts/eval/agent_infer.py`、`run_direct_rollout_eval.py`、`score.py` | 部署与评分链路；新版动作协议必须另做适配 |
| `03_sapr_rag/scripts/eval/paired_bootstrap.py` | 可复用统计入口，先核实输入格式 |
| `config/paths.py`、`config/env_*.sh` | 路径和设备配置；不硬编码个人服务器目录 |

特别注意：wrapper 名称里的 `canonical_sft` 指 SFT 起点，不意味着使用 `sapr_rag_canonical_scheduler`。当前 launcher 默认 `sapr_rag_scheduler`；后者保留多轮训练，而 `SaprCanonicalScheduler` 明确只返回最终轮 completion。不能直接换成 final-turn-only scheduler，再声称前面每个决策已通过 GRPO 训练。

在核对版本中，B wrapper 为 E14 `checkpoint-4150` 起点，启用 Evidence Agent、关闭 OPSD 和 truncation reward、8 条 GRPO 采样。`run_grpo_opsd.sh` 的基础 reward 为 `1.0*sapr_f1 + 0.2*sapr_relevance + 0.05*sapr_format`。这些是代码线索，不替代真实 B 的 args/manifest；不得静默采用其他脚本的默认值。

先把实际 B 的完整配置转成新的只读 manifest。主实验保持原 reward、reference KL、PPO clip、loss reduction、学习率、batch、上下文和 evidence 行为；新增动作协议除外，且所有新对照统一采用它。

## 3. 核心算法的精确定义

### 3.1 状态与两条分支

`h_t` 是第 t 次动作选择**之前**学生实际看到的 token 前缀，包括问题、已发生的动作和可见 observation。还须保存生成模板、截断后 input_ids、attention/position 信息、已用检索数、剩余 turn/token 预算、随机种子和策略版本。

不能用动作生成后的 reasoning、当前 query 返回结果或未来 observation 构造当前状态。若学生只见 evidence 摘要，状态不能偷偷换成完整检索文档。不能仅按 question_id 缓存状态。

对同一 `h_t`、同一冻结 rollout 策略 `pi_old`：

- A 分支：强制当前动作是 Answer，只生成回答所需文本，禁止调用检索。
- S 分支：强制当前动作是 Search，生成 query、调用原检索器和原 Evidence Agent，然后按同一策略自然继续，直到 Answer 或原剩余预算耗尽。
- S 不是“固定只多搜一轮”：后续可能多次搜索；**所有新增检索都计成本**。第一版不使用另一套 teacher 续写，不扩充 S 的上限。
- 每个分支独立采样 `K=2` 次，保存所有结果并取均值。`K=2` 是 pilot 默认建议，不是精确估计的保证。禁止 best-of-K、只保留成功回答或两条分支使用不同 K。
- 搜索分支没有实际执行检索时，必须区分原因：模型产生不合法 query 是失败结果，保留零质量并计逻辑搜索尝试；服务故障是基础设施错误，按下节规则处理。不得把一次失败假报为成功检索，也不得仅因模型失败而删去该状态。

合法状态须同时允许 A 与 S：至少还允许一次检索和其后的回答。达到预算上限时 A 是强制动作，不进入二元标签集。

### 3.2 质量与额外成本

第一版只采用训练集 gold short answer 的 token F1，取值 `[0,1]`。复用并核对现有 normalization、多 gold answer 取分及 `<answer>` 抽取规则；没有有效答案或因模型行为耗尽预算，则质量为 0。gold 只进评分器，绝不进分支生成或学生输入。

```text
Q_A(h) = mean_k F1(answer_A,k, gold)
Q_S(h) = mean_k F1(answer_S,k, gold)
C_A(h) = 0
C_S(h) = mean_k additional_search_attempts_S,k
U_A(h) = Q_A(h)
U_S(h) = Q_S(h) - lambda_cost * C_S(h)
Delta(h) = U_S(h) - U_A(h)
```

`additional_search_attempts` 只计 h 之后选择 S 的逻辑检索动作，包括选择 S 后生成不合法 query 的失败尝试。历史检索是 sunk cost，不再扣除。重复 query 即使命中缓存也计一次逻辑动作，防止用缓存“免费重复”；另报实际 RPC 次数、缓存命中及延迟。技术重试不作为新决策，单列其真实运行成本。这是显式的逻辑动作成本代理，不等于实际成功 RPC 数。

`lambda_cost` 单位是“每额外一次检索折合的 F1 损失”，不是秒或美元。首轮固定建议 `0.03`；这是实验选择，不是已经找到的最优值。token 与 wall time 先作为测量指标，不在首版混入不同量纲的惩罚。

接口异常、GPU OOM、检索服务不可用不等于模型答错。保留失败记录，按固定重试规则重试；仍失败则该状态标为 `infra_invalid`，不生成软标签，并报告覆盖率。模型无答案、坏格式、超出预算等必须保留为零质量，不能删去以美化结果。

### 3.3 效用差转软标签

```text
q_S(h) = sigmoid(Delta(h) / tau)
q_A(h) = 1 - q_S(h)
```

首轮建议 `tau=0.05`。温度越小目标越硬；不采用状态内 advantage 标准化，否则可能消掉成本量纲。`q` 全部 detach。

例一：`Q_A=.70, Q_S=.85, C_S=2, lambda=.03`，则 `Delta=.09`，`q_S≈.8581`。
例二：`Q_A=Q_S=.85`、其余同上，则 `Delta=-.06`，`q_S≈.2315`。

这里 q 是人为定义的效用偏好分布，不是“Search 有 85.81% 概率答对”，也不是统计显著性或置信度。相同质量且搜索有成本时自然偏向 Answer；都答不出也偏向 Answer，所以低能力模型可能过早放弃。必须监控双零质量比例，不能把更少搜索自动当作成功。

等价地，`q=softmax([U_A,U_S]/tau)` 是局部有限动作问题 `max_q sum_a q_a U_a + tau*H(q)` 的解。这只为软目标提供解释；分支价值是有限样本、给定当前策略和预算的估计，不是真实最优 Q，也不保证全局策略改进或严格因果识别。

### 3.4 软标签转 loss

在同一合法动作前缀上，取两个动作 logits `z_A,z_S`，定义：

```text
p_theta(a | h) = exp(z_a) / (exp(z_A) + exp(z_S))
L_dec(h) = -q_A(h)*log p_theta(A|h) - q_S(h)*log p_theta(S|h)
L_total = L_GRPO + beta_dec * mean_valid_states L_dec(h)
```

`L_dec = KL(q || p_theta) + H(q)`，q 停止梯度时两者梯度相同。它是**二动作正向分布匹配**，不同于仓库旧 OPSD 把 sampled-token teacher log-ratio 写入 advantage。不要用旧 OPSD 开关冒充这个 loss，也不要向所有 query token 广播 q。

对于 `u=z_S-z_A`，`dL_dec/du=p_S-q_S`。例如模型 `p_S=.40` 而标签 `.8581`，梯度下降会提高 Search；标签 `.2315` 时会降低 Search。这应成为单元测试。

首轮 `beta_dec=0.1`，loss 按有效状态平均，不按整条轨迹 token 数平均。记录两项 loss、decision 梯度范数、动作概率变化和 clip/KL。旧 OPSD teacher 关闭；保留 GRPO 原有 reference KL，不再重复加一个同义的 reference KL。

## 4. 动作概率如何真实落到当前模型上

### 4.1 第一版选择：显式单 token 决策字段

原模型自由 reasoning 之后才输出 `<query>` 或 `<answer>`，不能直接把首 token logits 叫 Search／Answer 概率，也不能比较整段 query/answer 的累计 logprob。

主实现选择一个显式、公共的决策前缀，例如先输出 `<decision>S</decision>` 或 `<decision>A</decision>`，随后生成原来的 reasoning 和对应 `<query>`／`<answer>` 内容。**S/A 必须在实际 tokenizer 和实际前缀下各对应一个明确 token**。优先复用词表，不增加 special token 或扩大 embedding；若不满足，先另选经过测试的单 token 代号并写进 manifest，而不是对多 token 字符串硬套二分类公式。

工程契约：

1. scheduler 在每个决策点提供固定 assistant 前缀 `<decision>`；只对 A/S 两个合法 token 做归一化。决策温度固定为 1，训练从这个分布采样；不对它再用内容生成的 top-p/top-k/temperature。
2. 选择一个动作 token 后，scheduler 注入固定闭合文本和换行，再让模型生成 reasoning 与对应标签内容。固定前缀、闭合符是环境/模板 token，loss mask 为 0；真实采样动作 token 有 GRPO mask。
3. Query 内容只能由 S 执行；A 不准调用检索。出现动作标签冲突、无合法结束标签或内容超限，记录模型格式失败，质量按失败处理，不偷偷重采样到成功为止。
4. 分支探测只替换这个动作 token，其他 prompt、可见历史、内容生成设置保持一致。强制 token 不参与 probe 梯度；probe 根本不进入 GRPO 组。
5. 剩余预算仅允许回答时强制 A：其选择概率为 1、选择 token 的 policy-gradient/decision mask 为 0，回答内容仍按原规则训练。
6. 端到端保存 token IDs，不把多个生成段 decode 后重新 tokenize 来猜边界。内容生成须继续同一 assistant turn；如果推理接口不能可靠 prefill/continue，先实现并测试，不用不同 chat 模板蒙混。
7. 每轮可能需要一次短动作调用加一次内容调用。部署不需要 teacher 或分支评估，但**不能声称零额外延迟**；计入动作调用、prefill 与新增 token 的实际开销。

由于选择发生在当前 reasoning 之前，这也改变了模型计算顺序，是明确的实验因素。格式对照必须共享这一因素；若格式改造已经严重退化，不推进软标签训练。

### 4.2 所有新实验共享的格式热启动

从训练问题的历史策略 rollout 中抽取至多 2048 个有效 reasoner turns，用原有最终 query/answer 类型作为行为克隆标签，在原可见前缀下补入决策字段。这个标签不代表动作最优，仅用于学会新协议。

- 不使用最终评测集，不加入 gold／未来观察，不把 Evidence Agent turn 当动作决策。
- 尽量覆盖两类动作，保存采样权重和问题 ID；原内容可用于保留格式/生成能力，固定注入 token 不设生成目标。
- 所有新对照使用同一个 warmstart checkpoint、同一 reference 配置、相同 optimizer/scheduler 初始化方式；不让只有完整方法获得额外 SFT。
- warmstart 配置、步数与数据必须在训练前冻结。2048 是建议上限，资源不足不得静默降量。
- 在独立训练源 validation 上，检查协议有效率、EM/F1、检索数及 max-turn。协议冲突率超过 1% 或宏 F1 相对原协议下降超过 1 个百分点，先修协议/热启动，不进入主训练；阈值是工程门槛而非统计定理。

### 4.3 与 GRPO 的一致性：不能漏改 old/ref logprob

在合法二元选择处，`log p_current`、`log p_old` 以及 reference 项必须采用**同一二元归一化和同一合法动作集**。直接使用全词表动作 token logprob，而实际 rollout 从两词分布采样，会使 importance ratio 错配。

内容 token 仍用原分布。建议保存 `token_kind=decision|content|injected|observation`；二元位置替换当前/旧/参考 logprob，其他可训练位置保持原计算。对 decision 位置可计算精确二元 reference KL；具体 reduction 与系数沿用基线，P0/P1/P2/P3 完全一致并写清。

不要改变原 GRPO 组定义：每题仍只有原来 G 条自然轨迹。probe 只是构造 auxiliary target，不能当作额外 rollouts 扩大 G、改变均值方差或基础 reward。

## 5. 在线标签生命周期、隔离与预算

每个 rollout batch 使用一个固定 `policy_version`。普通轨迹与分支续写在相同权重版本下生成；分支完成之前禁止 rollout 服务被下一步 adapter 同步覆盖。

第一版对每个被选中的问题：先按种子随机选择其 G 条自然轨迹之一，再在其中所有合法 decision states 中均匀选一个。选择不能看 reward、正确性、Delta 或是否重复。没有合法状态时记为无覆盖，不从成功轨迹补选。

默认 `probe_prompt_fraction=0.25`、每题至多一个状态、`K_A=K_S=2`。这是为控制成本而定义的监督分布；报告分数据集、轮次、成功/失败的实际覆盖。它不是对所有状态的无偏监督；若要宣称全状态目标，必须额外设计抽样加权实验。

q 只在本轮 rollout batch 的既定优化窗口内复用，窗口结束丢弃。训练配置必须记录数据复用次数、`steps_per_generation` 和策略陈旧度。离线诊断集不能一直复用却称为在线蒸馏。

复用原 rollout 服务和检索服务，新增有上限的 probe 队列；不要每个 GPU worker 都重复生成同一状态标签。为 fork 分配独立 uuid，深拷贝 history、seen_queries、预算和 request config，确保不会污染主轨迹的 `_traj`／`rollout_infos`。

缓存 key 至少包含：policy_version、实际 prefix token hash、模板版本、剩余预算、动作、分支 seed、检索/语料/evidence 指纹。保留结果原文用于本机审计但不 push。相同 query 在不同上下文或不同模型版本下不能误复用。

训练成本估算：若 N 个问题入 batch，按固定比例无放回选择 `floor(0.25*N)` 个问题，则新增分支数上限为 `floor(0.25*N)*1*(2+2)`；不足一个可选问题的小 batch 先合并成逻辑抽样窗口，不能永远没有监督。S 每条还可能含多轮检索和 evidence 调用。不能把这误算成“每题只多两 token”。先测 20 个状态，报告 tokens、GPU 小时、检索次数、峰值显存和 250-step 预计 wall time，用户认可后再扩大。P0/P1 不生成 probes，P3 beta_dec=0 的关闭对照也关闭 probe 队列。

## 6. 数据契约与日志

以下为建议 JSONL schema；字段名可在实现时统一命名，但语义与审计信息不能省略。数值样例只是格式示例。

```json
{
  "schema_version": "decision_distill_v1",
  "question_id": "train_source:id",
  "source": "hotpotqa",
  "split": "train",
  "trajectory_id": "main_uuid",
  "state_id": "main_uuid:decision_2",
  "policy_version": "rollout_version_id",
  "prefix_hash": "sha256",
  "prefix_tokens_ref": "local_record_id",
  "decision_position": 123,
  "remaining_search_budget": 3,
  "remaining_turn_budget": 4,
  "action_token_ids": {"answer": 100, "search": 101},
  "answer_branches": [
    {"seed": 11, "f1": 0.7, "extra_searches": 0, "status": "ok"},
    {"seed": 12, "f1": 0.7, "extra_searches": 0, "status": "ok"}
  ],
  "search_branches": [
    {"seed": 21, "f1": 0.85, "extra_searches": 2, "status": "ok"},
    {"seed": 22, "f1": 0.85, "extra_searches": 2, "status": "ok"}
  ],
  "lambda_cost": 0.03,
  "tau": 0.05,
  "delta": 0.09,
  "q_search": 0.8581489,
  "valid": true
}
```

`100/101` 不是实际 tokenizer ID，严禁照抄。分支记录还需保留终止原因、答案抽取、tokens、检索步骤、耗时、错误类型以及 schema 指向的本机完整 token 数据。gold 独立存放于 scorer 可访问区域，不能整个记录对象直接送进 student renderer。

每次训练至少记录：`L_GRPO/L_dec`、有效状态数、q/Delta 分布、每类动作概率、两条分支 F1 与方差、双零率、Search 有效执行率、标签熵、probe 覆盖/失败率、policy version 差、Query 重复率、平均搜索数、预算耗尽率、答案率、各模块耗时、GPU 小时与显存。

产物位于唯一新 run 目录，含 `run_config.yaml`、`raw_output.jsonl`、`trajectories.jsonl`、`metrics.json`、`badcases.jsonl`、`experiment_note.md`，另加 `decision_states.jsonl`、`branch_probes.jsonl`、`runtime_manifest.json`、`split_ids.json`、`gate_decision.json`。大数据、原始私有文本、checkpoint、日志、密钥不入 git。

## 7. 代码改动清单：保持可独立回退

在干净的项目分支 `codex/cost-sensitive-decision-distillation` 开发；如果分支已存在先检查，不强行重建。保留普通 GRPO 的关闭路径。

| 建议新增／修改位置 | 工作内容 | 交付验收 |
|---|---|---|
| `03_sapr_rag/scripts/grpo/build_decision_targets.py`（新增） | F1/成本聚合、q 构造与校验，可离线调用 | 纯 CPU 数值测试 |
| `03_sapr_rag/scripts/grpo/run_decision_probe.py`（新增） | 抽样、同状态 A/S fork、版本锁、诊断报告 | 分支隔离、预算和失败语义 |
| `03_sapr_rag/scripts/grpo/plugin.py` 或独立导入模块 | 默认关闭的新 decision scheduler，保存全部决策前缀 | 原 scheduler 行为不变 |
| `03_sapr_rag/scripts/grpo/build_decision_warmstart.py`（新增） | 仅协议行为克隆数据 | 无 gold/未来泄漏、双动作覆盖 |
| `03_sapr_rag/scripts/grpo/run_decision_distill.py`（新增） | 配置解析、dry-run、阶段执行与预算检查 | 不隐式覆盖输出、不启动未授权服务 |
| `03_sapr_rag/scripts/eval/run_decision_eval.py`（新增） | 部署同协议、严格预算、成本记录 | 评测时不读取 gold 做动作决策 |
| `04_experiments/run_configs/run_YYYYMMDD_decision_*.yaml`（新增） | 各组冻结配置 | 组间差异可机器比较 |
| `docs/cost_sensitive_decision_distillation_report.md`（新增） | 事实、失败、门槛判断、最终结果 | 不篡改旧 tracker 结论 |

配套 ms-swift 需要检查并按实际接口修改：

- `swift/rl_core/data.py`：结构化 decision metadata，不能被 collator 丢掉；不要凭文档猜类名。
- `swift/rlhf_trainers/grpo_trainer.py`：对状态做可微前向、二元 CE、动作位 old/current/ref logprob 处理、全局有效状态计数。
- `swift/rlhf_trainers/args_mixin.py`：默认关闭的参数和互斥检查。
- `swift/infer_engine/grpo_vllm_engine.py`／`swift/rollout/multi_turn.py`：仅在现有接口无法保留分段 token/logprob 时最小修改。
- 新增独立单测；更新迁移说明和增量 patch，记录实际部署补丁哈希。

不可把 CE 混进已有 teacher advantage，然后仍声称实现了本文公式。建议用训练模型对独立 state microbatch 前向，在决策前缀最后位置取 A/S logits；这样无需让 q 依赖完整未来 trajectory。prefix batch 的 token budget 必须限制并记录，不能 OOM 后悄悄丢长状态。

DDP／梯度累积：定义一个 optimizer update 中所有有效状态的总和除以总数；不能平均各 rank 的均值而给小 rank 更大权重。考虑 DDP 已有 world-size 平均和 trainer 累积缩放，写单卡／多卡等价测试。某 rank 零状态时仍参加必要 collective，不能死锁；全局零状态时 `L_dec=0`，不除零。

关闭新功能且保持旧协议时，GRPO 前向、advantage 和 mask 应与原实现一致；新协议 P0 则是单独的 matched baseline，不能要求它与旧协议逐位相同。

## 8. 训练伪代码与配置契约

```python
# 伪代码：不是已有 Python API。
for rollout_window in training:
    snapshot = freeze_rollout_version(student)
    trajectories = sample_natural_grpo_groups(snapshot, train_questions)
    states = select_states_before_observing_scores(trajectories, fraction=0.25)
    with no_grad_and_same_rollout_version(snapshot):
        probes = fork_answer_and_search(states, samples_per_action=2)
        targets = build_targets(probes, lambda_cost=0.03, tau=0.05)
    release_rollout_version_lock()

    # probes 不参与下面的组内均值/方差；q 在本窗口固定、无梯度。
    rewards = original_reward(trajectories)
    for optimizer_batch in existing_optimization_schedule(trajectories):
        loss_grpo = matched_grpo_loss(optimizer_batch, rewards)
        logp_binary = student_decision_logsoftmax(targets.assigned_prefixes)
        loss_dec = global_state_mean(-(targets.q.detach() * logp_binary).sum(-1))
        backward_with_correct_accumulation(loss_grpo + 0.1 * loss_dec)
    discard_targets_for_this_window()
```

必须把 targets 分配到具体 optimizer update，避免在窗口里的每个微批都重复整批 CE。原框架若多 epoch 重用 rollout，必须显式记录 q 的同等复用策略；不自行增加更新轮数。

建议配置草案如下，`null` 必填项须在 preflight 后填好；解析器对未解析环境变量、未知字段和缺失预算直接报错。以下不是可以直接传给现有 shell 脚本的配置。

```yaml
run_name: decision_distill_pilot_SEED_DATE
date: null
server: null
gpu: null
model: null
adapter: null
dataset: null
retriever: null
corpus: null
top_k: 3
max_steps: 250                    # optimizer steps，不是搜索轮数
temperature: null                 # 原内容采样温度，从匹配 B 配置填入
num_candidates: 8                 # GRPO G，不是 probe K
method: cost_sensitive_decision_distillation
baseline: matched_explicit_decision_grpo
output_path: null
log_path: null
metric_path: null
note: expected_result_pending_execution
seed: 42
max_gpu_hours: null
max_wall_hours: null
train_split_hash: null
validation_split_hash: null
original_protocol_manifest: null
max_decision_turns: 6
max_search_actions: 5
max_context_tokens: null
max_completion_tokens: null
evidence_agent_manifest: null
enable_opsd: false
enable_reward: true
advantage_mode: sequence
action_credit_mode: off
dynamic_sample: false
overlong_filter: false
loss_type: grpo
decision:
  enabled: true
  protocol: explicit_single_token
  action_token_ids: null
  selector_temperature: 1.0
  selector_top_p: 1.0
  probe_prompt_fraction: 0.25
  states_per_prompt: 1
  branch_samples_per_action: 2
  lambda_cost: 0.03
  target_temperature: 0.05
  loss_coefficient: 0.1
  targets_lifetime: one_rollout_window
  max_probe_concurrency: null
  deployment_action: argmax
```

`top_k=3`、最多 5 次搜索／6 个决策位是待核对的协议起点；如果权威基线不同，先统一更新所有组的 manifest，不直接更换历史评测口径。上下文总长和单轮／整条 completion 长度不是同一个量，必须分别核实预算含义。

完成实现后的拟定 CLI 契约（**当前仓库尚无这些命令，不可现在照抄运行**）：

```bash
python 03_sapr_rag/scripts/grpo/run_decision_distill.py --config CONFIG.yaml --phase preflight --dry-run
python 03_sapr_rag/scripts/grpo/run_decision_probe.py --config CONFIG.yaml --mode diagnostic
python 03_sapr_rag/scripts/grpo/run_decision_distill.py --config CONFIG.yaml --phase smoke
python 03_sapr_rag/scripts/grpo/run_decision_distill.py --config CONFIG.yaml --phase pilot
python 03_sapr_rag/scripts/eval/run_decision_eval.py --config CONFIG.yaml --checkpoint CHECKPOINT
```

`preflight` 只读且不启动 GPU 服务；其余 phase 必须有已确认的预算和资源。每个 CLI 支持 `--help`、配置展开、唯一输出目录、失败退出码和可审计 resume；复用 run 仅在配置/代码一致且明确 resume 时允许。

## 9. 必须通过的测试，不能以 loss 下降代替

### 9.1 CPU 数值与协议单测

1. 上述两个 q 数值正确；Delta=0 时 q=.5；tau<=0、NaN、越界 F1 拒绝；成本增加时 q_S 不增。
2. CE 与 KL 差固定 H(q)，自动微分与 `p-q` 一致；q、分支生成无梯度，学生梯度有限且非零。
3. 只在合法公共前缀取动作 logits；动作 token ID 验证、left padding、截断、多轮、空 batch、预算终点都覆盖。
4. 采样二元概率与 trainer old/current/ref logprob 一致；同权重时 ratio=1。注入和 observation token 的 mask 为 0。
5. 改未来 observation／分支返回文档，不改变原 `h_t`；在 gold 中放哨兵字符串，student/probe 输入不得出现。
6. 不同 fork 的 history、seen_queries、预算互不污染；主轨迹生成结果在 probes 开关下保持相同种子行为或解释可复现的 RNG 隔离。
7. scorer 在大小写、标点、冠词、多 gold、空答案、标签异常上与现有评测一致；F1 与百分制不混用。
8. probe 数量不改变 GRPO G、reward 归一化、训练自然轨迹数量；模型失败保留，服务失败另记。
9. 缓存必须因策略、预算、prefix 或检索版本改变而失效；无合法状态不伪造标签。
10. 关闭功能回归通过；不同 rank 状态数、梯度累积、全零状态与单进程参考数值一致。

### 9.2 GPU smoke：先 1–2 optimizer steps

- 实际 adapter 加载进训练和 rollout；更新后同步成功，可用固定输入输出/logprob 与权重校验，而不只是看日志“loaded”。
- 输出至少一例 A 优、一例 S 优的真实 probe；没有某类则报告缺失，不编造。
- decoder 的 token IDs 与重算 logits 对齐；loss backward、optimizer step 后学生变化，冻结生成快照无梯度。
- 训练关 probes 的部署路径可以单独启动，不需要 gold、teacher 字段或训练数据。
- 记录显存/吞吐/额外调用量。任何 token 对齐、版本锁、DDP 或泄漏测试失败即停止，不开 pilot。

## 10. 分阶段执行与继续／停止门槛

时间安排是按依赖顺序的时间盒，不是对任何 GPU 的耗时承诺；总预算由用户确认。

### 阶段 A：协议与小规模诊断（建议第 1–2 天）

1. 完成预检，复制 B manifest，先冻结 training/validation 划分，读最近论文，确认不是已有方法的直接重命名。已用于既往 SFT 的题即使从本轮 RL 留出，也不能宣称模型从未见过，须记录前序暴露。
2. 先完成第 7 节中协议、warmstart、probe 所需的最小实现和对应测试；完成协议热启动与格式对照检查，再用新协议模型构造主诊断。阶段 B 才接完整联合 trainer。可额外审计旧轨迹，但不要把两种策略的 branch values 混在一起。
3. 从 training split 固定分层抽取 450 个问题（每数据集 150），诊断模式明确设置 `probe_prompt_fraction=1.0`，每题按第 5 节的轨迹/状态抽样规则取一个合法状态，K=2。训练仍使用 0.25。无合法状态不补抽成功样本；记录实际覆盖数。
4. 对预先随机选定的 20% 状态追加独立 K=2 重采样，检查 Delta 符号/q 稳定性，不能按首轮“好看”结果选复测状态。
5. 统计双零率、`Delta>=.025`／`Delta<=-.025` 比例、状态轮次、模型原选择与 q 偏好的不一致、S 的额外成本、超时与格式失败，并展示正例/反例。

默认继续门槛：有效状态至少 300，基础设施未解决失败不超过 5%，正/负明确效用区各至少 10%，复测的两次均明确有方向的状态至少 30 个且其符号一致率至少 70%，协议门槛通过。样本不足记证据不足，不能用空分母过关。满足条件只说明值得 pilot，不证明算法有效。单侧信号、大量双零或不稳定时，不以无限增大 K 救方案；报告原因和增加采样的预算，申请一次明确调整或停止。

### 阶段 B：实现与 smoke（建议第 2–3 天）

完成第 7–9 节；拿到训练显存、速度、250-step 预算估算。用关闭功能和新协议 P0 做回归，再跑 P3 的 1–2 steps。产物标记 `debug_result`，不可进入论文主表。

### 阶段 C：250-step 匹配 pilot（建议第 3–4 天）

先运行 P0/P1/P3；三组从同一协议 warmstart 起点，固定 seed、reference、optimizer 初始化、数据顺序、预算和唯一 250-step 选择点。禁止为每组各挑最优 checkpoint。pilot validation 从训练源单独留出并完全排除训练，建议每集 500 题，固定 ID；资源不足先修改已批准方案，不能边看结果边缩样本。

预注册工程门槛满足其一才考虑扩展，均要求相对 P0 和 P1 判断，且没有单集 F1 下降超过 1 个百分点：

- 质量路线：macro F1 至少 +0.5 个百分点，平均检索次数增加不超过 5%。
- 效率路线：平均检索次数至少减少 10%，macro F1 下降不超过 0.3 个百分点，且 F1 配对差值的 95% 置信区间下界不低于 -0.003（F1 用 0–1 标度）。

这些只是预算决策，不是录用标准。置信区间太宽时标为证据不足，不挑另一个 seed 宣布成功。确认优势来自软标签前，还须补 P2 与必要消融。

### 阶段 D：机制对照与全量（建议第 4–6 天及获准延长期）

只有 pilot 合格且用户同意预算，才运行 P2、消融以及统一 1000-step 扩展。根据已冻结的 validation 规则锁定配置/checkpoint，最后做三集完整评测。优先补 P0/P3 第二 seed，而不是无限扫超参。历史 B 的全量结果只能背景参考，不能冒充这次匹配训练对照。

如果已有官方 dev 被长期用于选择方法，应如实称 validation；不要把它包装成 untouched test。最终测试集不可参与构造 soft labels、阈值选择或 checkpoint 选择。

## 11. 对照实验：把收益来源拆清楚

| 组 | 新动作协议 | 目标 | 要回答的问题 |
|---|---|---|---|
| B-legacy | 否 | 历史/复现 GRPO | 历史背景，协议变化影响 |
| P0 | 是 | GRPO，beta_dec=0 | 同协议、同热启动的主要基线 |
| P1 | 是 | GRPO；轨迹 reward 另扣 lambda_cost*搜索数 | 简单搜索惩罚是否已经足够 |
| P2 | 是 | GRPO + 同分支数据的局部动作 policy-gradient | 收益来自额外分支评估，还是软分布投影 |
| P3 | 是 | GRPO + 本文二元软标签 CE | 完整候选方法 |
| P4 | 是 | 不做联合蒸馏，训练源拟合质量/效用预测与早停阈值 | 简单 stopping controller 与相关工作近邻 |

P1 保留基础 reward 其他项，明确全轨迹成本惩罚与 P3 的局部成本标签不同。不要给 P1 同时增加额外长度、重复或覆盖惩罚。

P2 使用与 P3 同样的状态抽样、K、冻结策略和 U_A/U_S，但不用 sigmoid 目标。可实现如下**预注册的局部 policy-gradient 对照**：

```text
b(h) = sum_a p_old(a|h) * U_a(h)
A_local(a,h) = stop_gradient(U_a(h) - b(h))
r_a = p_theta(a|h) / stop_gradient(p_old(a|h))
L_local_RL = -sum_a stop_gradient(p_old(a|h))
                 * min(r_a*A_local, clip(r_a,1-eps,1+eps)*A_local)
L_P2 = L_GRPO + beta_local * mean_h L_local_RL
```

这是本文拟定的二动作枚举 PPO-style 对照，不冒称标准 GRPO 或已有论文方法。数值稳定用 log-space；action support 始终相同。P2/P3 都获得相同数量额外分支评估，但训练后状态自然不同，不能假称逐状态数据完全相同。两者辅助梯度尺度不同，`beta_local` 与 `beta_dec` 不能因数值相等就说公平；只允许在 training calibration states 上按预注册梯度范数匹配，或给两者相同且很小的 validation 调参预算，并披露。

P4 优先忠实复现已有论文的公开方法；若时间不够，只实现并标注“简化 stopping baseline”，不能挂论文名当复现结果。预测器仅使用部署可见信息，训练标签仅来自 training split 的 probes；推理调用开销全部计入。验证集用来定阈值，不用测试答案。

首轮必要消融：P3 `lambda_cost=0`（质量监督但无成本）、`beta_dec=0`（即 P0）、hard label（Delta>0 为 S，平局按预定规则 A）；资源允许再做 K=1 vs 2 或小范围 tau 敏感性。不同时引入置信度门控、更多奖励、不同教师、树搜索和动态 K。

至少报告两种预算口径：相同自然 rollout/optimizer steps，以及实际总 GPU 小时/生成 token 数。P3 用了额外 probe，不能声称与不 probe 的 P0 算力相同；若主张训练效率，必须在同总计算预算下再比较。

## 12. 评测与部署：什么才算有效

部署流程：`问题和已有 evidence -> 同一学生的 A/S 选择 -> S 时生成 query并检索 -> A 时生成答案`。默认 A/S 用 argmax、平局选 A，内容 decoding 沿用统一评测配置；所有新组一致。训练软目标只影响学到的参数，推理不计算 Delta/q、不读取 gold、不生成两条候选分支。

必须报告：

- 每集及宏平均 EM/F1，Cover-EM 仅作次要兼容指标；样本数与失败比例。
- 每题逻辑搜索次数、实际检索 RPC、重复 query 率、预算耗尽率、回答率、完整 reasoner+evidence+selector token 数、延迟 P50/P95。
- 训练自然 rollout/probe/反向计算分别的成本，峰值显存及总 GPU 小时。
- F1—平均搜索数 Pareto 图；不要只挑更短的轨迹或更便宜的例子。
- 以问题 ID 配对 bootstrap，建议 10000 次；宏平均按数据集分层重采样。seed 方差另报，问题 bootstrap 不能替代多 seed。
- 至少各展示改善、无变化和退化案例，区分“选择不当”“query 不好”“evidence 丢信息”“已有证据但答案生成失败”。

长度/检索下降但 F1 明显下降，是过度停止而非胜利。F1 上升但推理成本大涨，只能按权衡报告。增加训练成本换部署收益可能有价值，但必须计算或讨论部署多少次才能抵消额外训练开销。

证据覆盖是辅助诊断，不作本方案的答案质量替代。模型闭卷记忆答对可以获得 F1；因此本方法优化的是 benchmark 回答质量和成本，不自动证明答案有证据支撑。若论文要主张可信 grounding，另设可靠的引用/证据支撑评测并披露成本，不能靠标题命中认定。

## 13. 失败退路与禁止的“补救”

| 观察 | 判断与动作 |
|---|---|
| 协议热启动就退化或 token 接口做不对 | 停止联合训练；修正确性或回到原协议，不能把格式噪声当算法失败/成功 |
| 大多数 A/S 都为零分，S 也无提升 | 主要瓶颈可能是检索/生成能力；不继续强化“少搜”，退回 B，整理诊断 |
| K=2 的效用方向不稳定 | 记录不确定性；只能在明确预算内批准一次更大 K 诊断，不无限加采样 |
| P3 不如 P1 | 简单成本惩罚足够或新监督无用；不声称蒸馏有独立创新收益 |
| P3 不优于 P2/P4 | 分支数据或早停机制可能解释收益；缩小论文主张，必要时停止算法路线 |
| 250-step 有效，1000-step 失效 | 如实记录泛化/稳定性失败，不能仅选 pilot 最优点掩盖全量结果 |
| OOM、服务故障、同步失败 | 保存现场、停止当前阶段，报告资源或实现问题；禁止悄悄变更 baseline |

停止主线后恢复已有 B 推理配置，并保留新代码默认关闭。需要继续研究新的 query/evidence 教师、门控或训练顺序时另立假设、另请用户确认；不自动扩大当前任务。

## 14. 相关工作与新颖性边界

截至 2026-09-20 核对到以下直接近邻；这是立项检查，不是完整查新。

- [Predicting Partial Answer Quality and Utility in Agentic Retrieval-Augmented Generation](https://arxiv.org/abs/2609.16453)：已研究逐轮强制回答探测、答案质量/效用预测与早停。arXiv 页面作者注明 CIKM 2026 full paper 接收。不能把“中间回答探测”或“依据收益停止检索”本身当首次贡献。需进一步读全文和代码，对比同状态双动作采样、显式成本软目标以及联合训练是否已被覆盖。
- [Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](https://aclanthology.org/2026.acl-long.1751/)：ACL 2026 已收录，直接涉及 Agentic RAG 的蒸馏与策略优化结合。不能把“RL+蒸馏用于 RAG”当作独立创新；具体 PPO/GRPO 和 loss 需按原文区分。
- [Know When to Stop, Where to Restart: Accelerating Multi-Turn Agentic On-Policy Distillation](https://arxiv.org/abs/2609.14636)：STRIDE 研究 OPD 训练 rollout 的停止/重启与效率，当前所查页面为预印本。训练加速的 stop 与本文部署时 Search/Answer stop 目标不同，但都应比较监督与计算成本。

第一阶段还须检索 value-of-information retrieval、cost-sensitive stopping、counterfactual action distillation、branch policy improvement 等近邻。发现实质重复时立即告知用户，不通过换缩写包装原创。

若结果支持，可准确写成：“我们研究在学生访问状态上估计检索相对立即回答的成本敏感局部效用，并将其投影为二元决策监督；在匹配协议和计算成本分析下验证其效果。”不能预先写“首次”“理论最优”“保证改进”或“足以录用”。

## 15. 交付清单与给执行 AI 的启动指令

完成每阶段都更新报告：配置与代码 SHA、执行命令、资源、输入/输出路径及哈希、真实状态、门槛判断、下一步。区分 `expected_result`（本文计划）、`debug_result`（smoke）、`real_result`（完整可审计实验）、`failed_run`。

最终交付至少包括：

1. 可运行实现、默认关闭开关、单测结果、增量 ms-swift 补丁和迁移说明。
2. P0/P1/P2/P3 的精确配置差异、shared warmstart、split IDs 与不可变 manifest；缺组如实说明。
3. 原始结果本机路径、公开允许的小型汇总、配对统计、成本表、Pareto 图、失败案例。
4. `docs/cost_sensitive_decision_distillation_report.md`，明确支持/不支持哪些假设，是否值得继续。
5. 重现命令与已知限制。未做 GPU 实测的部分明确写“未运行”，不填预计指标当实测值。

可直接发给服务器端 AI：

> 请阅读 AGENTS.md 和 docs/cost_sensitive_decision_distillation_execution_plan.md，先核对当前 SAPR-RAG、ms-swift 补丁、真实基线协议、checkpoint、检索服务与可用 GPU。先提交预检和阶段预算，不直接启动长训练。按计划实现显式二元决策、同状态 A/S 分支效用标签及 GRPO+decision CE，保留旧功能关闭回归并完成所有正确性测试。获得资源与预算确认后，依次做诊断、1–2 步 smoke 和匹配 250-step pilot；只有通过预注册门槛才申请全量实验。不得泄露 gold、把 probe 混进 GRPO 组、改变对照口径或选择性报告结果。逐阶段写报告，方法无效也应如实交付。
