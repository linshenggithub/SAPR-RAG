# ICASSP 双方向研究入口

本目录用于并行推进两条可独立成文的研究方向。实验事实以
`docs/experiment_tracker.md` 为权威来源，本目录只记录论文命题、待验证假设、
实验矩阵和会话分工。

## 1. 两个方向

| 方向 | 核心问题 | 当前证据 | 方案 |
|---|---|---|---|
| 方向一 | 能否用无 reward 的 Action-Causal OPSD 替代离线 DPO | 纯 OPSD 在三个数据集上均高于 E15 DPO | [direction_1_action_causal_opsd.md](direction_1_action_causal_opsd.md) |
| 方向二 | 能否修复 GRPO 在多轮 Agentic RAG 中的动作信用分配 | GRPO 有效，但 20%–28% 组内零方差，序列 advantage 被广播给所有动作 | [direction_2_action_causal_grpo.md](direction_2_action_causal_grpo.md) |

## 2. 当前统一事实

统一结果口径为完整 dev 集，数值顺序为 EM / F1 / Cover-EM：

| 方法 | HotpotQA | 2Wiki | MuSiQue |
|---|---|---|---|
| E14 canonical SFT | .4373/.5513/.4748 | .4051/.4513/.4188 | .1651/.2405/.1841 |
| E15 SFT→DPO | .4140/.5281/.4304 | .4187/.4656/.4230 | .1585/.2459/.1676 |
| D SFT→纯分动作 OPSD | .4462/.5703/.5030 | .4948/.5548/.5270 | .1758/.2717/.2085 |
| B SFT→GRPO-only | .4629/.5837/.5026 | .5161/.5654/.5314 | .1808/.2794/.2056 |
| C/E16 SFT→GRPO+OPSD | .4636/.5816/.5025 | .5154/.5659/.5307 | .1837/.2786/.2089 |

当前结论：

1. 纯 OPSD 明显优于 DPO，方向一具有结果基础；
2. GRPO 是 E16 整体收益的主要来源；
3. OPSD 叠加在 GRPO 上几乎没有额外终点收益；
4. B/C/D 的三源训练问题与三个 dev 集规范化 question 精确重叠为 0；
5. 方向一仍缺 Vanilla OPSD 和动作消融；
6. 方向二必须改造动作级 advantage，不能仅增加 sequence reward。

## 3. 会话分工

### 会话 O：方向一

负责：

- DPO→OPSD 的状态分布偏移叙事；
- Vanilla OPSD、Answer-only、Query-only 和 Query+Answer 消融；
- OPSD 显著性与训练成本统计；
- 方向一论文方法和实验章节。

不负责：

- ms-swift GRPO advantage 核心逻辑；
- Action-Causal GRPO 实现。

### 会话 G：方向二

负责：

- sequence-level GRPO 的失败诊断；
- Query/Answer 动作级 advantage；
- 势函数证据增量、动作归一化和 zero-std 分析；
- ms-swift 核心实现、测试、pilot 和正式实验；
- 方向二论文方法和实验章节。

不负责：

- OPSD teacher prompt 与 OPSD 消融实验；
- 方向一论文叙事。

## 4. 并行协作规则

两个会话共享同一工作区时，必须遵守：

1. 不执行 `git reset --hard`、`git checkout --` 或覆盖他人修改；
2. 提交时只 `git add` 自己负责的文件，提交信息使用中文；
3. 方向一的新脚本统一使用 `run_icassp_opsd_*` 前缀；
4. 方向二的新脚本统一使用 `run_icassp_action_causal_grpo_*` 前缀；
5. 输出目录必须包含方向和实验编号，禁止复用现有正式目录；
6. `docs/experiment_tracker.md` 由一个指定会话统一写入，另一个会话先把结果
   写在本目录的方向文档中；
7. 修改共享核心文件前先检查 `git diff`，发现对方未提交改动时不得覆盖；
8. 两方向不得共用 rollout 端口、NCCL group port 或同一训练 GPU。

更稳妥的做法是为两个方向创建独立 git worktree；若继续共用当前目录，则必须
严格执行上述文件所有权约束。

## 5. 统一实验规范

- 起点优先统一为 E14 canonical SFT `checkpoint-4150`；
- 训练数据使用官方 train-derived 三源数据；
- 所有最终结论使用完整 dev 集；
- 记录 EM、F1、Cover-EM、回答率、max-turn、平均轮数和空 evidence 率；
- 主要差值使用 paired bootstrap 置信区间和双侧 p 值；
- 明确记录训练问题数、rollout 数、总生成 token 和 GPU 小时；
- 小样本和中间 checkpoint 只用于筛选，不进入论文主表；
- 新实验必须通过数据重叠检查并记录结果。

## 6. 决策门

### 方向一升级为主线

需要证明 Action-Causal OPSD 不仅优于 DPO，还优于 Vanilla OPSD 或
Answer-only OPSD。否则只能主张“OPSD 是 DPO 的无 reward 替代方案”。

### 方向二升级为主线

需要 Action-Causal GRPO 在完整 dev 上稳定超过 B，并同时改善动作过程指标；
若只有 reward 数值变化、最终指标和 Query 行为不变，则不构成方法贡献。

### 最终选题

两个方向完成首轮关键实验后再选择主线：

- 方向一优势：已有正结果、叙事稳定、实现风险较低；
- 方向二优势：算法创新上限更高，但需要真实增益和严格消融。
