# 文献阅读笔记规范

本规范吸收 `juliye2025/evil-read-arxiv` 中 `paper-analyze` 与 `extract-paper-images` 的做法，并针对本仓库的 Agentic RAG 研究目标做了改造。目标不是生成泛泛摘要，而是形成可直接服务周报、中期报告、论文 related work、方法设计和实验复现的中文深度笔记。

## 1. 单篇论文阅读流程

每篇论文按以下顺序处理：

1. 确认一手来源：优先使用 ACL Anthology、OpenReview、arXiv、官方项目页和官方 GitHub。
2. 提取论文元信息：标题、作者、年份、venue、paper/pdf/code/dataset 链接、任务、数据集、模型、检索器。
3. 获取图表：优先按 caption 定位并裁剪关键图表，即 Figure 1 / Figure 2 / Table 1 / Table 2；普通 PDF 页面截图只能作为排查材料，不能作为笔记正文的默认图。
4. 阅读正文结构：至少覆盖摘要、引言、方法、实验、消融、局限或附录。
5. 写中文深度笔记：每个关键判断标注 `[论文明确提出]` 或 `[基于方法/实验设置推断]`。
6. 映射到本课题：说明该论文支持哪个共性问题、暴露哪个逻辑缺口、可转化成 SAPR-RAG 的哪个模块。

## 2. 图表处理规范

关键图表保存到：

```text
01_literature/paper_notes/images/<note_stem>/key_figures/
```

每个关键图表目录必须包含：

```text
index.md
```

`index.md` 记录图片文件名、匹配到的 caption、页面和大小。笔记正文中优先引用：

1. Figure 1：motivation / problem setting；
2. Figure 2：method / framework；
3. Table 1：main results / dataset statistics；
4. Table 2：ablation / diagnostic results。

若自动裁剪失败，应人工检查 PDF 并补图；不要用无意义的首页截图、logo、箭头图标或随机页面图替代关键图表。

图注必须说明：

- 图表来自哪篇论文；
- 图表在论文中承担什么论证作用；
- 该图与本课题的 query/evidence/stop reward 或 failure diagnosis 有什么关系。

## 3. 深度笔记固定结构

每篇笔记使用以下结构：

```markdown
# Paper Title

## 0. 阅读结论
## 1. 核心信息
## 2. 摘要与问题定义
## 3. 图表速读
## 4. 方法拆解
## 5. 实验设计与结果解读
## 6. 论文贡献、局限与证据强度
## 7. 与本课题的关系
## 8. 可复现与代码阅读线索
## 9. 可用于写作的中文表述
## 10. 后续行动
```

## 4. 深度分析要求

### 0. 阅读结论

用 3 到 5 条 bullet 给出结论：

- 这篇论文在谱系中的位置；
- 它真正解决了什么；
- 它没有解决什么；
- 对 SAPR-RAG 的直接启发。

### 2. 摘要与问题定义

必须回答：

- 作者认为旧方法的问题是什么？
- 论文把问题定义成 reward、policy、retrieval、evidence、faithfulness、benchmark 还是 data synthesis？
- 该问题是否与 ReasonRAG badcase 同构？

### 4. 方法拆解

至少拆成：

- 输入输出；
- 核心状态或轨迹表示；
- 动作空间或模块；
- 训练信号、奖励或标注方式；
- 推理流程；
- 与 ReasonRAG 的接口差异。

若论文有公式，使用 Markdown LaTeX：

```markdown
行内公式：$U(d \mid q, s_t)$

块级公式：
$$
\theta^* = \arg\min_\theta L(\theta)
$$
```

### 5. 实验设计与结果解读

不能只写“实验有效”，必须说明：

- 数据集为什么能验证该方法；
- baseline 是否公平；
- 指标是否只看最终答案，还是能看过程质量；
- 主结果说明了什么；
- 消融实验证明了哪个组件；
- 还有哪些没有被验证。

### 6. 证据强度

使用三类标签：

- `[论文明确提出]`：论文正文、摘要、实验或结论直接说明。
- `[基于方法/实验设置推断]`：作者没有直接说，但从方法边界可稳妥推出。
- `[本课题 badcase 对齐]`：来自 ReasonRAG 复现与 badcase 分析，不应伪装成论文原文结论。

## 5. 与 SAPR-RAG 的映射方式

每篇笔记都要明确映射到以下至少一项：

- Query Reward
- Evidence Reward
- Stop Reward
- Step Entailment / Faithfulness Reward
- Repair Mechanism
- Failure Bank / Hop-Level Evaluation
- State-Conditioned Reranker / Retriever Distillation

## 6. 写作素材要求

“可用于写作的中文表述”不是翻译原文，而是生成可直接放进周报、中期报告或论文 related work 的中文句子。写作素材必须区分：

- 直接证据层：论文已经明确证明或指出的问题；
- 综合推断层：本课题基于多篇论文共同边界提出的逻辑缺口。

## 7. 禁止事项

- 不要只复述摘要。
- 不要把方法名误写成论文标题。
- 不要把 `[基于方法/实验设置推断]` 写成 `[论文明确提出]`。
- 不要长段复制论文原文。
- 不要只放图不解释图。
- 不要只写“对我的研究有启发”，必须说明启发到哪个模块和如何落地。
