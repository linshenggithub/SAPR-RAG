# Gate 0: Typed vs Scalar Branch Selection Analysis

## Goal
在已有的 ReasonRAG MCTS 推理结果上，验证 typed transition evaluation 是否能区分标量 PRM 无法区分的分支。

## Key Finding (Pre-analysis)
- reward_data 中 ~72.5% 的分支点，children 的 Q 值完全相同
- 标量 PRM 在大多数分支点上无法做出有信息量的选择

## Data
- Source: `/home/mayi/RAG/ReasonRAG/output/hotpotqa/reward_data{0,1,2,3}.json`
- Format: MCTS tree with Q values, rewards, children_ids per node
- Total: ~6500+ trajectories with ~10000+ branch points

## Steps
1. `parse_trees.py` - 解析 MCTS 树，提取分支点
2. `compute_phi_q.py` - 计算 φ_q（NER-based，不需要 LLM）
3. `analyze_results.py` - 对比 typed vs scalar 的区分能力

## Metrics
- Branch discrimination rate: typed 能区分多少 scalar 无法区分的分支
- Correlation with answer quality: typed 偏好的分支是否导致更好的最终答案
