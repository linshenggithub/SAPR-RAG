#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 实验 E17：Evidence-Attributed GRPO（动作级证据信用分配）。
#
# 设计为实验 B（run_canonical_sft_grpo_control_s1000.sh，GRPO-only）的严格单变量
# 增量：唯一新增的是 ACTION_CREDIT_MODE=query_evidence 打开的“逐轮证据增益”附加优势。
# 其余全部与 B 对齐——同一 E14 canonical SFT 起点、同一剥离 teacher 字段的三源数据、
# 同一 reward（sapr_f1/relevance/format）、同一 rollout / Evidence Agent / 步数 / 采样。
# 保持 ENABLE_OPSD=false 以隔离出“动作级信用”本身的贡献（不引入 OPSD teacher）。
#
# 机制（见 ms-swift swift/rl_core/action_credit.py 与 docs/ms_swift_local_patches.md）：
#   A_token(t) = A_outcome(seq)                 # 现有 GRPO 序列优势，广播到全部 token
#              + coef * A_action(turn(t))       # 新增：只加到对应 <query> 轮的 token
# 其中 A_action 来自 sapr_query_evidence_gain（reward 权重恒 0，仅把逐轮
# “本轮新覆盖 gold evidence 数 / gold 总数”向量写入 rollout_infos），在 prompt 组内
# 按 turn-slot 归一化后 scatter 到该 query 轮的 token。
#
# COEF pilot 建议先扫 0.1 / 0.2 / 0.4（改 ACTION_CREDIT_COEF 覆盖），本脚本默认 0.2。
exec env \
    INIT_ADAPTER="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft_canonical_fp16/checkpoint-4150" \
    ENABLE_OPSD=false \
    ENABLE_REWARD=true \
    OPD_USE_GRPO_ADVANTAGE=true \
    ACTION_CREDIT_MODE=query_evidence \
    ACTION_CREDIT_COEF="${ACTION_CREDIT_COEF:-0.2}" \
    ACTION_CREDIT_SCALE="${ACTION_CREDIT_SCALE:-group_turn}" \
    ACTION_CREDIT_INFOS_KEY=query_evidence_gain \
    TEACHER_ACTION_SCOPE=multi \
    ENABLE_TRUNCATION_REWARD=false \
    ENABLE_EVIDENCE_AGENT=true \
    MAX_STEPS=1000 \
    MAX_COMPLETION_LENGTH=4096 \
    PER_DEVICE_BATCH_SIZE=2 \
    GRADIENT_ACCUMULATION_STEPS=4 \
    STEPS_PER_GENERATION=8 \
    NUM_GENERATIONS=8 \
    SAVE_STEPS=250 \
    SAVE_TOTAL_LIMIT=8 \
    DATASET="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_musique_train_grpo_noteacher.jsonl" \
    RUN_NAME="action_credit_canonical_sft_3src_s1000_coef${ACTION_CREDIT_COEF:-0.2}_20260906" \
    ROLLOUT_PORT=8030 \
    VLLM_GROUP_PORT=21240 \
    ROLLOUT_GPU=7 \
    ROLLOUT_PORT2=8031 \
    VLLM_GROUP_PORT2=21242 \
    ROLLOUT_GPU2=1 \
    TRAIN_DEVICES=2,3,4,5,6 \
    NPROC_PER_NODE=5 \
    bash "$SCRIPT_DIR/launch_action_scoped_opsd_worker.sh"
