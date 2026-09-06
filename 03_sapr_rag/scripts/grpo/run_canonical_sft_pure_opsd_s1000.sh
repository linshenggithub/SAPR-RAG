#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 实验 D：E14 canonical-answer SFT -> 纯分动作 OPSD（无 RL reward）。
# 归因对照：与 E16 (run_canonical_sft_multi_opsd_s1000.sh) 唯一差异是关闭 GRPO 任务
# reward 且丢弃 GRPO 组内 advantage，只保留 Query/Answer 分动作 teacher log-ratio 信号。
# 这对应 OPSD 原论文 (arXiv:2601.18734) 的纯自蒸馏形式。
exec env \
    INIT_ADAPTER="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft_canonical_fp16/checkpoint-4150" \
    ENABLE_OPSD=true \
    ENABLE_REWARD=false \
    OPD_USE_GRPO_ADVANTAGE=false \
    TEACHER_ACTION_SCOPE=multi \
    TEACHER_QUERY_KL_COEF=0.01 \
    TEACHER_EVIDENCE_KL_COEF=0.0 \
    TEACHER_ANSWER_KL_COEF=0.03 \
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
    DATASET="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_musique_train_multi_opsd.jsonl" \
    RUN_NAME=pure_opsd_canonical_sft_q001_a003_3src_s1000_20260906 \
    ROLLOUT_PORT=8030 \
    VLLM_GROUP_PORT=21241 \
    ROLLOUT_GPU=7 \
    TRAIN_DEVICES=2,3,4,5,6 \
    NPROC_PER_NODE=5 \
    bash "$SCRIPT_DIR/launch_action_scoped_opsd_worker.sh"
