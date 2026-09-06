#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 实验 B：E14 canonical-answer SFT -> matched GRPO-only control（关闭 teacher）。
# 与 E16（run_canonical_sft_multi_opsd_s1000.sh）唯一差异：ENABLE_OPSD=false，
# 数据集换为已剥离 teacher 字段的同源三源数据。其余起点/reward/rollout/
# Evidence Agent/步数/采样全部保持一致，用于隔离出 OPSD teacher 的独立贡献。
exec env \
    INIT_ADAPTER="$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft_canonical_fp16/checkpoint-4150" \
    ENABLE_OPSD=false \
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
    RUN_NAME=grpo_control_canonical_sft_3src_s1000_20260906 \
    ROLLOUT_PORT=8030 \
    VLLM_GROUP_PORT=21240 \
    ROLLOUT_GPU=7 \
    TRAIN_DEVICES=2,3,4,5,6 \
    NPROC_PER_NODE=5 \
    bash "$SCRIPT_DIR/launch_action_scoped_opsd_worker.sh"
