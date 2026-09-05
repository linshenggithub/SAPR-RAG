#!/usr/bin/env bash
# 等 canonical-SFT→DPO 训练结束后，自动用 6 卡评测三数据集，并生成与 E14 / SFT+DPO 的对比表。
# 依赖：worker 上 tmux session `dpo_canonical` 为训练进程；GPU0 retrieval daemon 保持运行。
set -euo pipefail

ROOT=/mlx_devbox/users/mayi.summer/playground/SAPR-RAG
DPO_OUT=$ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft_canonical_dpo
EVAL_OUT=$ROOT/data/eval_results/sft_canonical_dpo_3src_6gpu_20260905
EVAL_LAUNCH=$ROOT/03_sapr_rag/scripts/eval/launch_sft_canonical_ckpt4150_6gpu_eval.sh
TRAIN_SESSION=dpo_canonical
LOG=$EVAL_OUT/orchestrator.log

mkdir -p "$EVAL_OUT"
echo "[orch-start] $(date -Is)" | tee "$LOG"

# 1. 等训练 session 结束
while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  echo "[wait-train] $(date -Is) dpo still running" | tee -a "$LOG"
  sleep 120
done
echo "[train-done] $(date -Is)" | tee -a "$LOG"

# 2. 选最新 checkpoint（DPO 1 epoch，取 step 最大的）
CKPT=$(ls -d "$DPO_OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [[ -z "${CKPT:-}" || ! -s "$CKPT/adapter_model.safetensors" ]]; then
  echo "[error] no valid DPO checkpoint under $DPO_OUT" | tee -a "$LOG"
  exit 2
fi
echo "[ckpt] $CKPT" | tee -a "$LOG"

# 3. 复用 6 卡评测脚本
CKPT="$CKPT" OUT="$EVAL_OUT" \
  GPUS_CSV=1,2,3,4,5,6 NUM_SHARDS=6 \
  SESSION_PREFIX=dpo_canonical_6gpu STOP_OLD=false \
  bash "$EVAL_LAUNCH" | tee -a "$LOG"

# 4. 等评测 monitor 完成（三数据集 metrics.json 齐全）
while true; do
  done_all=1
  for ds in hotpotqa 2wikimultihopqa musique; do
    [[ -s "$EVAL_OUT/$ds/metrics.json" ]] || done_all=0
  done
  [[ "$done_all" == 1 ]] && break
  echo "[wait-eval] $(date -Is)" | tee -a "$LOG"
  sleep 120
done
echo "[eval-done] $(date -Is)" | tee -a "$LOG"

# 5. 生成对比表
python "$ROOT/03_sapr_rag/scripts/eval/compare_canonical_dpo_vs_baselines.py" \
  --dpo_dir "$EVAL_OUT" \
  --output "$EVAL_OUT/comparison.md" 2>&1 | tee -a "$LOG"
echo "[all-done] $(date -Is) comparison=$EVAL_OUT/comparison.md" | tee -a "$LOG"
