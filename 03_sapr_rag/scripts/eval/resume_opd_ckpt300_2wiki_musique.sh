#!/usr/bin/env bash
# 补跑 OPD checkpoint-300 在 2wiki + musique 上的全量评测（hotpotqa 已完成，保留）。
# 复用已 ready 的检索服务（不重启），分片错峰启动，避免并发冲垮检索服务。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CKPT="${CKPT:-$ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/opd/opd_sft14b_failed_em_spg2_s500_20260904/v0-20260904-023807/checkpoint-300}"
OUT="${OUT:-$ROOT/data/eval_results/opd_ckpt300_fulldev_agentinfer_20260905}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8100}"
GPUS_CSV="${GPUS_CSV:-2,3,4,5,6}"
NUM_SHARDS="${NUM_SHARDS:-5}"
SESSION_PREFIX="${SESSION_PREFIX:-opd_ckpt300_resume}"
COHORT_SIZE="${COHORT_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-512}"
STAGGER="${STAGGER:-15}"

IFS=',' read -r -a GPUS <<<"$GPUS_CSV"
DATASETS=(2wikimultihopqa musique)

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPUS[$shard]}"
  session="${SESSION_PREFIX}_shard${shard}"
  tmux has-session -t "$session" 2>/dev/null && { echo "[skip] $session exists"; continue; }
  tmux new-session -d -s "$session" "
    set -euo pipefail
    cd '$ROOT'
    export NO_PROXY=127.0.0.1,localhost
    export no_proxy=127.0.0.1,localhost
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
    export CUDA_VISIBLE_DEVICES=$gpu
    echo '[shard-start]' \$(date -Is) shard=$shard gpu=$gpu | tee -a '$OUT/resume.log'
    for ds in ${DATASETS[*]}; do
      mkdir -p '$OUT'/\$ds/shards '$OUT'/\$ds/logs
      echo '[dataset-start]' \$(date -Is) shard=$shard dataset=\$ds | tee -a '$OUT/resume.log'
      python 03_sapr_rag/scripts/eval/agent_infer.py \
        --backend vllm \
        --input_jsonl data/eval/\$ds/dev.jsonl \
        --output_jsonl '$OUT'/\$ds/shards/shard_${shard}.jsonl \
        --lora_path '$CKPT' \
        --retrieval_url '$RETRIEVAL_URL' \
        --top_k 3 --max_turns 6 --max_tokens '$MAX_TOKENS' \
        --evidence_max_tokens 128 \
        --gpu_memory_utilization 0.85 --max_model_len 8192 \
        --vllm_dtype bfloat16 --vllm_disable_custom_all_reduce \
        --cohort_size '$COHORT_SIZE' \
        --num_shards '$NUM_SHARDS' --shard_id '$shard' --resume \
        2>&1 | tee '$OUT'/\$ds/logs/shard_${shard}.eval.log
      echo '[dataset-done]' \$(date -Is) shard=$shard dataset=\$ds | tee -a '$OUT/resume.log'
    done
    echo '[shard-done]' \$(date -Is) shard=$shard gpu=$gpu | tee -a '$OUT/resume.log'
  "
  echo "[launch] shard=$shard gpu=$gpu session=$session"
  sleep "$STAGGER"
done

monitor_session="${SESSION_PREFIX}_monitor"
if ! tmux has-session -t "$monitor_session" 2>/dev/null; then
  tmux new-session -d -s "$monitor_session" "
    set -euo pipefail
    cd '$ROOT'
    while true; do
      running=0
      for shard in \$(seq 0 $((NUM_SHARDS - 1))); do
        tmux has-session -t '${SESSION_PREFIX}_shard'\${shard} 2>/dev/null && running=1
      done
      [[ \"\$running\" == 0 ]] && break
      sleep 60
    done
    for ds in ${DATASETS[*]} hotpotqa; do
      echo '[merge-start]' \$(date -Is) dataset=\$ds | tee -a '$OUT/resume.log'
      python 03_sapr_rag/scripts/eval/merge_shards.py \
        --shard_dir '$OUT'/\$ds/shards \
        --output '$OUT'/\$ds/results.jsonl \
        2>&1 | tee '$OUT'/\$ds/logs/merge.log
      expected=\$(wc -l < data/eval/\$ds/dev.jsonl)
      rows=\$(wc -l < '$OUT'/\$ds/results.jsonl)
      echo \"[merge-check] dataset=\$ds rows=\$rows expected=\$expected\" | tee -a '$OUT/resume.log'
      python 03_sapr_rag/scripts/eval/score.py \
        --input '$OUT'/\$ds/results.jsonl \
        --output '$OUT'/\$ds/metrics.json \
        2>&1 | tee '$OUT'/\$ds/logs/score.log
      echo '[score-done]' \$(date -Is) dataset=\$ds | tee -a '$OUT/resume.log'
    done
    echo '[all_done]' \$(date -Is) | tee -a '$OUT/resume.log'
  "
  echo "[launch] monitor=$monitor_session"
fi
tmux ls | grep "$SESSION_PREFIX" || true
