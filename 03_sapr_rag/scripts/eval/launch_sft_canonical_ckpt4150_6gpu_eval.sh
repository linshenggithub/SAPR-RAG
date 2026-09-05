#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CKPT="${CKPT:-$ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/sft_canonical_fp16/checkpoint-4150}"
OUT="${OUT:-$ROOT/data/eval_results/sft_canonical_ckpt4150_3src_6gpu_20260904}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8100}"
GPUS_CSV="${GPUS_CSV:-1,2,3,4,5,6}"
NUM_SHARDS="${NUM_SHARDS:-6}"
SESSION_PREFIX="${SESSION_PREFIX:-sft_canonical_ckpt4150_6gpu}"
STOP_OLD="${STOP_OLD:-true}"
OLD_SESSION="${OLD_SESSION:-sft_canonical_ckpt4150_direct_eval}"
COHORT_SIZE="${COHORT_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-512}"

IFS=',' read -r -a GPUS <<<"$GPUS_CSV"
DATASETS=(hotpotqa 2wikimultihopqa musique)

require_file() {
  [[ -s "$1" ]] || {
    echo "ERROR: missing file: $1" >&2
    exit 2
  }
}

require_file "$CKPT/adapter_model.safetensors"
require_file "$CKPT/adapter_config.json"
for ds in "${DATASETS[@]}"; do
  require_file "$ROOT/data/eval/$ds/dev.jsonl"
done

mkdir -p "$OUT"
{
  echo "started_at=$(date -Is)"
  echo "root=$ROOT"
  echo "ckpt=$CKPT"
  echo "out=$OUT"
  echo "retrieval_url=$RETRIEVAL_URL"
  echo "gpus=$GPUS_CSV"
  echo "num_shards=$NUM_SHARDS"
  echo "cohort_size=$COHORT_SIZE"
  echo "max_tokens=$MAX_TOKENS"
} >"$OUT/config.txt"

if [[ "$STOP_OLD" == "true" ]]; then
  tmux kill-session -t "$OLD_SESSION" 2>/dev/null || true
fi

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPUS[$shard]}"
  session="${SESSION_PREFIX}_shard${shard}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[launch] session exists: $session"
    continue
  fi
  tmux new-session -d -s "$session" "
    set -euo pipefail
    cd '$ROOT'
    export NO_PROXY=127.0.0.1,localhost
    export no_proxy=127.0.0.1,localhost
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
    export CUDA_VISIBLE_DEVICES=$gpu
    echo '[shard-start]' \$(date -Is) shard=$shard gpu=$gpu | tee -a '$OUT/run.log'
    for ds in ${DATASETS[*]}; do
      mkdir -p '$OUT'/\$ds/shards '$OUT'/\$ds/logs
      echo '[dataset-start]' \$(date -Is) shard=$shard dataset=\$ds | tee -a '$OUT/run.log'
      python 03_sapr_rag/scripts/eval/agent_infer.py \
        --backend vllm \
        --input_jsonl data/eval/\$ds/dev.jsonl \
        --output_jsonl '$OUT'/\$ds/shards/shard_${shard}.jsonl \
        --lora_path '$CKPT' \
        --retrieval_url '$RETRIEVAL_URL' \
        --top_k 3 \
        --max_turns 6 \
        --max_tokens '$MAX_TOKENS' \
        --evidence_max_tokens 128 \
        --gpu_memory_utilization 0.85 \
        --max_model_len 8192 \
        --vllm_dtype bfloat16 \
        --vllm_disable_custom_all_reduce \
        --cohort_size '$COHORT_SIZE' \
        --num_shards '$NUM_SHARDS' \
        --shard_id '$shard' \
        --resume \
        2>&1 | tee '$OUT'/\$ds/logs/shard_${shard}.eval.log
      echo '[dataset-done]' \$(date -Is) shard=$shard dataset=\$ds | tee -a '$OUT/run.log'
    done
    echo '[shard-done]' \$(date -Is) shard=$shard gpu=$gpu | tee -a '$OUT/run.log'
  "
  echo "[launch] shard=$shard gpu=$gpu session=$session"
done

monitor_session="${SESSION_PREFIX}_monitor"
if ! tmux has-session -t "$monitor_session" 2>/dev/null; then
  tmux new-session -d -s "$monitor_session" "
    set -euo pipefail
    cd '$ROOT'
    echo '[monitor-start]' \$(date -Is) | tee -a '$OUT/run.log'
    while true; do
      running=0
      for shard in \$(seq 0 $((NUM_SHARDS - 1))); do
        if tmux has-session -t '${SESSION_PREFIX}_shard'\${shard} 2>/dev/null; then
          running=1
        fi
      done
      [[ \"\$running\" == 0 ]] && break
      sleep 60
    done
    for ds in ${DATASETS[*]}; do
      echo '[merge-start]' \$(date -Is) dataset=\$ds | tee -a '$OUT/run.log'
      python 03_sapr_rag/scripts/eval/merge_shards.py \
        --shard_dir '$OUT'/\$ds/shards \
        --output '$OUT'/\$ds/results.jsonl \
        2>&1 | tee '$OUT'/\$ds/logs/merge.log
      expected=\$(wc -l < data/eval/\$ds/dev.jsonl)
      rows=\$(wc -l < '$OUT'/\$ds/results.jsonl)
      unique=\$(python - '$OUT'/\$ds/results.jsonl <<'PY'
import json, sys
ids = []
with open(sys.argv[1]) as f:
    for line in f:
        if line.strip():
            ids.append(json.loads(line).get('id'))
print(len(set(ids)))
PY
)
      echo \"[merge-check] dataset=\$ds rows=\$rows unique=\$unique expected=\$expected\" | tee -a '$OUT/run.log'
      [[ \"\$rows\" == \"\$expected\" && \"\$unique\" == \"\$expected\" ]]
      python 03_sapr_rag/scripts/eval/score.py \
        --input '$OUT'/\$ds/results.jsonl \
        --output '$OUT'/\$ds/metrics.json \
        2>&1 | tee '$OUT'/\$ds/logs/score.log
      echo '[score-done]' \$(date -Is) dataset=\$ds | tee -a '$OUT/run.log'
    done
    echo '[all_done]' \$(date -Is) | tee -a '$OUT/run.log'
  "
  echo "[launch] monitor=$monitor_session"
fi

tmux ls | grep "$SESSION_PREFIX" || true
