#!/usr/bin/env bash
# 串行评估 Reward-v3 的多个 checkpoint（rollout 走训练同款 scheduler + GPU 检索 8100）。
# 与基线 rawdoc_sft_vs_reward_v2_ckpt300_200 口径一致：dev 前 200 条、strict answer、score.py 带 behavior。
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWIFT_ROOT="$(cd "$PROJ_ROOT/../ms-swift" && pwd)"
SCRIPT_DIR="$PROJ_ROOT/03_sapr_rag/scripts/eval"
GRPO_DIR="$PROJ_ROOT/03_sapr_rag/scripts/grpo"

MERGED_MODEL="${MERGED_MODEL:-}"
V3_ROOT="${V3_ROOT:-}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8100}"
ROLLOUT_GPU="${ROLLOUT_GPU:-1}"
PORT="${PORT:-8011}"
N_SUBSET="${N_SUBSET:-200}"
ENABLE_EVIDENCE_AGENT="${ENABLE_EVIDENCE_AGENT:-false}"
MULTI_TURN_SCHEDULER="${MULTI_TURN_SCHEDULER:-sapr_rag_scheduler}"
LABEL_PREFIX="${LABEL_PREFIX:-reward_v3}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-300,400,500}"

[ -n "$MERGED_MODEL" ] || {
  echo "ERROR: MERGED_MODEL must point to the merged SFT model." >&2
  exit 2
}
[ -n "$V3_ROOT" ] || {
  echo "ERROR: V3_ROOT must point to the Reward-v3 run directory." >&2
  exit 2
}

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$PROJ_ROOT/data/eval_results/hotpotqa/reward_v3_ckpt_sweep_${N_SUBSET}_${RUN_TAG}}"
FULL_DEV="$PROJ_ROOT/data/eval/hotpotqa/dev.jsonl"
SUBSET="$OUT_ROOT/dev_first${N_SUBSET}.jsonl"

mkdir -p "$OUT_ROOT"
head -n "$N_SUBSET" "$FULL_DEV" > "$SUBSET"
echo "[sweep] subset=$SUBSET lines=$(wc -l < "$SUBSET")"
echo "[sweep] merged_model=$MERGED_MODEL"
echo "[sweep] retrieval=$RETRIEVAL_URL rollout_gpu=$ROLLOUT_GPU port=$PORT"
echo "[sweep] out_root=$OUT_ROOT"

IFS=',' read -r -a STEPS <<< "$CHECKPOINT_STEPS"
LABELS=()
CKPTS=()
for STEP in "${STEPS[@]}"; do
  LABELS+=("${LABEL_PREFIX}_ckpt${STEP}")
  CKPTS+=("$V3_ROOT/checkpoint-${STEP}")
done

for i in "${!LABELS[@]}"; do
  LABEL="${LABELS[$i]}"
  CKPT="${CKPTS[$i]}"
  OUT_DIR="$OUT_ROOT/$LABEL"
  mkdir -p "$OUT_DIR/logs"
  RAW="$OUT_DIR/results.raw.jsonl"
  STRICT="$OUT_DIR/results.strict.jsonl"
  METRICS="$OUT_DIR/metrics.with_behavior.json"
  STATUS="$OUT_DIR/status.txt"

  echo "==================== [$LABEL] ===================="
  echo "ckpt=$CKPT" | tee "$STATUS"

  # 起 rollout 服务
  ROLL_LOG="$OUT_DIR/logs/rollout.log"
  nohup setsid env \
    SAPR_RAG_ROOT="$PROJ_ROOT" \
    SWIFT_ROOT="$SWIFT_ROOT" \
    DEVICE_BACKEND=cuda \
    ROLLOUT_DEVICES="$ROLLOUT_GPU" \
    PORT="$PORT" \
    SAPR_ENABLE_EVIDENCE_AGENT="$ENABLE_EVIDENCE_AGENT" \
    MULTI_TURN_SCHEDULER="$MULTI_TURN_SCHEDULER" \
    BASE_MODEL="$MERGED_MODEL" \
    MODEL_TYPE=qwen2 \
    TEMPLATE_TYPE=qwen2_5 \
    INIT_ADAPTER=none \
    ADAPTER_PATH="$CKPT" \
    VLLM_MAX_MODEL_LEN=8192 \
    VLLM_GPU_MEM_UTIL=0.55 \
    bash "$GRPO_DIR/run_rollout_opsd.sh" \
    >"$ROLL_LOG" 2>&1 < /dev/null &
  ROLL_PID="$!"
  echo "$ROLL_PID" > "$OUT_DIR/rollout.pid"

  # 等健康
  start_ts="$(date +%s)"
  ready=0
  while true; do
    if curl -s --max-time 5 "http://127.0.0.1:${PORT}/health/" >/dev/null 2>&1; then
      ready=1; break
    fi
    if ! kill -0 "$ROLL_PID" 2>/dev/null; then
      echo "[$LABEL] rollout died"; tail -n 80 "$ROLL_LOG"; break
    fi
    if [ $(( $(date +%s) - start_ts )) -ge 1500 ]; then
      echo "[$LABEL] rollout health timeout"; tail -n 120 "$ROLL_LOG"; break
    fi
    sleep 10
  done

  if [ "$ready" = "1" ]; then
    echo "[$LABEL] rollout ready, running inference" | tee -a "$STATUS"
    python "$SCRIPT_DIR/run_direct_rollout_eval.py" \
      --input_jsonl "$SUBSET" \
      --output_jsonl "$RAW" \
      --rollout_url "http://127.0.0.1:${PORT}" \
      --batch_size 64 \
      --max_turns 6 \
      --max_tokens 512 \
      2>&1 | tee "$OUT_DIR/logs/eval.log"

    # strict answer 提取
    python - "$RAW" "$STRICT" <<'PY'
import json, re, sys
from pathlib import Path
raw_path, strict_path = map(Path, sys.argv[1:])
answer_re = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
with raw_path.open() as fi, strict_path.open("w") as fo:
    for line in fi:
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("raw_output") or ""
        trace = row.get("trace") or []
        messages = ((trace[-1] or {}).get("messages") or []) if trace else []
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                text = msg.get("content") or text
                break
        m = answer_re.search(text or "")
        if m:
            row["answer"] = m.group(1).strip()
            if row.get("error") == "no_strict_answer_tag":
                row.pop("error", None)
        else:
            row["answer"] = None
            row["error"] = row.get("error") or "no_strict_answer_tag"
        fo.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

    # 答案质量（score.py，与基线同口径）+ 重复率行为指标（从 behavior 聚合，补齐基线字段）
    python - "$STRICT" "$METRICS" "$SCRIPT_DIR/score.py" <<'PY'
import json, sys, importlib.util
from pathlib import Path
strict_path, metrics_path, score_py = sys.argv[1], sys.argv[2], sys.argv[3]

spec = importlib.util.spec_from_file_location("score", score_py)
score = importlib.util.module_from_spec(spec); spec.loader.exec_module(score)

rows = [json.loads(l) for l in Path(strict_path).open() if l.strip()]
metrics = score.evaluate(rows)

n = len(rows)
answered = sum(1 for r in rows if r.get("answer") is not None)
q_sum = search_sum = exact_dup_rows = intercept_sum = 0
exact_repeat_rows = 0
for r in rows:
    b = r.get("behavior") or {}
    q_sum += int(b.get("num_queries") or 0)
    search_sum += int(b.get("actual_search_count") or 0)
    intercept_sum += int(b.get("intercepted_repeat_count") or 0)
    if b.get("has_exact_duplicate"):
        exact_repeat_rows += 1
    if int(b.get("exact_duplicate_count") or 0) > 0:
        exact_dup_rows += 1
metrics.update({
    "answer_rate": round(answered / n, 4) if n else 0.0,
    "avg_num_queries": round(q_sum / n, 3) if n else 0.0,
    "avg_actual_search_count": round(search_sum / n, 3) if n else 0.0,
    "exact_repeat_rate": round(exact_repeat_rows / n, 4) if n else 0.0,
    "intercepted_repeat_rate": round(exact_dup_rows / n, 4) if n else 0.0,
    "avg_intercepted_repeat_count": round(intercept_sum / n, 3) if n else 0.0,
})
Path(metrics_path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print(json.dumps(metrics, indent=2, ensure_ascii=False))
PY
    echo "[$LABEL] metrics written to $METRICS"
  fi

  # 停 rollout（释放显存给下一个 ckpt）
  if kill -0 "$ROLL_PID" 2>/dev/null; then
    pgid="$(ps -o pgid= -p "$ROLL_PID" | tr -d ' ' || true)"
    [ -n "${pgid:-}" ] && kill -TERM "-$pgid" 2>/dev/null || kill -TERM "$ROLL_PID" 2>/dev/null || true
    sleep 15
    [ -n "${pgid:-}" ] && kill -KILL "-$pgid" 2>/dev/null || true
  fi
  echo "[$LABEL] done, rollout stopped" | tee -a "$STATUS"
  sleep 5
done

echo "==================== SUMMARY ===================="
for LABEL in "${LABELS[@]}"; do
  M="$OUT_ROOT/$LABEL/metrics.with_behavior.json"
  echo "--- $LABEL ---"
  [ -f "$M" ] && cat "$M" || echo "(no metrics)"
  echo
done
echo "out_root=$OUT_ROOT"
