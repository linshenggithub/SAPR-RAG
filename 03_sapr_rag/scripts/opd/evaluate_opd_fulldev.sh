#!/usr/bin/env bash
# 用指定的 OPD LoRA checkpoint（默认 checkpoint-300）在 HotpotQA / 2Wiki / MuSiQue
# 三个数据集的【全量 dev】上评测 Agentic RAG，并计算 EM / Cover-EM / F1。
#
# 设计：
#   - 共享一个常驻检索服务（GPU0，端口 8100）。
#   - 三个数据集各起一个独立的 7B+OPD rollout server，分别占用一张训练空闲卡，
#     并行评测以充分利用 8 卡（符合“并行最大化利用 GPU”的偏好）。
#   - 每个数据集跑完后用 score.py 计算指标，最后汇总成一个 summary.json。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
GRPO_DIR="$PROJ_ROOT/03_sapr_rag/scripts/grpo"
EVAL_DIR="$PROJ_ROOT/03_sapr_rag/scripts/eval"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

BASE_MODEL="${BASE_MODEL:-$PROJ_ROOT/03_sapr_rag/models/Qwen2.5-7B-Instruct}"
OPD_ADAPTER="${OPD_ADAPTER:-$PROJ_ROOT/03_sapr_rag/saves/qwen2_5_7b/lora/opd/opd_sft14b_failed_em_spg2_s500_20260904/v0-20260904-023807/checkpoint-300}"
RUN_NAME="${RUN_NAME:-opd_ckpt300_fulldev_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$PROJ_ROOT/data/eval_results/$RUN_NAME}"

RETRIEVAL_GPU="${RETRIEVAL_GPU:-0}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TURNS="${MAX_TURNS:-6}"
MAX_TOKENS="${MAX_TOKENS:-512}"

# dataset -> 全量 dev 文件（注意 2wiki 目录名是 2wikimultihopqa）
declare -A DEV_FILES=(
    [hotpotqa]="$PROJ_ROOT/data/eval/hotpotqa/dev.jsonl"
    [2wikimultihopqa]="$PROJ_ROOT/data/eval/2wikimultihopqa/dev.jsonl"
    [musique]="$PROJ_ROOT/data/eval/musique/dev.jsonl"
)
# dataset -> 分配的 GPU 和 rollout 端口（并行）
declare -A DS_GPU=( [hotpotqa]=2 [2wikimultihopqa]=3 [musique]=4 )
declare -A DS_PORT=( [hotpotqa]=8051 [2wikimultihopqa]=8052 [musique]=8053 )
DATASETS=(hotpotqa 2wikimultihopqa musique)

ROLLOUT_PIDS=()
cleanup() {
    for pid in "${ROLLOUT_PIDS[@]:-}"; do
        [[ -n "$pid" ]] || continue
        if kill -0 "$pid" 2>/dev/null; then
            pgid="$(ps -o pgid= -p "$pid" | tr -d ' ' || true)"
            [[ -n "$pgid" ]] && kill -TERM "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

mkdir -p "$OUT_ROOT/logs"
[[ -d "$BASE_MODEL" ]] || { echo "ERROR: missing base model: $BASE_MODEL" >&2; exit 2; }
[[ -d "$OPD_ADAPTER" ]] || { echo "ERROR: missing OPD adapter: $OPD_ADAPTER" >&2; exit 2; }
for ds in "${DATASETS[@]}"; do
    [[ -f "${DEV_FILES[$ds]}" ]] || { echo "ERROR: missing dev file for $ds: ${DEV_FILES[$ds]}" >&2; exit 2; }
done

echo "[eval] OPD adapter: $OPD_ADAPTER"
echo "[eval] output root: $OUT_ROOT"

# 1) 共享检索服务
env RETRIEVAL_GPU="$RETRIEVAL_GPU" RETRIEVAL_PORT="$RETRIEVAL_PORT" \
    bash "$GRPO_DIR/retrieval_service.sh" start
env RETRIEVAL_GPU="$RETRIEVAL_GPU" RETRIEVAL_PORT="$RETRIEVAL_PORT" WAIT_TIMEOUT=1800 \
    bash "$GRPO_DIR/retrieval_service.sh" wait

# 2) 为每个数据集拉起独立 rollout server（并行）
for ds in "${DATASETS[@]}"; do
    gpu="${DS_GPU[$ds]}"; port="${DS_PORT[$ds]}"
    echo "[eval] launching rollout for $ds on GPU$gpu port$port"
    # 每个 rollout 独立的 vLLM 编译缓存目录，避免并发启动时 torch.compile
    # 缓存目录 os.rename 写入竞争导致 EngineCore 初始化失败。
    cache_root="$OUT_ROOT/vllm_cache/$ds"
    mkdir -p "$cache_root"
    setsid env \
        DEVICE_BACKEND=cuda \
        VLLM_CACHE_ROOT="$cache_root" \
        BASE_MODEL="$BASE_MODEL" \
        INIT_ADAPTER="$OPD_ADAPTER" \
        ROLLOUT_DEVICES="$gpu" \
        PORT="$port" \
        SAPR_RETRIEVAL_URL="http://127.0.0.1:${RETRIEVAL_PORT}" \
        SAPR_TOP_K=3 \
        SAPR_ENABLE_EVIDENCE_AGENT=true \
        MULTI_TURN_SCHEDULER=sapr_rag_scheduler \
        VLLM_MAX_MODEL_LEN=8192 \
        VLLM_GPU_MEM_UTIL=0.88 \
        bash "$GRPO_DIR/run_rollout_opsd.sh" \
        >"$OUT_ROOT/logs/rollout_${ds}.log" 2>&1 < /dev/null &
    ROLLOUT_PIDS+=("$!")
    # 错峰启动，进一步降低并发初始化时的资源争用
    sleep 20
done

# 3) 等待所有 rollout server ready
for ds in "${DATASETS[@]}"; do
    port="${DS_PORT[$ds]}"
    started="$(date +%s)"
    until curl -fsS --max-time 5 "http://127.0.0.1:${port}/health/" >/dev/null 2>&1; do
        if (( $(date +%s) - started >= 1800 )); then
            echo "ERROR: rollout $ds health timeout" >&2
            tail -n 120 "$OUT_ROOT/logs/rollout_${ds}.log" >&2 || true
            exit 4
        fi
        sleep 10
    done
    echo "[eval] rollout $ds READY"
done

# 4) 并行评测三个数据集
eval_pids=()
for ds in "${DATASETS[@]}"; do
    port="${DS_PORT[$ds]}"
    out_dir="$OUT_ROOT/$ds"; mkdir -p "$out_dir"
    (
        python "$EVAL_DIR/run_direct_rollout_eval.py" \
            --input_jsonl "${DEV_FILES[$ds]}" \
            --output_jsonl "$out_dir/results.jsonl" \
            --rollout_url "http://127.0.0.1:${port}" \
            --batch_size "$BATCH_SIZE" \
            --max_turns "$MAX_TURNS" \
            --max_tokens "$MAX_TOKENS" \
            --resume \
            >"$out_dir/eval.log" 2>&1
        python "$EVAL_DIR/score.py" \
            --input "$out_dir/results.jsonl" \
            --output "$out_dir/metrics.json" \
            >"$out_dir/score.log" 2>&1
    ) &
    eval_pids+=("$!")
done
fail=0
for pid in "${eval_pids[@]}"; do wait "$pid" || fail=1; done

# 5) 汇总
python - "$OUT_ROOT" "${DATASETS[@]}" <<'PY'
import json, sys, os
out_root = sys.argv[1]; datasets = sys.argv[2:]
summary = {}
for ds in datasets:
    mp = os.path.join(out_root, ds, "metrics.json")
    if os.path.exists(mp):
        summary[ds] = json.load(open(mp))
    else:
        summary[ds] = {"error": "metrics.json missing"}
json.dump(summary, open(os.path.join(out_root, "summary.json"), "w"), ensure_ascii=False, indent=2)
print("=== OPD full-dev evaluation summary ===")
for ds, m in summary.items():
    print(ds, json.dumps(m, ensure_ascii=False))
PY

echo "[eval] done. summary: $OUT_ROOT/summary.json"
exit $fail
