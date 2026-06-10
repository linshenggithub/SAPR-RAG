#!/usr/bin/env bash
# 一键对所有 zeroshot/SFT/SFT+DPO 产物跑 DeepSeek LLM-judge。
#
# 前置：
#   export DEEPSEEK_API_KEY=sk-xxx
#
# 用法：
#   bash run_llm_judge_all.sh             # 全量跑 8 份
#   MAX_SAMPLES=1000 bash ...             # 每份采样 1000 sanity
#
# 输出：每个 merged.jsonl 同目录下 metrics_llm.json + judgments.jsonl
# Cache：data/llm_judge_cache/deepseek_judge_cache.jsonl（断点续算）

set -euo pipefail

PROJ_ROOT=/mlx_devbox/users/mayi.summer/playground/SAPR-RAG
SCRIPT="$PROJ_ROOT/03_sapr_rag/scripts/eval/llm_judge_deepseek.py"

MAX_SAMPLES="${MAX_SAMPLES:-0}"
CONCURRENCY="${CONCURRENCY:-32}"
EXTRA_ARGS=()
[[ "$MAX_SAMPLES" -gt 0 ]] && EXTRA_ARGS+=("--max_samples" "$MAX_SAMPLES")

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "[FATAL] export DEEPSEEK_API_KEY=sk-xxx 先设置 API key"
    exit 1
fi

# 8 份 merged.jsonl 路径
MERGED_LIST=(
    "$PROJ_ROOT/data/eval_results/hotpotqa/zeroshot_20260608_193355/merged.jsonl"
    "$PROJ_ROOT/data/eval_results/hotpotqa/20260608_175824/merged.jsonl"
    "$PROJ_ROOT/data/eval_results/hotpotqa/sft_dpo_20260610_145349/merged.jsonl"
    "$PROJ_ROOT/data/eval_results/2wikimultihopqa/zeroshot_20260609_184744/merged.jsonl"
    "$PROJ_ROOT/data/eval_results/2wikimultihopqa/sft_20260609_232951/merged.jsonl"
    "$PROJ_ROOT/data/eval_results/musique/zeroshot_20260609_173552/merged.jsonl"
    "$PROJ_ROOT/data/eval_results/musique/sft_20260610_112233/merged.jsonl"
    "$PROJ_ROOT/data/eval_results/musique/sft_dpo_20260610_142115/merged.jsonl"
)

# 2Wiki SFT+DPO 还没跑完，跳过；如有 merged 出现自动加上
TWO_WIKI_DPO=$(ls "$PROJ_ROOT"/data/eval_results/2wikimultihopqa/sft_dpo_*/merged.jsonl 2>/dev/null | head -1 || true)
if [[ -n "$TWO_WIKI_DPO" ]]; then
    MERGED_LIST+=("$TWO_WIKI_DPO")
    echo "[info] 检测到 2Wiki SFT+DPO merged，加入评估"
fi

echo "[plan] 共 ${#MERGED_LIST[@]} 份产物，concurrency=$CONCURRENCY  max_samples=$MAX_SAMPLES"

n_idx=0
for merged in "${MERGED_LIST[@]}"; do
    n_idx=$((n_idx + 1))
    echo ""
    echo "============================================================"
    echo "[$n_idx/${#MERGED_LIST[@]}] $merged"
    echo "============================================================"
    if [[ ! -f "$merged" ]]; then
        echo "  ⚠️  跳过：文件不存在"
        continue
    fi
    python "$SCRIPT" \
        --merged "$merged" \
        --concurrency "$CONCURRENCY" \
        "${EXTRA_ARGS[@]}"
done

echo ""
echo "============================================================"
echo "[ALL DONE] 汇总所有 metrics_llm.json："
echo "============================================================"
for merged in "${MERGED_LIST[@]}"; do
    m="$(dirname $merged)/metrics_llm.json"
    if [[ -f "$m" ]]; then
        echo "--- $m ---"
        python -c "import json; d=json.load(open('$m')); print(f'  llm_acc={d[\"llm_acc_deepseek\"]}  n_correct={d[\"n_correct\"]}/{d[\"n_judged\"]}')"
    fi
done
