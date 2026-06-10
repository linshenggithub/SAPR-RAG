#!/usr/bin/env bash
# 准备 GRPO 全量混合训练数据：HotpotQA train + 2Wiki train，共 cap 7321 题。
#
# 流程：
#   1. HotpotQA train.jsonl 已在本地（90k 行，data/raw/hotpotqa/train.jsonl）
#   2. 2Wiki train.jsonl 若不在本地则从 HF 下载（167k 行，60MB）
#   3. 混合 + cap + shuffle → data/grpo/hotpotqa_2wiki_train.jsonl
#
# 用法：
#   bash prepare_grpo_train_data.sh
#   FAST=1 bash prepare_grpo_train_data.sh   # 跳过 corpus 可达性预过滤（更快但留 1% 不可达噪声）
set -euo pipefail

PROJ_ROOT=/mlx_devbox/users/mayi.summer/playground/SAPR-RAG
TRAIN_2WIKI="$PROJ_ROOT/data/raw/2wikimultihopqa/train.jsonl"
OUT="$PROJ_ROOT/data/grpo/hotpotqa_2wiki_train.jsonl"
CORPUS="$PROJ_ROOT/data/corpus/wiki18_extended.jsonl"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. 下载 2Wiki train（若缺失）─────────────────────────────────────────
if [[ -s "$TRAIN_2WIKI" ]]; then
    echo "[prepare] 2Wiki train.jsonl 已存在: $(wc -l < "$TRAIN_2WIKI") 行"
else
    echo "[prepare] 2Wiki train.jsonl 不存在，从 HF 下载..."
    mkdir -p "$(dirname "$TRAIN_2WIKI")"
    python - <<PY
from huggingface_hub import hf_hub_download
import shutil
cached = hf_hub_download(
    repo_id="RUC-NLPIR/FlashRAG_datasets",
    filename="2wikimultihopqa/train.jsonl",
    repo_type="dataset",
)
shutil.copy2(cached, "$TRAIN_2WIKI")
print(f"   downloaded -> $TRAIN_2WIKI")
PY
    echo "[prepare] 2Wiki train.jsonl: $(wc -l < "$TRAIN_2WIKI") 行"
fi

# ── 2. 构建混合训练集 ─────────────────────────────────────────────────────
CORPUS_ARG=()
if [[ "${FAST:-0}" != "1" && -s "$CORPUS" ]]; then
    CORPUS_ARG=(--corpus "$CORPUS")
    echo "[prepare] 启用 corpus 可达性预过滤"
fi

echo "[prepare] building mixed train set..."
python "$SCRIPT_DIR/build_grpo_dataset_mixed.py" \
    --output "$OUT" \
    --max_total 7321 \
    --ratios 0.5,0.5 \
    --seed 42 \
    "${CORPUS_ARG[@]}"

echo "[prepare] done -> $OUT"
echo "[prepare] 启动训练命令："
echo "  DATASET=$OUT VLLM_PORT=<port> DEEPSPEED=none bash $SCRIPT_DIR/run_grpo.sh"
