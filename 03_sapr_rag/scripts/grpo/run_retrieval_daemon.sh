#!/usr/bin/env bash
# 启动检索 daemon（GRPO rollout 的检索后端）。
# 占用 1 张 GPU 加载 BGE encoder；FAISS 索引 mmap 到 RAM。
#
# 用法：
#   GPU=7 PORT=8100 bash run_retrieval_daemon.sh
#   默认 GPU=7 PORT=8100，日志写到 logs/retrieval_daemon.log
#
# 在 server 模式 GRPO 里，建议把 daemon 放在 rollout vllm 所占之外的卡，避免显存争抢。
set -euo pipefail

PROJ_ROOT=/mlx_devbox/users/mayi.summer/playground/SAPR-RAG
SCRIPT_DIR="$PROJ_ROOT/03_sapr_rag/scripts/grpo"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

GPU="${GPU:-7}"
PORT="${PORT:-8100}"
HOST="${HOST:-127.0.0.1}"
TEXT_TRUNCATE="${TEXT_TRUNCATE:-500}"

echo "[run_retrieval_daemon] GPU=$GPU port=$PORT text_truncate=$TEXT_TRUNCATE"
echo "[run_retrieval_daemon] log -> $LOG_DIR/retrieval_daemon.log"

CUDA_VISIBLE_DEVICES="$GPU" \
python "$SCRIPT_DIR/retrieval_daemon.py" \
    --host "$HOST" \
    --port "$PORT" \
    --device cuda:0 \
    --text_truncate "$TEXT_TRUNCATE" \
    2>&1 | tee "$LOG_DIR/retrieval_daemon.log"
