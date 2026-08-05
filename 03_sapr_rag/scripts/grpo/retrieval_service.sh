#!/usr/bin/env bash
# SAPR-RAG retrieval 常驻服务管理器（增量脚本，不改 run_retrieval_daemon_flexible.sh）。
#
# 目的：把 BGE+FAISS 检索器做成脱离 SSH 会话的长驻 GPU 服务，避免每次训练冷启动
# （faiss 读 68GB mmap 索引 + datasets 解析语料）都要等十几分钟。
#
# 用法：
#   bash retrieval_service.sh start     # 若未存活则后台拉起（立即返回，不等 ready）
#   bash retrieval_service.sh wait      # 轮询 /health 直到 ready（或超时）
#   bash retrieval_service.sh status    # 查看存活 / health / GPU
#   bash retrieval_service.sh stop      # 停止常驻服务
#   bash retrieval_service.sh restart   # stop 后 start
#
# 约定：
#   - 训练 / smoke 脚本启动前应先 `wait`，存活即复用、绝不重复加载索引。
#   - 具体设备通过 RETRIEVAL_GPU / RETRIEVAL_DEVICES 显式指定，避免把某台机器布局写死。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${SAPR_RAG_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# 注意：部分运行环境可能导出 PORT=<外部端口>，故用 RETRIEVAL_PORT 专属变量，避免被污染。
RETRIEVAL_GPU="${RETRIEVAL_GPU:-0}"
RETRIEVAL_DEVICES="${RETRIEVAL_DEVICES:-$RETRIEVAL_GPU}"
PORT="${RETRIEVAL_PORT:-8100}"
HOST="${HOST:-127.0.0.1}"
SVC_LOG="$LOG_DIR/retrieval_service.log"
PID_FILE="$LOG_DIR/retrieval_service.pid"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1200}"   # 最长等 20 分钟 ready

ACTION="${1:-status}"

health() {
    curl -s --max-time 3 "http://$HOST:$PORT/health" 2>/dev/null
}

is_alive() {
    health | grep -q '"status"' 2>/dev/null
}

do_start() {
    if is_alive; then
        echo "[svc] already alive on $HOST:$PORT — reuse, skip start."
        health; echo
        return 0
    fi
    # 清理陈旧 pid 文件对应的死进程
    if [ -f "$PID_FILE" ] && ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        rm -f "$PID_FILE"
    fi
    echo "[svc] launching retrieval daemon on GPU$RETRIEVAL_DEVICES port=$PORT (detached) ..."
    nohup setsid env GPU="$RETRIEVAL_GPU" RETRIEVAL_DEVICES="$RETRIEVAL_DEVICES" \
        DEVICE_BACKEND=cuda PORT="$PORT" HOST="$HOST" \
        bash "$SCRIPT_DIR/run_retrieval_daemon_flexible.sh" \
        >"$SVC_LOG" 2>&1 < /dev/null &
    disown
    echo $! > "$PID_FILE"
    echo "[svc] launched pid=$(cat "$PID_FILE"), log=$SVC_LOG"
    echo "[svc] run 'retrieval_service.sh wait' to block until ready."
}

do_wait() {
    echo "[svc] waiting for /health (timeout=${WAIT_TIMEOUT}s) ..."
    local start_ts=$(date +%s)
    while true; do
        if is_alive; then
            echo "[svc] READY:"; health; echo
            return 0
        fi
        if [ -f "$PID_FILE" ] && ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "[svc] ERROR: daemon process died before ready. tail log:" >&2
            tail -n 30 "$SVC_LOG" >&2
            return 3
        fi
        local now=$(date +%s)
        if [ $((now - start_ts)) -ge "$WAIT_TIMEOUT" ]; then
            echo "[svc] ERROR: timeout after ${WAIT_TIMEOUT}s. tail log:" >&2
            tail -n 30 "$SVC_LOG" >&2
            return 4
        fi
        sleep 10
    done
}

do_status() {
    echo "=== retrieval service status ==="
    if [ -f "$PID_FILE" ]; then
        local pid; pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "pid $pid: RUNNING"
        else
            echo "pid $pid: DEAD (stale pid file)"
        fi
    else
        echo "(no pid file)"
    fi
    echo "--- /health ---"
    if is_alive; then health; echo; else echo "(not responding on $HOST:$PORT)"; fi
    echo "--- processes ---"
    pgrep -af retrieval_daemon.py || echo "(none)"
    echo "--- GPU$RETRIEVAL_GPU ---"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed -n "$((RETRIEVAL_GPU+1))p"
    else
        echo "(nvidia-smi not available)"
    fi
}

do_stop() {
    echo "[svc] stopping ..."
    pkill -f retrieval_daemon.py && echo "[svc] sent SIGTERM to retrieval_daemon.py" || echo "[svc] no retrieval_daemon.py process"
    rm -f "$PID_FILE"
}

case "$ACTION" in
    start)   do_start ;;
    wait)    do_wait ;;
    status)  do_status ;;
    stop)    do_stop ;;
    restart) do_stop; sleep 2; do_start ;;
    *) echo "usage: $0 {start|wait|status|stop|restart}" >&2; exit 2 ;;
esac
