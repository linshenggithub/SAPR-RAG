# 持久 faiss-cpu 检索环境记录

> **历史回退方案，不是当前正式配置。** 截至 2026-09-05，H20 Worker
> 已验证 `faiss-gpu 1.14.3 + CUDA 12.9`，正式训练使用 BGE GPU +
> FAISS GPU 常驻服务。请优先阅读
> [`retrieval_service_gpu_runbook.md`](retrieval_service_gpu_runbook.md)。
> 仅当 GPU FAISS 无法启动或显存不足时，才使用本文的 FAISS CPU 方案。

日期：2026-08-10

## 结论

持久环境路径通过环境变量指定：

```bash
export SAPR_RETRIEVAL_ENV=/path/to/persistent/sapr_faiss_cpu_py311
```

不要把环境放到节点临时目录；换 worker 或清理临时盘后会丢失。

## 当前验证结果

默认验证端口为 `8100`。当前方案让 BGE 编码运行在指定 GPU，
FAISS 索引保持在 CPU。

健康检查结果：

```json
{
  "status": "ok",
  "n_vectors": 22352695,
  "n_docs": 22352695,
  "text_truncate": 500,
  "bge_device": "cuda:0",
  "faiss_device": "cpu",
  "faiss_gpu_id": null,
  "faiss_gpu_fp16": null
}
```

实际检索请求也通过：

```bash
curl -fsS -X POST http://127.0.0.1:8100/search \
  -H 'content-type: application/json' \
  -d '{"query":"Who founded Apple?","top_k":3}'
```

top-3 返回包含 `History of Apple Inc.`、`Steve Wozniak`、`Apple Inc.`。

## 创建环境

worker 的 `/usr/bin/python3 -m venv` 缺 `ensurepip`，直接创建 venv 会失败。因此使用 `virtualenv`：

```bash
python3 -m pip install --user -U virtualenv

ENV_DIR="${SAPR_RETRIEVAL_ENV:?set SAPR_RETRIEVAL_ENV first}"
mkdir -p "$(dirname "$ENV_DIR")"
rm -rf "$ENV_DIR"
python3 -m virtualenv --system-site-packages -p /usr/bin/python3 "$ENV_DIR"
```

说明：

- `--system-site-packages` 是有意使用的：当前 worker 的用户 Python 目录已有可用的 `faiss`、`torch`、`transformers`、`datasets`、`fastapi`、`uvicorn`。
- 这样不用在项目目录里重新安装一整套 PyTorch，环境体积小，创建快。
- 环境目录应位于持久存储，不会被节点临时目录清理。

依赖验证：

```bash
ENV_DIR="${SAPR_RETRIEVAL_ENV:?set SAPR_RETRIEVAL_ENV first}"
"$ENV_DIR/bin/python" - <<'PY'
import site, sys
print("python", sys.executable)
print("version", sys.version.split()[0])
print("ENABLE_USER_SITE", site.ENABLE_USER_SITE)
for name in ["faiss", "torch", "transformers", "datasets", "fastapi", "uvicorn", "numpy", "pydantic"]:
    mod = __import__(name)
    print(name, getattr(mod, "__version__", "unknown"), getattr(mod, "__file__", ""))
PY
```

本次验证到的版本：

```text
faiss 1.14.1
torch 2.8.0+cu128
transformers 4.56.2
datasets 4.8.4
fastapi 0.136.3
uvicorn 0.46.0
numpy 2.2.6
pydantic 2.12.3
```

## 启动检索服务

默认训练插件使用：

```text
http://127.0.0.1:8100
```

如果 `8100` 没有被占用，按实际资源指定检索设备：

```bash
ENV_DIR="${SAPR_RETRIEVAL_ENV:?set SAPR_RETRIEVAL_ENV first}"
REPO="${SAPR_RAG_ROOT:?set SAPR_RAG_ROOT first}"
LOG=$REPO/03_sapr_rag/scripts/grpo/logs/retrieval_bge_gpu0_faiss_cpu_8100.log

nohup env CUDA_VISIBLE_DEVICES=0 "$ENV_DIR/bin/python" "$REPO/03_sapr_rag/scripts/grpo/retrieval_daemon.py" \
  --host 127.0.0.1 \
  --port 8100 \
  --device cuda:0 \
  --faiss_device cpu \
  --text_truncate 500 \
  > "$LOG" 2>&1 &
```

如果 `8100` 已被占用，可以先用 `8101` 验证：

```bash
ENV_DIR="${SAPR_RETRIEVAL_ENV:?set SAPR_RETRIEVAL_ENV first}"
REPO="${SAPR_RAG_ROOT:?set SAPR_RAG_ROOT first}"
LOG=$REPO/03_sapr_rag/scripts/grpo/logs/retrieval_bge_gpu0_faiss_cpu_8101.log

nohup env CUDA_VISIBLE_DEVICES=0 "$ENV_DIR/bin/python" "$REPO/03_sapr_rag/scripts/grpo/retrieval_daemon.py" \
  --host 127.0.0.1 \
  --port 8101 \
  --device cuda:0 \
  --faiss_device cpu \
  --text_truncate 500 \
  > "$LOG" 2>&1 &
```

健康检查：

```bash
curl -fsS http://127.0.0.1:8100/health
```

若使用 `8101` 验证：

```bash
curl -fsS http://127.0.0.1:8101/health
```

检索检查：

```bash
curl -fsS -X POST http://127.0.0.1:8100/search \
  -H 'content-type: application/json' \
  -d '{"query":"Who founded Apple?","top_k":3}'
```

## 训练时的端口

如果检索服务跑在默认 `8100`，训练脚本不用额外改。

如果临时跑在其他端口，例如 `8101`，训练前需要显式设置：

```bash
export SAPR_RETRIEVAL_URL=http://127.0.0.1:8101
```

## 为什么不用 FAISS GPU

H20 上试过 `faiss-gpu-cu12`，可以看到 GPU，但把索引搬到 GPU 时失败：

```text
CUDA error 209 no kernel image is available for execution on the device
```

这说明当前 wheel 没有适配 H20 的计算架构。除非换成确认支持 H20 的 FAISS GPU 包，或自行编译带对应架构的 FAISS，否则不要把训练依赖切回 FAISS GPU。

当前稳定方案是：

```text
BGE GPU0 + FAISS CPU
```

它能跑通，但训练会慢。后续若要继续 Reward-v3 小训练，建议先用较小步数确认奖励方向，不要直接跑 500 step。
