# SAPR-RAG 正式检索服务运行手册

- **适用范围**：Merlin H20 Worker 上的正式 GRPO/OPSD/OPD 训练与三数据集评测
- **当前推荐方案**：单进程常驻 BGE GPU + FAISS GPU daemon，端口 `8100`
- **最后核验日期**：2026-09-05
- **已验证 Worker**：`4216626`

本文是跨会话执行检索服务的权威入口。其他 AI 或研究者在新 Worker 上
启动正式训练/评测前，应先完整阅读本文，不要直接照搬旧的 FAISS CPU 文档。

## 1. 当前正式配置

```text
GPU0:
  BGE query encoder: cuda:0
  FAISS IndexFlatIP: gpu:0, fp32
  HTTP daemon: 127.0.0.1:8100

其他进程:
  rollout / eval / train
      -> HTTP 8100
      -> 唯一 retrieval daemon
```

| 项目 | 当前值 |
|---|---|
| BGE | 项目内的 `models/bge-base-en-v1.5` |
| FAISS index | `data/index/bge_extended_Flat.index` |
| corpus | `data/corpus/wiki18_extended.jsonl` |
| index 类型 | `IndexFlatIP` |
| 向量数 / 维度 | 22,352,695 / 768 |
| FAISS 存储精度 | fp32 |
| Top-K | 3 |
| 正文截断 | 500 字符 |
| GPU0 稳态显存 | 约 68.9 GiB |
| 服务端口 | 8100 |

为什么只启动一个 daemon：

- Flat index 约 68GB，每个进程各加载一份会直接耗尽内存或显存；
- 训练和评测只通过 HTTP 访问，不需要复制 index；
- daemon 可跨多个训练或评测阶段复用，避免反复冷启动；
- 新建 rollout server 或切换 checkpoint 时不要重启 retrieval daemon。

## 2. 固定路径

```bash
REPO="$(git rev-parse --show-toplevel)"
WORKSPACE="$(dirname "$REPO")"
GRPO_DIR=$REPO/03_sapr_rag/scripts/grpo

FAISS_ENV=$WORKSPACE/envs/sapr_micromamba_root/envs/sapr_faiss_gpu_cuda129_py311
FAISS_SITE=$FAISS_ENV/lib/python3.11/site-packages

BGE_MODEL=$REPO/models/bge-base-en-v1.5
FAISS_INDEX=$REPO/data/index/bge_extended_Flat.index
CORPUS=$REPO/data/corpus/wiki18_extended.jsonl
```

环境和数据位于共享盘，换 Worker 后仍然存在；服务进程、GPU 显存和
`tmux` 会话不会跨 Worker 保留，必须在新 Worker 上重新启动。

## 3. 登录 Worker

所有 `mlx worker` 命令必须关闭颜色：

```bash
NO_COLOR=1 TERM=dumb mlx worker login <worker_id>
```

后续命令必须在 Worker shell 内执行。不要在 Master 节点直接运行
`faiss.get_num_gpus()`；Master 没有 Worker GPU，GPU FAISS 可能直接 abort。

进入 Worker 后，切换到仓库根目录并确认位置：

```bash
cd SAPR-RAG
git rev-parse --show-toplevel
```

## 4. 启动前检查

### 4.1 检查资产

```bash
REPO="$(git rev-parse --show-toplevel)"
WORKSPACE="$(dirname "$REPO")"
FAISS_SITE=$WORKSPACE/envs/sapr_micromamba_root/envs/sapr_faiss_gpu_cuda129_py311/lib/python3.11/site-packages

test -f data/index/bge_extended_Flat.index
test -f data/corpus/wiki18_extended.jsonl
test -f models/bge-base-en-v1.5/config.json
test -d "$FAISS_SITE"

du -h data/index/bge_extended_Flat.index
wc -l data/corpus/wiki18_extended.jsonl
```

预期：

```text
index: 约 64GB
corpus: 22,352,695 行
```

### 4.2 检查 GPU0

先直接检查 GPU。只有 `nvidia-smi` 失败且当前 Worker 存在已验证的实际
驱动库时，才设置 `LD_PRELOAD`：

```bash
NVML_FIX=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.183.06
if ! nvidia-smi >/dev/null 2>&1 && [ -f "$NVML_FIX" ]; then
  export LD_PRELOAD="$NVML_FIX"
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.total \
  --format=csv,noheader
```

`535.183.06` 是当前 H20 Worker 已验证的修复版本，不应无条件用于不同
驱动版本的新 Worker，也不要全局写入 `.bashrc`。

GPU0 至少需要约 72GiB 可用显存。若 GPU0 已有大进程，不要继续加载 index；
先确认该进程是否属于其他会话。

### 4.3 检查端口和现有服务

必须先查健康接口，避免重复加载：

```bash
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

curl -sS http://127.0.0.1:8100/health
ss -ltnp | grep ':8100' || true
pgrep -af retrieval_daemon.py || true
```

判定：

- `/health` 返回 `"status":"ok"`：直接复用，不要重启；
- 有进程但 health 尚未 ready：检查日志，等待冷启动；
- 端口被其他程序占用：不要另起第二个 daemon；
- `curl` 返回 403：通常是 localhost 被代理，重新设置 `NO_PROXY/no_proxy`。

## 5. 正式启动命令

当前训练环境负责 `torch/transformers/datasets/FastAPI`，持久 FAISS 环境
只提供 GPU FAISS。两者通过 `FAISS_PYTHON_SITE` 组合；不要把整个训练
环境切换到 micromamba 的 FAISS Python。

```bash
REPO="$(git rev-parse --show-toplevel)"
WORKSPACE="$(dirname "$REPO")"
FAISS_SITE=$WORKSPACE/envs/sapr_micromamba_root/envs/sapr_faiss_gpu_cuda129_py311/lib/python3.11/site-packages
cd "$REPO"

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

NVML_FIX=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.183.06
if ! nvidia-smi >/dev/null 2>&1 && [ -f "$NVML_FIX" ]; then
  export LD_PRELOAD="$NVML_FIX"
fi

export RETRIEVAL_GPU=0
export RETRIEVAL_DEVICES=0
export RETRIEVAL_PORT=8100

export FAISS_DEVICE=gpu
export FAISS_GPU_ID=0
export FAISS_GPU_FP16=false
export FAISS_GPU_TEMP_MB=2048

export PYTHON_EXECUTABLE=/usr/bin/python
export FAISS_PYTHON_SITE="$FAISS_SITE"

bash 03_sapr_rag/scripts/grpo/retrieval_service.sh start
WAIT_TIMEOUT=1200 \
  bash 03_sapr_rag/scripts/grpo/retrieval_service.sh wait
```

说明：

- `RETRIEVAL_DEVICES=0` 会设置 `CUDA_VISIBLE_DEVICES=0`；
- 进程内部只看到一张卡，所以 `FAISS_GPU_ID` 必须写局部编号 `0`；
- 当前已验证配置为 fp32，`FAISS_GPU_FP16=false`；
- 不要省略 `FAISS_DEVICE=gpu`，脚本默认值仍是 `cpu`；
- `FAISS_GPU_TEMP_MB=2048` 防止 FAISS 申请过大的临时显存池；
- 服务通过 `setsid + nohup` 脱离 SSH，不需要额外手工放入 tmux。

## 6. 启动成功标准

### 6.1 健康检查

```bash
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

curl -sS http://127.0.0.1:8100/health | python -m json.tool
```

必须确认：

```json
{
  "status": "ok",
  "n_vectors": 22352695,
  "n_docs": 22352695,
  "text_truncate": 500,
  "bge_device": "cuda:0",
  "faiss_device": "gpu",
  "faiss_gpu_id": 0,
  "faiss_gpu_fp16": false
}
```

只看到 `"status":"ok"` 不够，还要核对 `n_vectors/n_docs` 和 GPU 字段。

### 6.2 实际查询

```bash
curl -sS -X POST http://127.0.0.1:8100/search \
  -H 'content-type: application/json' \
  -d '{"query":"Who founded Apple?","top_k":3}' \
  | python -m json.tool
```

必须返回 3 个含 `title/text/score` 的文档。

### 6.3 进程和显存

```bash
bash 03_sapr_rag/scripts/grpo/retrieval_service.sh status

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

当前已验证服务的进程参数应包含：

```text
--device cuda:0
--faiss_device gpu
--faiss_gpu_id 0
```

GPU0 显存约为 68.9GiB。空闲时利用率为 0% 正常；是否成功应以
health、查询返回和显存占用共同判断。

## 7. 启动训练

训练 wrapper 不负责创建 retrieval daemon，只会检查 `8100/health`。
必须先完成第 5–6 节。

以 E16 为例：

```bash
REPO="$(git rev-parse --show-toplevel)"
tmux new-session -d -s canonical_opsd \
  "cd '$REPO' && \
   export NO_PROXY=127.0.0.1,localhost,::1 && \
   export no_proxy=\$NO_PROXY && \
   bash 03_sapr_rag/scripts/grpo/run_canonical_sft_multi_opsd_s1000.sh"
```

如果第 4.2 节设置了 `LD_PRELOAD`，`tmux` 会继承当前 shell 的环境，无需
在命令中再次硬编码。

检查：

```bash
tmux ls
tail -f 03_sapr_rag/scripts/grpo/logs/opsd_canonical_sft_q001_a003_3src_s1000_20260905/train.log
```

正式训练应使用对应实验的顶层 wrapper，不要直接临时拼
`swift rlhf` 参数。当前 E16 布局：

```text
GPU0    retrieval daemon
GPU1    预留评测
GPU2-6  train
GPU7    rollout + Evidence Agent
```

## 8. 启动正式评测

评测同样复用 `8100`，不要为每个数据集复制 FAISS index。

单 checkpoint 三数据集：

```bash
MODE=full \
CHECKPOINT_STEP=<step> \
RUN_ROOT=<包含 checkpoint-* 的运行目录> \
ROLLOUT_GPU=<空闲 GPU> \
ROLLOUT_PORT=<空闲端口> \
OUT_ROOT=<评测输出目录> \
bash 03_sapr_rag/scripts/eval/eval_action_opsd_3src.sh
```

只评一个数据集：

```bash
DATASETS_CSV=hotpotqa \
MODE=full \
CHECKPOINT_STEP=<step> \
RUN_ROOT=<运行目录> \
ROLLOUT_GPU=<空闲 GPU> \
ROLLOUT_PORT=<空闲端口> \
OUT_ROOT=<评测输出目录> \
bash 03_sapr_rag/scripts/eval/eval_action_opsd_3src.sh
```

多卡并行评测时，每个数据集使用独立 `ROLLOUT_GPU/ROLLOUT_PORT/OUT_ROOT`，
但都访问同一个 `http://127.0.0.1:8100`。

## 9. 服务管理

```bash
# 状态
bash 03_sapr_rag/scripts/grpo/retrieval_service.sh status

# 等待 ready
WAIT_TIMEOUT=1200 \
  bash 03_sapr_rag/scripts/grpo/retrieval_service.sh wait

# 停止
bash 03_sapr_rag/scripts/grpo/retrieval_service.sh stop

# 重启 GPU 服务
# 先 stop，再完整重跑第 5 节的环境变量和 start/wait 命令
```

`stop` 会中断所有正在使用检索服务的训练和评测。执行前必须确认没有其他
会话正在访问端口 8100。不要在新 shell 中直接执行裸 `restart`：环境变量
没有保留时，它会按脚本默认值启动 FAISS CPU。

## 10. 常见故障

### 10.1 health 返回 403

原因：localhost 请求经过代理。

```bash
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
```

### 10.2 `No platform detected` 或 `nvidia-smi` 失败

先确认当前 shell 已登录 Worker。若 `nvidia-smi` 仍失败，查找与内核驱动
匹配的实际 NVML 库；当前已验证 Worker 可使用：

```bash
test -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.183.06
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.183.06
```

新 Worker 若没有该文件，不要照抄版本号；先用 `uname -r` 和
`ldconfig -p | grep libnvidia-ml` 核对。不要把该设置全局写入 `.bashrc`。

### 10.3 `This faiss build does not support GPU indices`

说明加载了系统 `faiss-cpu`，没有注入 GPU FAISS：

```bash
REPO="$(git rev-parse --show-toplevel)"
WORKSPACE="$(dirname "$REPO")"
export FAISS_PYTHON_SITE="$WORKSPACE/envs/sapr_micromamba_root/envs/sapr_faiss_gpu_cuda129_py311/lib/python3.11/site-packages"
export PYTHON_EXECUTABLE=/usr/bin/python
```

### 10.4 CUDA `no kernel image` / error 209

说明误用了旧 `faiss-gpu-cu12 1.14.1`。当前 H20 必须使用持久环境中的
FAISS GPU 1.14.3，不要修改训练主环境中的 FAISS。

### 10.5 FAISS clone OOM

检查：

```bash
pgrep -af retrieval_daemon.py
nvidia-smi
```

常见原因：

- 已有一个 daemon，重复加载；
- GPU0 被其他模型占用；
- 未设置 `FAISS_GPU_TEMP_MB=2048`；
- 使用 fp32 时 GPU 可用显存不足约 72GiB。

不要通过反复启动解决 OOM；先确认现有进程归属。

### 10.6 训练启动时报 retrieval health 失败

```bash
curl -sS http://127.0.0.1:8100/health
tail -n 100 03_sapr_rag/scripts/grpo/logs/retrieval_service.log
pgrep -af retrieval_daemon.py
```

训练脚本不会自动修复或重启 retrieval。

## 11. 不再推荐的旧方案

`docs/retrieval_faiss_cpu_env_worker_setup_20260810.md` 记录的是历史
“BGE GPU + FAISS CPU”回退方案。它适合 GPU FAISS 不可用时排障，但不是
当前 H20 正式训练的首选。

`docs/grpo_opsd_pipeline_overview.md` 中“FAISS index 仍在 CPU mmap”的
描述也是历史状态。当前正式配置应以本文和 `/health` 返回值为准。

## 12. 相关代码

- 服务管理：`03_sapr_rag/scripts/grpo/retrieval_service.sh`
- 启动包装：`03_sapr_rag/scripts/grpo/run_retrieval_daemon_flexible.sh`
- daemon：`03_sapr_rag/scripts/grpo/retrieval_daemon.py`
- HTTP client：`03_sapr_rag/scripts/grpo/retrieval_client.py`
- 训练 pipeline：`docs/grpo_opsd_pipeline_overview.md`
- 索引构建：`docs/index_build.md`
- 实验总表：`docs/experiment_tracker.md`
