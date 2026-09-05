# BGE wiki18_extended FAISS 索引构建文档

> 本文记录索引的**构建历史**。第 7–8 节的部署位置和共享盘阻塞是旧状态，
> 不代表当前运行环境。正式 H20 Worker 检索服务的环境、GPU 部署和启动命令
> 见 [`retrieval_service_gpu_runbook.md`](retrieval_service_gpu_runbook.md)。

记录 64GB BGE FAISS 索引的完整构建流程，作为重建（IVF/IVF-PQ/HNSW）和迁移的事实底稿。

## 1. 产物总览

| 产物 | 路径 | 大小 | 说明 |
|---|---|---|---|
| `wiki18_extended.jsonl` | `data/corpus/` | 14GB | 22,352,695 行；每行 `{id, contents}` |
| `bge_extended_Flat.index` | `data/index/` | 64GB | FAISS `IndexFlatIP`，22,352,695 × 768 fp32 |
| `bge_extended_Flat.meta.json` | `data/index/` | 345B | 编码参数 + n_vectors / dim |
| 中间产物 `_chunks_extended/chunk_*.npy` | `data/index/_chunks_extended/` | ~32GB | 448 个 chunk，每 chunk 50,000 vec × 768 fp32 ≈ 70MB |

doc_id（即 FAISS 内部行号）= jsonl 行号（0-indexed），构建期严格保序。

## 2. 数据来源与合并

由 [build_extended_corpus.py](../03_sapr_rag/data/build_v1/00_prep/build_extended_corpus.py) 顺序拼接：

| 段 | 来源 | 行数 | 字段约定 |
|---|---|---|---|
| 0 ～ ~21M | FlashRAG `wiki18_100w.jsonl` | ~21M | `{id, contents="title\n正文"}` |
| ~21M ～ ~22.35M | ReasonRAG `RAG_extend_corpus`（parquet）| ~1.35M | `{id, title, contents}` → 拼成 `title\n正文` |

合并规则（[build_extended_corpus.py:38-46](../03_sapr_rag/data/build_v1/00_prep/build_extended_corpus.py)）：
- `contents = "title\n正文"`，title 去引号
- 跳过 title+text 都为空的行
- 全局 doc_id 自增（`n_base + n_extend`）

## 3. 编码参数

[build_extended_index.py](../03_sapr_rag/data/build_v1/00_prep/build_extended_index.py) `_encode_chunks`：

| 参数 | 取值 | 备注 |
|---|---|---|
| 模型 | `bge-base-en-v1.5` | 跟在线 retriever、step3 检索完全一致 |
| pooling | CLS（`last_hidden_state[:, 0, :]`）| BGE 官方推荐 |
| L2 normalize | 是（fp32 阶段）| FAISS IP = cosine |
| `max_seq_len` | 256 | 与 ReasonRAG 对齐 |
| `batch_size` | 256 | H20 100GB 实跑值；可调到 512 |
| `dtype` | fp16（forward）| chunk 落盘转 fp32 |
| `chunk_size` | 50,000 | 每个 chunk_*.npy 约 70MB |

**doc 不加任何 prefix**，只有 query 端在线检索时加 `"Represent this sentence for searching relevant passages: "`（[agent_infer.py:64](../03_sapr_rag/scripts/eval/agent_infer.py)）。

## 4. 索引参数

[build_extended_index.py](../03_sapr_rag/data/build_v1/00_prep/build_extended_index.py) `_build_faiss`：

```python
index = faiss.IndexFlatIP(768)  # 22.35M × 768 × 4B ≈ 64GB
for chunk_path in sorted(out_dir.glob("chunk_*.npy")):
    arr = _l2_normalize(np.load(chunk_path).astype(np.float32))
    index.add(arr)
faiss.write_index(index, str(index_out))
```

二次 L2 normalize 是防御性的（fp16 → fp32 期间可能丢精度）。

## 5. 实测耗时与资源（4× H20 100GB）

来自 [data/logs/prep/encode_chunks.log](../data/logs/prep/encode_chunks.log) 与 [data/logs/prep/build_index.log](../data/logs/prep/build_index.log)：

| 阶段 | 耗时 | 资源 |
|---|---|---|
| **encode**（4 卡 torchrun，`--encode-only`）| **3,141 秒（52 分钟）** | 4× H20，每卡 ~5.6M vec |
| **build FAISS**（rank 0 单进程）| **195 秒** | 单核 CPU，瞬时内存 ~64GB |

448 chunk × 50,000 = 22,400,000 槽位，实际写入 22,352,695（少的是空文档）。

单卡 H20 100GB 预估 2-3h；单卡 RTX 3090 8GB 24-48h（`launch_build_index.sh` 头注释）。

## 6. 启动方式（多卡 H20，已验证）

```bash
source config/env_local.sh

# Phase A: encode（torchrun 多卡数据并行）
# Phase B: rank 0 单进程把 chunk_*.npy 拼成 IndexFlatIP
NPROC_PER_NODE=4 BATCH_SIZE=256 \
CHUNKS_DIR=/tmp/sapr_chunks \
INDEX_OUT=/tmp/sapr_index/bge_extended_Flat.index \
bash 03_sapr_rag/data/build_v1/00_prep/launch_build_index.sh
```

实际启动命令分两阶段（[launch_build_index.sh:115-131](../03_sapr_rag/data/build_v1/00_prep/launch_build_index.sh)）：

```bash
# Phase A
torchrun --nproc_per_node=4 build_extended_index.py \
    --chunks-dir <CHUNKS_DIR> --index-out <INDEX_OUT> \
    --batch-size 256 --max-seq-len 256 --dtype fp16 --encode-only

# Phase B
python build_extended_index.py \
    --chunks-dir <CHUNKS_DIR> --index-out <INDEX_OUT> --build-only
```

多卡分片策略（[build_extended_index.py:301-303](../03_sapr_rag/data/build_v1/00_prep/build_extended_index.py)）：
每个 rank 都流式扫整个 corpus，但只处理 `chunk_idx % world_size == rank` 的 chunk。
IO 上 4 倍冗余（4 个 rank 各扫一遍 14GB jsonl ≈ 4 × 30s），相对总时长 52min 可忽略。

重跑安全：每个 chunk_*.npy 是独立产物，崩溃后重跑会跳过尺寸正确的现有 chunk。

## 7. 当前部署位置（截至本次记录）

```
master /tmp/sapr_archive/index/
├── bge_extended_Flat.index     (64GB)
└── bge_extended_Flat.meta.json (345B)
```

```json
{
  "n_vectors": 22352695,
  "dim": 768,
  "encoder_model": ".../models/bge-base-en-v1.5",
  "corpus_path": ".../data/corpus/wiki18_extended.jsonl",
  "chunk_size": 50000,
  "batch_size": 256,
  "max_seq_len": 256,
  "dtype": "fp16",
  "world_size": 1
}
```

`world_size: 1` 是写 meta 时 rank 0 build phase 的 env，不是 encode phase（encode 实际是 4 卡）。

## 8. 节点可见性（关键约束）

| 路径 | master | worker (GPU 节点) |
|---|---|---|
| `/tmp/sapr_archive/` (`/dev/nvme0n1p3`, 1.7T) | ✓ | ✗ 节点本地盘 |
| `/mlx_devbox/users/.../playground/` (`/dev/vdak`, 125G) | ✓ | ✓ |
| `/home/tiger/` (`/dev/vdak`, 125G) | ✓ | ✓ 与 mlx_devbox 同盘 |

**symlink 不能跨节点工作**——symlink 只是字符串重定向，target 必须在当前节点存在。

**当前阻塞**：worker 跑推理需要 64G index，但 mlx_devbox 共享盘只有 53G 空闲（清理后）。

## 9. 重建为更小索引的参数预算

如果要走"重建小索引"方案（方案 C），可复用本流程，**只换 `_build_faiss`**：

| 类型 | 大小估算 | recall vs Flat | 备注 |
|---|---|---|---|
| `IndexFlatIP`（当前）| 64GB | 100% | 22.35M × 768 × 4B |
| `IndexHNSWFlat(M=32)` | ~70GB | ~99% | 增加 graph，**反而更大** |
| `IndexIVFFlat(nlist=4096)` | ~64GB | 95-99% | 只省查询时间，不省存储 |
| `IndexIVFPQ(nlist=4096, m=64, nbits=8)` | **~1.5GB** | 85-95% | 每个 vec 压成 64 字节 |
| `IndexIVFPQ(nlist=4096, m=128, nbits=8)` | ~3GB | 90-97% | 每个 vec 128 字节，更稳 |

PQ 类索引必须 `train`（取 ~256k 样本），训练 + add 在 64GB 内存机器上约 1-2 小时。

复用 chunk_*.npy 重建脚本骨架：

```python
import faiss, numpy as np
chunks = sorted(Path("_chunks_extended").glob("chunk_*.npy"))
quantizer = faiss.IndexFlatIP(768)
index = faiss.IndexIVFPQ(quantizer, 768, 4096, 64, 8)
index.metric_type = faiss.METRIC_INNER_PRODUCT
# train with sample
sample = np.concatenate([np.load(c).astype("float32") for c in chunks[:6]], axis=0)
index.train(sample)
for c in chunks:
    index.add(np.load(c).astype("float32"))
faiss.write_index(index, "bge_extended_IVFPQ.index")
```

**前置条件**：保留现有 `_chunks_extended/` 总计 ~32GB。当前已被删除（migrate 后），重建需要重跑 encode 阶段（52 min on 4× H20）。

## 10. 已知问题与未来改进

1. **`_fetch` 顺序扫 jsonl** ([agent_infer.py:104-115](../03_sapr_rag/scripts/eval/agent_infer.py))：每次检索回查文档要 `for idx, line in enumerate(f)` 全表扫描 22M 行，单次 ~秒级。FlashRAG / R3-RAG 用 `datasets.load_dataset` mmap 实现 O(1) lookup，更优。
2. **doc store 与 index 强绑定**：jsonl 行号即 doc_id，任何对 corpus 的增删都会破坏对齐，必须重建 index。
3. **fp16 编码 + fp32 落盘**：当前实现 forward 用 fp16 但 chunk 存 fp32，存盘其实多了 2x 空间。如果接受 ~0.5% recall 损失，可改 fp16 落盘 → index 减半到 32GB。

## 11. 相关文件索引

- [build_extended_corpus.py](../03_sapr_rag/data/build_v1/00_prep/build_extended_corpus.py) - corpus 合并脚本
- [build_extended_index.py](../03_sapr_rag/data/build_v1/00_prep/build_extended_index.py) - encode + build 主脚本
- [launch_build_index.sh](../03_sapr_rag/data/build_v1/00_prep/launch_build_index.sh) - shell 包装，含多卡启动
- [download_assets.sh](../03_sapr_rag/data/build_v1/00_prep/download_assets.sh) - 前置：下 wiki18_100w + RAG_extend_corpus + bge-base
- [data/logs/prep/encode_chunks.log](../data/logs/prep/encode_chunks.log) - encode 阶段实测日志
- [data/logs/prep/build_index.log](../data/logs/prep/build_index.log) - build 阶段实测日志
- [agent_infer.py:67-115](../03_sapr_rag/scripts/eval/agent_infer.py) - 推理端 retriever 实现（消费方）
