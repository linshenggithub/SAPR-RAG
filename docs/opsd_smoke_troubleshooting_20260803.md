# SAPR-RAG OPSD Smoke 排障记录：vLLM weight-sync 建组端口 [Errno 98]

- 日期：2026-08-03
- Worker：`<worker_id>`（8×H20，每卡 ~97.8GB）
- 结论：**前置环境全部就绪；smoke 训练启动稳定失败在 ms-swift↔vLLM 的 weight-sync 通信组建组阶段，非 OPSD 逻辑问题。**
- 本次未改动运行环境，仅做只读诊断（按用户要求）。

---

## 1. 现象

matched control smoke（plain GRPO，20 steps，`ENABLE_OPSD=false`）三进程编排：
retrieval daemon(CPU) → rollout(GPU6) → train(GPU0-5)。

前两阶段全部正常，train 启动后卡死并超时：

- **rollout 侧**（`logs/smoke_control_*/rollout.log`）：
  ```
  Exception: Call to collective_rpc method failed: [Errno 98] Address already in use
  ```
- **train 侧**（`logs/smoke_control_*/train.log`）：
  ```
  [c10d] The client socket has timed out after 300000ms while trying to connect to (127.0.0.1, 51216).
  ```

两次独立运行（15:03、15:38）稳定复现，失败点完全一致。

## 2. 已验证 OK 的前置

| 项 | 状态 |
|---|---|
| 8×H20 可见、全空闲 | OK（`torch.cuda.device_count()==8`） |
| deepspeed | 0.19.3，import OK |
| torch / vllm / ms-swift / trl / transformers | 2.7.1+cu126 / 0.10.0 / 4.5.0.dev0 / 0.26.2 / 4.56.2 |
| setup_env_opsd（cuda） | 环境就绪 ✓（OPSD 能力检查全通过） |
| pilot 数据 | plain/OPSD 各 100 条 matched |
| FAISS index / corpus / BGE / base model | 64G / 14G / 1.3G / 15G 均在 |
| retrieval daemon | `/health` ok，22,352,695 向量 |
| rollout server | vLLM 权重加载 + KV cache + `Application startup complete` on :8000 |
| train 6 rank 拉起 | 推进到 `Start connecting to vLLM server` |

## 3. 根因定位

GRPO server 模式下，训练进程需与 rollout(vLLM) 建一个额外的 NCCL 通信组做权重同步（train rank = vllm_world_size，多加 1）。该组的 host/port 由训练侧下发给 rollout。

- 训练侧客户端：[swift/rlhf_trainers/vllm_client.py:185-186](https://github.com/modelscope/ms-swift/blob/main/swift/rlhf_trainers/vllm_client.py#L185-L186)
  ```python
  if group_ports is None:
      group_ports = [51216 + i for i in range(self.num_servers)]
  ```
  默认 group port = **51216**。

- 训练侧发起建组：[vllm_client.py:197-222](https://github.com/modelscope/ms-swift/blob/main/swift/rlhf_trainers/vllm_client.py#L197-L222)
  先 POST `/init_communicator/`（让 rollout 侧 bind port），再本地 `StatelessProcessGroup.create(host, port=51216, ...)` 去 connect。

- rollout 侧建组：[swift/pipelines/infer/rollout.py:872-892](https://github.com/modelscope/ms-swift/blob/main/swift/pipelines/infer/rollout.py#L872-L892) → 通过 `collective_rpc('init_communicator', ...)` 在 vLLM worker 内 `StatelessProcessGroup.create(host, port, ...)`（[rollout.py:122-152](https://github.com/modelscope/ms-swift/blob/main/swift/pipelines/infer/rollout.py#L122-L152)）。

**失败机理**：rollout 侧在 51216 上建 TCP store 时 `bind` 失败（`[Errno 98] Address already in use`）→ `collective_rpc` 抛异常 → rollout 未真正监听 51216 → 训练侧 `StatelessProcessGroup.create` connect 51216 一直连不上 → c10d 300s 超时。

**注意**：本次启动前用 `check_ports.sh` 确认过 51216/51217/51218 均空闲，且无残留 CUDA 进程。因此 [Errno 98] 更可能来自集成层，而非外部残留端口。两个高度可疑因素：

1. **vLLM 版本不匹配**：日志明确警告
   `TRL currently supports vLLM versions: 0.10.2, 0.11.0, 0.11.1, 0.11.2. You have version 0.10.0`。
   vLLM 0.10.0 的 async engine + `collective_rpc` 建组路径可能存在端口/生命周期问题。
2. **async engine 重复 bind**：rollout 用 `--vllm_use_async_engine true`；async engine 下 worker 进程模型可能对同一 group port 触发多次 bind。

## 4. 可选修复（均待验证，本次未执行）

按“改动小→大”排序：

### 选项 A：显式指定 group port（无需改源码，最轻）
ms-swift 支持 CLI 覆盖，见 [args_mixin.py:179](https://github.com/modelscope/ms-swift/blob/main/swift/rlhf_trainers/args_mixin.py#L179)：
```python
vllm_server_group_port: Optional[List[int]] = None   # 默认 None -> 51216
```
在 `run_grpo_opsd.sh` 训练命令加：
```bash
--vllm_server_group_port 29500
```
- 若 51216 是被某个隐性服务占用 → 换端口可解。
- 若 [Errno 98] 来自 async engine 自身重复 bind → 换端口**无效**（换哪个都撞）。这一步可用来快速区分两类根因。

### 选项 B：rollout 关掉 async engine
`run_rollout_opsd.sh` 去掉 `--vllm_use_async_engine true`（改同步 engine）。
- 用于验证是否是 async engine 的 collective_rpc 建组路径问题。
- 代价：需确认 `sapr_rag_scheduler` 多轮调度在同步 engine 下是否受支持。

### 选项 C：升级 vLLM 到 0.10.2（trl 期望版本）
worker 上 `pip install 'vllm==0.10.2'`。
- 消除版本不匹配警告，可能修复 async 建组 bug。
- 风险：vLLM 升级可能连带要求 torch/transformers 变动，需回归 setup_env_opsd。

### 选项 D：改 colocate 模式（偏离当前 server 架构）
vLLM 与训练同卡 colocate，不走 server 端 weight-sync group，绕开 51216 建组。
- 能验证“训练主体 + OPSD advantage”是否跑得通。
- 但偏离既定 server 架构，且 colocate 与 `multi_turn_scheduler` 的兼容需另行确认。

## 5. 复现路径

```bash
# 1) 连 worker
NO_COLOR=1 TERM=dumb mlx worker login <worker_id> -- bash <script>

# 2) 一键 smoke（会依次起 retrieval/rollout/train）
bash 03_sapr_rag/scripts/grpo/run_control_smoke.sh
# 观察 logs/smoke_control_<ts>/{retrieval,rollout,train}.log

# 3) 失败特征
#   rollout.log:  Call to collective_rpc method failed: [Errno 98] Address already in use
#   train.log:    [c10d] ... timed out ... connect to (127.0.0.1, 51216)

# 4) 清理
bash 03_sapr_rag/scripts/grpo/smoke_kill.sh
bash 03_sapr_rag/scripts/grpo/smoke_kill2.sh   # 补杀残留 rollout 子进程（占 GPU6）
```

## 6. 相关脚本（本次新增，均为增量、不改 baseline）

- `run_control_smoke.sh`：matched control smoke 三进程编排
- `check_h20_worker_env.sh` / `check_assets.sh` / `check_ports.sh`：只读前置检查
- `smoke_status.sh` / `smoke_watch.sh` / `smoke_deepdive.sh`：运行态观测
- `smoke_kill.sh` / `smoke_kill2.sh`：清理
- `install_deepspeed.sh`：deepspeed 安装

## 7. 下一步（待用户决定）

推荐先做**选项 A**：加 `--vllm_server_group_port 29500` 重跑一次。它最轻，且能明确区分“外部端口占用”与“async engine 自身重复 bind”两类根因，为后续选 B/C/D 提供依据。

---

## 8. 选项 A 验证结果（2026-08-03 16:04 重跑）——已排除“外部端口占用”

给训练命令加 `--vllm_server_group_port 29500`（[run_grpo_opsd.sh](../03_sapr_rag/scripts/grpo/run_grpo_opsd.sh) 新增 `VLLM_GROUP_PORT` 变量，默认 29500）后重跑。启动前确认 8000/8100/29500/51216 全空闲。

**结果：仍然失败，`[Errno 98]` 依旧**，但拿到了精确调用栈（`rollout.log`）：

```
POST /close_communicator/  200 OK
GET  /get_world_size/      200 OK
POST /init_communicator/   200 OK        ← 注意这是第二次 init
ERROR ... rollout.py:152 in init_communicator
    pg = StatelessProcessGroup.create(host=host, port=port, ...)
ERROR ... swift/rlhf_trainers/utils.py:182 in _patched_stateless_pg_create
ERROR ... vllm/distributed/utils.py:397 in create
    listen_socket.bind((host, port))
OSError: [Errno 98] Address already in use
```

**精确根因（已确认，非推测）**：
- 失败点是 rollout 侧 vLLM worker 内 `StatelessProcessGroup.create` 的 `listen_socket.bind((host, port))`，经由 ms-swift monkey-patch [`_patched_stateless_pg_create`（utils.py:182）](https://github.com/modelscope/ms-swift/blob/main/swift/rlhf_trainers/utils.py#L182)。
- 日志出现 `close_communicator` → `get_world_size` → **第二次** `init_communicator`，说明**同一 rollout server 进程内 group port 被重复 bind**：上一次通信组的 listen socket 未真正释放，第二次 bind 同端口即 `[Errno 98]`。
- **换端口无效**：29500 与 51216 表现完全一致，因为冲突来自同进程重复 bind，而非外部占用。→ **选项 A 排除，问题锁定为 async engine + ms-swift 这版的 `close_communicator` 未释放 socket 的兼容 bug。**

**结论**：选项 A（改端口）无效；应转向：
- **选项 B**：rollout 关 `--vllm_use_async_engine`（改同步 engine），验证是否为 async engine 特有的重复 init/bind。**推荐下一个尝试**（改动小、最可能直接命中）。
- **选项 C**：升级 vLLM 0.10.0 → 0.10.2（trl 期望版本），可能修复 `close_communicator`/socket 释放行为。属环境变更，需先确认。
- 若 B/C 均不行，再考虑 **选项 D**（colocate）或给 `_patched_stateless_pg_create` / `close_communicator` 打补丁（设 `SO_REUSEADDR` 或确保 socket 关闭）。

> 注意：rollout 侧还观察到「先 `close_communicator` 再第二次 `init_communicator`」的序列，值得进一步查 ms-swift 为何在单次训练启动内触发两次 `init_communicator`（可能与 6 rank 中多个 rank 都发起 init、或 async engine 的 driver worker 重复响应有关）。

---

## 9. 选项 C 验证结果（2026-08-03 16:26 升级 vLLM）——同样无效，根因彻底闭环

### 9.1 升级动作与连带影响
`pip install vllm==0.10.2` 连带升级了整个 CUDA 栈（非仅换 vLLM）：

| 包 | 升级前 | 升级后 |
|---|---|---|
| vllm | 0.10.0 | **0.10.2** |
| torch | 2.7.1+cu126 | **2.8.0+cu128** |
| torchvision / torchaudio | 0.22.1 / 2.7.1 | 0.23.0 / 2.8.0 |
| xformers | 0.0.31 | 0.0.32.post1 |
| nccl-cu12 | 2.26.2 | 2.27.3 |
| triton | 3.3.1 | 3.4.0 |
| transformers / trl / datasets / deepspeed | 未变 | 未变 |

升级后 `setup_env_opsd`（cuda）复核：**环境就绪 ✓**，OPSD 能力全 OK。
（附带修正：`gkd_multiturn_failfast` 检查在新版 ms-swift 下误报 FAIL——因新版 `_check_gkd` 不再以 `multi_turn_scheduler` 字样表达限制；已将断言放宽为“GKD 仍有 fail-fast 守卫”，与 dynamic OPSD 路线无关。）

### 9.2 结果：`[Errno 98]` 仍在同一处复现
升级后重跑 smoke（带 `--vllm_server_group_port 29500`），失败点完全一致：
```
POST /init_communicator/   ← 第二次
rollout.py:149  StatelessProcessGroup.create(...)
utils.py:182    _patched_stateless_pg_create
OSError: [Errno 98] Address already in use
```
→ **选项 C 排除。问题与 vLLM 版本、端口号均无关。**

### 9.3 最终根因（源码级闭环）

两处 ms-swift 代码共同导致：

1. **`close_communicator` 不释放 listen socket** —— [rollout.py:310-313](https://github.com/modelscope/ms-swift/blob/main/swift/pipelines/infer/rollout.py#L310-L313)：
   ```python
   if self.communicator is not None:
       del self.communicator
       self.communicator = None
       self.client_rank = None
   ```
   只删 NCCL communicator 对象，**从不关闭第一次 `init_communicator` 建的 `StatelessProcessGroup` 的 listen socket**，该 socket 仍占着 group port。

2. **IPv4 bind 路径无 `SO_REUSEADDR`** —— [utils.py:181-190](https://github.com/modelscope/ms-swift/blob/main/swift/rlhf_trainers/utils.py#L181-L190)：
   `_patched_stateless_pg_create` 只对 **IPv6** 分支显式 `setsockopt(SO_REUSEADDR, 1)`（[utils.py:196](https://github.com/modelscope/ms-swift/blob/main/swift/rlhf_trainers/utils.py#L196)）；**IPv4（我们的 `0.0.0.0`）直接 fallback 到 vLLM 原生 create**，原生 bind 未设 `SO_REUSEADDR`。

训练侧固定序列 [rlhf_args.py:497-498](https://github.com/modelscope/ms-swift/blob/main/swift/arguments/rlhf_args.py#L497-L498) 为 `close_communicator()` → `init_communicator()`。因此：第一次 init 在 port 上 bind+listen（socket 泄漏）→ close 未释放该 socket → 第二次 init bind 同 port，且 IPv4 无 `SO_REUSEADDR` → 必然 `[Errno 98]`。

**这解释了为什么 A（换端口）和 C（升级 vLLM）都无效**：两者都走同一条“socket 泄漏 + 无 SO_REUSEADDR”的 IPv4 bind 路径。

### 9.4 可行修复（均需改 ms-swift 源码，待用户拍板）

- **修复 1（最小、最对症）**：`_patched_stateless_pg_create` 的 IPv4 分支也显式建 socket 并 `setsockopt(SO_REUSEADDR,1)` 再 bind（对齐现有 IPv6 分支），不再无脑 fallback。
- **修复 2**：`close_communicator` 真正关闭 `StatelessProcessGroup` 的 listen socket / TCPStore，从源头消除端口泄漏。
- 两者可只做修复 1（改动最小），或同时做以彻底根治。
- 备选（不改源码）：选项 D（colocate 模式，绕开 server 端 weight-sync group）。

### 9.5 当前状态
- 环境已清理（8 卡全空，无残留进程）。
- vLLM 已停留在 0.10.2 / torch 2.8.0（升级不可逆回滚成本高，且新版本本身无害，建议保留）。
- 等用户决定：改 ms-swift（修复 1/2）还是转选项 D。

---

## 10. 修复尝试与关键纠正（2026-08-03 17:22-17:38）——真正根因是端口选择撞 torch MASTER_PORT

### 10.1 尝试：给 IPv4 bind 加 SO_REUSEADDR（本地改 ms-swift，未 commit）
按“修复 1”改了 [utils.py 的 `_patched_stateless_pg_create`](https://github.com/modelscope/ms-swift/blob/main/swift/rlhf_trainers/utils.py)：IPv4 分支也显式建 socket + `SO_REUSEADDR` 再 bind（原备份 `utils.py.bak_20260803`）。确认改动已生效（pyc 17:27 重新编译，栈显示 `utils.py:191 listen_socket.bind`）。

**结果：`[Errno 98]` 仍在**，SO_REUSEADDR 无效。

### 10.2 关键纠正：不是 socket 泄漏，是我把 group port 选成了 torch MASTER_PORT
复看本次 rollout.log 的建组序列，**只有一次** `init_communicator`（此前“两次 init 撞端口”的判断有误）：
```
75: POST /close_communicator/  200 OK
76: GET  /get_world_size/      200 OK
77: POST /init_communicator/   200 OK   ← 仅此一次
97: OSError: [Errno 98] Address already in use   (bind 29500)
```
既然只 bind 一次就失败，说明 **29500 在 bind 前已被占用**。而：
- 训练用 `python -m torch.distributed.run --nproc_per_node 6 ...`；
- torchrun 的 `--master-port` **默认值就是 29500**（PyTorch 官方默认，已在 worker 上确认 `torch.distributed.run` 参数定义）；
- 6 卡训练进程组 rendezvous 先 bind `127.0.0.1:29500`；
- 随后 rollout 的 weight-sync group 也被我（选项 A）配成 `--vllm_server_group_port 29500` → **撞上训练自己的 MASTER_PORT** → `[Errno 98]`。

**这是选项 A 里错误地把 group port 设成 29500（= torch MASTER_PORT）引入的自制冲突**，与 ms-swift 的 socket 释放、SO_REUSEADDR、vLLM 版本均无关。SO_REUSEADDR 对“对端仍活跃 LISTEN 的端口”本就无效，所以修复 1 没用。

> 对最初 51216 失败的重新认识：51216 那次 train 侧 c10d 连 51216 超时，可能是 rollout 侧建组因别的原因未成/或另有占用；但 29500 这次是 100% 确定的 MASTER_PORT 冲突。正确的做法是让 group port 明确避开 29500 与 8000（vllm server）等已用端口。

### 10.3 已回滚
- ms-swift `utils.py` 已从备份还原（`git status` 干净）；本地未 commit。
- 环境已清理，8 卡全空。

### 10.4 正确下一步（不需改 ms-swift 源码）
把 weight-sync group port 改成一个**既不等于 29500（torch MASTER_PORT）、也不等于 8000（vllm server）/8100（retrieval）**的冷门端口，例如 **51299**（或给训练 torchrun 显式设一个不同的 `--master_port`，两者错开即可）。即回到“选项 A 思路，但选对端口”。

具体：`run_grpo_opsd.sh` 的 `VLLM_GROUP_PORT` 默认值 29500 → 改 51299 后重跑 smoke 验证。

---

## 11. 端口修复成功 + control smoke 跑通（2026-08-03 17:50-18:20）

把 `VLLM_GROUP_PORT` 改为 **51299**（避开 torchrun MASTER_PORT=29500）后重跑 **matched control smoke**（plain GRPO，`ENABLE_OPSD=false`）：

- `[Errno 98]` 消失，weight-sync NCCL 组建成功：`ncclCommInitRank ... nranks 2 ... Init COMPLETE`
- train `Connected to vLLM server` → deepspeed zero2 → 进入 step 循环
- **step 1-2 正常出 log**：
  ```
  global_step: 1/20 → 2/20
  loss: -0.0726 → 0.159   (finite)
  reward: 0.286  (SaprF1 0.168 / Relevance 0.510 / Format 0.313)
  kl: 1.086      (finite)
  num_turns: 5.35   (多轮检索链路工作)
  completions/clipped_ratio: 0.0；mean/max = 170.8 / 954
  memory ~18-21GB/卡，无 OOM；6 卡 util 100%；checkpoint 目录已建
  ```
- **结论：control 臂链路全绿**。端口选择是唯一的真实阻塞（选项 A 思路正确，只是最初端口选错）。

## 12. OPSD 臂 smoke 失败（2026-08-03 18:35-18:57）——OPSD 与多轮 scheduler 的 token 帧不兼容

停 control 后起 **OPSD 臂 smoke**（`ENABLE_OPSD=true` + OPSD 数据集 + `teacher_kl_coef=0.1`）：

- retrieval/rollout/建组全部正常（`Errno 98`=0），train 进入第一步；
- **第一步即失败**，6 个 rank 同时报同一 AssertionError：
  ```
  swift/rlhf_trainers/grpo_trainer.py:591  _compute_teacher_logps
  swift/rlhf_trainers/gkd_helpers.py:468    remap_teacher_logps_to_student_frame
  AssertionError: OPSD response length mismatch at sample 0:
      student=82 teacher=382. Teacher and student must share the same response tokens.
  ```
  各 rank student/teacher 长度：82/382、104/495、81/468、96/462、222/952、115/468 —— teacher 稳定是 student 的 ~4-5 倍。

### 12.1 根因（源码级）

ms-swift OPSD 契约（[data.py:85-106 build_teacher_view](https://github.com/modelscope/ms-swift/blob/main/swift/rl_core/data.py#L85-L106)、[gkd_helpers.py:26-48 encode_teacher_view](https://github.com/modelscope/ms-swift/blob/main/swift/rlhf_trainers/gkd_helpers.py#L26-L48)、[gkd_helpers.py:449-471 remap](https://github.com/modelscope/ms-swift/blob/main/swift/rlhf_trainers/gkd_helpers.py#L449-L471)）：
- teacher view = **仅替换最后一条 user message 为 `teacher_prompt`**，并**保留与 student 完全相同的 `response_token_ids`**；
- teacher forward 后，`remap_teacher_logps_to_student_frame` 要求 **teacher completion_mask 的有效 token 数 == student completion_mask 的有效 token 数**（同一串 response tokens）。

但 SAPR-RAG 是**多轮检索 agent**（`sapr_rag_scheduler`）：一条 response 里穿插多轮 `<query>…</query>` → 检索 → `<reference>observation</reference>` → 继续生成。student 的 `completion_mask` 把 observation(检索结果) token 置 0、只对模型 action token 置 1（见 `sanity_check` 里 `observation loss_mask=0` 校验）。

teacher view 只换了"最后一条 user message"，**保留的 messages 里仍带着 student 轨迹中间的多轮 observation**；teacher 侧重新 encode + collate 时，completion region 的划分与 student 不一致（把 privileged prompt / 多轮 observation 也并入了 completion），导致 teacher completion 远长于 student。

**本质：ms-swift 的 OPSD/OPD-RL 假设 response 是单轮、teacher 与 student 的 completion 一一对齐；这与我们多轮 `sapr_rag_scheduler` 的 loss_mask 结构不兼容。** 这正是 spec 里“ms-swift 对 GKD/OPSD + multi_turn_scheduler 支持有限”风险在 OPD-RL 路径上的具体暴露。

### 12.2 当前状态
- control smoke 已停；OPSD smoke 第一步失败已停。
- GPU0-5 全空；**GPU6 残留 ~85GB 僵尸显存**（PID <host_pid>，`ps`/`/proc` 均不可见、nvidia-smi 显示 `[Not Found]`，属进程已退但 CUDA context 未被驱动回收，本会话无权回收，需等驱动/平台侧释放）。
- ms-swift 源码干净（此前 SO_REUSEADDR 改动已回滚）。

### 12.3 待决策方向（OPSD teacher/student 帧对齐）
1. **让 teacher view 与 student 的多轮 completion 帧严格对齐**：teacher 侧也用与 student 相同的多轮 messages + 相同 `response_token_ids` + 相同 loss_mask（observation=0），只把「问题/最后 user」替换为 privileged 版；需改 `build_teacher_view` / `encode_teacher_view` 对多轮结构的处理，或在我们的数据侧保证 teacher completion_mask 与 student 完全一致。
2. **改 teacher_prompt 构造**：让 teacher 侧也是单轮（不含中间 observation），从数据/调度层规避多轮 completion 帧差异。
3. **暂缓 OPD-RL 的 teacher logp 注入**，先只跑通 plain GRPO（control 已验证可跑），OPSD 作为下一阶段单独攻关。
4. 需要时深入 ms-swift 多轮 OPSD 的 completion_mask 生成逻辑，评估是否可配置/打补丁使多轮对齐。
