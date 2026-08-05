# ms-swift Local Patches

本文记录 SAPR-RAG 当前依赖的 `ms-swift` 源码补丁。`ms-swift` 是独立仓库，补丁不直接混入 SAPR-RAG 代码树，因此需要在本仓库保留可复现说明和 patch 文件。

## 1. OPSD teacher view 透传 response_loss_mask

### 1.1 背景

SAPR-RAG 的 OPSD 训练使用 multi-turn GRPO rollout。student 侧 rollout 里包含模型生成的 response token，同时 scheduler 可能产生 observation / reference 相关上下文。teacher 侧使用 `teacher_prompt` 替换 prompt，但必须对同一串 student on-policy response token 计算 logprob。

如果 teacher view 没有拿到与 student 相同的 `response_loss_mask`，teacher 和 student 会在 completion 有效 token 数上不一致，导致 token-frame remap 失败。

典型错误形态：

```text
OPSD response length mismatch: student=<n_student> teacher=<n_teacher>
Teacher and student must share the same response tokens.
```

### 1.2 修改位置

外部仓库文件：

```text
ms-swift/swift/rl_core/data.py
```

函数：

```text
OnPolicySample.to_teacher_template_dict()
```

### 1.3 修改内容

在 teacher-side template dict 中，除了透传 `response_token_ids`，还需要透传 `response_loss_mask`：

```python
if self.response_token_ids:
    d['response_token_ids'] = self.response_token_ids
    if self.response_loss_mask:
        d['response_loss_mask'] = self.response_loss_mask
```

这样 `encode_teacher_view()` 会读取 `response_loss_mask`，teacher view 和 student view 使用相同的 completion mask。

### 1.4 Patch 文件

本仓库保存了可直接应用的 patch：

```text
patches/ms-swift/0001-opsd-teacher-response-loss-mask.patch
```

在外部 `ms-swift` 仓库根目录执行：

```bash
git apply /path/to/SAPR-RAG/patches/ms-swift/0001-opsd-teacher-response-loss-mask.patch
```

如果 `ms-swift` 与 SAPR-RAG 是同级目录，也可以从 SAPR-RAG 根目录执行：

```bash
git -C ../ms-swift apply "$(pwd)/patches/ms-swift/0001-opsd-teacher-response-loss-mask.patch"
```

### 1.5 验证方式

最小静态验证：

```bash
python -m py_compile ../ms-swift/swift/rl_core/data.py
```

功能验证：

```text
1. 构造带 teacher_prompt 的 OPSD 数据；
2. 启动 GRPO + OPSD 训练；
3. 确认不再出现 OPSD response length mismatch；
4. 确认 logging.jsonl 中出现有限值 teacher_kl。
```

### 1.6 为什么不改 SAPR-RAG 侧绕过

这个问题发生在 `ms-swift` 的 teacher template dict 重建阶段。SAPR-RAG 侧只能提供 `response_loss_mask`，无法控制 teacher encode 阶段是否读取它。因此正确修复点在 `ms-swift/swift/rl_core/data.py`。

## 2. 不纳入补丁的本地文件

本地可能存在类似：

```text
ms-swift/swift/rlhf_trainers/utils.py.bak_*
```

这类文件是调试备份，不是可复现补丁，不应提交到 SAPR-RAG，也不应作为外部依赖修改记录。
