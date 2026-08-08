# OPSD 多轮适配方案评估（2026-08-03）

## 背景
matched control smoke（plain GRPO）已在 8×H20 环境跑通；OPSD 臂第一步即失败：
```
gkd_helpers.py:468 remap_teacher_logps_to_student_frame
AssertionError: OPSD response length mismatch at sample 0: student=82 teacher=382.
```
teacher completion 稳定是 student 的 ~4-5 倍。

## 精确根因（已定位到具体代码差异，非"多轮根本不兼容"）

对比 student 与 teacher 两条 encode 路径，差异只有一处——**teacher 侧漏传了 `response_loss_mask`**：

- **student**：`encode_sample`（`ms-swift/swift/rlhf_trainers/utils.py:718-728`）
  ```python
  loss_mask = sample.response_loss_mask or None
  data['messages'] = replace_assistant_response_with_ids(
      msgs, sample.response_token_ids, loss_mask, non_thinking_prefix_ids=prefix_ids)
  ```
  多轮 response 里 observation(检索结果) token 的 loss_mask=0，故 student completion_mask 只含模型 action token（=82）。

- **teacher**：`encode_teacher_view`（`ms-swift/swift/rlhf_trainers/gkd_helpers.py:36-45`）
  ```python
  loss_mask = teacher_row.get('response_loss_mask') or None   # ← 恒为 None
  ```
  但 `teacher_row` 来自 `to_teacher_template_dict`（`ms-swift/swift/rl_core/data.py:108-122`），该函数**只放了 `response_token_ids`，没放 `response_loss_mask`**：
  ```python
  d = self._standard_fields()
  d['messages'] = self.teacher_messages
  if self.response_token_ids:
      d['response_token_ids'] = self.response_token_ids   # 没有 d['response_loss_mask']
  ```
  → teacher 侧 `loss_mask=None` → 整段 response（含所有 observation token）都进 completion_mask → teacher completion=382。

**结论**：不是"多轮结构无法对齐"，而是 **ms-swift 的 `to_teacher_template_dict` 在 OPSD 路径下遗漏了 `response_loss_mask` 透传**。student 用 loss_mask 屏蔽了 observation，teacher 没用，导致两边 completion token 数不一致，`remap_teacher_logps_to_student_frame` 的 assert 失败。

## 候选修复

### 方案 A（推荐）：teacher dict 透传 response_loss_mask（最小改动）
在 `ms-swift/swift/rl_core/data.py` 的 `to_teacher_template_dict` 补一行，与 student 的 `to_template_dict` 对齐：
```python
if self.response_token_ids:
    d['response_token_ids'] = self.response_token_ids
    if self.response_loss_mask:                 # 新增
        d['response_loss_mask'] = self.response_loss_mask
```
配合 `encode_teacher_view` 已有的 `teacher_row.get('response_loss_mask')` 读取逻辑，teacher 就会用与 student 相同的 loss_mask 屏蔽 observation，两边 completion 帧对齐。
- 改动量：1-3 行；本地改不 commit。
- 风险：低。teacher 与 student 用同一 `response_loss_mask` 正是 OPSD "同 response tokens" 契约的应有之义。
- 验证：改后重跑 OPSD smoke，看 `remap` assert 是否通过、`teacher_kl` 是否 finite。

### 方案 B：数据侧规避（不改 ms-swift）
把 teacher_prompt 及 OPSD 数据构造成单轮（response 不含中间 observation）。
- 代价：偏离"teacher 看 privileged evidence + 同一多轮 response"的设计，且 student 本身是多轮，仍会不一致。基本不可行，仅备录。

### 方案 C：暂缓 OPSD，先产出 plain GRPO
control 已验证可跑，先在 100 条 pilot 上跑完 plain GRPO 出基线，OPSD 待方案 A 验证后再上。

## 建议顺序
先做**方案 A**（1-3 行、最对症、低风险），本地改 ms-swift 不 commit，重跑 OPSD smoke 验证；若 A 通过则 OPSD 臂打通；若 A 暴露新的多轮对齐问题，再评估 B/C。

## 附：环境现状
- GPU0-5 空闲；GPU6 残留约 85GB 不可见显存占用（进程已退、CUDA context 未回收，需平台/驱动侧回收）。
- ms-swift 源码干净（SO_REUSEADDR 试验已回滚，备份 utils.py.bak_20260803 保留）。
- 端口修复保留：`run_grpo_opsd.sh` VLLM_GROUP_PORT=51299。
