# SAPR-RAG Training (LLaMA-Factory)

## Files

- `sft_lora.yaml` — 推荐起跑：SFT + LoRA r=16，混合 sapr_reasoning + sapr_evidence
- `sft_full.yaml` — 全参 SFT 备选（需要 8×A100 + DeepSpeed ZeRO-3）
- `dpo_lora.yaml` — 续 SFT 的 LoRA adapter，用 sapr_proguide_dpo 偏好对

## Pre-flight

1. clone LLaMA-Factory（任意目录均可，下面以 `~/LLaMA-Factory` 为例）：
   ```
   git clone https://github.com/hiyouga/LLaMA-Factory.git ~/LLaMA-Factory
   cd ~/LLaMA-Factory && pip install -e ".[torch,metrics]"
   ```

2. DPO 的偏好数据还没注册（待办）：
   - 把 `data/raw/proguide_dpo.parquet` 转成 alpaca-pref 格式 jsonl
   - 在 `dataset_info.json` 里注册 `sapr_proguide_dpo`

## Run

```bash
# SFT (LoRA)
llamafactory-cli train \
  /mlx_devbox/users/mayi.summer/playground/SAPR-RAG/03_sapr_rag/scripts/train/sft_lora.yaml

# DPO (LoRA, 续 SFT)
llamafactory-cli train \
  /mlx_devbox/users/mayi.summer/playground/SAPR-RAG/03_sapr_rag/scripts/train/dpo_lora.yaml
```

## Key choices

| 字段 | 选值 | 理由 |
|---|---|---|
| 基座 | Qwen2.5-7B-Instruct | 与 ReasonRAG 论文实验对齐，方便对照 |
| template | qwen | 触发 Qwen2 chat template |
| cutoff_len | 4096 | reasoning 样本含完整历史最长 ~1.3k tokens，给 evidence 的长 reference 留足余量 |
| LoRA r/α | 16/32 | 两者均为常用偏强配置，足以学动作风格 |
| epochs | 1.0 | 数据量 27.7w 行已足够，多 epoch 易过拟合 R3 表面措辞 |
| lr | 1e-4 (LoRA) / 1e-5 (full) / 5e-6 (DPO) | 沿用 LLaMA-Factory 官方推荐 |
| grad_accum | 16 | 单卡 bsz=1 时等效 batch 16；多卡可下调 |
| bf16 + ckpt | on | 7B 单卡 80G 必须 |
| val_size | 0.01 | 仅看 loss 曲线，不依赖 val 选模 |

## After SFT

`adapter_name_or_path` 在 `dpo_lora.yaml` 里已指向 `saves/qwen2_5_7b/lora/sft`，DPO
会从 SFT 的 LoRA 继续训练（同一份 adapter 上叠加偏好优化）。

## Without GPU

当前 devbox `nvidia-smi` 无输出。需要先申请 GPU 资源（建议 1×A100/H100 80G 起步）才能 run。
