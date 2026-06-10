#!/usr/bin/env python3
"""把 ReasonRAG RAG_ProGuide parquet 转为 LLaMA-Factory DPO 格式 jsonl。

源字段：instruction(system) / input(user query) / chosen / rejected / prompt(已拼好,丢弃)
目标字段：system / instruction / input / chosen / rejected （alpaca + DPO）

用法：
    python build_dpo.py
    输出：out/sapr_proguide_dpo.jsonl
"""
import json
import os
import pandas as pd

PROJ_ROOT = "/mlx_devbox/users/mayi.summer/playground/SAPR-RAG"
SRC = f"{PROJ_ROOT}/data/raw/proguide_dpo.parquet"
DST = f"{PROJ_ROOT}/03_sapr_rag/data/sft_build/out/sapr_proguide_dpo.jsonl"

os.makedirs(os.path.dirname(DST), exist_ok=True)

df = pd.read_parquet(SRC)
print(f"[build_dpo] 输入 {len(df)} 行")

# 简单校验
required = ["instruction", "input", "chosen", "rejected"]
for col in required:
    assert col in df.columns, f"missing {col}"

# 检查空值
n_empty_chosen = (df["chosen"].astype(str).str.len() == 0).sum()
n_empty_rejected = (df["rejected"].astype(str).str.len() == 0).sum()
n_same = (df["chosen"] == df["rejected"]).sum()
print(f"[build_dpo] empty_chosen={n_empty_chosen}  empty_rejected={n_empty_rejected}  same_pairs={n_same}")

with open(DST, "w") as f:
    n = 0
    for _, row in df.iterrows():
        rec = {
            "system": str(row["instruction"]),  # ReasonRAG 的 instruction 实际是 system prompt
            "instruction": str(row["input"]),   # ReasonRAG 的 input 是 "Question: ..." 即 user 输入
            "input": "",
            "chosen": str(row["chosen"]),
            "rejected": str(row["rejected"]),
        }
        # 跳过空 chosen/rejected 或 chosen==rejected 的退化对
        if not rec["chosen"].strip() or not rec["rejected"].strip():
            continue
        if rec["chosen"] == rec["rejected"]:
            continue
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1

print(f"[build_dpo] 已写出 {n} 条到 {DST}")
