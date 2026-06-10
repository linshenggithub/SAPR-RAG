#!/usr/bin/env python3
"""下载 2Wiki / MuSiQue 的 dev 集，统一成 agent_infer.py 能吃的 jsonl 格式。

数据来源：huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets
   - 2WikiMultihopQA: subset='2wikimultihopqa'，dev 12,576 题
   - MuSiQue:         subset='musique'，          dev  2,417 题
   - HotpotQA:        subset='hotpotqa'           dev  7,405 题（已有，跳过）

FlashRAG 数据集已经被它们预处理成 4 字段 jsonl：
   {"id", "question", "golden_answers", "metadata"}
agent_infer.py 期望的字段是 `question`（必填）+ `id`/`golden_answers`/`metadata`（选填），
完全兼容，所以本脚本只是"下载并落到指定目录"，不做字段重写。

用法：
   python download_eval_data.py --dataset 2wikimultihopqa
   python download_eval_data.py --dataset musique
   python download_eval_data.py --all          # 同时下两个
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJ_ROOT = Path("/mlx_devbox/users/mayi.summer/playground/SAPR-RAG")
EVAL_ROOT = PROJ_ROOT / "data" / "eval"

# FlashRAG_datasets 在 HF Hub 上的子集路径模板
HF_REPO = "RUC-NLPIR/FlashRAG_datasets"
DATASETS = {
    "2wikimultihopqa": {
        "out_dir": "2wiki",
        "subset": "2wikimultihopqa",
        "splits": ["dev"],
        "expected_dev": 12576,
    },
    "musique": {
        "out_dir": "musique",
        "subset": "musique",
        "splits": ["dev"],
        "expected_dev": 2417,
    },
}


def download_one(name: str):
    cfg = DATASETS[name]
    out_dir = EVAL_ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # FlashRAG 在 HF 上的目录结构：<repo>/<subset>/{train.jsonl, dev.jsonl, ...}
    from huggingface_hub import hf_hub_download
    import shutil

    for split in cfg["splits"]:
        remote_filename = f"{cfg['subset']}/{split}.jsonl"
        out_path = out_dir / f"{split}.jsonl"

        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[skip] {out_path} already exists ({out_path.stat().st_size:,} bytes)")
        else:
            print(f"[download] {HF_REPO}/{remote_filename} -> {out_path}")
            # 不指定 local_dir，hf_hub_download 会落到 HF cache，之后 copy 到我们目标位置
            cached = hf_hub_download(
                repo_id=HF_REPO,
                filename=remote_filename,
                repo_type="dataset",
            )
            shutil.copy2(cached, out_path)
            print(f"   copied {cached} -> {out_path}  ({out_path.stat().st_size:,} bytes)")

        # 验证：行数 + 字段
        verify(out_path, expected_count=cfg.get(f"expected_{split}"))


def verify(path: Path, expected_count=None):
    with open(path) as f:
        rows = [json.loads(l) for l in f]

    n = len(rows)
    print(f"[verify] {path.name}: {n} 行")
    if expected_count is not None and n != expected_count:
        print(f"   [WARN] 行数 {n} ≠ 预期 {expected_count}（可能 FlashRAG 数据更新了，检查无误就忽略）")

    sample = rows[0]
    required = ["question"]
    optional = ["id", "golden_answers", "metadata"]
    print(f"   字段：{sorted(sample.keys())}")
    for k in required:
        assert k in sample, f"缺必填字段 {k}: {sample}"
    for k in optional:
        if k not in sample:
            print(f"   [INFO] 缺选填字段 {k}（agent_infer.py 仍可工作，评估时会用 row index 兜底）")

    print(f"   样本: question={sample['question'][:80]!r}")
    if "golden_answers" in sample:
        print(f"          golden_answers={sample['golden_answers']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS.keys()), default=None)
    ap.add_argument("--all", action="store_true", help="下载所有数据集")
    args = ap.parse_args()

    if args.all:
        names = list(DATASETS.keys())
    elif args.dataset:
        names = [args.dataset]
    else:
        ap.error("必须指定 --dataset 或 --all")

    for n in names:
        print(f"\n========== {n} ==========")
        download_one(n)
    print("\n[done] 所有数据集已就绪。")


if __name__ == "__main__":
    main()
