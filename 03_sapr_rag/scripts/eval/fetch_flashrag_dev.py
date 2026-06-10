"""从 FlashRAG 官方仓库拉 dev 集到 data/eval/。

FlashRAG dataset 仓库：RUC-NLPIR/FlashRAG_datasets
每个数据集目录下有 dev.jsonl，字段：{id, question, golden_answers}
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


PROJ_ROOT = Path("/mlx_devbox/users/mayi.summer/playground/SAPR-RAG")
EVAL_DIR = PROJ_ROOT / "data/eval"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["hotpotqa", "2wikimultihopqa", "musique"])
    p.add_argument("--split", default="dev")
    args = p.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        remote = f"{ds}/{args.split}.jsonl"
        print(f"[fetch] {remote} ...")
        local = hf_hub_download(
            repo_id="RUC-NLPIR/FlashRAG_datasets",
            filename=remote,
            repo_type="dataset",
            local_dir=str(EVAL_DIR),
        )
        print(f"  -> {local}")


if __name__ == "__main__":
    main()
