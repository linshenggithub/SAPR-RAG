#!/usr/bin/env python3
"""Download the complete train splits required by action-specific OPSD.

This intentionally rejects the 15k FlashRAG 2Wiki subset. The expected inputs
are the official-scale 2Wiki train split (~167k) and MuSiQue train split (~20k).
The Hub repositories expose concrete data files, so this downloader avoids
legacy dataset scripts that are unsupported by datasets>=4.
"""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJ_ROOT = Path(os.environ.get("SAPR_RAG_ROOT", Path(__file__).resolve().parents[3]))
SOURCES = {
    "2wiki": {
        "repo": "xanhho/2WikiMultihopQA",
        "filename": "train.parquet",
        "format": "parquet",
        "output": PROJ_ROOT / "data/raw/2wikimultihopqa_full/train.jsonl",
        "minimum_rows": 160_000,
    },
    "musique": {
        "repo": "bdsaglam/musique",
        "filename": "musique_ans_v1.0_train.jsonl",
        "format": "jsonl",
        "output": PROJ_ROOT / "data/raw/musique/train.jsonl",
        "minimum_rows": 19_000,
    },
}
HOTPOT_PATH = PROJ_ROOT / "data/raw/hotpotqa/train.jsonl"


def line_count(path):
    with Path(path).open("rb") as f:
        return sum(1 for _ in f)


def verify_existing(name, path, minimum_rows):
    if not path.is_file():
        return False
    count = line_count(path)
    if count < minimum_rows:
        raise RuntimeError(
            f"{name} at {path} has only {count:,} rows; expected at least "
            f"{minimum_rows:,}. This is probably a sampled subset, not the full train split.")
    print(f"[prepare] {name}: existing full split, rows={count:,}, path={path}")
    return True


def download_source(name, config, overwrite=False):
    output = config["output"]
    if output.exists() and not overwrite:
        if verify_existing(name, output, config["minimum_rows"]):
            return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    print(f"[prepare] downloading {config['repo']}/{config['filename']} -> {output}")
    try:
        cached = hf_hub_download(
            repo_id=config["repo"],
            filename=config["filename"],
            repo_type="dataset",
        )
        count = 0
        with temporary.open("w", encoding="utf-8") as writer:
            if config["format"] == "jsonl":
                with open(cached, encoding="utf-8") as reader:
                    for line in reader:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                        count += 1
            elif config["format"] == "parquet":
                import pyarrow.parquet as pq
                parquet = pq.ParquetFile(cached)
                for batch in parquet.iter_batches(batch_size=1024):
                    columns = batch.to_pydict()
                    for index in range(batch.num_rows):
                        row = {key: value[index] for key, value in columns.items()}
                        writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                        count += 1
            else:
                raise ValueError(f"unsupported source format: {config['format']}")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if count < config["minimum_rows"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{config['repo']} returned only {count:,} train rows; refusing to use a subset "
            f"where at least {config['minimum_rows']:,} rows are expected.")
    temporary.replace(output)
    print(f"[prepare] {name}: rows={count:,}, path={output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources", default="2wiki,musique",
        help="Comma-separated sources to prepare: 2wiki,musique")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not HOTPOT_PATH.is_file():
        raise FileNotFoundError(f"HotpotQA train data is missing: {HOTPOT_PATH}")
    hotpot_rows = line_count(HOTPOT_PATH)
    if hotpot_rows < 90_000:
        raise RuntimeError(f"HotpotQA train has only {hotpot_rows:,} rows: {HOTPOT_PATH}")
    print(f"[prepare] hotpotqa: rows={hotpot_rows:,}, path={HOTPOT_PATH}")

    requested = [item.strip() for item in args.sources.split(",") if item.strip()]
    unknown = set(requested) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    for name in requested:
        download_source(name, SOURCES[name], overwrite=args.overwrite)


if __name__ == "__main__":
    main()
