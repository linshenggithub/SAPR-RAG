"""合并 DP shard 输出到一个 jsonl，按 id 排序。"""

import argparse
import glob
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    shards = sorted(glob.glob(str(Path(args.shard_dir) / "shard_*.jsonl")))
    print(f"[merge] {len(shards)} shard files")

    rows = []
    for s in shards:
        with open(s) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        print(f"  {s}: cumulative {len(rows)}")

    # 按 id（或 int(id)）排序，混合类型时降级为字符串
    try:
        rows.sort(key=lambda r: int(r["id"]))
    except (ValueError, TypeError):
        rows.sort(key=lambda r: str(r["id"]))

    with open(args.output, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[merge] wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
