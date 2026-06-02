from datasets import load_dataset

ds = load_dataset("reasonrag/RAG_extend_corpus", split="train")
print(ds)
print(ds.column_names)

ds.to_json(
    "data/rag_extend_corpus.jsonl",
    lines=True,
    force_ascii=False
)
PY

检查：

head -n 1 data/rag_extend_corpus.jsonl
wc -l data/rag_extend_corpus.jsonl

正常字段应该类似：

{"id": "...", "title": "...", "contents": "..."}
2. 合并 wiki18.jsonl 和 rag_extend_corpus.jsonl

新建脚本：

cat > merge_reasonrag_corpus.py <<'PY'
import argparse
import gzip
import hashlib
import json
from pathlib import Path

def open_text(path, mode="rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")

def get_contents(obj):
    contents = obj.get("contents")
    if contents is None:
        contents = obj.get("text")
    if contents is None:
        contents = obj.get("content")
    return str(contents).strip() if contents is not None else ""

def content_hash(title, contents):
    # 用 title + contents 去重，比只用 contents 稍微保守
    s = (str(title or "").strip() + "\n" + str(contents or "").strip()).lower()
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def write_jsonl(obj, fout):
    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki", required=True, help="原始 wiki18.jsonl 或 wiki18_100w.jsonl")
    parser.add_argument("--extend", required=True, help="RAG_extend_corpus 导出的 jsonl")
    parser.add_argument("--out", required=True, help="合并后的输出 jsonl")
    parser.add_argument("--no_dedup", action="store_true", help="不做内容去重")
    args = parser.parse_args()

    seen = set()
    n_wiki = n_ext = n_skip = n_write = 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open_text(args.out, "wt") as fout:
        # 1. 写入原始 wiki corpus
        with open_text(args.wiki, "rt") as fin:
            for i, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                contents = get_contents(obj)
                if not contents:
                    n_skip += 1
                    continue

                title = obj.get("title", "")
                h = content_hash(title, contents)

                if not args.no_dedup:
                    if h in seen:
                        n_skip += 1
                        continue
                    seen.add(h)

                # 保留 wiki 原始 id，不改动
                if "contents" not in obj:
                    obj["contents"] = contents

                write_jsonl(obj, fout)
                n_wiki += 1
                n_write += 1

                if n_write % 500000 == 0:
                    print(f"[progress] written={n_write}, skipped={n_skip}")

        # 2. 写入扩展 corpus
        with open_text(args.extend, "rt") as fin:
            for i, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                contents = get_contents(obj)
                if not contents:
                    n_skip += 1
                    continue

                title = obj.get("title", "")
                h = content_hash(title, contents)

                if not args.no_dedup:
                    if h in seen:
                        n_skip += 1
                        continue
                    seen.add(h)

                # 给扩展语料 id 加前缀，避免和 wiki 原始 id 冲突
                old_id = obj.get("id", i)
                obj["id"] = f"extend_{old_id}"
                obj["contents"] = contents

                write_jsonl(obj, fout)
                n_ext += 1
                n_write += 1

                if n_write % 500000 == 0:
                    print(f"[progress] written={n_write}, skipped={n_skip}")

    print("done")
    print(f"wiki_written={n_wiki}")
    print(f"extend_written={n_ext}")
    print(f"skipped={n_skip}")
    print(f"total_written={n_write}")
    print(f"output={args.out}")

if __name__ == "__main__":
    main()