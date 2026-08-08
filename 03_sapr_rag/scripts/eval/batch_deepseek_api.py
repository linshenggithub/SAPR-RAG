#!/usr/bin/env python3
"""通用 DeepSeek/DMXAPI 批式调用脚本。

用途：
  - 给一批 prompt/messages 做并发 API 调用
  - 逐条落盘 output + usage，方便估算真实成本
  - 支持断点续跑，避免重复扣费

输入 JSONL 支持两种格式：
  1) prompt 字段：
     {"id": "1", "prompt": "Question: ..."}

  2) OpenAI messages 字段：
     {"id": "1", "messages": [{"role": "user", "content": "..."}]}

默认走低成本短答案配置：model=deepseek-chat, temperature=0, max_tokens=128。

示例：
  export DEEPSEEK_API_KEY=sk-xxx
  python 03_sapr_rag/scripts/eval/batch_deepseek_api.py \
    --input prompts.jsonl \
    --output outputs.jsonl \
    --limit 100 \
    --concurrency 16 \
    --max_tokens 128
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


DEFAULT_DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_DMXAPI_URL = "https://www.dmxapi.cn/v1/chat/completions"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="输入 JSONL，包含 prompt 或 messages")
    p.add_argument("--output", required=True, help="输出 JSONL，支持断点续跑")
    p.add_argument("--meta", default=None, help="汇总信息 JSON，默认 output 同名 .meta.json")
    p.add_argument("--id_field", default="id")
    p.add_argument("--prompt_field", default="prompt")
    p.add_argument("--messages_field", default="messages")
    p.add_argument("--system_field", default="system", help="可选 system 字段名，仅 prompt 模式使用")
    p.add_argument("--default_system", default="", help="prompt 模式下的默认 system prompt")
    p.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    p.add_argument("--provider", choices=["deepseek", "dmxapi"], default="deepseek")
    p.add_argument("--base_url", default=None, help="覆盖 chat/completions URL")
    p.add_argument("--api_key_env", default=None, help="默认 deepseek=DEEPSEEK_API_KEY, dmxapi=DMXAPI_API_KEY")
    p.add_argument("--env_path", default=None, help="可选 .env 文件路径；仅把缺失的环境变量加载进当前进程")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--retry_base_delay", type=float, default=1.5)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="0=全量；>0 只跑前 N 条")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument("--dry_run", action="store_true", help="只解析输入和估算任务，不调用 API")
    p.add_argument("--progress_every", type=int, default=100)
    p.add_argument("--input_yuan_per_mtok", type=float, default=1.0, help="成本估算：输入元/百万 token")
    p.add_argument("--output_yuan_per_mtok", type=float, default=2.0, help="成本估算：输出元/百万 token")
    return p.parse_args()


def load_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON parse failed: {e}") from e
            rows.append(obj)
            if limit and len(rows) >= limit:
                break
    return rows


def stable_key(row: Dict[str, Any], args: argparse.Namespace) -> str:
    row_id = row.get(args.id_field)
    if row_id not in (None, ""):
        return str(row_id)
    material = {
        "prompt": row.get(args.prompt_field),
        "messages": row.get(args.messages_field),
        "system": row.get(args.system_field),
    }
    s = json.dumps(material, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


def load_completed(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = obj.get("key")
            if key:
                done.add(str(key))
    return done


def build_messages(row: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, str]]:
    messages = row.get(args.messages_field)
    if messages is not None:
        if not isinstance(messages, list):
            raise ValueError(f"messages field must be list, got {type(messages).__name__}")
        return messages

    prompt = row.get(args.prompt_field)
    if prompt is None:
        raise ValueError(f"row must contain either {args.messages_field!r} or {args.prompt_field!r}")

    out: List[Dict[str, str]] = []
    system = row.get(args.system_field) or args.default_system
    if system:
        out.append({"role": "system", "content": str(system)})
    out.append({"role": "user", "content": str(prompt)})
    return out


def get_endpoint(args: argparse.Namespace) -> Tuple[str, str]:
    if args.env_path:
        load_dotenv(Path(args.env_path))
    if args.provider == "dmxapi":
        key_env = args.api_key_env or "DMXAPI_API_KEY"
        url = args.base_url or os.environ.get("DMXAPI_CHAT_URL") or DEFAULT_DMXAPI_URL
    else:
        key_env = args.api_key_env or "DEEPSEEK_API_KEY"
        url = args.base_url or os.environ.get("DEEPSEEK_CHAT_URL") or DEFAULT_DEEPSEEK_URL
    api_key = os.environ.get(key_env)
    if not api_key:
        raise RuntimeError(f"环境变量 {key_env} 未设置")
    return url, api_key


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def call_api(
    *,
    url: str,
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    retry_base_delay: float,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error: Optional[str] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                obj = resp.json()
                choice = obj.get("choices", [{}])[0]
                msg = choice.get("message", {}) or {}
                usage = obj.get("usage", {}) or {}
                return {
                    "ok": True,
                    "content": msg.get("content") or "",
                    "reasoning_content": msg.get("reasoning_content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "usage": {
                        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                        "completion_tokens": int(usage.get("completion_tokens") or 0),
                        "total_tokens": int(usage.get("total_tokens") or 0),
                    },
                    "error": None,
                }
            last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
            if resp.status_code not in RETRYABLE_STATUS:
                break
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < max_retries - 1:
            time.sleep(min(60.0, retry_base_delay * (2 ** attempt)))

    return {
        "ok": False,
        "content": "",
        "reasoning_content": "",
        "finish_reason": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "error": last_error or "unknown_error",
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    meta_path = Path(args.meta) if args.meta else output_path.with_suffix(output_path.suffix + ".meta.json")

    rows = load_jsonl(input_path, args.limit)
    completed = load_completed(output_path) if args.skip_existing else set()

    tasks = []
    for idx, row in enumerate(rows):
        key = stable_key(row, args)
        if key in completed:
            continue
        messages = build_messages(row, args)
        tasks.append((idx, key, row, messages))

    print(
        f"[plan] input={len(rows)} completed={len(completed)} todo={len(tasks)} "
        f"model={args.model} provider={args.provider} max_tokens={args.max_tokens} "
        f"concurrency={args.concurrency}",
        flush=True,
    )

    if args.dry_run:
        for idx, key, row, messages in tasks[:3]:
            print(json.dumps({"idx": idx, "key": key, "messages": messages}, ensure_ascii=False)[:2000])
        return 0

    url, api_key = get_endpoint(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "input_rows": len(rows),
        "skipped_existing": len(rows) - len(tasks),
        "todo": len(tasks),
        "succeeded": 0,
        "failed": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    lock = Lock()
    started = time.time()

    def worker(task):
        idx, key, row, messages = task
        result = call_api(
            url=url,
            api_key=api_key,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
        )
        usage = result["usage"]
        out = {
            "key": key,
            "idx": idx,
            "id": row.get(args.id_field),
            "ok": result["ok"],
            "content": result["content"],
            "reasoning_content": result["reasoning_content"],
            "finish_reason": result["finish_reason"],
            "usage": usage,
            "error": result["error"],
            "model": args.model,
            "input": row,
        }
        return out

    done = 0
    with output_path.open("a", encoding="utf-8", buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(worker, task) for task in tasks]
            for fut in as_completed(futs):
                out = fut.result()
                usage = out["usage"]
                with lock:
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    done += 1
                    if out["ok"]:
                        stats["succeeded"] += 1
                    else:
                        stats["failed"] += 1
                    stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                    stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                    stats["total_tokens"] += int(usage.get("total_tokens") or 0)

                    if done % args.progress_every == 0 or done == len(tasks):
                        elapsed = max(time.time() - started, 1e-6)
                        print(
                            f"[progress] {done}/{len(tasks)} ok={stats['succeeded']} "
                            f"fail={stats['failed']} rate={done / elapsed:.2f}/s "
                            f"in_tok={stats['prompt_tokens']} out_tok={stats['completion_tokens']}",
                            flush=True,
                        )

    elapsed = time.time() - started
    estimated_yuan = (
        stats["prompt_tokens"] / 1_000_000 * args.input_yuan_per_mtok
        + stats["completion_tokens"] / 1_000_000 * args.output_yuan_per_mtok
    )
    meta = {
        **stats,
        "model": args.model,
        "provider": args.provider,
        "url": url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "elapsed_sec": round(elapsed, 2),
        "throughput_qps": round(len(tasks) / elapsed, 4) if elapsed > 0 else 0.0,
        "input_yuan_per_mtok": args.input_yuan_per_mtok,
        "output_yuan_per_mtok": args.output_yuan_per_mtok,
        "estimated_yuan": round(estimated_yuan, 4),
        "output": str(output_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[done]", json.dumps(meta, ensure_ascii=False), flush=True)
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
