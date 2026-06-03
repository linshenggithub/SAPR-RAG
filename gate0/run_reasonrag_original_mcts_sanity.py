#!/usr/bin/env python3
"""受控运行 ReasonRAG original GPT-4o MCTS 小样本验证。

这个入口用于验证原始 ReasonRAG MCTS 在强模型下是否仍出现重复分支问题。
它只负责小样本 sanity，不替代正式 Gate0-B。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.paths import (  # noqa: E402
    BGE_INDEX_PATH,
    BGE_MODEL_PATH,
    HOTPOTQA_DEV_PATH,
    REASONRAG_ROOT,
    WIKI_CORPUS_PATH,
)


DEFAULT_OUT_DIR = REPO_ROOT / "gate0" / "data" / "reasonrag_original_gpt4o_mcts_sanity"
DMXAPI_BASE_URL = "https://www.dmxapi.cn/v1"
DEFAULT_RETRIEVAL_SERVICE_URL = "http://127.0.0.1:18080"
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def path_arg(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")


def load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DMXAPI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY or DMXAPI_API_KEY before formal run.")
    return key


class RemoteRetriever:
    """使用长驻 HTTP 检索服务，避免每次实验重复加载 64G FAISS index。"""

    def __init__(self, base_url: str, top_k: int, max_content_chars: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.top_k = top_k
        self.max_content_chars = max_content_chars

    def _post_retrieve(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        payload = {
            "query": query,
            "top_k": top_k or self.top_k,
            "max_content_chars": self.max_content_chars,
        }
        req = urllib.request.Request(
            f"{self.base_url}/retrieve",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with NO_PROXY_OPENER.open(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"retrieval service HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"retrieval service connection failed: {exc}") from exc

    def search(self, query: str, return_score: bool = False):
        response = self._post_retrieve(query, self.top_k)
        docs = []
        scores = []
        for item in response.get("results", []):
            docs.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title", ""),
                    "contents": item.get("contents", ""),
                }
            )
            scores.append(float(item.get("score", 0.0)))
        if return_score:
            return docs, scores
        return docs

    def batch_search(self, queries: list[str]):
        return [self.search(query, return_score=False) for query in queries]


def check_retrieval_service(base_url: str) -> None:
    req = urllib.request.Request(f"{base_url.rstrip('/')}/health", method="GET")
    try:
        with NO_PROXY_OPENER.open(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"retrieval service health check failed: {exc}") from exc
    if payload.get("status") != "ok":
        raise RuntimeError(f"retrieval service health check returned unexpected payload: {payload}")


def validate_local_retrieval_service_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("--retrieval-service-url must start with http:// or https://")
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("--retrieval-service-url must point to localhost for this sanity script")


def read_jsonl_slice(path: Path, start_index: int, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if line_index < start_index:
                continue
            if len(rows) >= n:
                break
            rows.append(json.loads(line))
    return rows


def build_config(args: argparse.Namespace, save_dir: Path) -> dict[str, Any]:
    return {
        "data_dir": str(args.hotpotqa_dev_path.parent.parent),
        "dataset_name": "hotpotqa",
        "split": ["dev"],
        "test_sample_num": args.start_index + args.num_examples,
        "random_sample": False,
        "index_path": str(args.bge_index_path),
        "corpus_path": str(args.wiki_corpus_path),
        "model2path": {
            "bge": str(args.bge_model_path),
            "gpt-4o": args.model,
        },
        "generator_model": args.model,
        "retrieval_method": "bge",
        "framework": "openai",
        "openai_setting": {
            "api_key": None,
            "base_url": args.base_url,
        },
        "gpu_id": None,
        "faiss_gpu": False,
        "metrics": ["em", "f1", "acc", "recall", "precision"],
        "retrieval_topk": args.retrieval_topk,
        "retrieval_batch_size": 64,
        "retrieval_query_max_length": 128,
        "retrieval_use_fp16": False,
        "save_intermediate_data": True,
        "save_note": "reasonrag_original_gpt4o_mcts_sanity",
        "save_dir": str(save_dir),
        "generation_params": {
            "do_sample": True,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_stage_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp}\t{message}\n")
        f.flush()


def run_dry_run(args: argparse.Namespace) -> None:
    selected = read_jsonl_slice(args.hotpotqa_dev_path, args.start_index, args.num_examples)
    estimated_calls = args.num_examples * args.max_rollouts * 2
    summary = {
        "label": "dry_run",
        "purpose": "ReasonRAG original GPT-4o MCTS sanity config check",
        "num_examples": args.num_examples,
        "start_index": args.start_index,
        "data": {
            "dataset": "hotpotqa/dev",
            "path": str(args.hotpotqa_dev_path),
        },
        "retrieval": {
            "retrieval_service_url": args.retrieval_service_url,
            "corpus_path": str(args.wiki_corpus_path),
            "index_path": str(args.bge_index_path),
            "bge_model_path": str(args.bge_model_path),
            "faiss_gpu": False,
            "retrieval_topk": args.retrieval_topk,
        },
        "source": {
            "reasonrag_root": str(args.reasonrag_root),
            "pipeline": str(args.reasonrag_root / "pipeline" / "reasonrag_pipeline.py"),
        },
        "mcts": {
            "max_iter": args.max_iter,
            "max_children": args.max_children,
            "max_rollouts": args.max_rollouts,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "model": args.model,
        },
        "api": {
            "base_url": args.base_url,
            "key_env": "OPENAI_API_KEY or DMXAPI_API_KEY",
            "estimated_upper_bound_calls": estimated_calls,
        },
        "selected_items": [
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "golden_answers": row.get("golden_answers", []),
            }
            for row in selected
        ],
    }
    write_json(args.out_dir / "dry_run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_formal(args: argparse.Namespace) -> None:
    api_key = load_api_key()
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ.setdefault("OPENAI_BASE_URL", args.base_url)

    sys.path.insert(0, str(args.reasonrag_root))
    from flashrag.config import Config  # noqa: WPS433
    from flashrag.utils import get_dataset  # noqa: WPS433
    from pipeline.reasonrag_pipeline import ReasonRAGPipeline  # noqa: WPS433

    run_dir = args.out_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_stdout_path = run_dir / "raw_stdout.log"
    stage_log_path = run_dir / "stage.log"

    config_dict = build_config(args, run_dir)
    safe_config = json.loads(json.dumps(config_dict))
    safe_config["openai_setting"]["api_key"] = None
    write_json(run_dir / "run_config.json", safe_config)

    started_at = time.time()
    append_stage_log(stage_log_path, "start formal run")
    with raw_stdout_path.open("w", encoding="utf-8") as raw_stdout, contextlib.redirect_stdout(raw_stdout):
        append_stage_log(stage_log_path, "imported dependencies")
        config = Config(config_dict=config_dict)
        append_stage_log(stage_log_path, "built FlashRAG Config")
        dataset = get_dataset(config)["dev"]
        if args.start_index:
            dataset.data = dataset.data[args.start_index : args.start_index + args.num_examples]
        append_stage_log(stage_log_path, f"loaded dataset: {len(dataset)} examples")
        retriever = None
        if args.retrieval_service_url:
            check_retrieval_service(args.retrieval_service_url)
            append_stage_log(stage_log_path, f"checked retrieval service: {args.retrieval_service_url}")
            retriever = RemoteRetriever(
                base_url=args.retrieval_service_url,
                top_k=args.retrieval_topk,
                max_content_chars=args.retrieval_max_content_chars,
            )
            append_stage_log(stage_log_path, "initialized RemoteRetriever")
        pipeline = ReasonRAGPipeline(
            config,
            prompt_template=None,
            retriever=retriever,
            max_iter=args.max_iter,
            max_children=args.max_children,
            max_rollouts=args.max_rollouts,
            beta=args.beta,
        )
        if not hasattr(pipeline.prompt_template, "check_prompt_length"):
            pipeline.prompt_template.check_prompt_length = lambda _prompt: False
            append_stage_log(stage_log_path, "patched missing PromptTemplate.check_prompt_length")
        append_stage_log(stage_log_path, "initialized ReasonRAGPipeline")

        records = []
        for idx, item in enumerate(dataset):
            item_start = time.time()
            status = "ok"
            error = None
            append_stage_log(stage_log_path, f"start item {idx}: {item.id}")
            try:
                pipeline.search(item)
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                append_stage_log(stage_log_path, f"failed item {idx}: {error}")
            records.append(
                {
                    "index": idx,
                    "id": item.id,
                    "question": item.question,
                    "golden_answers": item.golden_answers,
                    "status": status,
                    "error": error,
                    "elapsed_sec": round(time.time() - item_start, 3),
                    "output": item.output,
                }
            )
            write_json(run_dir / "progress.json", records)
            append_stage_log(stage_log_path, f"finished item {idx}: {status}")

    summary = {
        "label": "sanity_result",
        "run_dir": str(run_dir),
        "elapsed_sec": round(time.time() - started_at, 3),
        "num_examples": args.num_examples,
        "num_ok": sum(1 for r in records if r["status"] == "ok"),
        "num_failed": sum(1 for r in records if r["status"] != "ok"),
        "raw_stdout_path": str(raw_stdout_path),
        "stage_log_path": str(stage_log_path),
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--base-url", default=DMXAPI_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=7)
    parser.add_argument("--max-children", type=int, default=2)
    parser.add_argument("--max-rollouts", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--retrieval-topk", type=int, default=3)
    parser.add_argument("--retrieval-max-content-chars", type=int, default=8000)
    parser.add_argument("--retrieval-service-url", default=DEFAULT_RETRIEVAL_SERVICE_URL)
    parser.add_argument("--reasonrag-root", type=path_arg, default=path_arg(REASONRAG_ROOT))
    parser.add_argument("--hotpotqa-dev-path", type=path_arg, default=path_arg(HOTPOTQA_DEV_PATH))
    parser.add_argument("--wiki-corpus-path", type=path_arg, default=path_arg(WIKI_CORPUS_PATH))
    parser.add_argument("--bge-index-path", type=path_arg, default=path_arg(BGE_INDEX_PATH))
    parser.add_argument("--bge-model-path", type=path_arg, default=path_arg(BGE_MODEL_PATH))
    parser.add_argument("--out-dir", type=path_arg, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    require_dir(args.reasonrag_root, "ReasonRAG root")
    require_file(args.reasonrag_root / "pipeline" / "reasonrag_pipeline.py", "ReasonRAG pipeline")
    require_file(args.hotpotqa_dev_path, "HotpotQA dev")
    if args.retrieval_service_url:
        validate_local_retrieval_service_url(args.retrieval_service_url)
    else:
        require_file(args.wiki_corpus_path, "Wiki corpus")
        require_file(args.bge_index_path, "BGE index")
        require_dir(args.bge_model_path, "BGE model")
    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run(args)
    else:
        run_formal(args)


if __name__ == "__main__":
    main()
