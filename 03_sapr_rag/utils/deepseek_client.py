"""DeepSeek-V3 API client for SAPR-R v1 offline data construction.

Why exist:
    SAPR-R v1 数据构造管线（step2 reasoning_steps 生成 / step4 cls 标注）需要
    大批量调 LLM。这里封装一个并发安全 + 自动重试 + 可选 JSON 强约束的
    OpenAI 兼容 client，可同时支持：
      - DeepSeek 官方 API（base_url=https://api.deepseek.com/v1, model=deepseek-chat）
      - DMXAPI 中转（base_url=https://www.dmxapi.cn/v1, model=deepseek-v3 等）

    并发策略：ThreadPoolExecutor + 限流（max_workers 建议 ≤50），与 ReasonRAG
    pipeline 原 multiprocessing 风格一致；retry 用指数退避，专门处理 429 / 5xx /
    网络抖动。

Usage:
    >>> client = DeepSeekClient.from_env()
    >>> # 单条调用
    >>> resp = client.chat([{"role": "user", "content": "Hi"}])
    >>> # 并发批处理（保持输出顺序）
    >>> outputs = client.chat_batch(prompts_list, max_workers=20)
    >>> # 强 JSON 输出
    >>> obj = client.chat_json([{"role": "user", "content": "返回 {\"x\": 1}"}])
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# 默认重试目标（OpenAI / 网络层面常见瞬时错误）
_RETRYABLE_KEYWORDS = (
    "rate limit",
    "rate_limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "overloaded",
)


@dataclass
class DeepSeekConfig:
    """配置项；仅记录与 OpenAI SDK 调用相关的参数。"""

    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    timeout: float = 60.0
    max_retries: int = 5
    retry_base_delay: float = 1.5
    retry_max_delay: float = 30.0
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    extra_create_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallStats:
    """跨线程累计计数；供批处理后 sanity check / 成本估计。"""

    requests: int = 0
    succeeded: int = 0
    failed: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class DeepSeekClient:
    """OpenAI 兼容的 DeepSeek 调用封装；线程安全。"""

    def __init__(self, config: DeepSeekConfig) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "需要 openai>=1.0：pip install 'openai>=1.30'"
            ) from e

        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
        self.stats = CallStats()
        self._stats_lock = threading.Lock()

    # ---------- 构造 ----------

    @classmethod
    def from_env(
        cls,
        env_path: Optional[Path] = None,
        prefer: str = "deepseek",
    ) -> "DeepSeekClient":
        """从环境变量 / .env 文件构造。

        prefer:
          - "deepseek": 优先 DEEPSEEK_API_KEY → DMXAPI_API_KEY
          - "dmxapi"  : 优先 DMXAPI_API_KEY → DEEPSEEK_API_KEY
        """
        if env_path is not None and env_path.exists():
            _load_dotenv_into_os(env_path)

        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        dmxapi_key = os.environ.get("DMXAPI_API_KEY")

        if prefer == "deepseek":
            order = [
                ("deepseek", deepseek_key, "https://api.deepseek.com/v1", "deepseek-chat"),
                ("dmxapi", dmxapi_key,
                 os.environ.get("DMXAPI_BASE_URL", "https://www.dmxapi.cn/v1"),
                 os.environ.get("DMXAPI_MODEL", "deepseek-v3")),
            ]
        else:
            order = [
                ("dmxapi", dmxapi_key,
                 os.environ.get("DMXAPI_BASE_URL", "https://www.dmxapi.cn/v1"),
                 os.environ.get("DMXAPI_MODEL", "deepseek-v3")),
                ("deepseek", deepseek_key, "https://api.deepseek.com/v1", "deepseek-chat"),
            ]

        for name, key, base_url, model in order:
            if key:
                logger.info("DeepSeekClient using provider=%s model=%s", name, model)
                return cls(DeepSeekConfig(
                    api_key=key,
                    base_url=base_url,
                    model=os.environ.get("DEEPSEEK_MODEL", model),
                ))

        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY 或 DMXAPI_API_KEY；"
            "请在 03_sapr_rag/.env 或环境变量里设置"
        )

    # ---------- 单调用 ----------

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """阻塞调用，返回 assistant 文本。失败抛 RuntimeError。"""
        return self._chat_with_retry(
            list(messages),
            temperature=temperature if temperature is not None else self.config.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.default_max_tokens,
            response_format=response_format,
            extra_kwargs=extra_kwargs or {},
        )

    def chat_json(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """要求模型返回合法 JSON；解析失败时按可重试错误重试。"""
        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                raw = self.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    extra_kwargs=extra_kwargs,
                )
                return json.loads(raw)
            except json.JSONDecodeError as e:
                last_err = e
                logger.warning(
                    "chat_json parse failure (attempt %d/%d): %s",
                    attempt + 1, self.config.max_retries, e,
                )
                with self._stats_lock:
                    self.stats.retries += 1
                time.sleep(_jitter_delay(
                    base=self.config.retry_base_delay,
                    attempt=attempt,
                    cap=self.config.retry_max_delay,
                ))
        raise RuntimeError(f"chat_json 在 {self.config.max_retries} 次后仍未解析成功") from last_err

    # ---------- 批量并发 ----------

    def chat_batch(
        self,
        prompts: Sequence[Sequence[Dict[str, str]]],
        *,
        max_workers: int = 20,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        on_failure: str = "raise",
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> List[Optional[str]]:
        """并发批处理；输出顺序与输入对齐。

        on_failure:
          - "raise":  任一失败抛出
          - "skip" :  失败位置返回 None，不中断
        """
        if on_failure not in {"raise", "skip"}:
            raise ValueError(f"on_failure must be raise|skip, got {on_failure}")

        results: List[Optional[str]] = [None] * len(prompts)
        done_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    self._chat_with_retry,
                    list(p),
                    temperature=temperature if temperature is not None else self.config.default_temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.config.default_max_tokens,
                    response_format=response_format,
                    extra_kwargs={},
                ): idx
                for idx, p in enumerate(prompts)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    if on_failure == "raise":
                        raise
                    logger.error("batch idx=%d failed: %s", idx, e)
                    results[idx] = None
                done_count += 1
                if progress_cb is not None:
                    progress_cb(done_count, len(prompts))
        return results

    def chat_json_batch(
        self,
        prompts: Sequence[Sequence[Dict[str, str]]],
        *,
        max_workers: int = 20,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        on_failure: str = "skip",
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> List[Optional[Any]]:
        """JSON 批处理；解析失败的样本（即便已重试）按 on_failure 处理。"""
        results: List[Optional[Any]] = [None] * len(prompts)
        done_count = 0

        def _one(p: Sequence[Dict[str, str]]) -> Any:
            return self.chat_json(
                p,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_one, p): idx for idx, p in enumerate(prompts)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    if on_failure == "raise":
                        raise
                    logger.error("batch json idx=%d failed: %s", idx, e)
                    results[idx] = None
                done_count += 1
                if progress_cb is not None:
                    progress_cb(done_count, len(prompts))
        return results

    # ---------- 内部 ----------

    def _chat_with_retry(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]],
        extra_kwargs: Dict[str, Any],
    ) -> str:
        with self._stats_lock:
            self.stats.requests += 1

        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                kwargs: Dict[str, Any] = dict(self.config.extra_create_kwargs)
                kwargs.update(extra_kwargs)
                kwargs.update(dict(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ))
                if response_format is not None:
                    kwargs["response_format"] = response_format

                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                with self._stats_lock:
                    self.stats.succeeded += 1
                    if usage is not None:
                        self.stats.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                        self.stats.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                return content
            except Exception as e:
                last_err = e
                if not _is_retryable(e) or attempt == self.config.max_retries - 1:
                    with self._stats_lock:
                        self.stats.failed += 1
                    raise
                with self._stats_lock:
                    self.stats.retries += 1
                delay = _jitter_delay(
                    base=self.config.retry_base_delay,
                    attempt=attempt,
                    cap=self.config.retry_max_delay,
                )
                logger.warning(
                    "DeepSeek call failed (attempt %d/%d): %s; sleeping %.1fs",
                    attempt + 1, self.config.max_retries, e, delay,
                )
                time.sleep(delay)

        with self._stats_lock:
            self.stats.failed += 1
        raise RuntimeError("retry exhausted") from last_err


# -------------- helpers --------------

def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in _RETRYABLE_KEYWORDS)


def _jitter_delay(*, base: float, attempt: int, cap: float) -> float:
    raw = base * (2 ** attempt)
    raw = min(raw, cap)
    return raw * (0.5 + random.random() * 0.5)


def _load_dotenv_into_os(path: Path) -> None:
    """轻量 .env 解析；不引入 python-dotenv 依赖。"""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


__all__ = ["DeepSeekClient", "DeepSeekConfig", "CallStats"]
