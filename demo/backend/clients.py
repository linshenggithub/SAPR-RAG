from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class UpstreamError(RuntimeError):
    pass


class VLLMClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        timeout = httpx.Timeout(timeout_seconds, connect=10.0)
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self.client.aclose()

    async def health(self) -> dict[str, Any]:
        try:
            response = await self.client.get(f"{self.base_url}/models")
            response.raise_for_status()
            response.json()
            return {"ok": True}
        except (httpx.HTTPError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        stop: list[str],
    ) -> str:
        payload = self._payload(messages, max_tokens=max_tokens, stop=stop, stream=False)
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions", json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise UpstreamError(f"model request failed: {exc}") from exc

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        stop: list[str],
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, max_tokens=max_tokens, stop=stop, stream=True)
        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    content = (choices[0].get("delta") or {}).get("content")
                    if content:
                        yield content
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise UpstreamError(f"streaming model request failed: {exc}") from exc

    def _payload(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        stop: list[str],
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": stop,
            "stream": stream,
        }


class RetrieverClient:
    def __init__(self, base_url: str, timeout_seconds: float):
        self.base_url = base_url.rstrip("/")
        timeout = httpx.Timeout(timeout_seconds, connect=10.0)
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self.client.aclose()

    async def health(self) -> dict[str, Any]:
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return {"ok": True, **response.json()}
        except (httpx.HTTPError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    async def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        try:
            response = await self.client.post(
                f"{self.base_url}/search_batch",
                json={"queries": [query], "top_k": top_k},
            )
            response.raise_for_status()
            results = response.json()["results"]
            if not results or not isinstance(results[0], list):
                raise ValueError("retriever returned an invalid result shape")
            return results[0]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise UpstreamError(f"retrieval request failed: {exc}") from exc
