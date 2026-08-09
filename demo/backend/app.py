from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import SAPRDemoAgent
from .clients import RetrieverClient, UpstreamError, VLLMClient
from .config import Settings


DEMO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = DEMO_ROOT / "frontend"
LOGGER = logging.getLogger("sapr_demo")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)


class CooldownLimiter:
    def __init__(self, cooldown_seconds: float):
        self.cooldown_seconds = cooldown_seconds
        self.last_request: dict[str, float] = {}
        self.lock = asyncio.Lock()

    async def check(self, key: str) -> float:
        if self.cooldown_seconds <= 0:
            return 0.0
        async with self.lock:
            now = time.monotonic()
            retry_after = self.cooldown_seconds - (now - self.last_request.get(key, 0.0))
            if retry_after > 0:
                return retry_after
            self.last_request[key] = now
            return 0.0


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model = VLLMClient(
            settings.model_url, settings.model_name, settings.request_timeout_seconds
        )
        retriever = RetrieverClient(
            settings.retrieval_url, settings.request_timeout_seconds
        )
        app.state.model = model
        app.state.retriever = retriever
        app.state.agent = SAPRDemoAgent(
            model,
            retriever,
            max_turns=settings.max_turns,
            top_k=settings.top_k,
            max_tokens=settings.max_tokens,
            evidence_max_tokens=settings.evidence_max_tokens,
        )
        app.state.semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        app.state.limiter = CooldownLimiter(settings.cooldown_seconds)
        yield
        await model.close()
        await retriever.close()

    app = FastAPI(
        title="SAPR-RAG Demo",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
            "style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'self'"
        )
        return response

    @app.get("/api/health")
    async def health(request: Request):
        model_health, retrieval_health = await asyncio.gather(
            request.app.state.model.health(), request.app.state.retriever.health()
        )
        return {
            "status": "ok" if model_health["ok"] and retrieval_health["ok"] else "degraded",
            "model": {"ok": model_health["ok"]},
            "retriever": {
                "ok": retrieval_health["ok"],
                "n_vectors": retrieval_health.get("n_vectors"),
                "n_docs": retrieval_health.get("n_docs"),
            },
        }

    @app.post("/api/chat/stream")
    async def chat_stream(body: ChatRequest, request: Request):
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Question cannot be empty.")
        if len(question) > settings.question_max_chars:
            raise HTTPException(
                status_code=422,
                detail=f"Question cannot exceed {settings.question_max_chars} characters.",
            )

        peer = request.client.host if request.client else "unknown"
        if settings.trust_cloudflare_ip:
            peer = request.headers.get("cf-connecting-ip", peer)
        retry_after = await request.app.state.limiter.check(peer)
        if retry_after > 0:
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {retry_after:.1f} seconds before asking again.",
                headers={"Retry-After": str(max(1, int(retry_after)))},
            )

        async def events():
            yield _sse({"type": "queued"})
            try:
                async with request.app.state.semaphore:
                    yield _sse({"type": "started"})
                    async for event in request.app.state.agent.run(question):
                        if await request.is_disconnected():
                            return
                        yield _sse(event)
            except UpstreamError as exc:
                LOGGER.warning("Demo upstream request failed: %s", exc)
                yield _sse({"type": "error", "code": "upstream_error"})
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Unexpected demo request failure")
                yield _sse(
                    {
                        "type": "error",
                        "code": "internal_error",
                    }
                )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    app.mount("/static", StaticFiles(directory=FRONTEND_ROOT), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(FRONTEND_ROOT / "index.html")

    return app


app = create_app()
