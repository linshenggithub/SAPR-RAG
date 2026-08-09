from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
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


class ClientLimiter:
    MAX_TRACKED_CLIENTS = 10000

    def __init__(
        self,
        cooldown_seconds: float,
        requests_per_window: int,
        window_seconds: float,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.requests: dict[str, deque[float]] = {}
        self.salt = secrets.token_bytes(16)
        self.last_cleanup = 0.0
        self.lock = asyncio.Lock()

    async def check(self, key: str) -> float:
        digest = hashlib.blake2b(
            key[:256].encode("utf-8", errors="replace"),
            key=self.salt,
            digest_size=16,
        ).hexdigest()
        async with self.lock:
            now = time.monotonic()
            cutoff = now - self.window_seconds
            cleanup_interval = min(60.0, self.window_seconds)
            if now - self.last_cleanup >= cleanup_interval:
                for client, timestamps in list(self.requests.items()):
                    while timestamps and timestamps[0] <= cutoff:
                        timestamps.popleft()
                    if not timestamps:
                        del self.requests[client]
                self.last_cleanup = now

            if digest not in self.requests and len(self.requests) >= self.MAX_TRACKED_CLIENTS:
                return cleanup_interval
            history = self.requests.setdefault(digest, deque())

            if history and self.cooldown_seconds > 0:
                retry_after = self.cooldown_seconds - (now - history[-1])
                if retry_after > 0:
                    return retry_after

            if len(history) >= self.requests_per_window:
                return max(1.0, history[0] + self.window_seconds - now)

            history.append(now)
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
        app.state.limiter = ClientLimiter(
            settings.cooldown_seconds,
            settings.requests_per_window,
            settings.rate_window_seconds,
        )
        yield
        await model.close()
        await retriever.close()

    app = FastAPI(
        title="SAPR-RAG Demo",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    if settings.allowed_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=list(settings.allowed_hosts),
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
        if request.method in {"POST", "PUT", "PATCH"}:
            raw_length = request.headers.get("content-length")
            try:
                content_length = int(raw_length) if raw_length is not None else 0
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )
            if content_length > settings.max_request_bytes:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body is too large."},
                )

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
            "style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'self'"
        )
        forwarded_proto = request.headers.get("x-forwarded-proto", "")
        if request.url.scheme == "https" or forwarded_proto == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/health")
    async def health(request: Request):
        model_health, retrieval_health = await asyncio.gather(
            request.app.state.model.health(), request.app.state.retriever.health()
        )
        return {
            "status": "ok" if model_health["ok"] and retrieval_health["ok"] else "degraded",
            "model": {"ok": model_health["ok"]},
            "retriever": {"ok": retrieval_health["ok"]},
        }

    @app.get("/robots.txt", include_in_schema=False)
    async def robots():
        return PlainTextResponse("User-agent: *\nDisallow: /\n")

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
