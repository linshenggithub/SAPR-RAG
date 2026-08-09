from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    model_url: str
    model_name: str
    retrieval_url: str
    max_turns: int
    top_k: int
    max_tokens: int
    evidence_max_tokens: int
    question_max_chars: int
    request_timeout_seconds: float
    max_concurrent_requests: int
    cooldown_seconds: float
    requests_per_window: int
    rate_window_seconds: float
    max_request_bytes: int
    trust_cloudflare_ip: bool
    allowed_origins: tuple[str, ...]
    allowed_hosts: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        origins = tuple(
            item.strip()
            for item in os.getenv("SAPR_DEMO_ALLOWED_ORIGINS", "").split(",")
            if item.strip()
        )
        hosts = tuple(
            item.strip()
            for item in os.getenv("SAPR_DEMO_ALLOWED_HOSTS", "").split(",")
            if item.strip()
        )
        return cls(
            model_url=os.getenv("SAPR_DEMO_MODEL_URL", "http://127.0.0.1:8001/v1").rstrip("/"),
            model_name=os.getenv("SAPR_DEMO_MODEL_NAME", "sapr-sft"),
            retrieval_url=os.getenv(
                "SAPR_DEMO_RETRIEVAL_URL", "http://127.0.0.1:8100"
            ).rstrip("/"),
            max_turns=_env_int("SAPR_DEMO_MAX_TURNS", 6, 1, 10),
            top_k=_env_int("SAPR_DEMO_TOP_K", 3, 1, 10),
            max_tokens=_env_int("SAPR_DEMO_MAX_TOKENS", 512, 64, 2048),
            evidence_max_tokens=_env_int(
                "SAPR_DEMO_EVIDENCE_MAX_TOKENS", 128, 32, 512
            ),
            question_max_chars=_env_int(
                "SAPR_DEMO_QUESTION_MAX_CHARS", 500, 50, 2000
            ),
            request_timeout_seconds=_env_float(
                "SAPR_DEMO_REQUEST_TIMEOUT_SECONDS", 300.0, 10.0
            ),
            max_concurrent_requests=_env_int(
                "SAPR_DEMO_MAX_CONCURRENT_REQUESTS", 1, 1, 8
            ),
            cooldown_seconds=_env_float("SAPR_DEMO_COOLDOWN_SECONDS", 5.0, 0.0),
            requests_per_window=_env_int(
                "SAPR_DEMO_REQUESTS_PER_WINDOW", 20, 1, 1000
            ),
            rate_window_seconds=_env_float(
                "SAPR_DEMO_RATE_WINDOW_SECONDS", 3600.0, 60.0
            ),
            max_request_bytes=_env_int(
                "SAPR_DEMO_MAX_REQUEST_BYTES", 4096, 512, 65536
            ),
            trust_cloudflare_ip=os.getenv("SAPR_DEMO_TRUST_CLOUDFLARE_IP", "0") == "1",
            allowed_origins=origins,
            allowed_hosts=hosts,
        )
