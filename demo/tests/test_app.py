import json
import unittest

from fastapi.testclient import TestClient

from demo.backend.app import create_app
from demo.backend.config import Settings


class FakeAgent:
    async def run(self, question):
        yield {"type": "query", "query": "test query", "turn": 0}
        yield {"type": "answer_delta", "delta": "test answer", "turn": 0}
        yield {"type": "done", "answer": "test answer", "turns": 1, "history": []}


def test_settings():
    return Settings(
        model_url="http://127.0.0.1:9/v1",
        model_name="test-model",
        retrieval_url="http://127.0.0.1:9",
        max_turns=2,
        top_k=3,
        max_tokens=128,
        evidence_max_tokens=64,
        question_max_chars=100,
        request_timeout_seconds=10.0,
        max_concurrent_requests=1,
        cooldown_seconds=0.0,
        requests_per_window=20,
        rate_window_seconds=3600.0,
        max_request_bytes=4096,
        trust_cloudflare_ip=False,
        allowed_origins=(),
        allowed_hosts=(),
    )


class DemoAppTests(unittest.TestCase):
    def test_index_and_security_headers(self):
        with TestClient(create_app(test_settings())) as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("SAPR-RAG", response.text)
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertEqual(response.headers["x-robots-tag"], "noindex, nofollow, noarchive")
        self.assertIn("default-src 'self'", response.headers["content-security-policy"])

    def test_public_metadata_is_minimal(self):
        with TestClient(create_app(test_settings())) as client:
            openapi = client.get("/openapi.json")
            robots = client.get("/robots.txt")
            health = client.get("/api/health")

        self.assertEqual(openapi.status_code, 404)
        self.assertEqual(robots.status_code, 200)
        self.assertIn("Disallow: /", robots.text)
        self.assertNotIn("n_vectors", health.text)
        self.assertNotIn("n_docs", health.text)

    def test_trusted_host_and_forwarded_https(self):
        settings = test_settings()
        settings = Settings(**{**settings.__dict__, "allowed_hosts": ("rag.example.com",)})
        with TestClient(create_app(settings), base_url="https://rag.example.com") as client:
            accepted = client.get("/", headers={"X-Forwarded-Proto": "https"})
            rejected = client.get("/", headers={"Host": "attacker.example"})

        self.assertEqual(accepted.status_code, 200)
        self.assertIn("max-age=31536000", accepted.headers["strict-transport-security"])
        self.assertEqual(rejected.status_code, 400)

    def test_stream_endpoint_emits_sse_without_reasoning_text(self):
        app = create_app(test_settings())
        with TestClient(app) as client:
            app.state.agent = FakeAgent()
            response = client.post("/api/chat/stream", json={"question": "Question?"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(payloads[0]["type"], "queued")
        self.assertEqual(payloads[-1]["type"], "done")
        self.assertNotIn("raw", response.text)

    def test_rejects_oversized_question_before_streaming(self):
        with TestClient(create_app(test_settings())) as client:
            response = client.post("/api/chat/stream", json={"question": "x" * 101})

        self.assertEqual(response.status_code, 422)

    def test_rejects_oversized_request_body(self):
        settings = test_settings()
        settings = Settings(**{**settings.__dict__, "max_request_bytes": 32})
        with TestClient(create_app(settings)) as client:
            response = client.post(
                "/api/chat/stream",
                content=b"x" * 33,
                headers={"Content-Type": "application/json"},
            )

        self.assertEqual(response.status_code, 413)

    def test_rate_limit_does_not_store_raw_client_address(self):
        settings = test_settings()
        settings = Settings(
            **{
                **settings.__dict__,
                "requests_per_window": 1,
                "trust_cloudflare_ip": True,
            }
        )
        app = create_app(settings)
        with TestClient(app) as client:
            app.state.agent = FakeAgent()
            first = client.post(
                "/api/chat/stream",
                json={"question": "Question?"},
                headers={"CF-Connecting-IP": "203.0.113.25"},
            )
            second = client.post(
                "/api/chat/stream",
                json={"question": "Question?"},
                headers={"CF-Connecting-IP": "203.0.113.25"},
            )
            stored_keys = tuple(app.state.limiter.requests)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertNotIn("203.0.113.25", stored_keys)


if __name__ == "__main__":
    unittest.main()
