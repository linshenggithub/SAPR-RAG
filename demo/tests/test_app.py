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
        trust_cloudflare_ip=False,
        allowed_origins=(),
    )


class DemoAppTests(unittest.TestCase):
    def test_index_and_security_headers(self):
        with TestClient(create_app(test_settings())) as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("SAPR-RAG", response.text)
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertIn("default-src 'self'", response.headers["content-security-policy"])

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


if __name__ == "__main__":
    unittest.main()
