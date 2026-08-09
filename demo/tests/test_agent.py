import unittest

from demo.backend.agent import SAPRDemoAgent


class FakeModel:
    def __init__(self, streams, replies):
        self.streams = iter(streams)
        self.replies = iter(replies)

    async def chat_stream(self, messages, *, max_tokens, stop):
        for chunk in next(self.streams):
            yield chunk

    async def chat(self, messages, *, max_tokens, stop):
        return next(self.replies)


class FakeRetriever:
    def __init__(self):
        self.queries = []

    async def search(self, query, top_k):
        self.queries.append((query, top_k))
        return [
            {"title": "Example", "text": "A supporting passage.", "score": 0.91}
        ]


class SAPRDemoAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_retrieval_evidence_and_streamed_answer(self):
        model = FakeModel(
            streams=[
                ["Need retrieval. <que", "ry>bridge entity", "</query>"],
                ["So the answer is <answer>final ", "answer", "</answer>"],
            ],
            replies=["Relevant evidence: <evidence>supporting fact</evidence>"],
        )
        retriever = FakeRetriever()
        agent = SAPRDemoAgent(
            model,
            retriever,
            max_turns=3,
            top_k=3,
            max_tokens=512,
            evidence_max_tokens=128,
        )

        events = [event async for event in agent.run("A multi-hop question?")]

        self.assertEqual(retriever.queries, [("bridge entity", 3)])
        self.assertIn("query", [event["type"] for event in events])
        self.assertIn("documents", [event["type"] for event in events])
        self.assertIn("evidence", [event["type"] for event in events])
        deltas = "".join(
            event["delta"] for event in events if event["type"] == "answer_delta"
        )
        self.assertEqual(deltas, "final answer")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["answer"], "final answer")

    async def test_invalid_action_returns_public_error(self):
        agent = SAPRDemoAgent(
            FakeModel(streams=[["No structured action"]], replies=[]),
            FakeRetriever(),
            max_turns=1,
            top_k=3,
            max_tokens=512,
            evidence_max_tokens=128,
        )

        events = [event async for event in agent.run("Question?")]

        self.assertEqual(events[-1], {"type": "error", "code": "invalid_agent_action"})


if __name__ == "__main__":
    unittest.main()
