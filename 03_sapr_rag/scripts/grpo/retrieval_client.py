#!/usr/bin/env python
"""检索 daemon 的轻量客户端，供 GRPO rollout scheduler / reward 函数 import 使用。

用法：
    from retrieval_client import RetrievalClient
    rc = RetrievalClient("http://127.0.0.1:8100")
    rc.wait_until_ready()                       # 训练启动前阻塞等待 daemon 就绪
    docs = rc.search("who founded Apple", top_k=3)
    batch = rc.search_batch(["q1", "q2"], top_k=3)
"""
import time

import requests


class RetrievalClient:
    def __init__(self, base_url="http://127.0.0.1:8100", timeout=30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def wait_until_ready(self, max_wait=1800, interval=5):
        """轮询 /health 直到 daemon 加载完索引+corpus（首次冷启动可能数分钟）。"""
        t0 = time.time()
        while time.time() - t0 < max_wait:
            try:
                r = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return r.json()
            except requests.RequestException:
                pass
            time.sleep(interval)
        raise TimeoutError(f"retrieval daemon not ready after {max_wait}s @ {self.base_url}")

    def search(self, query, top_k=3):
        r = self._session.post(
            f"{self.base_url}/search",
            json={"query": query, "top_k": top_k},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["results"]

    def search_batch(self, queries, top_k=3):
        r = self._session.post(
            f"{self.base_url}/search_batch",
            json={"queries": queries, "top_k": top_k},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["results"]
