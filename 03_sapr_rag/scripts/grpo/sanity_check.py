#!/usr/bin/env python3
"""GRPO plugin sanity 验证（不启真训练）。

step 1: 检索 daemon /health + 单查询（需 daemon 已起；可 --skip_daemon 跳过）。
step 2: mock rollout_infos + dataset 行，验证 Reward-v2 各 ORM 的值域和方向。

用法：
  python sanity_check.py                 # 跑全部
  python sanity_check.py --skip_daemon   # 只验 reward 逻辑
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plugin import (
    SaprF1ORM,
    SaprFormatORM,
    SaprMaxTurnORM,
    SaprMarginalRelevanceORM,
    SaprRagScheduler,
    SaprRelevanceORM,
    SaprRepeatQueryORM,
    SaprTurnCostORM,
)
from retrieval_client import RetrievalClient


def check_daemon(url):
    print(f"[sanity] daemon /health @ {url} ...")
    rc = RetrievalClient(base_url=url)
    rc.wait_until_ready(max_wait=60, interval=3)
    docs = rc.search("who founded Apple Inc", top_k=3)
    assert docs and "title" in docs[0], f"unexpected search result: {docs}"
    print(f"[sanity]   OK, top-1 title={docs[0]['title']!r} score={docs[0].get('score')}")


def check_rewards():
    f1 = SaprF1ORM()
    rel = SaprRelevanceORM()
    marginal_rel = SaprMarginalRelevanceORM()
    fmt = SaprFormatORM()
    turn_cost = SaprTurnCostORM()
    repeat_query = SaprRepeatQueryORM()
    max_turn = SaprMaxTurnORM()

    # mock：检到 gold title + answer 正确 + 格式合法
    good_completion = "So the answer is <answer>Steve Wozniak</answer>"
    good_kwargs = {
        "golden_answers": [["Steve Wozniak"]],
        "gold_titles": [["Steve Wozniak", "Apple Inc."]],
        "gold_sup_sents": [["Steve Wozniak is an American inventor.",
                            "Apple Inc. is an American company."]],
        "rollout_infos": [{
            "retrieved_steps": [
                {"turn": 1, "query": "who founded Apple",
                 "docs": [
                     {"title": "Steve Wozniak", "text": "Steve Wozniak is an American inventor."},
                     {"title": "Apple Inc.", "text": "Apple Inc. is an American company."},
                 ]},
            ]
        }],
    }
    # mock：多轮轨迹允许前序 query，只要最后一个协议标签是 answer 即合法
    mult_turn_completion = (
        "Need retrieval. So the next query is <query>who founded Apple</query>"
        " Reference: <reference>Steve Wozniak founded Apple.</reference>"
        " So the answer is <answer>Steve Wozniak</answer>"
    )
    # mock：检索全错 + answer 错 + 缺 answer 标签（格式非法）
    bad_completion = "So the next query is <query>foobar</query>"
    bad_kwargs = {
        "golden_answers": [["Steve Wozniak"]],
        "gold_titles": [["Steve Wozniak", "Apple Inc."]],
        "gold_sup_sents": [["Steve Wozniak is an American inventor.",
                            "Apple Inc. is an American company."]],
        "rollout_infos": [{
            "retrieved_steps": [
                {"turn": 1, "query": "foobar",
                 "docs": [{"title": "Banana", "text": "A banana is a fruit."}]},
            ]
        }],
    }
    answer_shortcut_kwargs = {
        **bad_kwargs,
        "rollout_infos": [{
            "retrieved_steps": [{
                "turn": 1,
                "query": "unrelated",
                "docs": [{
                    "title": "Unrelated page",
                    "text": "The page happens to mention Steve Wozniak.",
                }],
            }],
        }],
    }
    repeated_kwargs = {
        **good_kwargs,
        "rollout_infos": [{
            "retrieved_steps": [
                {
                    "turn": turn,
                    "query": "Who founded Apple?" if turn % 2 else "who founded apple",
                    "docs": good_kwargs["rollout_infos"][0]["retrieved_steps"][0]["docs"],
                }
                for turn in range(1, 7)
            ],
        }],
    }
    exhausted_kwargs = {
        **repeated_kwargs,
        "rollout_infos": [{
            "retrieved_steps": repeated_kwargs["rollout_infos"][0]["retrieved_steps"][:5],
            "num_turns": 6,
        }],
    }

    f1_good = f1([good_completion], **good_kwargs)[0]
    f1_bad = f1([bad_completion], **bad_kwargs)[0]
    rel_good = rel([good_completion], **good_kwargs)[0]
    rel_bad = rel([bad_completion], **bad_kwargs)[0]
    rel_shortcut = rel([bad_completion], **answer_shortcut_kwargs)[0]
    marginal_good = marginal_rel([good_completion], **good_kwargs)[0]
    marginal_bad = marginal_rel([bad_completion], **bad_kwargs)[0]
    marginal_repeated = marginal_rel([bad_completion], **repeated_kwargs)[0]
    fmt_good = fmt([good_completion], **good_kwargs)[0]
    fmt_multi_turn = fmt([mult_turn_completion], **good_kwargs)[0]
    fmt_bad = fmt([bad_completion], **bad_kwargs)[0]
    turn_good = turn_cost([good_completion], **good_kwargs)[0]
    turn_repeated = turn_cost([bad_completion], **repeated_kwargs)[0]
    repeat_good = repeat_query([good_completion], **good_kwargs)[0]
    repeat_repeated = repeat_query([bad_completion], **repeated_kwargs)[0]
    max_turn_good = max_turn([good_completion], **exhausted_kwargs)[0]
    max_turn_bad = max_turn([bad_completion], **exhausted_kwargs)[0]

    print(f"[sanity] f1:        good={f1_good:.3f}  bad={f1_bad:.3f}")
    print(f"[sanity] relevance: good={rel_good:.3f}  bad={rel_bad:.3f}  answer_shortcut={rel_shortcut:.3f}")
    print(f"[sanity] marginal:  good={marginal_good:.3f}  bad={marginal_bad:.3f}  repeated={marginal_repeated:.3f}")
    print(f"[sanity] format:    good={fmt_good:.3f}  multi_turn={fmt_multi_turn:.3f}  bad={fmt_bad:.3f}")
    print(f"[sanity] turn_cost: good={turn_good:.3f}  repeated={turn_repeated:.3f}")
    print(f"[sanity] repeat:    good={repeat_good:.3f}  repeated={repeat_repeated:.3f}")
    print(f"[sanity] max_turn:  answered={max_turn_good:.3f}  unanswered={max_turn_bad:.3f}")

    for name, v in [("f1_good", f1_good), ("f1_bad", f1_bad),
                    ("rel_good", rel_good), ("rel_bad", rel_bad),
                    ("fmt_good", fmt_good), ("fmt_multi_turn", fmt_multi_turn),
                    ("fmt_bad", fmt_bad)]:
        assert 0.0 <= v <= 1.0, f"{name} out of range: {v}"

    assert f1_good > f1_bad, "f1 方向错：answer 正确应高于错误"
    assert rel_good > rel_bad, "relevance 方向错：检到 gold 应高于检错"
    assert rel_shortcut == 0.0, "relevance 不应再用 gold answer 文本作为命中捷径"
    assert marginal_good == 1.0 and marginal_bad == 0.0, "新增证据奖励基础方向错误"
    assert marginal_good > marginal_repeated, "重复检索同一证据后，新增证据奖励应低于及时回答"
    assert fmt_good == 1.0 and fmt_multi_turn == 1.0 and fmt_bad == 0.0, "format 方向错"
    assert turn_good == 0.0 and turn_repeated == -5.0, "turn cost 方向或免费首轮错误"
    assert repeat_good == 0.0 and repeat_repeated == -3.0, "重复 query 惩罚或 cap 错误"
    assert max_turn_good == 0.0 and max_turn_bad == -1.0, "max-turn 未回答惩罚错误"
    print("[sanity] reward 值域 + 方向 全部通过 ✓")


class _FakeClient:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k=3):
        self.calls.append((query, top_k))
        return [{"title": "Apple Inc.", "text": "Apple Inc. was founded by Steve Wozniak."}]


class _FakeRequest:
    def __init__(self, uuid="traj-1"):
        self.uuid = uuid
        self.messages = [{"role": "user", "content": "Question: who founded Apple?"}]


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)
        self.token_ids = [1, 2, 3]


def check_duplicate_query_intercept():
    scheduler = SaprRagScheduler.__new__(SaprRagScheduler)
    scheduler.client = _FakeClient()
    scheduler.top_k = 3
    scheduler.use_evidence_agent = False
    scheduler._traj = {}

    req = _FakeRequest()
    first = scheduler.step(req, _FakeChoice("So the next query is <query>Who founded Apple?</query>"), 1)
    second = scheduler.step(req, _FakeChoice("So the next query is <query>who founded apple</query>"), 2)

    steps = second["rollout_infos"]["retrieved_steps"]
    assert len(scheduler.client.calls) == 1, "完全重复 query 不应再次调用检索服务"
    assert steps[0]["search_executed"] is True and steps[0]["exact_duplicate"] is False
    assert steps[1]["search_executed"] is False and steps[1]["exact_duplicate"] is True
    assert steps[1]["docs"] == [], "重复 query 的 docs 应为空，避免伪造新增证据"
    assert "duplicates a previous query" in req.messages[-1]["content"]
    print("[sanity] duplicate query runtime intercept 通过 ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("SAPR_RETRIEVAL_URL", "http://127.0.0.1:8100"))
    ap.add_argument("--skip_daemon", action="store_true")
    args = ap.parse_args()

    if not args.skip_daemon:
        check_daemon(args.url)
    check_rewards()
    check_duplicate_query_intercept()
    print("[sanity] ALL PASS")


if __name__ == "__main__":
    main()
