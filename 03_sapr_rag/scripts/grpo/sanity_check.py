#!/usr/bin/env python3
"""GRPO plugin sanity 验证（不启真训练）。

step 1: 检索 daemon /health + 单查询（需 daemon 已起；可 --skip_daemon 跳过）。
step 2: mock 一条 rollout_infos + dataset 行，调三个 ORM，断言值域 [0,1] 且方向合理。

用法：
  python sanity_check.py                 # 跑全部
  python sanity_check.py --skip_daemon   # 只验 reward 逻辑
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plugin import SaprF1ORM, SaprRelevanceORM, SaprFormatORM
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
    fmt = SaprFormatORM()

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

    f1_good = f1([good_completion], **good_kwargs)[0]
    f1_bad = f1([bad_completion], **bad_kwargs)[0]
    rel_good = rel([good_completion], **good_kwargs)[0]
    rel_bad = rel([bad_completion], **bad_kwargs)[0]
    fmt_good = fmt([good_completion], **good_kwargs)[0]
    fmt_bad = fmt([bad_completion], **bad_kwargs)[0]

    print(f"[sanity] f1:        good={f1_good:.3f}  bad={f1_bad:.3f}")
    print(f"[sanity] relevance: good={rel_good:.3f}  bad={rel_bad:.3f}")
    print(f"[sanity] format:    good={fmt_good:.3f}  bad={fmt_bad:.3f}")

    for name, v in [("f1_good", f1_good), ("f1_bad", f1_bad),
                    ("rel_good", rel_good), ("rel_bad", rel_bad),
                    ("fmt_good", fmt_good), ("fmt_bad", fmt_bad)]:
        assert 0.0 <= v <= 1.0, f"{name} out of range: {v}"

    assert f1_good > f1_bad, "f1 方向错：answer 正确应高于错误"
    assert rel_good > rel_bad, "relevance 方向错：检到 gold 应高于检错"
    assert fmt_good == 1.0 and fmt_bad == 0.0, "format 方向错"
    print("[sanity] reward 值域 + 方向 全部通过 ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("SAPR_RETRIEVAL_URL", "http://127.0.0.1:8100"))
    ap.add_argument("--skip_daemon", action="store_true")
    args = ap.parse_args()

    if not args.skip_daemon:
        check_daemon(args.url)
    check_rewards()
    print("[sanity] ALL PASS")


if __name__ == "__main__":
    main()
