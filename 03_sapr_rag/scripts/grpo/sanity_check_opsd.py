#!/usr/bin/env python3
"""GRPO / OPSD sanity 验证（不启真训练）。

step 1: 检索 daemon /health + 单查询（需 daemon 已起；可 --skip_daemon 跳过）。
step 2: mock 一条 rollout_infos + dataset 行，调三个 ORM，断言值域 [0,1] 且方向合理。
step 3: 可选检查 OPSD dataset 的 teacher_prompt 覆盖率、长度和 gold grounding。
step 4: 用 CPU synthetic tensor 验证 teacher log-ratio / KL / per-token advantage 注入。
step 5: mock SaprRagScheduler.step()，验证 observation token 的 loss_mask=0。

用法：
  python sanity_check.py                 # 跑全部
  python sanity_check.py --skip_daemon   # 只验 reward 逻辑
  python sanity_check.py --skip_daemon --dataset data/grpo/hotpotqa_2wiki_train_opsd.jsonl
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plugin
from plugin import SaprF1ORM, SaprRelevanceORM, SaprFormatORM
from retrieval_client import RetrievalClient


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _norm_text(s):
    return " ".join(str(s or "").lower().split())


def _contains_any(haystack, needles):
    h = _norm_text(haystack)
    return any(_norm_text(n) and _norm_text(n) in h for n in needles)


def _pct(n, d):
    return 0.0 if d == 0 else 100.0 * n / d


def _quantiles(values):
    if not values:
        return {}
    vals = sorted(values)
    def q(p):
        idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))
        return vals[idx]
    return {
        "min": vals[0],
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": vals[-1],
        "mean": statistics.mean(vals),
    }


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

    f1_good = f1([good_completion], **good_kwargs)[0]
    f1_bad = f1([bad_completion], **bad_kwargs)[0]
    rel_good = rel([good_completion], **good_kwargs)[0]
    rel_bad = rel([bad_completion], **bad_kwargs)[0]
    fmt_good = fmt([good_completion], **good_kwargs)[0]
    fmt_multi_turn = fmt([mult_turn_completion], **good_kwargs)[0]
    fmt_bad = fmt([bad_completion], **bad_kwargs)[0]

    print(f"[sanity] f1:        good={f1_good:.3f}  bad={f1_bad:.3f}")
    print(f"[sanity] relevance: good={rel_good:.3f}  bad={rel_bad:.3f}")
    print(f"[sanity] format:    good={fmt_good:.3f}  multi_turn={fmt_multi_turn:.3f}  bad={fmt_bad:.3f}")

    for name, v in [("f1_good", f1_good), ("f1_bad", f1_bad),
                    ("rel_good", rel_good), ("rel_bad", rel_bad),
                    ("fmt_good", fmt_good), ("fmt_multi_turn", fmt_multi_turn),
                    ("fmt_bad", fmt_bad)]:
        assert 0.0 <= v <= 1.0, f"{name} out of range: {v}"

    assert f1_good > f1_bad, "f1 方向错：answer 正确应高于错误"
    assert rel_good > rel_bad, "relevance 方向错：检到 gold 应高于检错"
    assert fmt_good == 1.0 and fmt_multi_turn == 1.0 and fmt_bad == 0.0, "format 方向错"
    print("[sanity] reward 值域 + 方向 全部通过 ✓")


def check_dataset(path, max_rows, min_answer_coverage, min_evidence_coverage, max_teacher_tokens):
    path = Path(path)
    print(f"[sanity] dataset teacher_prompt @ {path} ...")
    assert path.is_file(), f"dataset not found: {path}"

    total = 0
    teacher_nonempty = 0
    answer_denom = answer_hits = 0
    evidence_denom = evidence_hits = 0
    token_lengths = []
    char_lengths = []
    fallback = truncated = 0
    first_teacher_row = None

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            prompt = row.get("teacher_prompt")
            if prompt:
                teacher_nonempty += 1
                first_teacher_row = first_teacher_row or row
                char_lengths.append(len(prompt))
                token_lengths.append(int(row.get("teacher_prompt_tokens") or len(prompt.split())))
                fallback += int(bool(row.get("teacher_prompt_fallback")))
                truncated += int(bool(row.get("teacher_prompt_truncated")))

                answers = _as_list(row.get("golden_answers"))
                if answers:
                    answer_denom += 1
                    answer_hits += int(_contains_any(prompt, answers))

                evidence_needles = _as_list(row.get("gold_titles")) + _as_list(row.get("gold_sup_sents"))
                evidence_needles = [x for x in evidence_needles if x]
                if evidence_needles:
                    evidence_denom += 1
                    evidence_hits += int(_contains_any(prompt, evidence_needles))
            if max_rows and total >= max_rows:
                break

    assert total > 0, f"empty dataset: {path}"
    print(f"[sanity]   rows={total} teacher_nonempty={teacher_nonempty} ({_pct(teacher_nonempty, total):.1f}%)")

    if teacher_nonempty == 0:
        print("[sanity]   plain GRPO dataset: no teacher_prompt, skip OPSD grounding checks")
        return None

    tq = _quantiles(token_lengths)
    cq = _quantiles(char_lengths)
    print("[sanity]   teacher_tokens: "
          f"mean={tq['mean']:.1f} p50={tq['p50']} p90={tq['p90']} p95={tq['p95']} max={tq['max']}")
    print("[sanity]   teacher_chars:  "
          f"mean={cq['mean']:.1f} p50={cq['p50']} p90={cq['p90']} p95={cq['p95']} max={cq['max']}")
    print(f"[sanity]   fallback={fallback} truncated={truncated}")
    print(f"[sanity]   answer_coverage={answer_hits}/{answer_denom} ({_pct(answer_hits, answer_denom):.1f}%)")
    print(f"[sanity]   evidence_coverage={evidence_hits}/{evidence_denom} ({_pct(evidence_hits, evidence_denom):.1f}%)")

    assert teacher_nonempty == total, "OPSD dataset 存在 teacher_prompt 缺失行"
    if answer_denom:
        assert answer_hits / answer_denom >= min_answer_coverage, "teacher_prompt 中 gold answer 覆盖率不足"
    if evidence_denom:
        assert evidence_hits / evidence_denom >= min_evidence_coverage, "teacher_prompt 中 gold evidence 覆盖率不足"
    if max_teacher_tokens:
        too_long = sum(1 for x in token_lengths if x > max_teacher_tokens)
        assert too_long == 0, f"teacher_prompt 超过预算 {max_teacher_tokens}: {too_long} rows"
    return first_teacher_row


def check_teacher_view(row=None):
    print("[sanity] OPSD OnPolicySample teacher view ...")
    from swift.rl_core.data import OnPolicySample

    if row is None:
        row = {
            "messages": [
                {"role": "system", "content": "You are a RAG agent."},
                {"role": "user", "content": "Question: who founded Apple?"},
            ],
            "teacher_prompt": "Privileged evidence: Steve Wozniak co-founded Apple. Answer: Steve Wozniak.",
        }

    sample = OnPolicySample.from_row({
        "messages": row["messages"],
        "teacher_prompt": row.get("teacher_prompt"),
        "response_token_ids": [[101, 102, 103]],
        "response_loss_mask": [[1, 1, 1]],
    })
    assert sample.build_teacher_view(), "teacher_prompt 未生成 teacher view"
    assert sample.teacher_messages is not None
    assert sample.teacher_messages[-1]["role"] == "user"
    assert sample.teacher_messages[-1]["content"] == row.get("teacher_prompt")
    assert sample.messages[-1]["content"] != sample.teacher_messages[-1]["content"], "student prompt 被污染"
    assert sample.response_token_ids == [[101, 102, 103]], "teacher view 不应改变 response token ids"

    plain = OnPolicySample.from_row({"messages": row["messages"], "response_token_ids": [[1]]})
    assert not plain.build_teacher_view(), "plain sample 不应生成 teacher view"
    print("[sanity]   teacher_prompt 替换最后 user 且 response ids 保持一致 ✓")


def check_teacher_advantage():
    print("[sanity] OPD teacher log-ratio / KL / per-token advantage ...")
    import torch
    from swift.rl_core.advantage import (
        compute_teacher_kl_per_token,
        compute_teacher_logratio,
        expand_advantage_to_per_token,
    )

    teacher = torch.tensor([[-0.1, -0.3, -2.0], [-1.0, -0.2, -0.4]], dtype=torch.float32)
    policy = torch.tensor([[-0.4, -0.2, -1.5], [-0.8, -0.5, -0.9]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.float32)
    base_adv = torch.tensor([0.2, -0.1], dtype=torch.float32)
    coef = 0.1

    logratio = compute_teacher_logratio(teacher, policy, mask)
    expected_logratio = (teacher - policy) * mask
    assert torch.allclose(logratio, expected_logratio), "teacher log-ratio 计算不符合定义"

    kl = compute_teacher_kl_per_token(teacher, policy, mask)
    assert torch.isfinite(kl).all(), "teacher KL 出现非有限值"
    assert (kl >= -1e-6).all(), "k3 teacher KL 应非负"
    assert (kl[mask == 0] == 0).all(), "mask=0 token 的 teacher KL 应为 0"

    per_token = expand_advantage_to_per_token(base_adv, mask, teacher, policy, teacher_kl_coef=coef)
    expected = base_adv.unsqueeze(1).expand_as(mask) + coef * expected_logratio
    assert torch.allclose(per_token, expected), "per-token advantage 注入不符合 A + alpha * logratio"
    print("[sanity]   teacher_kl finite, token 对齐张量形状一致, advantage 注入正确 ✓")


class _FakeRetrievalClient:
    def __init__(self, *args, **kwargs):
        pass

    def wait_until_ready(self, *args, **kwargs):
        return None

    def search(self, query, top_k=3):
        return [
            {"title": "Steve Wozniak", "text": "Steve Wozniak co-founded Apple.", "score": 1.0},
            {"title": "Apple Inc.", "text": "Apple Inc. is a technology company.", "score": 0.9},
        ][:top_k]


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(1000, 1000 + max(1, len(text.split()))))


def check_scheduler_loss_mask():
    print("[sanity] SaprRagScheduler observation loss_mask ...")
    old_client = plugin.RetrievalClient
    plugin.RetrievalClient = _FakeRetrievalClient
    try:
        scheduler = plugin.SaprRagScheduler(infer_engine=None, tokenizer=_FakeTokenizer())
        infer_request = SimpleNamespace(
            uuid="sanity-uuid",
            messages=[{"role": "user", "content": "Question: who founded Apple?"}],
        )
        response_choice = SimpleNamespace(
            message=SimpleNamespace(content="Need evidence. <query>Apple founders</query>"),
            token_ids=[11, 12, 13],
        )
        result = scheduler.step(infer_request, response_choice, current_turn=1)
    finally:
        plugin.RetrievalClient = old_client

    token_ids = result["response_token_ids"]
    loss_mask = result["response_loss_mask"]
    assert len(token_ids) == len(loss_mask), "response_token_ids 和 response_loss_mask 长度不一致"
    assert loss_mask[:3] == [1, 1, 1], "模型动作 token 应 loss_mask=1"
    assert token_ids[:3] == [11, 12, 13], "原始 response token ids 被破坏"
    assert len(token_ids) > 3 and all(v == 0 for v in loss_mask[3:]), "observation token 应 loss_mask=0"
    assert "Reference: <reference>" in infer_request.messages[-1]["content"], "observation 未注入下一轮 user content"
    infos = result["rollout_infos"]
    assert infos["uuid"] == "sanity-uuid"
    assert infos["retrieved_steps"] and infos["retrieved_steps"][0]["query"] == "Apple founders"
    print("[sanity]   action token=1, observation token=0, rollout_infos 保存 ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("SAPR_RETRIEVAL_URL", "http://127.0.0.1:8100"))
    ap.add_argument("--skip_daemon", action="store_true")
    ap.add_argument("--dataset", default=None, help="可选：检查带 teacher_prompt 的 OPSD jsonl 数据")
    ap.add_argument("--max_rows", type=int, default=1000, help="dataset sanity 最多读取行数，0 表示全量")
    ap.add_argument("--min_answer_coverage", type=float, default=0.95)
    ap.add_argument("--min_evidence_coverage", type=float, default=0.95)
    ap.add_argument("--max_teacher_tokens", type=int, default=1536)
    args = ap.parse_args()

    if not args.skip_daemon:
        check_daemon(args.url)
    check_rewards()
    teacher_row = None
    if args.dataset:
        teacher_row = check_dataset(
            args.dataset,
            max_rows=args.max_rows,
            min_answer_coverage=args.min_answer_coverage,
            min_evidence_coverage=args.min_evidence_coverage,
            max_teacher_tokens=args.max_teacher_tokens,
        )
    check_teacher_view(teacher_row)
    check_teacher_advantage()
    check_scheduler_loss_mask()
    print("[sanity] ALL PASS")


if __name__ == "__main__":
    main()
