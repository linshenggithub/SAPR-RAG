#!/usr/bin/env python3
"""
Gate 0 L1: Re-label Q values with GPT-4o on an UNBIASED random sample.

目的:
- 在你已有的 ReasonRAG MCTS reward_data 树上,用 GPT-4o + 原 EVALUATION_PROMPT
  重新对 sibling children 打 Q.
- 与 Llama-70B-int4 的 Q 值做无偏对比,验证 "标量 PRM 在 67.5% 分支点撞车"
  到底是 judge 模型缺陷,还是 scalar PRM 的固有局限.

与 parse_trees.py 的关键区别:
- 这里的 trajectory 采样是 **完全随机** (默认 seed=2026),不再做 top-30 + random-20
  的有偏混合,以避免 q_same_rate 被高估.
- 直接从 reward_data{0..3}.json 解析,不依赖 sampled_trajectories.json,
  使用 **完整未截断** 的 response 文本 (parse_trees.py 截断到 1000 字符,
  会让 GPT-4o 评估失真).

输出:
- gate0/data/relabel_q_gpt4o_branches.jsonl   每个 child 一行,带 Q_old / Q_new
- gate0/data/relabel_q_gpt4o_stats.json       q_same_rate 对比汇总
- gate0/data/relabel_q_gpt4o_progress.jsonl   断点续跑用的进度文件

使用:
    # 干跑,只统计采样规模和成本估算,不调用 API
    python relabel_q_gpt4o.py --dry-run

    # 正式跑 (需要 OPENAI_API_KEY 或 DMXAPI key)
    export OPENAI_API_KEY=sk-xxx
    export OPENAI_BASE_URL=https://www.dmxapi.cn/v1
    python relabel_q_gpt4o.py --n-traj 50 --concurrency 8

    # 断点续跑 (自动跳过 progress 文件里已完成的 child_id)
    python relabel_q_gpt4o.py --n-traj 50 --resume

环境依赖:
    pip install openai tqdm
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 让脚本能直接 `python gate0/relabel_q_gpt4o.py` 运行：把仓库根加进 sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.paths import REASONRAG_OUTPUT_DIR  # noqa: E402

# ----------------------------------------------------------------------------
# 1) 路径与超参
# ----------------------------------------------------------------------------
# 仓库外路径：从 config/paths.py 读取（可被 SAPR_REASONRAG_OUTPUT_DIR 环境变量覆盖）
DEFAULT_REWARD_DIR = REASONRAG_OUTPUT_DIR
DEFAULT_OUT_DIR = Path(__file__).parent / "data"

EVALUATION_PROMPT = (
    "An agent is tasked with answering a question using a retrieval tool. \n"
    "    Critically assess its intermediate reasoning process to determine if it leads to the correct answer. \n"
    "    Identify all flaws, inconsistencies, and mistakes in the thought process. \n"
    "    Every imperfection, no matter how small, must be acknowledged. \n"
    "    Evaluate how effectively the reasoning supports the final answer and the overall accuracy of the response. \n"
    "    Ensure the evaluation is extremely harsh, leaving no leniency. \n"
    "    Even if the answer seems close to correct, do not award full marks to maintain strict grading standards. \n"
    "    Assign a score between [0, 100] based on the severity of flaws and the reasoning's accuracy in leading to the golden answer.\n"
    "Respond briefly and conclude with: So the score is [Score].\n"
)


# ----------------------------------------------------------------------------
# 2) 解析 reward_data 树(直接复用 parse_trees.py 的核心逻辑,但读完整字段)
# ----------------------------------------------------------------------------
def load_reward_data(reward_dir: Path) -> List[dict]:
    all_data = []
    for i in range(4):
        fpath = reward_dir / f"reward_data{i}.json"
        if not fpath.exists():
            print(f"[WARN] {fpath} not found, skipped")
            continue
        with open(fpath) as f:
            data = json.load(f)
        print(f"  loaded {fpath.name}: {len(data)} trajectories")
        all_data.extend(data)
    return all_data


def parse_node(node_dict: dict) -> Optional[dict]:
    if not isinstance(node_dict, dict):
        return None
    return {
        "Q": node_dict.get("Q"),
        "reward": node_dict.get("reward"),
        "step": node_dict.get("step"),
        "N": node_dict.get("N"),
        "parent_id": node_dict.get("parent_id"),
        "children_ids": node_dict.get("children_ids", []),
        "action_name": node_dict.get("action_name"),
        "query": node_dict.get("query", "") or "",
        "response": node_dict.get("response", "") or "",
    }


def extract_tree(output: dict) -> Dict[int, dict]:
    nodes = {}
    for key, value in output.items():
        if not key.startswith("intermediate_node_"):
            continue
        try:
            node_id = int(key.split("_")[-1])
        except ValueError:
            continue
        parsed = parse_node(value)
        if parsed is not None:
            parsed["node_id"] = node_id
            nodes[node_id] = parsed
    return nodes


def get_path_to_node(nodes: Dict[int, dict], node_id: int) -> List[dict]:
    """Return root->node path (inclusive)."""
    path = []
    seen = set()
    current = node_id
    while current in nodes and current not in seen:
        seen.add(current)
        path.append(nodes[current])
        current = nodes[current].get("parent_id", -1)
    return list(reversed(path))


def collect_thoughts(path_nodes: List[dict]) -> List[str]:
    """Recreate node.thoughts as ReasonRAG does:
    每个 node 的 response 串联起来. 根节点 step=0,通常 response 为空."""
    thoughts = []
    for n in path_nodes:
        resp = n.get("response", "") or ""
        if resp.strip():
            thoughts.append(resp.strip())
    return thoughts


def find_branch_points(nodes: Dict[int, dict]) -> List[dict]:
    """parent 至少有 2 个 valid children."""
    bps = []
    for nid, node in nodes.items():
        cids = node.get("children_ids", [])
        children = [nodes[c] for c in cids if c in nodes]
        if len(children) < 2:
            continue
        child_qs = [c["Q"] for c in children if c.get("Q") is not None]
        q_all_same = (len(set(child_qs)) == 1) if len(child_qs) >= 2 else False
        bps.append(
            {
                "parent_node_id": nid,
                "parent_Q": node.get("Q"),
                "children": children,
                "q_all_same": q_all_same,
                "child_Qs": child_qs,
            }
        )
    return bps


# ----------------------------------------------------------------------------
# 3) 无偏采样
# ----------------------------------------------------------------------------
def unbiased_sample(
    all_data: List[dict], n_traj: int, seed: int
) -> List[Tuple[int, dict]]:
    """随机抽 n_traj 条 *至少含 1 个 branch point* 的 trajectory.
    返回 [(global_idx, traj), ...]."""
    rng = random.Random(seed)
    candidates = []
    for idx, traj in enumerate(all_data):
        out = traj.get("output", {})
        nodes = extract_tree(out)
        if not nodes:
            continue
        bps = find_branch_points(nodes)
        if not bps:
            continue
        candidates.append((idx, traj))
    print(f"  candidates with >=1 branch point: {len(candidates)}")
    if len(candidates) < n_traj:
        print(f"[WARN] only {len(candidates)} candidates, returning all")
        return candidates
    return rng.sample(candidates, n_traj)


# ----------------------------------------------------------------------------
# 4) 构造 EVALUATION_PROMPT 的输入 (与 reasonrag_pipeline.evaluate_thoughts 严格一致)
# ----------------------------------------------------------------------------
def build_eval_user_prompt(
    question: str, golden_answers: List[str], thoughts: List[str]
) -> str:
    """Reproduces:
        question_thoughts = node.question
            + "\nGolden Answer: " + " or ".join(node.golden_answers)
            + "\nAgent Reasoning Process: " + " ".join(node.thoughts)
        user_prompt = "Question: {question_thoughts}"
    """
    qt = (
        question
        + "\nGolden Answer: "
        + " or ".join(golden_answers or [])
        + "\nAgent Reasoning Process: "
        + " ".join(thoughts or [])
    )
    return f"Question: {qt}"


def extract_last_number(s: str) -> Optional[float]:
    matches = re.findall(r"[+-]?\d+", s)
    if matches:
        return float(matches[-1])
    return None


# ----------------------------------------------------------------------------
# 5) GPT-4o 调用
# ----------------------------------------------------------------------------
def make_client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Need `pip install openai>=1.0`")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY (DMXAPI key works too)")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://www.dmxapi.cn/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_gpt4o(
    client, model: str, system_prompt: str, user_prompt: str, max_retries: int = 3
) -> Tuple[Optional[float], str, dict]:
    """Returns (Q_in_[0,1], raw_response, usage_dict)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            text = resp.choices[0].message.content or ""
            num = extract_last_number(text)
            if num is None:
                last_err = "no number in response"
                time.sleep(1.5 * (attempt + 1))
                continue
            q = max(0.0, min(1.0, num / 100.0))
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
            }
            return q, text, usage
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(2.0 * (attempt + 1))
    return None, f"[FAILED after {max_retries}] {last_err}", {}


# ----------------------------------------------------------------------------
# 6) 主流程
# ----------------------------------------------------------------------------
def collect_eval_jobs(
    sampled: List[Tuple[int, dict]],
) -> List[dict]:
    """把 sampled trajectories 展平成 child-level evaluation jobs."""
    jobs = []
    for traj_idx, traj in sampled:
        question = traj.get("question", "")
        golden = traj.get("golden_answers", [])
        nodes = extract_tree(traj.get("output", {}))
        bps = find_branch_points(nodes)
        for bp_local_idx, bp in enumerate(bps):
            for child in bp["children"]:
                path = get_path_to_node(nodes, child["node_id"])
                thoughts = collect_thoughts(path)
                jobs.append(
                    {
                        "traj_idx": traj_idx,
                        "bp_local_idx": bp_local_idx,
                        "parent_node_id": bp["parent_node_id"],
                        "parent_Q_llama": bp["parent_Q"],
                        "child_node_id": child["node_id"],
                        "child_step": child.get("step"),
                        "child_action_name": child.get("action_name"),
                        "child_Q_llama": child.get("Q"),
                        "child_query": child.get("query", ""),
                        "child_response_len": len(child.get("response", "")),
                        "q_all_same_llama": bp["q_all_same"],
                        "n_siblings": len(bp["children"]),
                        "question": question,
                        "golden_answers": golden,
                        "user_prompt": build_eval_user_prompt(
                            question, golden, thoughts
                        ),
                    }
                )
    return jobs


def aggregate_stats(records: List[dict]) -> dict:
    """根据 child-level Q_new 重算 q_same_rate."""
    by_bp: Dict[Tuple[int, int], List[dict]] = {}
    for r in records:
        if r.get("Q_gpt4o") is None:
            continue
        key = (r["traj_idx"], r["bp_local_idx"])
        by_bp.setdefault(key, []).append(r)

    n_bp = 0
    n_q_same_llama = 0
    n_q_same_gpt4o = 0
    n_q_same_both = 0
    n_q_same_llama_diff_gpt4o = 0
    n_q_diff_llama_same_gpt4o = 0

    for key, rows in by_bp.items():
        if len(rows) < 2:
            continue
        n_bp += 1
        q_llama = {r["child_Q_llama"] for r in rows}
        q_gpt4o = {round(r["Q_gpt4o"], 4) for r in rows}
        same_llama = len(q_llama) == 1
        same_gpt4o = len(q_gpt4o) == 1
        n_q_same_llama += int(same_llama)
        n_q_same_gpt4o += int(same_gpt4o)
        if same_llama and same_gpt4o:
            n_q_same_both += 1
        if same_llama and not same_gpt4o:
            n_q_same_llama_diff_gpt4o += 1
        if not same_llama and same_gpt4o:
            n_q_diff_llama_same_gpt4o += 1

    def rate(x):
        return round(x / n_bp, 4) if n_bp else None

    return {
        "n_branch_points_with_complete_eval": n_bp,
        "q_same_rate_llama": rate(n_q_same_llama),
        "q_same_rate_gpt4o": rate(n_q_same_gpt4o),
        "q_same_both": n_q_same_both,
        "q_same_llama_diff_gpt4o": n_q_same_llama_diff_gpt4o,
        "q_diff_llama_same_gpt4o": n_q_diff_llama_same_gpt4o,
        "n_records_total": len(records),
        "n_records_with_q": sum(1 for r in records if r.get("Q_gpt4o") is not None),
        "n_records_failed": sum(1 for r in records if r.get("Q_gpt4o") is None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-data-dir", type=Path, default=DEFAULT_REWARD_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-traj", type=int, default=50)
    ap.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="与 parse_trees.py(seed=42) 区分,做独立无偏采样",
    )
    ap.add_argument("--model", type=str, default="gpt-4o")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.out_dir / "relabel_q_gpt4o_branches.jsonl"
    out_stats = args.out_dir / "relabel_q_gpt4o_stats.json"
    out_progress = args.out_dir / "relabel_q_gpt4o_progress.jsonl"

    print("=" * 60)
    print("Gate 0 L1: GPT-4o re-labeling on UNBIASED random sample")
    print("=" * 60)

    print("\n[1/4] Loading reward_data ...")
    all_data = load_reward_data(args.reward_data_dir)
    print(f"  total trajectories: {len(all_data)}")

    print(f"\n[2/4] Sampling {args.n_traj} trajectories (seed={args.seed}) ...")
    sampled = unbiased_sample(all_data, args.n_traj, args.seed)
    print(f"  sampled: {len(sampled)} trajectories")

    print("\n[3/4] Building evaluation jobs ...")
    jobs = collect_eval_jobs(sampled)
    print(f"  total child-level jobs: {len(jobs)}")
    n_bps = len({(j["traj_idx"], j["bp_local_idx"]) for j in jobs})
    print(f"  total branch points:    {n_bps}")
    avg_prompt_len = sum(len(j["user_prompt"]) for j in jobs) / max(len(jobs), 1)
    print(f"  avg user_prompt length: {avg_prompt_len:.0f} chars")
    est_tokens = len(jobs) * (avg_prompt_len / 4 + 100)
    print(
        f"  rough cost estimate:  ~{est_tokens/1e6:.2f}M tokens "
        f"(~¥{est_tokens/1e6 * 25:.0f} via DMXAPI gpt-4o)"
    )

    if args.dry_run:
        print("\n[dry-run] exiting before API calls.")
        return

    # 4) 调 API,断点续跑
    done_keys = set()
    if args.resume and out_progress.exists():
        with open(out_progress) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_keys.add((rec["traj_idx"], rec["child_node_id"]))
                except Exception:  # noqa: BLE001
                    continue
        print(f"  resume: {len(done_keys)} child evaluations already done")

    pending = [
        j for j in jobs if (j["traj_idx"], j["child_node_id"]) not in done_keys
    ]
    print(f"\n[4/4] Calling GPT-4o on {len(pending)} pending jobs ...")
    if not pending:
        print("  nothing to do.")
    else:
        client = make_client()

        def worker(job):
            q, raw, usage = call_gpt4o(
                client,
                args.model,
                EVALUATION_PROMPT,
                job["user_prompt"],
                max_retries=args.max_retries,
            )
            return {
                **{k: v for k, v in job.items() if k != "user_prompt"},
                "Q_gpt4o": q,
                "raw_response": raw[:500],
                "usage": usage,
            }

        with open(out_progress, "a") as fp:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                fut_to_job = {ex.submit(worker, j): j for j in pending}
                for i, fut in enumerate(as_completed(fut_to_job), 1):
                    rec = fut.result()
                    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fp.flush()
                    if i % 20 == 0 or i == len(pending):
                        print(f"  [{i}/{len(pending)}] done  Q_new={rec['Q_gpt4o']}")

    # 5) 汇总
    all_records = []
    if out_progress.exists():
        with open(out_progress) as f:
            for line in f:
                try:
                    all_records.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue

    with open(out_jsonl, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = aggregate_stats(all_records)
    stats["sample"] = {
        "n_trajectories_sampled": len(sampled),
        "n_branch_points": n_bps,
        "n_children": len(jobs),
        "seed": args.seed,
        "model": args.model,
    }
    with open(out_stats, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("RESULT (unbiased random sample)")
    print("=" * 60)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nbranches:  {out_jsonl}")
    print(f"stats:     {out_stats}")


if __name__ == "__main__":
    main()
