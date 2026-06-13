"""ms-swift GRPO external plugin for SAPR-RAG.

承载两类注册（ms-swift --external_plugins 约定指向单文件）：
  1. multi_turns['sapr_rag_scheduler'] = SaprRagScheduler  —— 多轮 rollout 调度
  2. orms['sapr_f1' / 'sapr_relevance' / 'sapr_format']     —— 三个 reward

启动：
  swift rollout --multi_turn_scheduler sapr_rag_scheduler --external_plugins plugin.py ...
  swift rlhf   --reward_funcs sapr_f1 sapr_relevance sapr_format --external_plugins plugin.py ...

设计见同目录 IMPLEMENTATION.md。方案 A：只训 reason，检索文档作 observation 以 loss_mask=0 注入。
"""
import os
import re
import string
import sys
from collections import Counter
from typing import Dict, List

from swift.rewards import ORM, orms
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns

# RetrievalClient 与本文件同目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_client import RetrievalClient

# ─────────── 协议正则 / prompt（内联复制 agent_infer，避免 import faiss/torch 重依赖）───────────
RE_QUERY = re.compile(r"<query>(.*?)</query>", re.DOTALL)
RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

RETRIEVAL_DAEMON_URL = os.environ.get("SAPR_RETRIEVAL_URL", "http://127.0.0.1:8100")


# ═══════════════════════════════════════════════════════════════════
# §1 评分原语（内联复制 score.py / retrieval_recall.py，纯标准库）
# ═══════════════════════════════════════════════════════════════════
def normalize_answer(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1_score(pred, golds):
    p_toks = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        g_toks = normalize_answer(g).split()
        if not p_toks or not g_toks:
            best = max(best, float(p_toks == g_toks))
            continue
        common = Counter(p_toks) & Counter(g_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(p_toks)
        recall = num_same / len(g_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def norm_title(s: str) -> str:
    s = (s or "").lower().strip()
    return re.sub(r"\s+", " ", s)


def norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_final_answer(text):
    m = RE_ANSWER.search(text or "")
    return m.group(1).strip() if m else ""


# ═══════════════════════════════════════════════════════════════════
# §2 多轮调度器（方案 A：只训 reason）
# ═══════════════════════════════════════════════════════════════════
class SaprRagScheduler(MultiTurnScheduler):
    """每个 turn 模型只生成 reasoning 段（<query>/<answer> 收尾）。
    step() 解析 <query> → 检索 → 把 docs 作为 observation 以 loss_mask=0 注入下一轮。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = RetrievalClient(base_url=RETRIEVAL_DAEMON_URL)
        self.client.wait_until_ready()
        self.top_k = int(os.environ.get("SAPR_TOP_K", "3"))
        self._traj: Dict[str, List[Dict]] = {}  # uuid -> [{turn, query, docs}]

    # 出现 <answer> 即停；否则交母类判 max_turns / length
    def check_finished(self, infer_request, response_choice, current_turn) -> bool:
        if RE_ANSWER.search(response_choice.message.content or ""):
            return True
        return super().check_finished(infer_request, response_choice, current_turn)

    def _format_observation(self, docs) -> str:
        # 与 agent_infer.build_evidence_prompt 同款；doc.text 已在 daemon 截 [:500]
        ref = " ".join(f"{d.get('title','')}. {d.get('text','')}" for d in docs)
        return f" Reference: <reference>{ref}</reference>"

    def step(self, infer_request, response_choice, current_turn) -> Dict:
        text = response_choice.message.content or ""
        token_ids = list(response_choice.token_ids)
        loss_mask = [1] * len(token_ids)
        uuid = infer_request.uuid or "default"
        steps = self._traj.setdefault(uuid, [])

        m = RE_QUERY.search(text)
        if m:
            query = m.group(1).strip()
            try:
                docs = self.client.search(query, top_k=self.top_k)
            except Exception:
                docs = []
            obs = self._format_observation(docs)
            # 注入到对话历史，供下一轮 reason 看到
            infer_request.messages[-1]["content"] += obs
            # 注入 token，loss_mask=0（环境部分不参与训练）
            result_tokens = self.tokenizer.encode(obs, add_special_tokens=False)
            token_ids.extend(result_tokens)
            loss_mask.extend([0] * len(result_tokens))
            steps.append({"turn": current_turn, "query": query, "docs": docs})

        # 覆盖语义：rollout_infos 同名 key 覆盖不追加，每次写完整列表
        return {
            "infer_request": infer_request,
            "response_token_ids": token_ids,
            "response_loss_mask": loss_mask,
            "rollout_infos": {"retrieved_steps": list(steps), "uuid": uuid},
        }


multi_turns["sapr_rag_scheduler"] = SaprRagScheduler


# ═══════════════════════════════════════════════════════════════════
# §3 Reward 函数（三个 ORM）
# ═══════════════════════════════════════════════════════════════════
def _as_list(x):
    """dataset 列经 rows_to_batched 后可能是 list 或标量，统一成 list[str]。"""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


class SaprF1ORM(ORM):
    """主信号：末轮 answer 对 golden_answers 的 token 级 F1。值域 [0,1]。"""

    def __call__(self, completions, **kwargs) -> List[float]:
        golden_answers = kwargs.get("golden_answers")
        rewards = []
        for i, comp in enumerate(completions):
            golds = _as_list(golden_answers[i]) if golden_answers else []
            pred = parse_final_answer(comp)
            rewards.append(f1_score(pred, golds) if golds else 0.0)
        return rewards


orms["sapr_f1"] = SaprF1ORM


class SaprRelevanceORM(ORM):
    """检索相关性：检索 doc 对 gold supporting 的三级 OR 连续命中比例。值域 [0,1]。"""

    def _collect_docs(self, steps):
        """跨 turn 去重收集 (title, text)。"""
        docs, seen = [], set()
        for st in steps or []:
            for d in st.get("docs", []) or []:
                title = d.get("title", "")
                text = d.get("text", "")
                key = (title, text[:80])
                if key in seen:
                    continue
                seen.add(key)
                docs.append((title, text))
        return docs

    def __call__(self, completions, **kwargs) -> List[float]:
        rollout_infos = kwargs.get("rollout_infos")
        gold_titles = kwargs.get("gold_titles")
        gold_sup_sents = kwargs.get("gold_sup_sents")
        golden_answers = kwargs.get("golden_answers")

        rewards = []
        for i in range(len(completions)):
            info = rollout_infos[i] if rollout_infos else {}
            steps = (info or {}).get("retrieved_steps", []) if isinstance(info, dict) else []
            docs = self._collect_docs(steps)
            rtitles = set(norm_title(t) for t, _ in docs)
            rtexts = [norm_text(tx) for _, tx in docs]

            gtitles = _as_list(gold_titles[i]) if gold_titles else []
            gsents = _as_list(gold_sup_sents[i]) if gold_sup_sents else []
            ganswers = _as_list(golden_answers[i]) if golden_answers else []

            num_gold = len(gtitles)
            if num_gold == 0:
                rewards.append(0.0)  # 兜底（已在 build_grpo_dataset 预过滤）
                continue

            # 逐 gold 三级 OR 命中
            hits = 0
            for j, gt in enumerate(gtitles):
                hit = norm_title(gt) in rtitles
                if not hit and j < len(gsents):
                    gsn = norm_text(gsents[j])
                    hit = bool(gsn) and any(gsn in tx for tx in rtexts)
                if not hit:
                    # 第三级：gold answer 文本出现在任一 doc 正文
                    for ga in ganswers:
                        gan = norm_text(ga)
                        if gan and any(gan in tx for tx in rtexts):
                            hit = True
                            break
                hits += 1 if hit else 0

            rewards.append(hits / num_gold)
        return rewards


orms["sapr_relevance"] = SaprRelevanceORM


class SaprFormatORM(ORM):
    """格式：允许前序多轮 <query>，但最终协议标签必须是非空 <answer>。"""

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for comp in completions:
            comp = comp or ""
            events = []
            events.extend((m.start(), "query", m.group(1).strip()) for m in RE_QUERY.finditer(comp))
            events.extend((m.start(), "answer", m.group(1).strip()) for m in RE_ANSWER.finditer(comp))
            events.sort(key=lambda x: x[0])

            if not events:
                rewards.append(0.0)
                continue

            _, last_kind, last_text = events[-1]
            rewards.append(1.0 if last_kind == "answer" and bool(last_text) else 0.0)
        return rewards


orms["sapr_format"] = SaprFormatORM
