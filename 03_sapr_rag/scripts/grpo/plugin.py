"""ms-swift GRPO external plugin for SAPR-RAG.

承载两类注册（ms-swift --external_plugins 约定指向单文件）：
  1. multi_turns['sapr_rag_scheduler'] = SaprRagScheduler  —— 多轮 rollout 调度
  2. orms['sapr_f1' / 'sapr_relevance' / 'sapr_format']     —— 三个 reward

启动：
  swift rollout --multi_turn_scheduler sapr_rag_scheduler --external_plugins plugin.py ...
  swift rlhf   --reward_funcs sapr_f1 sapr_relevance sapr_format --external_plugins plugin.py ...

设计见同目录 IMPLEMENTATION.md。当前采用方案 B：检索文档作为下一轮 user observation 注入，
对齐 ms-swift VisualToolBoxScheduler 的多轮工具返回协议。
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
# §2 多轮调度器（方案 B：检索结果作为下一轮 user observation）
# ═══════════════════════════════════════════════════════════════════
class SaprRagScheduler(MultiTurnScheduler):
    """每个 turn 模型只生成 reasoning 段（<query>/<answer> 收尾）。
    step() 解析 <query> → 检索 → 把 docs 作为下一轮 user observation 注入。
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
        return (
            f"Reference: <reference>{ref}</reference>\n"
            "Use the reference to continue answering the original question. "
            "Do not repeat any previous query because it will return the same documents. "
            "If more evidence is needed, query a different entity, relation, or missing fact. "
            "If the answer is supported, conclude with: "
            "\"So the answer is <answer>answer</answer>\". "
            "If more retrieval is needed, conclude with: "
            "\"So the next query is <query>query</query>\"."
        )

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
            # 参考 ms-swift VisualToolBoxScheduler：工具/环境返回作为下一轮 user message。
            # 这样 reference 不再混入 assistant completion，也不需要 response_loss_mask=0。
            infer_request.messages.append({"role": "user", "content": obs})
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


def _support_sentence_variants(value):
    variants = []
    for item in _as_list(value):
        if isinstance(item, (list, tuple)):
            variants.extend(_support_sentence_variants(item))
        else:
            variants.extend(part.strip() for part in str(item).splitlines() if part.strip())
    return variants


def _extract_steps(info):
    if not isinstance(info, dict):
        return []
    steps = info.get("retrieved_steps", info.get("steps", [])) or []
    return steps if isinstance(steps, list) else []


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
    """检索相关性：首次覆盖的 unique gold evidence 比例。值域 [0,1]。"""

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

        rewards = []
        for i in range(len(completions)):
            info = rollout_infos[i] if rollout_infos else {}
            steps = _extract_steps(info)
            docs = self._collect_docs(steps)
            rtitles = set(norm_title(t) for t, _ in docs)
            rtexts = [norm_text(tx) for _, tx in docs]

            gtitles = _as_list(gold_titles[i]) if gold_titles else []
            gsents = _as_list(gold_sup_sents[i]) if gold_sup_sents else []

            num_gold = len(gtitles)
            if num_gold == 0:
                rewards.append(0.0)  # 兜底（已在 build_grpo_dataset 预过滤）
                continue

            # 每个 gold title 只计一次。supporting sentence 只作为 title 的
            # 对齐补充，不再用 gold answer 文本代替缺失 evidence。
            hits = 0
            for j, gt in enumerate(gtitles):
                hit = norm_title(gt) in rtitles
                if not hit and j < len(gsents):
                    normalized_sentences = [
                        norm_text(sentence)
                        for sentence in _support_sentence_variants(gsents[j])
                    ]
                    hit = any(
                        sentence and any(sentence in text for text in rtexts)
                        for sentence in normalized_sentences
                    )
                hits += 1 if hit else 0

            rewards.append(hits / num_gold)
        return rewards


orms["sapr_relevance"] = SaprRelevanceORM


class SaprTurnCostORM(ORM):
    """第一轮检索免费，之后每轮返回一个负奖励单位。"""

    def __call__(self, completions, **kwargs) -> List[float]:
        rollout_infos = kwargs.get("rollout_infos")
        rewards = []
        for i in range(len(completions)):
            info = rollout_infos[i] if rollout_infos else {}
            query_count = len(_extract_steps(info))
            rewards.append(float(-max(0, query_count - 1)))
        return rewards


orms["sapr_turn_cost"] = SaprTurnCostORM


class SaprRepeatQueryORM(ORM):
    """惩罚规范化后完全重复的 query，默认最多扣三个单位。"""

    @staticmethod
    def _normalize_query(query):
        query = re.sub(r"[^\w]+", " ", str(query).lower(), flags=re.UNICODE)
        return re.sub(r"\s+", " ", query).strip()

    def __call__(self, completions, **kwargs) -> List[float]:
        rollout_infos = kwargs.get("rollout_infos")
        cap = max(0, int(os.environ.get("SAPR_REPEAT_QUERY_CAP", "3")))
        rewards = []
        for i in range(len(completions)):
            info = rollout_infos[i] if rollout_infos else {}
            queries = [
                self._normalize_query(step.get("query", ""))
                for step in _extract_steps(info)
            ]
            queries = [query for query in queries if query]
            repeat_count = len(queries) - len(set(queries))
            rewards.append(float(-min(repeat_count, cap)))
        return rewards


orms["sapr_repeat_query"] = SaprRepeatQueryORM


class SaprMaxTurnORM(ORM):
    """惩罚耗尽 query 预算后仍没有有效 answer 的轨迹。"""

    def __call__(self, completions, **kwargs) -> List[float]:
        rollout_infos = kwargs.get("rollout_infos")
        max_turns = max(1, int(os.environ.get("SAPR_MAX_TURNS", "6")))
        rewards = []
        for i, completion in enumerate(completions):
            info = rollout_infos[i] if rollout_infos else {}
            num_turns = info.get("num_turns") if isinstance(info, dict) else None
            if num_turns is not None:
                exhausted = int(num_turns) >= max_turns
            else:
                # Compatibility fallback: the final assistant turn stops before
                # step(), so at most max_turns - 1 retrievals are recorded.
                exhausted = len(_extract_steps(info)) >= max(0, max_turns - 1)
            answered = bool(parse_final_answer(completion))
            rewards.append(-1.0 if exhausted and not answered else 0.0)
        return rewards


orms["sapr_max_turn"] = SaprMaxTurnORM


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
