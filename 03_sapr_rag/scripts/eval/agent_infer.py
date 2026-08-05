"""SAPR-RAG agentic 推理脚本。

vllm 加载 Qwen2.5-7B-Instruct + LoRA SFT adapter
BGE-base-en-v1.5 编码 query → FAISS IP 检索 wiki18_extended → 取 top-3 passage
两个角色循环：
  reasoning agent: 出 <query>...</query> 或 <answer>...</answer>
  evidence agent : 抽 <evidence>...</evidence>
最多 6 轮，超时强制收尾。

用法（单条交互）：
  python agent_infer.py --question "Are director of Move (1970) and Méditerranée (1963) from the same country?"
用法（批量）：
  python agent_infer.py --input_jsonl questions.jsonl --output_jsonl results.jsonl
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import faiss
import numpy as np
import requests
import torch
from transformers import AutoModel, AutoTokenizer


# ─────────── 路径 ───────────
PROJ_ROOT = Path(os.environ.get("SAPR_RAG_ROOT", Path(__file__).resolve().parents[3]))

BASE_MODEL = PROJ_ROOT / "03_sapr_rag/models/Qwen2.5-7B-Instruct"
LORA_ADAPTER = PROJ_ROOT / "03_sapr_rag/saves/qwen2_5_7b/lora/sft/checkpoint-1650"

BGE_PATH = PROJ_ROOT / "models/bge-base-en-v1.5"
INDEX_PATH = PROJ_ROOT / "data/index/bge_extended_Flat.index"
CORPUS_PATH = PROJ_ROOT / "data/corpus/wiki18_extended.jsonl"

# ─────────── 双角色 system prompt（与 build_sft.py 一致）───────────
REASONING_SYSTEM = (
    "You are an assistant for question answering with access to a retrieval tool. "
    "Upon receiving a question, your task is to:\n"
    "* Analyze and Decompose the Question: Break the question into smaller, manageable "
    "sub-questions to ensure all aspects are addressed.\n"
    "* Evaluate Your Knowledge: Assess each sub-question or component:\n"
    "- Identify parts you can confidently answer based on your existing knowledge.\n"
    "- Pinpoint parts that require additional information or verification through retrieval tools.\n"
    "* Conciseness: Ensure both queries and answers are concise, using nouns or short "
    "phrases whenever possible.\n"
    "* Respond Format:\n"
    "If your knowledge is sufficient to answer the question, conclude with:\n"
    '"So the answer is <answer>answer</answer>"\n'
    "If retrieval is necessary to provide a complete answer, conclude with:\n"
    '"So the next query is <query>query</query>"\n'
)

EVIDENCE_SYSTEM = (
    "You are an information retrieval assistant. Given a query and a reference document, "
    "extract a concise piece of evidence that directly answers the query. "
    'If no relevant evidence is found, output <evidence>None</evidence>. '
    "Otherwise, output the evidence in the format: "
    "Based on the query, the relevant evidence is <evidence>evidence_text</evidence>."
)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ─────────── retriever ───────────
class BGEFaissRetriever:
    def __init__(self, bge_path, index_path, corpus_path, device="cuda:0"):
        print(f"[retriever] loading BGE on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(bge_path)
        self.model = AutoModel.from_pretrained(bge_path).to(device).eval()
        self.device = device

        # mmap 模式：8 个 DP 进程通过 OS page cache 共享同一份 64GB 物理 RAM
        print(f"[retriever] loading FAISS index ({index_path}, mmap) ...")
        t0 = time.time()
        self.index = faiss.read_index(
            str(index_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
        )
        print(f"[retriever] index loaded in {time.time()-t0:.1f}s, "
              f"n_vectors={self.index.ntotal}, dim={self.index.d}")

        # FlashRAG 风格：HF datasets 把 jsonl 转 Arrow 缓存，O(1) 随机访问
        # 首次会扫一遍 jsonl 建 cache（~5-10 min for 14GB），后续进程共享 cache
        print(f"[retriever] loading corpus via HF datasets ({corpus_path}) ...")
        t0 = time.time()
        import datasets
        self.corpus = datasets.load_dataset(
            "json", data_files=str(corpus_path), split="train",
        )
        print(f"[retriever] corpus loaded in {time.time()-t0:.1f}s, "
              f"n_docs={len(self.corpus)}")

    @torch.no_grad()
    def encode(self, queries):
        prefixed = [BGE_QUERY_PREFIX + q for q in queries]
        enc = self.tokenizer(prefixed, padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(self.device)
        out = self.model(**enc)
        emb = torch.nn.functional.normalize(out.last_hidden_state[:, 0], p=2, dim=1)
        return np.ascontiguousarray(emb.cpu().numpy().astype("float32"))

    def search(self, query, top_k=3):
        return self.search_batch([query], top_k=top_k)[0]

    def search_batch(self, queries, top_k=3):
        """queries -> List[List[doc]]，一次 BGE 前向 + 一次 FAISS 2D search。"""
        if not queries:
            return []
        embs = self.encode(queries)
        scores, doc_ids = self.index.search(embs, top_k)
        results = []
        for row_ids, row_scores in zip(doc_ids, scores):
            row_ids = row_ids.tolist()
            docs = self._fetch(row_ids)
            results.append(
                [{"title": docs[d]["title"], "text": docs[d]["text"],
                  "score": float(s)}
                 for d, s in zip(row_ids, row_scores) if d in docs]
            )
        return results

    def _fetch(self, doc_ids):
        docs = {}
        for d in doc_ids:
            if d < 0:
                continue
            item = self.corpus[int(d)]
            raw = item.get("contents", "") or ""
            parts = raw.split("\n", 1)
            first = parts[0].strip().strip('"')
            text = (parts[1] if len(parts) > 1 else "").strip()[:500]
            docs[d] = {"title": first, "text": text}
        return docs


class HTTPRetriever:
    """Adapter for the persistent retrieval daemon used by GRPO rollout."""

    def __init__(self, base_url, timeout=60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        r = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        print(f"[retriever=http] health={r.json()}")

    def search(self, query, top_k=3):
        return self.search_batch([query], top_k=top_k)[0]

    def search_batch(self, queries, top_k=3):
        if not queries:
            return []
        r = self.session.post(
            f"{self.base_url}/search_batch",
            json={"queries": queries, "top_k": top_k},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["results"]


# ─────────── prompt 拼装 ───────────
def render_history(history):
    """history = [{query, evidence}, ...]"""
    if not history:
        return ""
    parts = []
    for h in history:
        parts.append(
            f"So the next query is <query>{h['query']}</query> "
            f"Based on the query, the relevant evidence is <evidence>{h['evidence']}</evidence>."
        )
    return "\n\n".join(parts)


def build_reasoning_prompt(question, history):
    instruction = f"Question: {question}"
    if history:
        instruction += "\nPrevious Thoughts: " + render_history(history)
    return REASONING_SYSTEM, instruction


def build_evidence_prompt(query, docs):
    reference = " ".join(f"{d['title']}. {d['text']}" for d in docs)
    instruction = (
        f"Question: {query}. Reference: <reference>{reference}</reference>"
    )
    return EVIDENCE_SYSTEM, instruction


# ─────────── 解析 ───────────
RE_QUERY = re.compile(r"<query>(.*?)</query>", re.DOTALL)
RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
RE_EVIDENCE = re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL)


def parse_action(text):
    """从 reasoning 输出里抓 query / answer。"""
    m = RE_ANSWER.search(text)
    if m:
        return {"type": "answer", "value": m.group(1).strip(), "raw": text}
    m = RE_QUERY.search(text)
    if m:
        return {"type": "query", "value": m.group(1).strip(), "raw": text}
    return {"type": "unknown", "value": text.strip(), "raw": text}


def parse_evidence(text):
    m = RE_EVIDENCE.search(text)
    return m.group(1).strip() if m else "None"


# ─────────── 推理 backend ───────────
class VLLMBackend:
    def __init__(self, base_model, lora_path, max_model_len, gpu_memory_utilization,
                 max_lora_rank=16):
        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams
        # lora_path=None -> zero-shot，纯 backbone，不挂 LoRA
        self.use_lora = lora_path is not None
        self.llm = LLM(
            model=str(base_model),
            enable_lora=self.use_lora,
            max_lora_rank=max_lora_rank,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="float16",
            enable_prefix_caching=True,
            enforce_eager=True,
            max_num_batched_tokens=2048,
        )
        if self.use_lora:
            from vllm.lora.request import LoRARequest
            self.lora_request = LoRARequest("sapr_sft", 1, str(lora_path))
        else:
            self.lora_request = None

    def chat(self, system, user, sampling, stop, max_tokens=None):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        sp = self._SamplingParams(
            max_tokens=max_tokens or sampling["max_tokens"],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            stop=stop,
        )
        out = self.llm.chat(messages, sampling_params=sp,
                            lora_request=self.lora_request,
                            use_tqdm=False)
        return out[0].outputs[0].text

    def chat_batch(self, convs, sampling, stop, max_tokens=None):
        """convs = [(system, user), ...] -> [text, ...]，一次性喂 vllm 走 continuous batching。"""
        if not convs:
            return []
        messages_list = [
            [{"role": "system", "content": s}, {"role": "user", "content": u}]
            for s, u in convs
        ]
        sp = self._SamplingParams(
            max_tokens=max_tokens or sampling["max_tokens"],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            stop=stop,
        )
        outs = self.llm.chat(messages_list, sampling_params=sp,
                             lora_request=self.lora_request,
                             use_tqdm=False)
        return [o.outputs[0].text for o in outs]


class RolloutHTTPBackend:
    """Call a running `swift rollout` server through /infer/."""

    def __init__(self, base_url, timeout=300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        r = self.session.get(f"{self.base_url}/health/", timeout=self.timeout)
        r.raise_for_status()
        print(f"[backend=rollout_http] health={r.json()}")

    def chat(self, system, user, sampling, stop, max_tokens=None):
        return self.chat_batch([(system, user)], sampling, stop, max_tokens=max_tokens)[0]

    def chat_batch(self, convs, sampling, stop, max_tokens=None):
        if not convs:
            return []
        infer_requests = [
            {"messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]}
            for system, user in convs
        ]
        payload = {
            "infer_requests": infer_requests,
            "request_config": {
                "max_tokens": max_tokens or sampling["max_tokens"],
                "temperature": sampling["temperature"],
                "top_p": sampling["top_p"],
                "stop": stop,
                "return_details": True,
            },
            "use_tqdm": False,
        }
        r = self.session.post(f"{self.base_url}/infer/", json=payload, timeout=self.timeout)
        r.raise_for_status()
        outputs = r.json()
        texts = []
        for out in outputs:
            response = out.get("response", {})
            choices = response.get("choices") or []
            text = ""
            if choices:
                text = ((choices[0].get("message") or {}).get("content") or "")
            # Multi-turn rollout may return the complete messages history; use final assistant if present.
            messages = out.get("messages") or []
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    text = msg.get("content") or text
                    break
            texts.append(text)
        return texts


class TransformersBackend:
    @staticmethod
    def _patch_qwen2_rope():
        """Avoid tiny RoPE matmul hitting CUBLAS_STATUS_INVALID_VALUE on some CUDA eval setups."""
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        except Exception:
            return

        if getattr(Qwen2RotaryEmbedding, "_sapr_rope_patched", False):
            return

        def forward(self, x, position_ids):
            inv_freq = self.inv_freq[None, :, None].float().to(x.device)
            pos = position_ids[:, None, :].float()
            device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq * pos).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

        Qwen2RotaryEmbedding.forward = forward
        Qwen2RotaryEmbedding._sapr_rope_patched = True

    def __init__(self, base_model, lora_path, device="cuda:0",
                 dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._patch_qwen2_rope()
        print(f"[backend=transformers] loading base {base_model} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(base_model))
        base = AutoModelForCausalLM.from_pretrained(
            str(base_model), torch_dtype=dtype, attn_implementation="eager",
        ).to(device).eval()
        if lora_path is not None:
            from peft import PeftModel
            print(f"[backend=transformers] attaching LoRA from {lora_path} ...")
            self.model = PeftModel.from_pretrained(base, str(lora_path)).eval()
        else:
            print("[backend=transformers] ZERO-SHOT (no LoRA)")
            self.model = base
        self.device = device

    @torch.no_grad()
    def chat(self, system, user, sampling, stop, max_tokens=None):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        seq_len = inputs["input_ids"].shape[1]
        pad_len = (-seq_len) % 8
        if pad_len:
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            pad_ids = torch.full((1, pad_len), pad_id, dtype=inputs["input_ids"].dtype, device=self.device)
            pad_mask = torch.zeros((1, pad_len), dtype=inputs["attention_mask"].dtype, device=self.device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad_ids], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], pad_mask], dim=1)
        gen_kwargs = dict(
            max_new_tokens=max_tokens or sampling["max_tokens"],
            do_sample=sampling["temperature"] > 0,
            temperature=max(sampling["temperature"], 1e-5),
            top_p=sampling["top_p"],
            pad_token_id=self.tokenizer.eos_token_id,
        )
        out_ids = self.model.generate(**inputs, **gen_kwargs)
        new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        # 软实现 stop：截到第一个 stop 字符串处（含 stop 本身）
        cut = len(text)
        for s in stop:
            i = text.find(s)
            if i >= 0:
                cut = min(cut, i + len(s))
        return text[:cut]

    def chat_batch(self, convs, sampling, stop, max_tokens=None):
        """transformers 路径退化为逐条循环，只为兼容性 sanity，不追吞吐。"""
        return [self.chat(s, u, sampling, stop, max_tokens=max_tokens)
                for s, u in convs]


# ─────────── agent loop ───────────
class SAPRAgent:
    def __init__(self, backend, sampling, retriever, max_turns=6,
                 evidence_max_tokens=128):
        self.backend = backend
        self.sampling = sampling
        self.retriever = retriever
        self.max_turns = max_turns
        self.evidence_max_tokens = evidence_max_tokens

    def _chat(self, system, user, stop, max_tokens=None):
        return self.backend.chat(system, user, self.sampling, stop,
                                 max_tokens=max_tokens)

    def run(self, question, top_k=3):
        return self.run_batch([question], top_k=top_k)[0]

    def run_batch(self, questions, top_k=3):
        """lockstep cohort 状态机：所有题按阶段同步推进，单阶段内 batched 喂 vllm。

        返回 List[result]，每个 result 结构与单条 run 完全一致
        （answer/history/trace[/error]）。
        """
        states = [{"question": q, "history": [], "trace": [],
                   "status": "running", "result": None,
                   "pending_query": None, "pending_docs": None}
                  for q in questions]

        for turn in range(self.max_turns):
            active = [s for s in states if s["status"] == "running"]
            if not active:
                break

            # ── 阶段 A：REASONING（batched）──
            convs = [build_reasoning_prompt(s["question"], s["history"])
                     for s in active]
            outs = self.backend.chat_batch(
                convs, self.sampling, stop=["</query>", "</answer>"])
            need_retrieve = []
            for s, text in zip(active, outs):
                # vllm 会吃掉 stop 字符串本身，补回去再 parse
                if "</query>" not in text and "<query>" in text:
                    text += "</query>"
                if "</answer>" not in text and "<answer>" in text:
                    text += "</answer>"
                action = parse_action(text)
                s["trace"].append({"turn": turn, "stage": "reason",
                                   "out": text, "parsed": action})
                if action["type"] == "answer":
                    s["result"] = {"answer": action["value"],
                                   "history": s["history"], "trace": s["trace"]}
                    s["status"] = "done"
                elif action["type"] == "query":
                    s["pending_query"] = action["value"]
                    need_retrieve.append(s)
                else:
                    s["result"] = {"answer": None, "history": s["history"],
                                   "trace": s["trace"],
                                   "error": "no_query_or_answer"}
                    s["status"] = "done"

            if not need_retrieve:
                continue

            # ── 阶段 B：RETRIEVE（batched encode + 2D FAISS）──
            queries = [s["pending_query"] for s in need_retrieve]
            docs_list = self.retriever.search_batch(queries, top_k=top_k)
            for s, docs in zip(need_retrieve, docs_list):
                s["pending_docs"] = docs
                s["trace"].append({"turn": turn, "stage": "retrieve",
                                   "query": s["pending_query"], "docs": docs})

            # ── 阶段 C：EVIDENCE（batched）──
            convs = [build_evidence_prompt(s["pending_query"], s["pending_docs"])
                     for s in need_retrieve]
            outs = self.backend.chat_batch(
                convs, self.sampling, stop=["</evidence>"],
                max_tokens=self.evidence_max_tokens)
            for s, ev_text in zip(need_retrieve, outs):
                if "</evidence>" not in ev_text and "<evidence>" in ev_text:
                    ev_text += "</evidence>"
                evidence = parse_evidence(ev_text)
                s["trace"].append({"turn": turn, "stage": "evidence",
                                   "out": ev_text, "parsed": evidence})
                s["history"].append({"query": s["pending_query"],
                                     "evidence": evidence})

        # 收尾：仍 running 的题 = 超出最大轮数
        for s in states:
            if s["status"] == "running":
                s["result"] = {"answer": None, "history": s["history"],
                               "trace": s["trace"],
                               "error": "max_turns_exceeded"}
                s["status"] = "done"

        return [s["result"] for s in states]


# ─────────── main ───────────
def _rewrite_clean(path):
    """去掉文件里无法解析的行（多为被 kill 截断的末尾半行），原地重写。"""
    good = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                json.loads(s)
            except json.JSONDecodeError:
                continue
            good.append(s)
    with open(path, "w") as f:
        for s in good:
            f.write(s + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--question", type=str, default=None)
    p.add_argument("--input_jsonl", type=str, default=None,
                   help="批量模式输入，每行 {question}")
    p.add_argument("--output_jsonl", type=str, default=None)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--max_turns", type=int, default=6)
    p.add_argument("--max_tokens", type=int, default=512,
                   help="reasoning agent 单轮生成上限")
    p.add_argument("--evidence_max_tokens", type=int, default=128,
                   help="evidence agent 单轮生成上限（只输出一句话，128 足够）")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--base_model", type=str, default=str(BASE_MODEL),
                   help="基础模型路径；可传入已 merge LoRA 的完整模型目录")
    p.add_argument("--lora_path", type=str, default=str(LORA_ADAPTER))
    p.add_argument("--retrieval_url", type=str, default=None,
                   help="复用常驻检索 daemon，例如 http://127.0.0.1:<port>；为空则本进程加载 BGE+FAISS")
    p.add_argument("--rollout_url", type=str, default=None,
                   help="复用 swift rollout 服务，例如 http://127.0.0.1:<port>")
    p.add_argument("--no_lora", action="store_true",
                   help="zero-shot：不挂 LoRA，纯 backbone 推理（4-setting 对照基准）")
    p.add_argument("--backend", type=str, default="vllm",
                   choices=["vllm", "transformers", "rollout_http"],
                   help="vllm: 快但环境苛刻; transformers: 慢但兼容性好")
    p.add_argument("--torch_dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="仅 transformers: 模型加载 dtype")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.5,
                   help="仅 vllm: 占多少显存，BGE encoder 还要 ~3G")
    p.add_argument("--max_model_len", type=int, default=8192,
                   help="仅 vllm: KV cache 上限；history 累积 + 长 evidence 时需要")
    p.add_argument("--shard_id", type=int, default=0,
                   help="DP 切片 id (0..num_shards-1)，本进程只跑 i%%num_shards==shard_id 的题")
    p.add_argument("--num_shards", type=int, default=1,
                   help="DP 切片总数")
    p.add_argument("--cohort_size", type=int, default=0,
                   help="批处理 cohort 大小；0=整 shard 作为一个 cohort（方案X）")
    p.add_argument("--resume", action="store_true",
                   help="断点续跑：读已有 output_jsonl 里完成的 id，跳过它们并以 append 追加")
    args = p.parse_args()

    lora_path = None if args.no_lora else args.lora_path
    print(f"[main] backend={args.backend}, "
          f"{'ZERO-SHOT (no LoRA)' if lora_path is None else 'lora='+str(lora_path)}")
    if args.backend == "vllm":
        backend = VLLMBackend(
            base_model=Path(args.base_model),
            lora_path=lora_path,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    elif args.backend == "rollout_http":
        if not args.rollout_url:
            raise ValueError("--backend rollout_http requires --rollout_url")
        backend = RolloutHTTPBackend(args.rollout_url)
    else:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        backend = TransformersBackend(
            base_model=Path(args.base_model),
            lora_path=lora_path,
            device="cuda:0",
            dtype=dtype_map[args.torch_dtype],
        )

    if args.retrieval_url:
        retriever = HTTPRetriever(args.retrieval_url)
    else:
        retriever = BGEFaissRetriever(
            BGE_PATH, INDEX_PATH, CORPUS_PATH, device="cuda:0",
        )

    agent = SAPRAgent(
        backend=backend,
        sampling={"max_tokens": args.max_tokens,
                  "temperature": args.temperature,
                  "top_p": args.top_p},
        retriever=retriever,
        max_turns=args.max_turns,
        evidence_max_tokens=args.evidence_max_tokens,
    )

    # 单条
    if args.question:
        res = agent.run(args.question, top_k=args.top_k)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    # 批量
    assert args.input_jsonl and args.output_jsonl, "批量需要 --input_jsonl --output_jsonl"
    with open(args.input_jsonl) as f:
        questions = [json.loads(l) for l in f]
    # DP 切片：每个进程只跑自己负责的 idx
    if args.num_shards > 1:
        questions = [(i, q) for i, q in enumerate(questions)
                     if i % args.num_shards == args.shard_id]
        print(f"[shard {args.shard_id}/{args.num_shards}] 负责 {len(questions)} 条")
    else:
        questions = list(enumerate(questions))

    # 断点续跑：读已完成 id，过滤掉；append 模式追加
    write_mode = "w"
    if args.resume and os.path.exists(args.output_jsonl):
        done_ids = set()
        with open(args.output_jsonl) as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # 被 kill 截断的末尾半行，跳过
                    continue
                if "id" in rec:
                    done_ids.add(rec["id"])
        before = len(questions)
        questions = [(i, q) for (i, q) in questions
                     if (q.get("id", i)) not in done_ids]
        write_mode = "a"
        print(f"[shard {args.shard_id}] resume: 已完成 {len(done_ids)}，"
              f"本 shard 剩余 {len(questions)}/{before}")
        # 末尾若有截断半行，重写干净（去掉无法解析的最后一行）
        _rewrite_clean(args.output_jsonl)

    with open(args.output_jsonl, write_mode) as fo:
        cohort_size = args.cohort_size if args.cohort_size > 0 else len(questions)
        n_done = 0
        t_start = time.time()
        for c0 in range(0, len(questions), cohort_size):
            cohort = questions[c0:c0 + cohort_size]
            qs = [q["question"] for _, q in cohort]
            t0 = time.time()
            try:
                results = agent.run_batch(qs, top_k=args.top_k)
            except Exception as e:
                # cohort 级容错：整批失败仍落盘，不丢已完成的其它 cohort
                results = [{"answer": None, "history": [], "trace": [],
                            "error": f"exception: {type(e).__name__}: {e}"}
                           for _ in cohort]
            dt = time.time() - t0
            for (orig_i, q), res in zip(cohort, results):
                res["id"] = q.get("id", orig_i)
                res["question"] = q["question"]
                res["gold"] = q.get("answer") or q.get("golden_answers")
                res["latency_s"] = round(dt / max(len(cohort), 1), 2)  # cohort 均摊
                fo.write(json.dumps(res, ensure_ascii=False) + "\n")
            fo.flush()
            n_done += len(cohort)
            tput = n_done / max(time.time() - t_start, 1e-6)
            print(f"[shard {args.shard_id}] cohort {c0}-{c0+len(cohort)-1} "
                  f"done in {dt:.1f}s | {n_done}/{len(questions)} "
                  f"| {tput:.2f} q/s", flush=True)


if __name__ == "__main__":
    main()
