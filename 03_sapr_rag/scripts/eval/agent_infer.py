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
import re
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


# ─────────── 路径 ───────────
PROJ_ROOT = Path("/mlx_devbox/users/mayi.summer/playground/SAPR-RAG")

BASE_MODEL = PROJ_ROOT / "03_sapr_rag/models/Qwen2.5-7B-Instruct"
LORA_ADAPTER = PROJ_ROOT / "03_sapr_rag/saves/qwen2_5_7b/lora/sft"

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

        print(f"[retriever] loading FAISS index ({index_path}) ...")
        t0 = time.time()
        self.index = faiss.read_index(str(index_path))
        print(f"[retriever] index loaded in {time.time()-t0:.1f}s, "
              f"n_vectors={self.index.ntotal}, dim={self.index.d}")

        self.corpus_path = corpus_path
        self._line_cache = {}  # doc_id -> {title, text}

    @torch.no_grad()
    def encode(self, queries):
        prefixed = [BGE_QUERY_PREFIX + q for q in queries]
        enc = self.tokenizer(prefixed, padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(self.device)
        out = self.model(**enc)
        emb = torch.nn.functional.normalize(out.last_hidden_state[:, 0], p=2, dim=1)
        return np.ascontiguousarray(emb.cpu().numpy().astype("float32"))

    def search(self, query, top_k=3):
        emb = self.encode([query])
        scores, doc_ids = self.index.search(emb, top_k)
        doc_ids = doc_ids[0].tolist()
        docs = self._fetch(doc_ids)
        return [{"title": docs[d]["title"], "text": docs[d]["text"], "score": float(s)}
                for d, s in zip(doc_ids, scores[0]) if d in docs]

    def _fetch(self, doc_ids):
        needed = [d for d in doc_ids if d >= 0 and d not in self._line_cache]
        if needed:
            needed_set = set(needed)
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx in needed_set:
                        d = json.loads(line)
                        raw = d.get("contents", "")
                        first = raw.split("\n", 1)[0].strip().strip('"')
                        text = raw[len(first):].strip()[:500]
                        self._line_cache[idx] = {"title": first, "text": text}
                        needed_set.discard(idx)
                        if not needed_set:
                            break
        return {d: self._line_cache[d] for d in doc_ids if d in self._line_cache}


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
        from vllm.lora.request import LoRARequest

        self._SamplingParams = SamplingParams
        self.llm = LLM(
            model=str(base_model),
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
        )
        self.lora_request = LoRARequest("sapr_sft", 1, str(lora_path))

    def chat(self, system, user, sampling, stop):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        sp = self._SamplingParams(
            max_tokens=sampling["max_tokens"],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            stop=stop,
        )
        out = self.llm.chat(messages, sampling_params=sp,
                            lora_request=self.lora_request,
                            use_tqdm=False)
        return out[0].outputs[0].text


class TransformersBackend:
    def __init__(self, base_model, lora_path, device="cuda:0",
                 dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print(f"[backend=transformers] loading base {base_model} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(base_model))
        base = AutoModelForCausalLM.from_pretrained(
            str(base_model), torch_dtype=dtype,
        ).to(device).eval()
        print(f"[backend=transformers] attaching LoRA from {lora_path} ...")
        self.model = PeftModel.from_pretrained(base, str(lora_path)).eval()
        self.device = device

    @torch.no_grad()
    def chat(self, system, user, sampling, stop):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = dict(
            max_new_tokens=sampling["max_tokens"],
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


# ─────────── agent loop ───────────
class SAPRAgent:
    def __init__(self, backend, sampling, retriever, max_turns=6):
        self.backend = backend
        self.sampling = sampling
        self.retriever = retriever
        self.max_turns = max_turns

    def _chat(self, system, user, stop):
        return self.backend.chat(system, user, self.sampling, stop)

    def run(self, question, top_k=3):
        history = []
        trace = []
        for turn in range(self.max_turns):
            sys, user = build_reasoning_prompt(question, history)
            text = self._chat(sys, user, stop=["</query>", "</answer>"])
            # vllm 会吃掉 stop 字符串本身，补回去再 parse
            if "</query>" not in text and "<query>" in text:
                text += "</query>"
            if "</answer>" not in text and "<answer>" in text:
                text += "</answer>"
            action = parse_action(text)
            trace.append({"turn": turn, "stage": "reason", "out": text, "parsed": action})

            if action["type"] == "answer":
                return {"answer": action["value"], "history": history, "trace": trace}
            if action["type"] != "query":
                return {"answer": None, "history": history, "trace": trace,
                        "error": "no_query_or_answer"}

            query = action["value"]
            docs = self.retriever.search(query, top_k=top_k)
            trace.append({"turn": turn, "stage": "retrieve", "query": query, "docs": docs})

            sys, user = build_evidence_prompt(query, docs)
            ev_text = self._chat(sys, user, stop=["</evidence>"])
            if "</evidence>" not in ev_text and "<evidence>" in ev_text:
                ev_text += "</evidence>"
            evidence = parse_evidence(ev_text)
            trace.append({"turn": turn, "stage": "evidence", "out": ev_text,
                          "parsed": evidence})

            history.append({"query": query, "evidence": evidence})

        return {"answer": None, "history": history, "trace": trace,
                "error": "max_turns_exceeded"}


# ─────────── main ───────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--question", type=str, default=None)
    p.add_argument("--input_jsonl", type=str, default=None,
                   help="批量模式输入，每行 {question}")
    p.add_argument("--output_jsonl", type=str, default=None)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--max_turns", type=int, default=6)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--lora_path", type=str, default=str(LORA_ADAPTER))
    p.add_argument("--backend", type=str, default="vllm",
                   choices=["vllm", "transformers"],
                   help="vllm: 快但环境苛刻; transformers: 慢但兼容性好")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.5,
                   help="仅 vllm: 占多少显存，BGE encoder 还要 ~3G")
    p.add_argument("--max_model_len", type=int, default=4096,
                   help="仅 vllm: KV cache 上限")
    args = p.parse_args()

    print(f"[main] backend={args.backend}, lora={args.lora_path}")
    if args.backend == "vllm":
        backend = VLLMBackend(
            base_model=BASE_MODEL,
            lora_path=args.lora_path,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        backend = TransformersBackend(
            base_model=BASE_MODEL,
            lora_path=args.lora_path,
            device="cuda:0",
        )

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
    with open(args.output_jsonl, "w") as fo:
        for i, q in enumerate(questions):
            t0 = time.time()
            res = agent.run(q["question"], top_k=args.top_k)
            res["id"] = q.get("id", i)
            res["question"] = q["question"]
            res["gold"] = q.get("answer") or q.get("golden_answers")
            res["latency_s"] = round(time.time() - t0, 2)
            fo.write(json.dumps(res, ensure_ascii=False) + "\n")
            fo.flush()
            print(f"[{i+1}/{len(questions)}] {res['latency_s']:.1f}s "
                  f"answer={res.get('answer')}")


if __name__ == "__main__":
    main()
