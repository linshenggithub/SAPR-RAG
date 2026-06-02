#!/usr/bin/env python3
"""
SAPR-E v0 End-to-End Test (30 samples).

Modifies ReasonRAG pipeline to:
1. Retrieve top-10 (instead of top-3)
2. Re-rank top-10 using SAPR-E v0 heuristic scorer
3. Feed top-3 selected docs to the model
4. Compare EM/F1 against baseline (retriever top-3)

Usage:
  conda activate reasonrag
  CUDA_VISIBLE_DEVICES=0 python run_sapr_e_e2e.py --num_examples 30 --mode sapr_e
  CUDA_VISIBLE_DEVICES=0 python run_sapr_e_e2e.py --num_examples 30 --mode baseline
"""

import os, sys, json, re, time, argparse, datetime
import numpy as np

REASONRAG_ROOT = os.environ.get("REASONRAG_ROOT", "/home/mayi/ReasonRAG")
RESEARCH_ROOT = os.environ.get("RESEARCH_ROOT", "/home/mayi/RAG/agentic-rag-process-optimization")
sys.path.insert(0, REASONRAG_ROOT)

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.dataset.dataset import Dataset as FlashRAGDataset
from flashrag.pipeline import BasicPipeline
from pipeline.reasonrag_pipeline import ReasonRAGPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--num_examples", type=int, default=30)
parser.add_argument("--mode", choices=["sapr_e", "baseline"], default="sapr_e")
parser.add_argument("--run_id", default=None)
parser.add_argument("--max_tokens", type=int, default=256)
parser.add_argument("--gpu_id", default="0")
parser.add_argument("--index_path", default=None)
parser.add_argument("--corpus_path", default="/nas/mayi/RAG/corpus/wiki18_extended.jsonl")
parser.add_argument("--bge_path", default="/nas/mayi/RAG/retrievers/bge-base-en-v1.5")
parser.add_argument(
    "--generator_path",
    default="/home/mayi/LLaMA-Factory/examples/merge_lora/output/qwen2.5-7B-lora-dpo-RAG-ProGuide",
)
parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
args = parser.parse_args()

SLICE_SIZE = args.num_examples
MODE = args.mode
RUN_ID = args.run_id or "{}_sapr_e_e2e_{}samples_maxtok{}".format(
    datetime.datetime.now().strftime("%Y%m%d"), SLICE_SIZE, args.max_tokens
)
OUTPUT_DIR = os.path.join(RESEARCH_ROOT, "04_experiments/logs",
    RUN_ID, MODE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── SAPR-E v0 Scorer ────────────────────────────────────────────
def _word_set(text):
    return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))

def _extract_entities(texts):
    ep = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b'
    ents = set()
    for t in texts:
        ents |= set(re.findall(ep, t))
    return set(e.lower() for e in ents)

def normalize_title(title):
    t = re.sub(r'[^a-z0-9\s]', '', title.strip().lower())
    return re.sub(r'\s+', ' ', t).strip()

def sapr_e_score(question, history_thoughts, subquery, doc):
    score = 0.0
    doc_text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
    doc_words = _word_set(doc_text)
    if subquery:
        sw = _word_set(subquery)
        if sw: score += 2.0 * len(sw & doc_words) / len(sw)
    qw = _word_set(question)
    if qw: score += 1.0 * len(qw & doc_words) / len(qw)
    ents = _extract_entities([question] + ([subquery] if subquery else []))
    if ents:
        hits = sum(1 for e in ents if e in doc_text)
        score += 1.5 * hits / len(ents)
    if history_thoughts:
        hw = _word_set(" ".join(history_thoughts))
        if doc_words and hw:
            score += 0.5 * len(doc_words - hw) / len(doc_words)
    dtn = normalize_title(doc.get("title", ""))
    for e in ents:
        if all(w in dtn for w in e.split()):
            score += 1.0; break
    return score

def infer_subquery(question, thoughts_list):
    full_text = " ".join(thoughts_list)
    m = re.search(r'So the next query is\s*(.*?)(?:\.|$)', full_text, re.IGNORECASE)
    if m and m.group(1).strip(): return m.group(1).strip()
    m = re.findall(r'<query>(.*?)</query>', full_text)
    if m and m[-1].strip(): return m[-1].strip()
    for pat in [r'(?:Determine|Find|Search for|Look up|Identify)\s+(.*?)(?:\.|$)']:
        m = re.findall(pat, full_text, re.IGNORECASE)
        if m: return m[-1].strip()
    m = re.findall(r'\d+\.\s+(.*?[?\.])', full_text)
    if m: return m[-1].strip()
    return question

def select_sapr_e_top3(question, history_thoughts, subquery, docs_raw):
    formatted = []
    for d in docs_raw:
        raw = d if isinstance(d, str) else d.get("contents", d.get("text", ""))
        parts = raw.split("\n", 1)
        title = parts[0].strip().strip('"') if parts else ""
        text = parts[1].strip() if len(parts) > 1 else raw
        formatted.append({"title": title, "text": text[:500], "raw": d})
    scored = [(sapr_e_score(question, history_thoughts, subquery, fd), fd) for fd in formatted]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [fd["raw"] for _, fd in scored[:3]]

# ── Config ───────────────────────────────────────────────────────
topk = 10 if MODE == "sapr_e" else 3
index_path = args.index_path or os.path.join(REASONRAG_ROOT, "indexes/bge_extended/bge_Flat.index")
config_dict = {
    "data_dir": os.path.join(REASONRAG_ROOT, "dataset/"),
    "dataset_name": "hotpotqa", "split": ["dev", "test"],
    "index_path": index_path,
    "retrieval_method": "bge",
    "corpus_path": args.corpus_path,
    "faiss_gpu": False,
    "model2path": {
        "bge": args.bge_path,
        "qwen2.5-instruct-ReasonRAG-lora": args.generator_path,
    },
    "model2pooling": {"bge": "cls"}, "method2index": {"bge": None},
    "generator_model": "qwen2.5-instruct-ReasonRAG-lora",
    "generator_batch_size": 1, "tensor_parallel_size": 1,
    "framework": "vllm", "gpu_id": args.gpu_id,
    "gpu_memory_utilization": args.gpu_memory_utilization,
    "generator_max_input_len": 8192,
    "generation_params": {
        "do_sample": False,
        "max_tokens": args.max_tokens,
    },
    "retrieval_topk": topk,
    "metrics": ["em", "f1", "acc"],
    "save_intermediate_data": True,
    "save_note": "sapr_e_e2e_{}".format(MODE),
    "save_dir": OUTPUT_DIR, "seed": 2024,
    "disable_save": False, "test_sample_num": None, "random_sample": False,
}

FORBIDDEN_DIR = os.path.join(REASONRAG_ROOT, "output")
assert not os.path.abspath(OUTPUT_DIR).startswith(os.path.abspath(FORBIDDEN_DIR)), \
    "SAFETY: save_dir must not be inside {}".format(FORBIDDEN_DIR)
assert args.max_tokens >= 128, \
    "max_tokens={} is too small for ReasonRAG routing markers; use 256.".format(args.max_tokens)
assert os.path.exists(REASONRAG_ROOT), "Missing REASONRAG_ROOT={}".format(REASONRAG_ROOT)
assert os.path.exists(index_path), "Missing index_path={}".format(index_path)
assert os.path.exists(args.corpus_path), "Missing corpus_path={}".format(args.corpus_path)
assert os.path.exists(args.bge_path), "Missing bge_path={}".format(args.bge_path)
assert os.path.exists(args.generator_path), "Missing generator_path={}".format(args.generator_path)

print("=" * 70)
print("SAPR-E End-to-End Test - mode={}".format(MODE))
print("Run ID: {}".format(RUN_ID))
print("Examples: {}, topk: {}, max_tokens: {}, gpu_id: {}".format(
    SLICE_SIZE, topk, args.max_tokens, args.gpu_id))
print("ReasonRAG root: {}".format(REASONRAG_ROOT))
print("Index: {}".format(index_path))
print("Generator: {}".format(args.generator_path))
print("Output: {}".format(OUTPUT_DIR))
print("=" * 70)

config = Config(config_dict=config_dict)
resolved_max_tokens = config["generation_params"]["max_tokens"]
assert resolved_max_tokens == args.max_tokens, \
    "Resolved max_tokens mismatch: {} != {}".format(resolved_max_tokens, args.max_tokens)
print("Resolved generation max_tokens: {}".format(resolved_max_tokens))
all_split = get_dataset(config)
dev_data = all_split["dev"]
sliced_data = FlashRAGDataset(
    config=config, dataset_path=dev_data.dataset_path,
    data=dev_data.data[:SLICE_SIZE], sample_num=None, random_sample=False,
)
print("Loaded {} examples".format(len(sliced_data)))

t0 = time.time()
pipeline = ReasonRAGPipeline(config, prompt_template=None, answer_format="answer",
                              max_iter=8, max_children=2, max_rollouts=64)
print("Pipeline ready ({:.1f}s)".format(time.time() - t0))

# ── Patch run_batch for SAPR-E mode ─────────────────────────────
if MODE == "sapr_e":
    retriever = pipeline.retriever
    max_iter = pipeline.max_iter
    stop_tokens = pipeline.stop_tokens
    begin_reasoning_prompt = pipeline.begin_reasoning_prompt
    document_analysis_prompt = pipeline.document_analysis_prompt
    reasoning_prompt = pipeline.reasoning_prompt
    answer_generation_prompt = pipeline.answer_generation_prompt
    generator = pipeline.generator

    def get_flags(responses):
        flags = []
        for r in responses:
            if "So the next query is" in r: flags.append("query")
            elif "So the answer is" in r: flags.append("answer")
            elif "<evidence>" in r: flags.append("evidence")
            else: flags.append("None")
        return flags

    def extract_query(response):
        m = re.search(r'So the next query is\s*(.*?)(?=\n|$)', response, re.IGNORECASE|re.DOTALL)
        if not m: return ""
        text = m.group(1).strip()
        return re.sub(r'</?(answer|query|evidence)>', '', text).strip()

    def extract_answer(response):
        prefix = "So the answer is"
        pred = response.split(prefix)[1].strip() if prefix in response else response
        am = re.findall(r'<answer>(.*?)</answer>', pred)
        pred = am[-1] if am else pred
        pred = re.sub(r'<answer.*?>.*?</answer>|<query.*?>.*?</query>|answer>|<answer', '', pred, flags=re.DOTALL)
        return pred.split('.')[0].strip() if '.' in pred else pred.strip()

    def sapr_e_run_batch(dataset):
        for item in dataset:
            item.update_output('finish_flag', False)
            item.update_output('iteration_count', 0)
            item.update_output('previous_thoughts', [])
            item.update_output('flag', None)
            item.update_output('query', None)
            item.update_output('answer', None)

        input_prompts = [begin_reasoning_prompt.get_string(question=item.question) for item in dataset]
        responses = generator.generate(input_prompts, stop=stop_tokens)
        for i, item in enumerate(dataset):
            item.previous_thoughts.append(responses[i])
            item.flag = get_flags([responses[i]])[0]
            item.query = extract_query(responses[i])
            item.answer = extract_answer(responses[i])
            if item.flag in ["finish", "answer"]:
                item.finish_flag = True

        for step in range(1, max_iter + 1):
            exist_items = [item for item in dataset if not item.finish_flag]
            if not exist_items: break

            active_questions = [item.question for item in exist_items]
            active_previous_thoughts = [item.previous_thoughts for item in exist_items]
            active_querys = [item.query for item in exist_items]

            # Use inferred_subquery when pipeline query is empty
            search_queries = []
            for i, item in enumerate(exist_items):
                q = active_querys[i]
                if not q or not q.strip():
                    q = infer_subquery(active_questions[i], active_previous_thoughts[i])
                search_queries.append(q)

            # Retrieve top-10
            result = retriever._batch_search(search_queries, num=10, return_score=True)
            docs_list, scores_list = result

            # SAPR-E rerank: top-3 from top-10
            reranked = []
            for i, item in enumerate(exist_items):
                subq = infer_subquery(active_questions[i], active_previous_thoughts[i])
                top3 = select_sapr_e_top3(
                    active_questions[i],
                    [t[:200] for t in active_previous_thoughts[i]],
                    subq, docs_list[i])
                reranked.append(top3)

            qt_list = [
                q + "\nPrevious Thoughts: " + " ".join(thoughts)
                for q, thoughts in zip(active_questions, active_previous_thoughts)
            ]

            input_prompts = []
            for i, item in enumerate(exist_items):
                if item.iteration_count >= max_iter - 1:
                    input_prompts.append(answer_generation_prompt.get_string(question=qt_list[i]))
                elif "query" in item.flag:
                    input_prompts.append(document_analysis_prompt.get_string(
                        question=qt_list[i], retrieval_result=reranked[i]))
                elif "evidence" in item.flag:
                    input_prompts.append(reasoning_prompt.get_string(question=qt_list[i]))
                else:
                    input_prompts.append(answer_generation_prompt.get_string(question=qt_list[i]))

            responses = generator.generate(input_prompts, stop=stop_tokens)
            for i, item in enumerate(exist_items):
                item.previous_thoughts.append(responses[i])
                item.iteration_count += 1
                item.flag = get_flags([responses[i]])[0]
                item.query = extract_query(responses[i])
                item.answer = extract_answer(responses[i])
                if item.flag in ["finish", "answer"] or item.iteration_count >= max_iter:
                    item.finish_flag = True

        for item in dataset:
            item.pred = item.answer
        return dataset

    pipeline.run_batch = sapr_e_run_batch

# ── Run ──────────────────────────────────────────────────────────
print("\nRunning inference (mode={})...".format(MODE))
t0 = time.time()
output_dataset = pipeline.run(sliced_data, batch_size=1, do_eval=True)
t_run = time.time() - t0
print("Done ({:.1f}s, {:.1f}min)".format(t_run, t_run/60))

# ── Results ──────────────────────────────────────────────────────
out = output_dataset.data
preds = [item.pred for item in out]
golds = [item.gold for item in out]

em = sum(1 for p, g in zip(preds, golds) if p.strip().lower() == g[0].strip().lower()) / len(golds)
f1_scores = []
for p, g in zip(preds, golds):
    pt = set(p.strip().lower().split())
    gt = set(g[0].strip().lower().split())
    if not pt or not gt: f1_scores.append(0.0)
    else:
        ov = len(pt & gt)
        f1_scores.append(2 * ov / (len(pt) + len(gt)))
f1 = sum(f1_scores) / len(f1_scores)

print("\n" + "=" * 70)
print("RESULTS (mode={}, n={})".format(MODE, len(golds)))
print("=" * 70)
print("EM: {:.4f} ({}/{})".format(em, int(em*len(golds)), len(golds)))
print("F1: {:.4f}".format(f1))
print("Runtime: {:.1f}s ({:.1f}min)".format(t_run, t_run/60))

results = {
    "run_id": RUN_ID, "mode": MODE, "num_examples": len(golds),
    "max_tokens": args.max_tokens, "retrieval_topk": topk,
    "em": round(em, 4), "f1": round(f1, 4),
    "runtime_s": round(t_run, 1), "label": "debug_result",
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved: {}".format(os.path.join(OUTPUT_DIR, "metrics.json")))
print("=" * 70)
