#!/usr/bin/env python3
"""Build SFT data for SAPR-RAG by converting R3 cold-start trajectories into
ReasonRAG-style two-role samples (LLaMA-Factory alpaca format).

Outputs two jsonl files (one row per record):
  - reasoning samples (one per R3 row): teaches the agent to decide the next
    action (<query> or <answer>) given the current state.
  - evidence samples (one per cache entry): teaches the extractor to produce
    <evidence> from a (query, retrieval_documents) pair.

Both use independent system columns; LLaMA-Factory dataset_info maps
{prompt: instruction, query: input, response: output, system: system}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[3]

# --- ReasonRAG system prompts (verbatim from pipeline/reasonrag_pipeline.py) ---
REASONING_SYSTEM = (
    "You are an assistant for question answering with access to a retrieval tool. "
    "Upon receiving a question, your task is to:\n"
    "* Analyze and Decompose the Question: Break the question into smaller, manageable "
    "sub-questions to ensure all aspects are addressed.\n"
    "* Evaluate Your Knowledge: Assess each sub-question or component:\n"
    "- Identify parts you can confidently answer based on your existing knowledge.\n"
    "- Pinpoint parts that require additional information or verification through retrieval tools.\n"
    "* Conciseness: Ensure both queries and answers are concise, using nouns or short phrases whenever possible.\n"
    "* Respond Format:\n"
    "If your knowledge is sufficient to answer the question, conclude with:\n"
    '"So the answer is <answer>answer</answer>"\n'
    "If retrieval is necessary to provide a complete answer, conclude with:\n"
    '"So the next query is <query>query</query>"\n'
)

EVIDENCE_SYSTEM = (
    "You are an information retrieval assistant. Your task is to extract "
    "relevant evidence from the provided Wikipedia documents based on the "
    "latest query.\n\n"
    "Instructions:\n"
    "* Identify key terms or concepts in the query.\n"
    "* Search the documents for evidence that supports the query.\n"
    "* Be concise: ideally one or two short sentences.\n"
    "* Response format:\n"
    "If relevant evidence is found, output:\n"
    '   Based on the query, the relevant evidence is <evidence>evidence</evidence>.\n'
    "If no relevant evidence is found, output:\n"
    '   <evidence>None</evidence>.\n'
)

EVIDENCE_USER_TEMPLATE = "Question: {query}. Reference: <reference>{reference}</reference>"

STEP_RE = re.compile(r"^Step\s+\d+\s*:\s*$", re.MULTILINE)

DEFAULT_GOLD_TRAIN_JSONLS = [
    PROJ_ROOT / "data/raw/hotpotqa/train.jsonl",
    PROJ_ROOT / "data/raw/2wikimultihopqa/train.jsonl",
    PROJ_ROOT / "data/raw/musique/train.jsonl",
]


# --------------------- R3 step parsing ---------------------------------------
def _grab(block, label, until_labels):
    idx = block.find(label)
    if idx < 0:
        return None
    start = idx + len(label)
    end = len(block)
    for ul in until_labels:
        j = block.find(ul, start)
        if 0 <= j < end:
            end = j
    return block[start:end].strip()


def parse_steps(text):
    text = (text or "").strip()
    parts = STEP_RE.split(text)
    out = []
    for chunk in parts[1:]:
        out.append({
            "analysis": _grab(
                chunk, "The problem analysis:",
                ["The retrieval query:", "The retrieval documents:", "The final answer:"],
            ),
            "query": _grab(
                chunk, "The retrieval query:",
                ["The retrieval documents:", "The final answer:"],
            ),
            "documents": _grab(chunk, "The retrieval documents:", ["The final answer:"]),
            "answer": _grab(chunk, "The final answer:", []),
        })
    return out


def extract_question(instruction):
    m = re.match(r"\s*The question:\s*(.*?)(?:\n|$)", instruction or "")
    return m.group(1).strip() if m else None


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question or "").strip()


def extract_gold_answer(row: dict):
    gold = row.get("golden_answers")
    if gold is None:
        gold = row.get("answer")
    if gold is None:
        gold = row.get("answer_aliases")
    if isinstance(gold, list):
        for item in gold:
            text = str(item).strip()
            if text:
                return text
        return None
    text = str(gold).strip()
    return text or None


def load_gold_answer_map(paths):
    """question -> canonical short answer from original training datasets."""
    gold_by_question = {}
    stats = {"files": 0, "rows": 0, "with_gold": 0, "duplicates": 0, "conflicts": 0}
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"[gold] skip missing: {path}", file=sys.stderr)
            continue
        stats["files"] += 1
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stats["rows"] += 1
                question = normalize_question(row.get("question", ""))
                gold = extract_gold_answer(row)
                if not question or not gold:
                    continue
                stats["with_gold"] += 1
                prev = gold_by_question.get(question)
                if prev is None:
                    gold_by_question[question] = gold
                elif prev == gold:
                    stats["duplicates"] += 1
                else:
                    # Keep the first source deterministically; report conflicts.
                    stats["conflicts"] += 1
    print(
        f"[gold] loaded questions={len(gold_by_question):,} stats="
        f"{json.dumps(stats, ensure_ascii=False)}",
        file=sys.stderr,
    )
    return gold_by_question


def cache_key(query: str, reference: str) -> str:
    h = hashlib.sha256()
    h.update(query.encode("utf-8"))
    h.update(b"\n--R3REF--\n")
    h.update(reference.encode("utf-8"))
    return h.hexdigest()[:32]


# --------------------- evidence cache loader ---------------------------------
def load_evidence_cache(cache_path: Path):
    """key -> evidence string (already extracted from <evidence>...</evidence>)."""
    cache = {}
    with cache_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = rec.get("key")
            ev = rec.get("evidence", "")
            if k:
                cache[k] = ev
    return cache


# --------------------- history rendering -------------------------------------
def render_history(hist_steps, ev_cache):
    """Render prior steps with cached evidence inserted.

    Each historical step that has a (query, documents) pair gets:
       <analysis>
       So the next query is <query>q</query> Based on the query,
       the relevant evidence is <evidence>ev</evidence>.
    """
    pieces = []
    for st in hist_steps:
        analysis = (st.get("analysis") or "").strip()
        query = (st.get("query") or "").strip()
        docs = (st.get("documents") or "").strip()
        if not query:
            continue
        seg = f"{analysis}\n\nSo the next query is <query>{query}</query>"
        if docs:
            ev = ev_cache.get(cache_key(query, docs), "")
            ev = ev if ev else "None"
            seg += f" Based on the query, the relevant evidence is <evidence>{ev}</evidence>."
        else:
            # retrieval failed in R3 → no documents → mark None
            seg += " Based on the query, the relevant evidence is <evidence>None</evidence>."
        pieces.append(seg.strip())
    return "\n\n".join(pieces)


def build_reasoning_input(question, hist_steps, ev_cache):
    s = f"Question: {question}"
    hist = render_history(hist_steps, ev_cache)
    if hist:
        s += f"\nPrevious Thoughts: {hist}"
    return s


def build_reasoning_output(cur_step):
    analysis = (cur_step.get("analysis") or "").strip()
    if cur_step.get("answer"):
        return f"{analysis}\n\nSo the answer is <answer>{cur_step['answer'].strip()}</answer>".strip()
    if cur_step.get("query"):
        return f"{analysis}\n\nSo the next query is <query>{cur_step['query'].strip()}</query>".strip()
    return None


def _tags_balanced(text):
    for tag in ("query", "answer", "evidence"):
        if text.count(f"<{tag}>") != text.count(f"</{tag}>"):
            return False
    return (text.count("</query>") + text.count("</answer>")) == 1


# --------------------- per-row reasoning sample ------------------------------
def convert_reasoning_row(instruction, output, ev_cache, gold_by_question=None, canonical_stats=None):
    question = extract_question(instruction)
    if not question:
        return None, "no_question"
    hist_steps = parse_steps(instruction)
    cur_steps = parse_steps(output)
    if not cur_steps:
        return None, "no_current_step"
    if len(cur_steps) != 1:
        return None, "output_not_single_step"
    cur = cur_steps[0]
    if cur.get("answer") and gold_by_question is not None:
        key = normalize_question(question)
        gold = gold_by_question.get(key)
        if gold:
            if canonical_stats is not None:
                canonical_stats["terminal_rows"] = canonical_stats.get("terminal_rows", 0) + 1
                if normalize_question(cur.get("answer", "")) != normalize_question(gold):
                    canonical_stats["answer_replaced"] = canonical_stats.get("answer_replaced", 0) + 1
                else:
                    canonical_stats["answer_already_same"] = canonical_stats.get("answer_already_same", 0) + 1
            cur = dict(cur)
            cur["answer"] = gold
        elif canonical_stats is not None:
            canonical_stats["terminal_rows"] = canonical_stats.get("terminal_rows", 0) + 1
            canonical_stats["missing_gold"] = canonical_stats.get("missing_gold", 0) + 1
    out_text = build_reasoning_output(cur)
    if out_text is None:
        return None, "current_step_no_query_or_answer"
    if not _tags_balanced(out_text):
        return None, "unbalanced_output_tags"
    rec = {
        "system": REASONING_SYSTEM,
        "instruction": build_reasoning_input(question, hist_steps, ev_cache),
        "input": "",
        "output": out_text,
    }
    return rec, "ok"


# --------------------- per-cache evidence sample ------------------------------
def make_evidence_sample(query, reference, evidence):
    """One SFT sample teaching evidence extraction.

    Uses the exact ReasonRAG DOCUMENT_ANALYSIS_PROMPT format. The output is
    the raw response shape: 'Based on the query, ... <evidence>X</evidence>.'
    or '<evidence>None</evidence>.' when no evidence.
    """
    if not evidence or evidence.strip() == "None":
        out = "<evidence>None</evidence>."
    else:
        out = f"Based on the query, the relevant evidence is <evidence>{evidence}</evidence>."
    return {
        "system": EVIDENCE_SYSTEM,
        "instruction": EVIDENCE_USER_TEMPLATE.format(query=query, reference=reference),
        "input": "",
        "output": out,
    }


# --------------------- main ---------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mlx_devbox/users/mayi.summer/playground/"
                    "SAPR-RAG/data/raw/r3_coldstart.parquet")
    ap.add_argument("--cache", default="/mlx_devbox/users/mayi.summer/playground/"
                    "SAPR-RAG/03_sapr_rag/data/sft_build/out/evidence_cache.jsonl")
    ap.add_argument("--out-reasoning", required=True)
    ap.add_argument("--out-evidence", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0=all R3 rows")
    ap.add_argument(
        "--canonical-answer",
        action="store_true",
        help="Replace terminal <answer> content with the original train-set gold answer.",
    )
    ap.add_argument(
        "--gold-train-jsonl",
        action="append",
        default=[],
        help=(
            "Original train jsonl used for question->gold lookup. Can be repeated. "
            "Defaults to HotpotQA/2Wiki/MuSiQue train files when --canonical-answer is set."
        ),
    )
    args = ap.parse_args()

    cache_path = Path(args.cache)
    print(f"[load] evidence cache: {cache_path}", file=sys.stderr)
    ev_cache = load_evidence_cache(cache_path)
    print(f"[load] cache entries: {len(ev_cache):,}", file=sys.stderr)

    import pandas as pd
    df = pd.read_parquet(args.src)
    if args.limit > 0:
        df = df.head(args.limit)
    print(f"[load] R3 rows: {len(df):,}", file=sys.stderr)

    gold_by_question = None
    canonical_stats = None
    if args.canonical_answer:
        gold_paths = args.gold_train_jsonl or [str(p) for p in DEFAULT_GOLD_TRAIN_JSONLS]
        gold_by_question = load_gold_answer_map(gold_paths)
        canonical_stats = {"terminal_rows": 0, "answer_replaced": 0,
                           "answer_already_same": 0, "missing_gold": 0}

    # --- reasoning samples ---
    stats_r = {"ok": 0}
    out_r = Path(args.out_reasoning)
    out_r.parent.mkdir(parents=True, exist_ok=True)
    with out_r.open("w") as f:
        for _, r in df.iterrows():
            rec, status = convert_reasoning_row(
                r["instruction"],
                r["output"],
                ev_cache,
                gold_by_question=gold_by_question,
                canonical_stats=canonical_stats,
            )
            stats_r[status] = stats_r.get(status, 0) + 1
            if rec is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[reasoning] wrote -> {out_r}  stats={json.dumps(stats_r, ensure_ascii=False)}",
          file=sys.stderr)
    if canonical_stats is not None:
        print(f"[canonical] stats={json.dumps(canonical_stats, ensure_ascii=False)}",
              file=sys.stderr)

    # --- evidence samples (one per cache entry whose (q, ref) is reached by
    #     the rows we just processed; for full run this == cache size) ---
    seen_keys = set()
    for _, r in df.iterrows():
        for src in (r.get("instruction") or "", r.get("output") or ""):
            for st in parse_steps(src):
                q = (st.get("query") or "").strip()
                d = (st.get("documents") or "").strip()
                if q and d:
                    seen_keys.add(cache_key(q, d))

    out_e = Path(args.out_evidence)
    out_e.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    miss = 0
    # We need (query, reference, evidence) → re-scan cache file for full records.
    with cache_path.open() as fc, out_e.open("w") as fo:
        for line in fc:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = rec.get("key")
            if k not in seen_keys:
                continue
            sample = make_evidence_sample(
                query=rec.get("query", ""),
                reference=rec.get("reference", ""),
                evidence=rec.get("evidence", ""),
            )
            fo.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
    miss = len(seen_keys) - written
    print(f"[evidence] wrote -> {out_e}  written={written:,}  missing_in_cache={miss:,}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
