#!/usr/bin/env python3
"""Convert R3-RAG cold-start data (Alpaca, plain-text steps) into ReasonRAG-style
SFT data (LLaMA-Factory alpaca format) that matches the RAG_ProGuide DPO schema.

R3 source (one row = one teacher-forcing step):
  instruction:
    The question: <Q>
    Step 1:
    The problem analysis: <a1>
    The retrieval query: <q1>
    The retrieval documents: <docs1>
    Step 2: ...
  output:
    Step k:
    The problem analysis: <ak>
    The retrieval query: <qk>          # non-terminal
      -- or --
    The final answer: <ans>            # terminal

ReasonRAG target:
  instruction: <fixed system prompt>
  input:       Question: <Q>
               Previous Thoughts: <rendered history with <query>/<evidence> tags>
  output:      <analysis> So the next query is <query>qk</query>
               -- or --
               <analysis> So the answer is <answer>ans</answer>
"""
import argparse
import json
import re
import sys

SYSTEM_PROMPT = (
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

EVIDENCE_MAX_CHARS = 300

STEP_RE = re.compile(r"^Step\s+\d+\s*:\s*$", re.MULTILINE)


def _grab(block, label, until_labels):
    """Extract text after `label:` up to the next label in `until_labels` (or end)."""
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
    """Parse an R3 instruction/output text into a list of step dicts.

    Returns list of {analysis, query, documents, answer}. The `question` is
    handled by the caller (only present in instruction's first line).
    """
    # Drop a leading "The question: ..." line if present.
    text = text.strip()
    # Split into "Step N:" chunks.
    parts = STEP_RE.split(text)
    # parts[0] is whatever precedes the first "Step" marker (question line, etc.)
    steps = []
    for chunk in parts[1:]:
        analysis = _grab(
            chunk, "The problem analysis:",
            ["The retrieval query:", "The retrieval documents:", "The final answer:"],
        )
        query = _grab(
            chunk, "The retrieval query:",
            ["The retrieval documents:", "The final answer:"],
        )
        documents = _grab(chunk, "The retrieval documents:", ["The final answer:"])
        answer = _grab(chunk, "The final answer:", [])
        steps.append({
            "analysis": analysis,
            "query": query,
            "documents": documents,
            "answer": answer,
        })
    return steps


def extract_question(instruction):
    m = re.match(r"\s*The question:\s*(.*?)(?:\n|$)", instruction)
    return m.group(1).strip() if m else None


def condense_evidence(documents):
    """Turn a raw R3 documents blob into a short evidence snippet.

    R3 has no evidence-extraction action, so for history rendering we use the
    first non-empty passage chunk, truncated. This keeps the ReasonRAG protocol
    shape while staying within cutoff length.
    """
    if not documents:
        return ""
    # Collapse whitespace, take leading slice.
    flat = re.sub(r"\s+", " ", documents).strip()
    if len(flat) > EVIDENCE_MAX_CHARS:
        flat = flat[:EVIDENCE_MAX_CHARS].rstrip() + "..."
    return flat


def render_history(hist_steps):
    """Render prior steps as ReasonRAG-style Previous Thoughts."""
    pieces = []
    for st in hist_steps:
        analysis = st.get("analysis") or ""
        query = st.get("query")
        if query:
            seg = f"{analysis}\n\nSo the next query is <query>{query}</query>"
            ev = condense_evidence(st.get("documents"))
            if ev:
                seg += f" Based on the query, the relevant evidence is <evidence>{ev}</evidence>."
            pieces.append(seg.strip())
    return "\n\n".join(pieces)


def build_input(question, hist_steps):
    s = f"Question: {question}"
    hist = render_history(hist_steps)
    if hist:
        s += f"\nPrevious Thoughts: {hist}"
    return s


def build_output(cur_step):
    analysis = (cur_step.get("analysis") or "").strip()
    if cur_step.get("answer"):
        return f"{analysis}\n\nSo the answer is <answer>{cur_step['answer'].strip()}</answer>".strip()
    if cur_step.get("query"):
        return f"{analysis}\n\nSo the next query is <query>{cur_step['query'].strip()}</query>".strip()
    return None


def _tags_balanced(text):
    """Exactly one closing tag among query/answer, each opened tag closed once."""
    for tag in ("query", "answer", "evidence"):
        if text.count(f"<{tag}>") != text.count(f"</{tag}>"):
            return False
    n_query = text.count("</query>")
    n_answer = text.count("</answer>")
    return (n_query + n_answer) == 1


def convert_row(instruction, output):
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
    out_text = build_output(cur)
    if out_text is None:
        return None, "current_step_no_query_or_answer"
    if not _tags_balanced(out_text):
        return None, "unbalanced_output_tags"
    rec = {
        "system": SYSTEM_PROMPT,
        "instruction": build_input(question, hist_steps),
        "input": "",
        "output": out_text,
    }
    return rec, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/r3.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.src)
    if args.limit > 0:
        df = df.head(args.limit)

    stats = {"ok": 0}
    with open(args.out, "w") as f:
        for _, r in df.iterrows():
            rec, status = convert_row(r["instruction"], r["output"])
            stats[status] = stats.get(status, 0) + 1
            if rec is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] wrote -> {args.out}", file=sys.stderr)
    print(f"[stats] {json.dumps(stats, ensure_ascii=False)}", file=sys.stderr)


if __name__ == "__main__":
    main()
