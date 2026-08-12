#!/usr/bin/env python3
"""Build full HotpotQA + 2Wiki + MuSiQue data for action-specific OPSD.

Query supervision comes only from successful R3 retrieval queries. Gold answers
and supporting evidence are exposed only to the answer teacher. Evidence targets
remain ordinary dataset columns so the runtime evidence agent can verify them
against its actual Top-3 documents; this script never creates a static evidence
teacher prompt.
"""
import argparse
import json
import os
import random
import re
import unicodedata
from collections import OrderedDict, defaultdict
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(os.environ.get("SAPR_RAG_ROOT", Path(__file__).resolve().parents[3]))
DEFAULT_TOKENIZER = PROJ_ROOT / "03_sapr_rag/models/Qwen2.5-7B-Instruct"
DEFAULT_SOURCES = {
    "hotpotqa": PROJ_ROOT / "data/raw/hotpotqa/train.jsonl",
    "2wiki": PROJ_ROOT / "data/raw/2wikimultihopqa_full/train.jsonl",
    "musique": PROJ_ROOT / "data/raw/musique/train.jsonl",
}
R3_DEFAULT = PROJ_ROOT / "data/raw/r3_coldstart.parquet"
PROMPT_VERSION = "sapr-action-opsd-v1"

REASONING_SYSTEM = (
    "You are an assistant for question answering with access to a retrieval tool. "
    "Analyze and decompose the question, then either issue one concise retrieval query "
    "or answer from sufficient evidence. The retrieval system is deterministic: never "
    "repeat a previous query; instead target a different entity, relation, or missing "
    "fact. End a retrieval turn with <query>query</query>. End the final turn with "
    "<answer>answer</answer>."
)


def normalize_text(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def question_key(value):
    return normalize_text(value).casefold()


def dedupe_text(values):
    result = []
    seen = set()
    for value in values or []:
        text = normalize_text(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def decode_json_field(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def iter_records(path):
    path = Path(path)
    if path.suffix == ".parquet":
        for batch in pd.read_parquet(path).to_dict("records"):
            yield batch
        return
    with path.open(encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            try:
                import ijson
                yield from ijson.items(f, "item")
            except ImportError:
                yield from json.load(f)
        else:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def context_map(metadata):
    context = decode_json_field(metadata.get("context")) or {}
    if isinstance(context, dict):
        titles = context.get("title") or []
        contents = context.get("sentences", context.get("content", context.get("text", []))) or []
        return {
            question_key(title): content if isinstance(content, list) else [content]
            for title, content in zip(titles, contents)
        }
    result = {}
    for item in context:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            content = item[1] if isinstance(item[1], list) else [item[1]]
            result[question_key(item[0])] = content
        elif isinstance(item, dict):
            title = item.get("title", "")
            content = item.get("sentences", item.get("content", item.get("text", [])))
            result[question_key(title)] = content if isinstance(content, list) else [content]
    return result


def hotpot_evidence(raw):
    metadata = raw.get("metadata") or raw
    supporting = decode_json_field(metadata.get("supporting_facts")) or []
    if isinstance(supporting, dict):
        pairs = zip(supporting.get("title") or [], supporting.get("sent_id") or [])
    else:
        pairs = (
            (item[0], item[1])
            for item in supporting
            if isinstance(item, (list, tuple)) and len(item) >= 2
        )
    contexts = context_map(metadata)
    evidence = OrderedDict()
    for title, sentence_id in pairs:
        title = normalize_text(title)
        if not title:
            continue
        item = evidence.setdefault(question_key(title), {"title": title, "sentences": []})
        sentences = contexts.get(question_key(title), [])
        sentence = ""
        if isinstance(sentence_id, int) and 0 <= sentence_id < len(sentences):
            sentence = normalize_text(sentences[sentence_id])
        if sentence and sentence.casefold() not in {x.casefold() for x in item["sentences"]}:
            item["sentences"].append(sentence)
    return list(evidence.values())


def musique_evidence(raw):
    metadata = raw.get("metadata") or raw
    decomposition = metadata.get("question_decomposition") or raw.get("question_decomposition") or []
    paragraphs = metadata.get("paragraphs") or raw.get("paragraphs") or []
    paragraph_by_idx = {
        item.get("idx", index): item
        for index, item in enumerate(paragraphs)
        if isinstance(item, dict)
    }
    evidence = OrderedDict()
    for hop in decomposition:
        paragraph = hop.get("support_paragraph")
        if not isinstance(paragraph, dict):
            paragraph = paragraph_by_idx.get(hop.get("paragraph_support_idx"))
        if not isinstance(paragraph, dict):
            continue
        title = normalize_text(paragraph.get("title", ""))
        text = normalize_text(paragraph.get("paragraph_text", paragraph.get("text", "")))
        if not title:
            continue
        item = evidence.setdefault(question_key(title), {"title": title, "sentences": []})
        if text and text.casefold() not in {x.casefold() for x in item["sentences"]}:
            item["sentences"].append(text)
    if not evidence:
        for paragraph in paragraphs:
            if not isinstance(paragraph, dict) or not paragraph.get("is_supporting"):
                continue
            title = normalize_text(paragraph.get("title", ""))
            text = normalize_text(paragraph.get("paragraph_text", paragraph.get("text", "")))
            if title:
                evidence[question_key(title)] = {"title": title, "sentences": [text] if text else []}
    return list(evidence.values())


def adapt_record(source, raw):
    question = normalize_text(raw.get("question", ""))
    answers = raw.get("golden_answers")
    if answers is None:
        answers = [raw.get("answer", "")]
        answers.extend(raw.get("answer_aliases") or [])
    answers = dedupe_text(answers if isinstance(answers, (list, tuple)) else [answers])
    evidence = musique_evidence(raw) if source == "musique" else hotpot_evidence(raw)
    if not question or not answers:
        return None
    return {
        "messages": [
            {"role": "system", "content": REASONING_SYSTEM},
            {"role": "user", "content": f"Question: {question}"},
        ],
        "golden_answers": answers,
        "gold_titles": [item["title"] for item in evidence],
        "gold_sup_sents": ["\n".join(item["sentences"]) for item in evidence],
        "source": source,
        "_question": question,
        "_evidence": evidence,
    }


class TokenCounter:
    def __init__(self, tokenizer_path):
        self.tokenizer = None
        self.name = "utf8_bytes_conservative"
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path), local_files_only=True, trust_remote_code=True)
            self.name = str(tokenizer_path)
        except Exception as exc:
            print(f"[build] WARNING: tokenizer unavailable, using UTF-8 byte budget: {exc}")

    def count(self, text):
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text.encode("utf-8"))


def load_r3_plans(path):
    frame = pd.read_parquet(path, columns=["instruction", "output"])
    plans = defaultdict(list)
    seen_queries = defaultdict(set)
    duplicate_queries = 0
    for instruction, output in zip(frame["instruction"], frame["output"]):
        question_match = re.match(r"The question:\s*(.*?)(?:\n|$)", str(instruction))
        query_match = re.search(r"The retrieval query:\s*(.*?)(?:\n|$)", str(output))
        if not question_match or not query_match:
            continue
        key = question_key(question_match.group(1))
        query = normalize_text(query_match.group(1))
        query_key = query.casefold()
        if not query or query_key in seen_queries[key]:
            duplicate_queries += int(bool(query))
            continue
        seen_queries[key].add(query_key)
        plans[key].append(query)
    print(
        f"[build] R3 plans: questions={len(plans):,}, queries={sum(map(len, plans.values())):,}, "
        f"removed_exact_duplicates={duplicate_queries:,}")
    return plans


def build_query_prompt(queries, counter, max_tokens):
    prefix = (
        "<privileged_query_guidance>\n"
        "The following queries come from a successful real-retrieval trajectory for this "
        "question. They do not contain the gold answer. Use them only as a search-plan "
        "reference and still adapt the next query to the documents already observed.\n"
    )
    suffix = "\n</privileged_query_guidance>"
    selected = []
    for index, query in enumerate(queries, 1):
        candidate = prefix + "\n".join(selected + [f"{index}. {query}"]) + suffix
        if counter.count(candidate) > max_tokens:
            break
        selected.append(f"{index}. {query}")
    return prefix + "\n".join(selected) + suffix, len(selected) < len(queries)


def build_answer_prompt(answers, evidence, counter, max_tokens):
    prefix = (
        "<privileged_answer_guidance>\n"
        "This information is available only when scoring final answer tokens. Preserve the "
        "actual retrieval history above and use the verified evidence to produce the answer.\n"
        f"Gold answer(s): {' | '.join(answers)}\n"
        "Verified supporting evidence:\n"
    )
    suffix = (
        "\nWhen the accumulated evidence is sufficient, end with "
        "<answer>answer</answer>.\n</privileged_answer_guidance>"
    )
    blocks = []
    for index, item in enumerate(evidence, 1):
        lines = [f"{index}. Title: {item['title']}"]
        lines.extend(f"   - {sentence}" for sentence in item["sentences"])
        block = "\n".join(lines)
        if counter.count(prefix + "\n".join(blocks + [block]) + suffix) > max_tokens:
            break
        blocks.append(block)
    prompt = prefix + ("\n".join(blocks) if blocks else "(not available)") + suffix
    if counter.count(prompt) > max_tokens:
        prompt = (
            "<privileged_answer_guidance>\n"
            f"Gold answer(s): {' | '.join(answers)}\n"
            "Use this only for final <answer>answer</answer> tokens.\n"
            "</privileged_answer_guidance>"
        )
    return prompt, len(blocks) < len(evidence)


def parse_source_overrides(values):
    result = dict(DEFAULT_SOURCES)
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or name not in result:
            raise ValueError(f"invalid --source {value!r}; expected one of NAME=PATH for {sorted(result)}")
        result[name] = Path(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(PROJ_ROOT / "data/grpo/hotpotqa_2wiki_musique_train_multi_opsd.jsonl"))
    parser.add_argument(
        "--source", action="append", default=[],
        help="Override a source path, e.g. --source 2wiki=/path/train.json")
    parser.add_argument(
        "--sources", default="hotpotqa,2wiki,musique",
        help="Comma-separated subset of hotpotqa,2wiki,musique")
    parser.add_argument("--r3_path", default=str(R3_DEFAULT))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--query_prompt_max_tokens", type=int, default=512)
    parser.add_argument("--answer_prompt_max_tokens", type=int, default=1536)
    parser.add_argument(
        "--limit_per_source", type=int, default=0,
        help="Use the first N valid rows per source for a smoke dataset; 0 keeps all rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    requested_sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    source_paths = parse_source_overrides(args.source)
    unknown = set(requested_sources) - set(source_paths)
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    for source in requested_sources:
        if not source_paths[source].is_file():
            raise FileNotFoundError(
                f"{source} train data is missing: {source_paths[source]}. "
                "Run prepare_action_opsd_train_data.py first or pass --source NAME=PATH.")
    if not Path(args.r3_path).is_file():
        raise FileNotFoundError(f"R3 cold-start data is missing: {args.r3_path}")

    random.seed(args.seed)
    counter = TokenCounter(args.tokenizer)
    plans = load_r3_plans(args.r3_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(int)
    query_matches = defaultdict(int)
    answer_truncated = defaultdict(int)
    query_truncated = defaultdict(int)
    skipped = defaultdict(int)

    with output.open("w", encoding="utf-8") as writer:
        for source in requested_sources:
            for raw in iter_records(source_paths[source]):
                row = adapt_record(source, raw)
                if row is None:
                    skipped[source] += 1
                    continue
                question = row.pop("_question")
                evidence = row.pop("_evidence")
                queries = plans.get(question_key(question))
                if queries:
                    prompt, truncated = build_query_prompt(
                        queries, counter, args.query_prompt_max_tokens)
                    row["teacher_query_prompt"] = prompt
                    row["teacher_query_plan_length"] = len(queries)
                    row["teacher_query_prompt_truncated"] = truncated
                    query_matches[source] += 1
                    query_truncated[source] += int(truncated)
                answer_prompt, truncated = build_answer_prompt(
                    row["golden_answers"], evidence, counter, args.answer_prompt_max_tokens)
                row["teacher_answer_prompt"] = answer_prompt
                row["teacher_answer_prompt_truncated"] = truncated
                row["teacher_prompt_version"] = PROMPT_VERSION
                row["teacher_evidence_runtime_only"] = True
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                totals[source] += 1
                if truncated:
                    answer_truncated[source] += 1
                if args.limit_per_source and totals[source] >= args.limit_per_source:
                    break

    print(f"[build] wrote {sum(totals.values()):,} rows -> {output}")
    for source in requested_sources:
        coverage = query_matches[source] / totals[source] if totals[source] else 0.0
        print(
            f"  {source}: rows={totals[source]:,}, skipped={skipped[source]:,}, "
            f"R3_query={query_matches[source]:,} ({coverage:.1%}), "
            f"query_truncated={query_truncated[source]:,}, "
            f"answer_truncated={answer_truncated[source]:,}")


if __name__ == "__main__":
    main()
