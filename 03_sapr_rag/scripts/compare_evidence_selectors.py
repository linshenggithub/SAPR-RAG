#!/usr/bin/env python3
"""
SAPR-E v0 Evidence Selector Comparison.

Compares three evidence selection strategies on the corrected
evidence_decision_points.jsonl from the query-fixed exporter:

1. Retriever top-3: select top-3 by retrieval score (baseline)
2. Static question-doc scorer: score doc relevance to original question
3. SAPR-E state-aware scorer: score using question + history + inferred_subquery + doc

No training. No GPU. Pure post-hoc analysis.

Metrics:
- gold supporting-doc hit@3 (at least one gold doc in top-3 selection)
- gold supporting-doc recall@3 (fraction of gold docs found in top-3)
- noise rate (fraction of selected docs that are NOT gold supporting docs)
- per-question badcases

Usage:
  python compare_evidence_selectors.py --input <evidence_decision_points.jsonl> --output_dir <dir>
"""

import json
import re
import os
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


def normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    t = title.strip().lower()
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def get_gold_titles(supporting_facts: dict) -> set:
    """Extract normalized gold supporting doc titles."""
    titles = supporting_facts.get("title", [])
    return set(normalize_title(t) for t in titles)


def select_top3_by_score(docs: List[dict]) -> List[dict]:
    """Selector 1: Retriever top-3 by retrieval score."""
    sorted_docs = sorted(docs, key=lambda d: d.get("score", 0), reverse=True)
    return sorted_docs[:3]


def static_question_doc_score(question: str, doc: dict) -> float:
    """Selector 2: Static relevance score between question and doc.

    Cheap heuristic: count overlap of question keywords with doc title + text.
    """
    q_words = set(re.findall(r'\b[a-z]{3,}\b', question.lower()))
    doc_text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
    doc_words = set(re.findall(r'\b[a-z]{3,}\b', doc_text))

    if not q_words:
        return 0.0

    overlap = len(q_words & doc_words)
    # Normalize by question length to avoid bias toward long questions
    return overlap / len(q_words)


def select_top3_static(question: str, docs: List[dict]) -> List[dict]:
    """Selector 2: Static question-doc scorer."""
    scored = [(static_question_doc_score(question, d), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:3]]


def sapr_e_state_aware_score(
    original_question: str,
    history_thoughts: List[str],
    inferred_subquery: str,
    doc: dict,
) -> float:
    """Selector 3: SAPR-E state-aware evidence utility score.

    Dimensions:
    1. Relevance to inferred_subquery (current information need)
    2. Novelty w.r.t. history (adds new information)
    3. Entity overlap (contains entities from question/subquery)
    4. Chain support (advances multi-hop reasoning)
    """
    score = 0.0

    # Concatenate document text
    doc_text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
    doc_words = set(re.findall(r'\b[a-z]{3,}\b', doc_text))

    # --- Dimension 1: Relevance to inferred_subquery ---
    if inferred_subquery:
        subq_words = set(re.findall(r'\b[a-z]{3,}\b', inferred_subquery.lower()))
        if subq_words:
            relevance = len(subq_words & doc_words) / len(subq_words)
            score += 2.0 * relevance  # highest weight

    # --- Dimension 2: Relevance to original question ---
    q_words = set(re.findall(r'\b[a-z]{3,}\b', original_question.lower()))
    if q_words:
        q_relevance = len(q_words & doc_words) / len(q_words)
        score += 1.0 * q_relevance

    # --- Dimension 3: Entity overlap ---
    # Extract potential entities (capitalized words in question and subquery)
    entity_pattern = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b'
    q_entities = set(re.findall(entity_pattern, original_question))
    if inferred_subquery:
        q_entities |= set(re.findall(entity_pattern, inferred_subquery))
    q_entities_lower = set(e.lower() for e in q_entities)

    entity_hits = 0
    for entity in q_entities_lower:
        if entity in doc_text:
            entity_hits += 1
    if q_entities_lower:
        score += 1.5 * (entity_hits / len(q_entities_lower))

    # --- Dimension 4: Novelty (not redundant with history) ---
    history_text = " ".join(history_thoughts).lower() if history_thoughts else ""
    history_words = set(re.findall(r'\b[a-z]{3,}\b', history_text))

    if doc_words and history_words:
        # New information = doc words NOT in history
        novel_words = doc_words - history_words
        novelty = len(novel_words) / len(doc_words) if doc_words else 0
        score += 0.5 * novelty

    # --- Dimension 5: Title entity match (strong signal for HotpotQA) ---
    doc_title_norm = normalize_title(doc.get("title", ""))
    # Check if any gold-relevant entity appears in doc title
    for entity in q_entities_lower:
        entity_words = entity.split()
        if all(w in doc_title_norm for w in entity_words):
            score += 1.0
            break

    return score


def select_top3_sapr_e(
    original_question: str,
    history_thoughts: List[str],
    inferred_subquery: str,
    docs: List[dict],
) -> List[dict]:
    """Selector 3: SAPR-E state-aware scorer."""
    scored = [
        (sapr_e_state_aware_score(original_question, history_thoughts, inferred_subquery, d), d)
        for d in docs
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:3]]


def compute_metrics(selected: List[dict], gold_titles: set) -> dict:
    """Compute metrics for a single selection."""
    selected_titles = set(normalize_title(d.get("title", "")) for d in selected)

    if not gold_titles:
        return {"hit@3": None, "recall@3": None, "noise_rate": None, "n_gold_found": 0, "n_selected": len(selected)}

    hits = selected_titles & gold_titles
    return {
        "hit@3": 1.0 if hits else 0.0,
        "recall@3": len(hits) / len(gold_titles) if gold_titles else 0.0,
        "noise_rate": 1.0 - (len(hits) / len(selected)) if selected else 0.0,
        "n_gold_found": len(hits),
        "n_gold_total": len(gold_titles),
        "n_selected": len(selected),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to evidence_decision_points.jsonl")
    parser.add_argument("--output_dir", required=True, help="Output directory for metrics and badcases")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    lines = open(args.input).readlines()
    data = [json.loads(l) for l in lines]

    # Separate retrieval steps and summaries
    retrieval_steps = [d for d in data if d.get("step", -1) > 0]
    summaries = {d["item_id"]: d for d in data if d.get("step") == -1}

    print("=" * 70)
    print("SAPR-E v0 Evidence Selector Comparison")
    print("=" * 70)
    print("Input: {}".format(args.input))
    print("Retrieval steps: {}".format(len(retrieval_steps)))
    print("Summaries: {}".format(len(summaries)))

    # Group retrieval steps by item_id
    item_steps = defaultdict(list)
    for d in retrieval_steps:
        item_steps[d["item_id"]].append(d)

    print("Unique items: {}".format(len(item_steps)))
    print()

    # Run selectors
    selectors = {
        "retriever_top3": lambda step: select_top3_by_score(step["retrieval_top10"]),
        "static_qd": lambda step: select_top3_static(step["original_question"], step["retrieval_top10"]),
        "sapr_e_v0": lambda step: select_top3_sapr_e(
            step["original_question"],
            step.get("history_thoughts", []),
            step.get("inferred_subquery", step["original_question"]),
            step["retrieval_top10"],
        ),
    }

    all_results = {name: {"per_step": [], "per_item": {}} for name in selectors}
    badcases = []

    for item_id, steps in item_steps.items():
        gold_titles = set()
        for s in steps:
            gold_titles |= get_gold_titles(s.get("supporting_facts", {}))

        for sel_name, sel_fn in selectors.items():
            item_hits = []
            for step in steps:
                selected = sel_fn(step)
                metrics = compute_metrics(selected, gold_titles)
                metrics["item_id"] = item_id
                metrics["step"] = step["step"]
                metrics["selector"] = sel_name
                all_results[sel_name]["per_step"].append(metrics)
                item_hits.append(metrics["hit@3"] or 0)

            # Per-item: any hit across any step
            all_results[sel_name]["per_item"][item_id] = {
                "any_hit": max(item_hits) if item_hits else 0,
                "avg_recall": sum(m["recall@3"] or 0 for m in all_results[sel_name]["per_step"]
                                  if m["item_id"] == item_id) / max(len(steps), 1),
            }

            # Badcases: item where retriever hit but SAPR-E didn't, or vice versa
            if sel_name == "sapr_e_v0":
                retr_any = all_results["retriever_top3"]["per_item"].get(item_id, {}).get("any_hit", 0)
                sapr_any = all_results["sapr_e_v0"]["per_item"].get(item_id, {}).get("any_hit", 0)
                if retr_any != sapr_any:
                    badcases.append({
                        "item_id": item_id,
                        "retriever_hit": retr_any,
                        "sapr_e_hit": sapr_any,
                        "gold_titles": list(gold_titles),
                        "question": steps[0]["original_question"] if steps else "",
                    })

    # Aggregate
    print("{:<20s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Selector", "Hit@3", "Recall@3", "NoiseRate", "ItemHit%"))
    print("-" * 65)

    aggregate = {}
    for sel_name in selectors:
        steps = all_results[sel_name]["per_step"]
        hit_vals = [m["hit@3"] for m in steps if m["hit@3"] is not None]
        recall_vals = [m["recall@3"] for m in steps if m["recall@3"] is not None]
        noise_vals = [m["noise_rate"] for m in steps if m["noise_rate"] is not None]

        item_hits = [v["any_hit"] for v in all_results[sel_name]["per_item"].values()]
        item_hit_pct = sum(item_hits) / len(item_hits) * 100 if item_hits else 0

        agg = {
            "selector": sel_name,
            "n_steps": len(steps),
            "hit@3_mean": round(sum(hit_vals) / len(hit_vals), 4) if hit_vals else None,
            "recall@3_mean": round(sum(recall_vals) / len(recall_vals), 4) if recall_vals else None,
            "noise_rate_mean": round(sum(noise_vals) / len(noise_vals), 4) if noise_vals else None,
            "item_hit_pct": round(item_hit_pct, 1),
            "n_items": len(all_results[sel_name]["per_item"]),
        }
        aggregate[sel_name] = agg

        print("{:<20s} {:>10.4f} {:>10.4f} {:>10.4f} {:>9.1f}%".format(
            sel_name,
            agg["hit@3_mean"] or 0,
            agg["recall@3_mean"] or 0,
            agg["noise_rate_mean"] or 0,
            agg["item_hit_pct"],
        ))

    # EM from summaries
    if summaries:
        em_vals = [s["em"] for s in summaries.values()]
        print("\nBaseline EM: {:.4f} ({}/{})".format(sum(em_vals)/len(em_vals), int(sum(em_vals)), len(em_vals)))

    # Write metrics.json
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "label": "debug_result",
            "aggregate": aggregate,
            "n_items": len(item_steps),
            "n_retrieval_steps": len(retrieval_steps),
        }, f, indent=2, ensure_ascii=False)
    print("\nMetrics: {}".format(metrics_path))

    # Write badcases
    badcases_path = os.path.join(args.output_dir, "badcases.jsonl")
    with open(badcases_path, "w") as f:
        for bc in badcases:
            f.write(json.dumps(bc, ensure_ascii=False) + "\n")
    print("Badcases: {} ({} items)".format(badcases_path, len(badcases)))

    # Write comparison table
    table_path = os.path.join(args.output_dir, "..", "..", "04_experiments", "tables", "20260530_sapr_evidence_v0_comparison_table.md")
    os.makedirs(os.path.dirname(table_path), exist_ok=True)

    with open(table_path, "w") as f:
        f.write("# SAPR-E v0 Evidence Selector Comparison\n\n")
        f.write("**Date**: 2026-05-30\n")
        f.write("**Label**: `debug_result`\n")
        f.write("**Dataset**: HotpotQA dev, 30 examples\n\n")
        f.write("| Selector | Hit@3 | Recall@3 | Noise Rate | Item Hit% |\n")
        f.write("|----------|------:|---------:|-----------:|----------:|\n")
        for sel_name in selectors:
            a = aggregate[sel_name]
            f.write("| {} | {:.4f} | {:.4f} | {:.4f} | {:.1f}% |\n".format(
                sel_name,
                a["hit@3_mean"] or 0,
                a["recall@3_mean"] or 0,
                a["noise_rate_mean"] or 0,
                a["item_hit_pct"],
            ))
        if badcases:
            f.write("\n## Badcases ({} items)\n\n".format(len(badcases)))
            for bc in badcases:
                f.write("- **{}**: retr={}, sapr={}\n".format(
                    bc["item_id"], bc["retriever_hit"], bc["sapr_e_hit"]))
    print("Table: {}".format(table_path))

    # SAPR-E signal check
    retr_hit = aggregate.get("retriever_top3", {}).get("hit@3_mean", 0) or 0
    sapr_hit = aggregate.get("sapr_e_v0", {}).get("hit@3_mean", 0) or 0
    retr_recall = aggregate.get("retriever_top3", {}).get("recall@3_mean", 0) or 0
    sapr_recall = aggregate.get("sapr_e_v0", {}).get("recall@3_mean", 0) or 0

    print("\n" + "=" * 70)
    if sapr_hit > retr_hit or sapr_recall > retr_recall:
        print("DIRECTIONAL SIGNAL DETECTED ✅")
        print("  SAPR-E hit@3: {:.4f} vs Retriever: {:.4f} (delta: {:+.4f})".format(
            sapr_hit, retr_hit, sapr_hit - retr_hit))
        print("  SAPR-E recall@3: {:.4f} vs Retriever: {:.4f} (delta: {:+.4f})".format(
            sapr_recall, retr_recall, sapr_recall - retr_recall))
    else:
        print("NO DIRECTIONAL SIGNAL ❌")
        print("  SAPR-E hit@3: {:.4f} vs Retriever: {:.4f} (delta: {:+.4f})".format(
            sapr_hit, retr_hit, sapr_hit - retr_hit))
        print("  SAPR-E recall@3: {:.4f} vs Retriever: {:.4f} (delta: {:+.4f})".format(
            sapr_recall, retr_recall, sapr_recall - retr_recall))
    print("=" * 70)

    return aggregate


if __name__ == "__main__":
    main()
