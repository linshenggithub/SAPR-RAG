#!/usr/bin/env python3
"""
SAPR-E v0 Evidence Selector Comparison — 6-way with ablations.

Selectors:
  1. retriever_top3     — baseline: top-3 by retrieval score
  2. static_qd          — keyword overlap with original question
  3. sapr_e_v0          — full 5-dim: subquery(×2) + question(×1) + entity(×1.5) + novelty(×0.5) + title_entity(×1)
  4. sapr_e_no_hist     — ablation: remove Dim4 (novelty/history)
  5. sapr_e_no_title    — ablation: remove Dim5 (title entity match)
  6. sapr_e_no_subquery — ablation: remove Dim1 (subquery relevance)

Usage:
  python compare_6way_selector_ablation.py --input <reretrieved.jsonl> --output_dir <dir>
"""

import json
import re
import os
import argparse
from collections import defaultdict
from typing import List, Dict, Set, Tuple


def normalize_title(title: str) -> str:
    t = title.strip().lower()
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def get_gold_titles(supporting_facts: dict) -> Set[str]:
    return set(normalize_title(t) for t in supporting_facts.get("title", []))


# ── Selector 1: Retriever top-3 ─────────────────────────────────
def select_retriever_top3(docs: List[dict]) -> List[dict]:
    return sorted(docs, key=lambda d: d.get("score", 0), reverse=True)[:3]


# ── Selector 2: Static question-doc ─────────────────────────────
def static_qd_score(question: str, doc: dict) -> float:
    q_words = set(re.findall(r'\b[a-z]{3,}\b', question.lower()))
    doc_text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
    doc_words = set(re.findall(r'\b[a-z]{3,}\b', doc_text))
    return len(q_words & doc_words) / len(q_words) if q_words else 0.0


def select_static_qd(question: str, docs: List[dict]) -> List[dict]:
    scored = [(static_qd_score(question, d), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:3]]


# ── Shared scoring dimensions ───────────────────────────────────
def _extract_entities(texts: List[str]) -> Set[str]:
    ep = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b'
    ents = set()
    for t in texts:
        ents |= set(re.findall(ep, t))
    return set(e.lower() for e in ents)


def _word_set(text: str) -> Set[str]:
    return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))


# ── Selectors 3-6: SAPR-E variants ──────────────────────────────
def sapr_e_score(
    question: str,
    history_thoughts: List[str],
    subquery: str,
    doc: dict,
    *,
    use_subquery: bool = True,
    use_question: bool = True,
    use_entity: bool = True,
    use_history: bool = True,
    use_title_entity: bool = True,
) -> float:
    """Configurable SAPR-E scorer. Set flags to False for ablation."""
    score = 0.0
    doc_text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
    doc_words = _word_set(doc_text)

    # Dim 1: subquery relevance (weight 2.0)
    if use_subquery and subquery:
        subq_words = _word_set(subquery)
        if subq_words:
            score += 2.0 * len(subq_words & doc_words) / len(subq_words)

    # Dim 2: question relevance (weight 1.0)
    if use_question:
        q_words = _word_set(question)
        if q_words:
            score += 1.0 * len(q_words & doc_words) / len(q_words)

    # Dim 3: entity overlap (weight 1.5)
    if use_entity:
        ent_sources = [question]
        if subquery:
            ent_sources.append(subquery)
        entities = _extract_entities(ent_sources)
        if entities:
            hits = sum(1 for e in entities if e in doc_text)
            score += 1.5 * hits / len(entities)

    # Dim 4: novelty vs history (weight 0.5)
    if use_history and history_thoughts:
        history_words = _word_set(" ".join(history_thoughts))
        if doc_words and history_words:
            novel = doc_words - history_words
            score += 0.5 * len(novel) / len(doc_words)

    # Dim 5: title entity match (weight 1.0)
    if use_title_entity:
        ent_sources = [question]
        if subquery:
            ent_sources.append(subquery)
        entities = _extract_entities(ent_sources)
        doc_title_norm = normalize_title(doc.get("title", ""))
        for e in entities:
            if all(w in doc_title_norm for w in e.split()):
                score += 1.0
                break

    return score


def _make_sapr_selector(*, use_subquery=True, use_question=True,
                         use_entity=True, use_history=True, use_title_entity=True):
    """Return a selector function with the given ablation flags."""
    def selector(step):
        return _select_top3_sapr(
            step["original_question"],
            step.get("history_thoughts", []),
            step.get("inferred_subquery", step["original_question"]),
            step["retrieval_top10"],
            use_subquery=use_subquery,
            use_question=use_question,
            use_entity=use_entity,
            use_history=use_history,
            use_title_entity=use_title_entity,
        )
    return selector


def _select_top3_sapr(question, history, subquery, docs, **flags):
    scored = [(sapr_e_score(question, history, subquery, d, **flags), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:3]]


# ── Metrics ─────────────────────────────────────────────────────
def compute_metrics(selected: List[dict], gold_titles: Set[str]) -> dict:
    sel_titles = set(normalize_title(d.get("title", "")) for d in selected)
    if not gold_titles:
        return {"hit@3": None, "recall@3": None, "noise_rate": None}
    hits = sel_titles & gold_titles
    return {
        "hit@3": 1.0 if hits else 0.0,
        "recall@3": len(hits) / len(gold_titles),
        "noise_rate": 1.0 - (len(hits) / len(selected)) if selected else 0.0,
    }


# ── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SAPR-E 6-way ablation comparison")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data = [json.loads(l) for l in open(args.input)]
    retrieval_steps = [d for d in data if d.get("step", -1) > 0]
    item_steps = defaultdict(list)
    for d in retrieval_steps:
        item_steps[d["item_id"]].append(d)

    # Define selectors
    selectors = {
        "retriever_top3": lambda s: select_retriever_top3(s["retrieval_top10"]),
        "static_qd": lambda s: select_static_qd(s["original_question"], s["retrieval_top10"]),
        "sapr_e_v0": _make_sapr_selector(),
        "sapr_e_no_hist": _make_sapr_selector(use_history=False),
        "sapr_e_no_title": _make_sapr_selector(use_title_entity=False),
        "sapr_e_no_subquery": _make_sapr_selector(use_subquery=False),
    }

    # Run comparison
    all_res = {n: {"per_step": [], "per_item": {}} for n in selectors}
    badcases = []

    for iid, steps in sorted(item_steps.items()):
        gold = set()
        for s in steps:
            gold |= get_gold_titles(s.get("supporting_facts", {}))

        for sn, sf in selectors.items():
            item_hits = []
            for step in steps:
                selected = sf(step)
                m = compute_metrics(selected, gold)
                m.update({"item_id": iid, "step": step["step"], "selector": sn})
                all_res[sn]["per_step"].append(m)
                item_hits.append(m["hit@3"] or 0)
            all_res[sn]["per_item"][iid] = {"any_hit": max(item_hits) if item_hits else 0}

        # Badcases: any selector differs from sapr_e_v0
        sapr_hit = all_res["sapr_e_v0"]["per_item"][iid]["any_hit"]
        for sn in ["retriever_top3", "static_qd"]:
            sn_hit = all_res[sn]["per_item"][iid]["any_hit"]
            if sapr_hit != sn_hit:
                badcases.append({
                    "item_id": iid, "sapr_e_v0": sapr_hit, sn: sn_hit,
                    "question": steps[0]["original_question"],
                    "gold_titles": sorted(gold),
                })

    # Aggregate
    print("=" * 75)
    print("SAPR-E v0 6-Way Ablation Comparison")
    print("=" * 75)
    print("{:<20s} {:>8s} {:>9s} {:>10s} {:>9s}".format(
        "Selector", "Hit@3", "Recall@3", "NoiseRate", "ItemHit%"))
    print("-" * 60)

    aggregate = {}
    for sn in selectors:
        steps = all_res[sn]["per_step"]
        hv = [m["hit@3"] for m in steps if m["hit@3"] is not None]
        rv = [m["recall@3"] for m in steps if m["recall@3"] is not None]
        nv = [m["noise_rate"] for m in steps if m["noise_rate"] is not None]
        ih = [v["any_hit"] for v in all_res[sn]["per_item"].values()]
        ihp = sum(ih) / len(ih) * 100 if ih else 0
        agg = {
            "selector": sn, "n_steps": len(steps),
            "hit@3_mean": round(sum(hv)/len(hv), 4) if hv else None,
            "recall@3_mean": round(sum(rv)/len(rv), 4) if rv else None,
            "noise_rate_mean": round(sum(nv)/len(nv), 4) if nv else None,
            "item_hit_pct": round(ihp, 1),
            "n_items": len(all_res[sn]["per_item"]),
        }
        aggregate[sn] = agg
        print("{:<20s} {:>8.4f} {:>9.4f} {:>10.4f} {:>8.1f}%".format(
            sn, agg["hit@3_mean"] or 0, agg["recall@3_mean"] or 0,
            agg["noise_rate_mean"] or 0, agg["item_hit_pct"]))

    # Deltas vs retriever baseline
    print("\n" + "=" * 75)
    print("Deltas vs retriever_top3 baseline:")
    baseline_hit = aggregate["retriever_top3"]["hit@3_mean"] or 0
    baseline_recall = aggregate["retriever_top3"]["recall@3_mean"] or 0
    baseline_item = aggregate["retriever_top3"]["item_hit_pct"] or 0
    for sn in selectors:
        if sn == "retriever_top3":
            continue
        a = aggregate[sn]
        dh = (a["hit@3_mean"] or 0) - baseline_hit
        dr = (a["recall@3_mean"] or 0) - baseline_recall
        di = (a["item_hit_pct"] or 0) - baseline_item
        print(f"  {sn:<20s}  hit@3 {dh:+.4f}  recall@3 {dr:+.4f}  itemHit {di:+.1f}pp")

    # Ablation deltas vs full SAPR-E
    print("\nAblation deltas vs sapr_e_v0 (full):")
    full_hit = aggregate["sapr_e_v0"]["hit@3_mean"] or 0
    full_recall = aggregate["sapr_e_v0"]["recall@3_mean"] or 0
    full_item = aggregate["sapr_e_v0"]["item_hit_pct"] or 0
    for sn in ["sapr_e_no_hist", "sapr_e_no_title", "sapr_e_no_subquery"]:
        a = aggregate[sn]
        dh = (a["hit@3_mean"] or 0) - full_hit
        dr = (a["recall@3_mean"] or 0) - full_recall
        di = (a["item_hit_pct"] or 0) - full_item
        dim = sn.replace("sapr_e_", "")
        print(f"  −{dim:<14s}  hit@3 {dh:+.4f}  recall@3 {dr:+.4f}  itemHit {di:+.1f}pp")

    # Signal check
    print("\n" + "=" * 75)
    if full_hit > baseline_hit:
        print("DIRECTIONAL SIGNAL: sapr_e_v0 > retriever_top3 ✅")
    else:
        print("NO DIRECTIONAL SIGNAL ❌")

    # Write metrics
    out = {
        "label": "debug_result",
        "aggregate": aggregate,
        "badcases": badcases,
        "n_items": len(item_steps),
        "n_retrieval_steps": len(retrieval_steps),
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics: {metrics_path}")

    # Write per-item table
    lines = ["# Per-Item 6-Way Ablation", ""]
    lines.append("| Item | Question | retr | static | v0 | no_hist | no_title | no_subq |")
    lines.append("|------|----------|:----:|:------:|:--:|:-------:|:--------:|:-------:|")
    for iid in sorted(item_steps.keys()):
        q = item_steps[iid][0]["original_question"][:40] + "..."
        mark = lambda sn: "✅" if all_res[sn]["per_item"][iid]["any_hit"] else "❌"
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            iid, q, mark("retriever_top3"), mark("static_qd"),
            mark("sapr_e_v0"), mark("sapr_e_no_hist"),
            mark("sapr_e_no_title"), mark("sapr_e_no_subquery")))
    tbl_path = os.path.join(args.output_dir, "per_item_table.md")
    with open(tbl_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Table: {tbl_path}")

    # Hit counts
    print("\nPer-selector item hit counts:")
    for sn in selectors:
        hits = sum(1 for v in all_res[sn]["per_item"].values() if v["any_hit"])
        print(f"  {sn}: {hits}/{len(item_steps)}")

    return aggregate


if __name__ == "__main__":
    main()
