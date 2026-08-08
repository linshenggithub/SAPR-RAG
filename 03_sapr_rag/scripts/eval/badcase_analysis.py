import argparse
import json
import os
import hashlib
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

sys.path.insert(0, str(Path(__file__).parent))
from ceiling_diagnostic import (
    load_dataset, extract_oracle_context, SYSTEM_PROMPT,
    normalize_answer, em_score, f1_score, cover_em_score
)

ERROR_CATEGORIES = [
    ("reasoning_chain_error", "推理链错误：多跳推理中逻辑推导错误，导致答案错误"),
    ("entity_confusion", "实体混淆：把相似名称/描述的实体搞混了"),
    ("context_misinterpretation", "上下文误读：从上下文中提取信息时理解错误"),
    ("parametric_knowledge_conflict", "参数知识冲突：模型依赖自身知识而非上下文，导致与上下文信息冲突"),
    ("answer_format_issue", "答案表述问题：事实正确但表述形式不同（如别名、缩写、语序不同）"),
    ("numeric_calculation_error", "数字/日期计算错误：涉及数量、日期、年龄等计算时出错"),
    ("missing_key_info", "遗漏关键信息：上下文中有答案所需信息，但模型没注意到/没用到"),
    ("multi_hop_bridge_error", "桥接实体错误：多跳中中间的桥接实体找错了"),
    ("other", "其他：以上分类都不适用"),
]

CLASSIFICATION_PROMPT_TEMPLATE = """你是一个多跳 QA 错误分析专家。请分析以下问答样本，判断模型答错的主要原因。

【问题】
{question}

【标准答案】
{gold_answer}

【模型回答】
{model_answer}

【上下文（模型可见）】
{context}

【错误分类体系】
{categories}

请按照以下步骤分析：
1. 先确认模型是否真的答错了（如果模型回答事实正确但表述不同，属于 answer_format_issue）
2. 如果确实答错，判断最主要的错误原因
3. 给出简要理由

请严格以 JSON 格式输出：
{{
  "is_truly_wrong": true/false,
  "error_category": "错误类型的 key",
  "rationale": "简短的分析理由（1-2 句话）"
}}
"""


def build_categories_text():
    lines = []
    for i, (key, desc) in enumerate(ERROR_CATEGORIES, 1):
        lines.append(f"{i}. `{key}`: {desc}")
    return "\n".join(lines)


def cache_key(model, dataset, qid, question, model_answer):
    s = f"error_classify|{model}|{dataset}|{qid}|{question}|{model_answer}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


def classify_error(api_key, model, question, gold_answer, model_answer, context):
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    categories_text = build_categories_text()
    user_prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer if isinstance(gold_answer, str) else ", ".join(gold_answer),
        model_answer=model_answer,
        context=context[:3000],
        categories=categories_text,
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个严谨的错误分析专家，输出严格的 JSON。"},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            return result
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {"error_category": "classification_failed", "rationale": f"API error: {str(e)}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["hotpotqa", "2wikimultihopqa", "musique"])
    parser.add_argument("--model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="data/eval_results/ceiling")
    parser.add_argument("--cache_dir", type=str, default="data/ceiling_cache")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    # 加载数据集
    samples = load_dataset(args.dataset)

    # 加载 oracle 结果，找出 badcase
    results_path = Path(args.output_dir) / f"{args.model}_{args.dataset}_oracle_results.jsonl"
    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}")
        print("Please run ceiling_diagnostic.py first.")
        sys.exit(1)

    results = {}
    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            results[r["id"]] = r

    # 识别 badcase（EM = 0 的样本）
    badcases = []
    for s in samples:
        qid = s["_id"] if args.dataset == "hotpotqa" else s["id"]
        if qid not in results:
            continue
        pred = results[qid]["prediction"]
        em = results[qid].get("em", em_score(pred, results[qid].get("gold", [])))
        if em == 0:
            gold_key = "answer" if args.dataset == "hotpotqa" else "golden_answers"
            gold = s[gold_key]
            if isinstance(gold, str):
                gold = [gold]
            badcases.append({
                "qid": qid,
                "question": s["question"],
                "gold_answer": gold,
                "model_answer": pred,
                "sample": s,
            })

    print(f"=== Badcase 错误分类 ===")
    print(f"  dataset: {args.dataset}")
    print(f"  oracle model: {args.model}")
    print(f"  total samples: {len(samples)}")
    print(f"  badcases (EM=0): {len(badcases)}")
    print(f"  error_rate: {len(badcases)/len(samples):.4f}")
    print(f"  classify with: {args.model}")
    print(f"  max_samples: {min(args.max_samples, len(badcases))}")
    print()

    # 限制样本数
    badcases = badcases[:args.max_samples]

    # 加载缓存
    cache_file = Path(args.cache_dir) / f"error_classify_{args.model}_{args.dataset}_cache.jsonl"
    cache = {}
    if cache_file.exists():
        with open(cache_file) as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["qid"]] = entry

    print(f"  cached: {sum(1 for b in badcases if b['qid'] in cache)}")
    print(f"  to_classify: {sum(1 for b in badcases if b['qid'] not in cache)}")
    print()

    # 分类
    results_list = []
    lock = __import__("threading").Lock()

    def process_one(bc):
        qid = bc["qid"]
        if qid in cache:
            return cache[qid]

        context = extract_oracle_context(bc["sample"], args.dataset)
        result = classify_error(
            api_key, args.model,
            bc["question"], bc["gold_answer"], bc["model_answer"], context
        )
        result["qid"] = qid
        result["question"] = bc["question"]
        result["gold_answer"] = bc["gold_answer"]
        result["model_answer"] = bc["model_answer"]

        with lock:
            with open(cache_file, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        return result

    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_one, bc): bc for bc in badcases}
        for future in as_completed(futures):
            result = future.result()
            results_list.append(result)
            completed += 1
            if completed % 20 == 0 or completed == len(badcases):
                # 实时统计
                cat_counts = {}
                for r in results_list:
                    cat = r.get("error_category", "unknown")
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
                print(f"[{completed}/{len(badcases)}] 分类进度")

    # 统计分布
    print()
    print("=== 错误类型分布 ===")
    print()

    cat_counts = {}
    truly_wrong_count = 0
    for r in results_list:
        cat = r.get("error_category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        if r.get("is_truly_wrong", True):
            truly_wrong_count += 1

    sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])
    total = len(results_list)

    print(f"{'错误类型':<35} {'数量':>6} {'占比':>8} {'描述'}")
    print("-" * 100)
    for cat, count in sorted_cats:
        desc = next((d for k, d in ERROR_CATEGORIES if k == cat), cat)
        pct = count / total * 100
        print(f"{cat:<35} {count:>6} {pct:>7.1f}%  {desc}")

    print()
    print(f"Truly wrong: {truly_wrong_count}/{total} ({truly_wrong_count/total*100:.1f}%)")
    print(f"Format issue (事实对但表述错): {total - truly_wrong_count}/{total} ({(total-truly_wrong_count)/total*100:.1f}%)")

    # 保存完整结果
    out_file = Path(args.output_dir) / f"error_analysis_{args.model}_{args.dataset}.json"
    with open(out_file, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "oracle_model": args.model,
            "classifier_model": args.model,
            "total_badcases_analyzed": total,
            "error_distribution": {k: v for k, v in sorted_cats},
            "truly_wrong": truly_wrong_count,
            "format_issue": total - truly_wrong_count,
            "samples": results_list,
        }, f, ensure_ascii=False, indent=2)

    print()
    print(f"完整结果已保存: {out_file}")


if __name__ == "__main__":
    main()
