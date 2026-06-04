# Corrected Case Findings

## Scope

- Baseline: `/home/mayi/SAPR-RAG/04_experiments/logs/20260604_reasonrag_baseline_30_service_gpu0/baseline/hotpotqa_2026_06_04_16_10_reasonrag_baseline/intermediate_data.json`
- SAPR-E v0: `/home/mayi/SAPR-RAG/04_experiments/logs/20260604_sapr_e_v0_30_service_gpu0/sapr_e_minimal_rerank/hotpotqa_2026_06_04_16_18_sapr_e_minimal_rerank/intermediate_data.json`
- Analysis output: `/home/mayi/SAPR-RAG/04_experiments/metrics/20260604_sapr_e_v0_30_case_analysis_corrected`

## Corrected Metrics

The earlier "gold document hit" analysis used the first line of `contents` as a title proxy. This is not always reliable. Some retrieved documents have an unrelated first line while their body contains the gold title or gold answer.

Corrected analysis therefore tracks three signals:

- First-line title hit: parsed first line of `contents` matches `supporting_facts.title`.
- Content gold-title hit: full retrieved document text contains a gold supporting title.
- Content gold-answer hit: full retrieved document text contains the gold answer.

| Metric | Baseline | SAPR-E v0 |
|---|---:|---:|
| EM count | 9 / 30 | 9 / 30 |
| Avg F1 | 0.4399 | 0.4126 |
| First-line gold-title hit steps | 23 | 29 |
| Content gold-title hit steps | 30 | 33 |
| Content gold-answer hit steps | 22 | 19 |

## Main Finding

After correction, SAPR-E v0 still retrieves slightly more documents whose content contains gold supporting titles, but it retrieves fewer steps whose content contains the gold answer. This better explains why v0 does not improve final EM/F1 on 30 examples.

## Key Cases

### dev_10 Regression

- Gold answer: `Kansas Song`
- Baseline answer: `Kansas Song`
- SAPR-E v0 answer: `I'm a Jayhawk`

First-line title matching previously underrated baseline. Baseline retrieved documents whose first lines were unrelated, for example `Steven Ronald Jensen`, but their body contained `Kansas Song` and `University of Kansas`.

SAPR-E v0 also retrieved content containing `Kansas Song`, but selected more strongly entity/topic-related distractors such as `University of Kansas` and `I'm a Jayhawk`. The model followed the distractor and answered incorrectly.

### dev_20 Regression

- Gold answer: `Pedro Rodriguez`
- Baseline answer: `Pedro Rodriguez`
- SAPR-E v0 answer: `Manuel Fittipaldi`

SAPR-E v0 retrieved more content containing the gold title and answer than baseline in early steps, but it also emphasized `Force India`, `Sergio Perez`, and broad `Formula One drivers from Mexico` context. The selected evidence became noisier for answer extraction.

### dev_3 Improvement

- Gold answer: `no`
- Baseline answer: empty
- SAPR-E v0 answer: `No`

SAPR-E v0 mainly improved ordering around `Esma Sultan Mansion`, `Laleli Mosque`, and neighborhood-related pages. This is a cleaner positive example for reranking.

### dev_13 Improvement

- Gold answer: `no`
- Baseline answer: `yes`
- SAPR-E v0 answer: `No`

This is not strong evidence for better document reranking: v0 reached the answer with fewer/no document-analysis steps in the aligned comparison, so the gain is partly due to trajectory change.

## Implication

The current v0 scorer has retrieval signal but is not answer-aligned enough. It tends to prefer topic/entity-related documents, and those documents can be strong distractors. The next v0 fix should target answer-supporting evidence, not only title/entity/query overlap.
