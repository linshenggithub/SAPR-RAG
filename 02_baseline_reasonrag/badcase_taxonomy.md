# Badcase Taxonomy

## Main Failure Types

1. Retrieval evidence problem
   - Relevant evidence is missing.
   - Gold evidence is ranked too low.
   - Retrieved documents are polluted by noisy entities.

2. Reasoning or relation error
   - The model retrieves partial evidence but fails to infer the correct relation.
   - Bridge entities are not connected into a complete evidence chain.

3. Process control error
   - The model stops too early.
   - The model answers without enough evidence.
   - The model repeatedly searches similar subqueries.

4. Query drift and retrieval coordination error
   - Generated subqueries are too broad, duplicated, or merged across hops.
   - Subquery generation loses key entities.

5. Evaluation alignment issue
   - The prediction may be semantically acceptable but mismatched with the exact gold answer format.

## Working Hypothesis

The two most important optimization targets are:

1. state-aware evidence utility modeling;
2. state-aware process reward for trajectory repair.

