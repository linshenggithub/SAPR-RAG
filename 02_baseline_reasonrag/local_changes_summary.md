# ReasonRAG Local Changes Summary

Local ReasonRAG path:

```text
/home/mayi/RAG/ReasonRAG
```

Current local work includes several exploratory components:

1. HTTP retriever service
   - `retrieval_server.py`
   - `start_retriever.sh`
   - `retrieval_service.sh`

2. Thesis automation scaffold
   - `thesis_auto/analyzer.py`
   - `thesis_auto/http_retriever.py`
   - `thesis_auto/official_reasonrag_pipeline.py`
   - `thesis_auto/official_retrieval_rewrite_pipeline.py`
   - `thesis_auto/modules/retrieval_filter/`
   - `thesis_auto/modules/verifier/`

3. Exploratory experiment configs
   - official ReasonRAG baseline configs
   - retrieval rewrite configs
   - verifier configs
   - retrieval filter configs

These components should not be blindly copied into this repository. Stable code can later be promoted into `03_sapr_rag/` or referenced from `02_baseline_reasonrag/scripts/`.

