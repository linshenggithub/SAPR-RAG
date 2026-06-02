python -m flashrag.retriever.index_builder \
  --retrieval_method bge \
  --model_path /BAAI/bge-base-en-v1.5 \
  --corpus_path indexes/wiki18.jsonl \
  --save_dir indexes/ \
  --use_fp16 \
  --max_length 512 \
  --batch_size 256 \
  --pooling_method mean \
  --faiss_type Flat