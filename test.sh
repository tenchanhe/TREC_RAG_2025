# python -m pyserini.index.lucene \
#     --collection JsonCollection \
#     --input test_data \
#     --index test_index \
#     --generator DefaultLuceneDocumentGenerator \
#     --threads 8 \
#     --storePositions \
#     --storeDocvectors \
#     --storeRaw

# TOPICS_FILE="/tmp2/TREC_RAG2025/topics/topics.rag24.test.txt"
# INDEX_DIR="indexes/bm25"
# RUN_DIR="runs/retrieval"

# python -m pyserini.search.lucene \
#     --flat \
#     --index $INDEX_DIR \
#     --encoder sentence-transformers/all-MiniLM-L6-v2 \
#     --topics $TOPICS_FILE \
#     --output $RUN_DIR/dense.txt \
#     --dense \
#     --hits 1000

# python -m pyserini.encode \
#   input   --corpus test_data/ \
#           --fields text \
#   output  --embeddings indexes/test \
#   encoder --encoder castorini/unicoil-msmarco-passage \
#           --fields text \
#           --batch 32 \
#           --fp16

# python -m pyserini.index.faiss \
#   --input indexes/test \
#   --output runs/test \
#   --hnsw \
#   --pq

  python -m pyserini.search.lucene \
  --index indexes/test \
  --topics msmarco-v2-passage-dev \
  --encoder castorini/unicoil-noexp-msmarco-passage \
  --output runs/run.msmarco-v2-passage-unicoil-noexp-0shot.dev.txt \
  --batch 144 --threads 36 \
  --hits 1000 \
  --impact