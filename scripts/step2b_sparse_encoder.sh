#!/bin/bash


# 設定目錄
CORPUS_DIR="data/corpus/processed/"
INDEX_DIR="indexes/sparse_encode"
LOG_DIR="logs/indexing"

# 創建必要的目錄
mkdir -p $INDEX_DIR
mkdir -p $LOG_DIR

# 記錄開始時間
echo "開始建立索引: $(date)" | tee -a $LOG_DIR/indexing.log

python -m pyserini.encode \
  input   --corpus $CORPUS_DIR \
          --fields text \
  output  --embeddings $INDEX_DIR \
  encoder --encoder castorini/unicoil-msmarco-passage \
          --fields text \
          --batch 32 \
          --fp16 \
          2>&1 | tee -a $LOG_DIR/indexing.log

python -m pyserini.encode \
  input   --corpus $CORPUS_DIR \
          --fields text \
          --delimiter "\n" \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings $INDEX_DIR \
          --to-faiss \
  encoder --encoder castorini/tct_colbert-v2-hnp-msmarco \
          --fields text \
          --batch 32 \
          --fp16