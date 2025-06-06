#!/bin/bash

# 設定目錄
TOPICS_FILE="data/topics/topics.rag24.test.txt"
INDEX_DIR="indexes/bm25"
RUN_DIR="runs/retrieval"
LOG_DIR="logs/retrieval"

# 創建必要的目錄
mkdir -p $RUN_DIR
mkdir -p $LOG_DIR

# 記錄開始時間
echo "開始檢索: $(date)" | tee -a $LOG_DIR/retrieval.log

# 使用 Pyserini 進行檢索
python -m pyserini.search.lucene \
    --index $INDEX_DIR \
    --topics $TOPICS_FILE \
    --output $RUN_DIR/bm25_run.txt \
    --bm25 \
    --hits 1000 \
    2>&1 | tee -a $LOG_DIR/retrieval.log

# 檢查檢索是否成功
if [ $? -eq 0 ]; then
    echo "檢索完成: $(date)" | tee -a $LOG_DIR/retrieval.log
else
    echo "檢索失敗: $(date)" | tee -a $LOG_DIR/retrieval.log
    exit 1
fi 