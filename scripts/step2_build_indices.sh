#!/bin/bash


# 設定目錄
CORPUS_DIR="data/corpus/processed"
INDEX_DIR="indexes/bm25"
LOG_DIR="logs/indexing"

# 創建必要的目錄
mkdir -p $INDEX_DIR
mkdir -p $LOG_DIR

# 記錄開始時間
echo "開始建立索引: $(date)" | tee -a $LOG_DIR/indexing.log

# 使用 Pyserini 建立 BM25 索引
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input $CORPUS_DIR \
    --index $INDEX_DIR \
    --generator DefaultLuceneDocumentGenerator \
    --threads 8 \
    --storePositions \
    --storeDocvectors \
    --storeRaw \
    2>&1 | tee -a $LOG_DIR/indexing.log

# 檢查索引是否成功建立
if [ $? -eq 0 ]; then
    echo "索引建立成功: $(date)" | tee -a $LOG_DIR/indexing.log
    
    # 顯示索引統計資訊
    echo "索引統計資訊:" | tee -a $LOG_DIR/indexing.log
    python -m pyserini.index.lucene \
        --index $INDEX_DIR \
        --stats \
        2>&1 | tee -a $LOG_DIR/indexing.log
else
    echo "索引建立失敗: $(date)" | tee -a $LOG_DIR/indexing.log
    exit 1
fi

# 記錄結束時間
echo "索引建立完成: $(date)" | tee -a $LOG_DIR/indexing.log
