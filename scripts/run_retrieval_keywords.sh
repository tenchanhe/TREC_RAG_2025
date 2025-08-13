#!/usr/bin/env bash

# --- 設定基本變數 ---
QUERY_FILE="data/topics/top10_topic.txt"
CONTENT_JSONL_FILE="data/segment/dense_10q.jsonl"
KEYWORD_JSONL_FILE="data/kg/keywords.jsonl"
CACHE_DIR="cache/segment_keywords_embeddings"
MODEL_NAME="pretrained_model/sentence-transformers/all-MiniLM-L6-v2/"

# --- 設定輸出目錄 ---
OUTPUT_DIR="runs/retrieval/segment_keywords"
mkdir -p "$OUTPUT_DIR"

# --- 定義要測試的權重 ---
CONTENT_WEIGHTS=(0.5 1.0 1.5)
KEYWORD_WEIGHTS=(0.5 1.0 1.5)

# --- 迴圈執行不同權重的檢索 ---
echo "開始進行多權重檢索實驗..."

for cw in "${CONTENT_WEIGHTS[@]}"; do
  for kw in "${KEYWORD_WEIGHTS[@]}"; do
    # 動態生成輸出檔案名稱
    OUTPUT_FILE="$OUTPUT_DIR/run_cw_${cw}_kw_${kw}.txt"
    CACHE_DIR="$CACHE_DIR/cw_${cw}_kw_${kw}"
    
    echo "-----------------------------------------------------"
    echo "執行中: Content Weight = $cw, Keyword Weight = $kw"
    echo "結果將儲存至: $OUTPUT_FILE"
    echo "-----------------------------------------------------"
    
    # 執行 Python 腳本
    python src/retrieval/dense_keyword.py \
      --query_file "$QUERY_FILE" \
      --content_jsonl_file "$CONTENT_JSONL_FILE" \
      --keyword_jsonl_file "$KEYWORD_JSONL_FILE" \
      --cache_dir "$CACHE_DIR" \
      --model_name "$MODEL_NAME" \
      --output_ranking_file "$OUTPUT_FILE" \
      --content_weight "$cw" \
      --keyword_weight "$kw"
      
    if [ $? -ne 0 ]; then
      echo "執行失敗: Content Weight = $cw, Keyword Weight = $kw"
      exit 1
    fi
  done
done

echo "所有實驗已完成！"
echo "結果檔案已儲存在 '$OUTPUT_DIR' 目錄中。"