#!/usr/bin/env bash

# --- 定義要測試的權重 ---
CONTENT_WEIGHTS=(0.5 1.0 1.5)
KEYWORD_WEIGHTS=(0.5 1.0 1.5)

# --- 迴圈執行不同權重的檢索 ---
echo "開始進行多權重檢索實驗..."

for cw in "${CONTENT_WEIGHTS[@]}"; do
  for kw in "${KEYWORD_WEIGHTS[@]}"; do
    
    # 執行 Python 腳本
    python3 -m src.retireval.dense_keyword_sentence \
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