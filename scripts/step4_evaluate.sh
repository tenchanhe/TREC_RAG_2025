#!/bin/bash

# 設定目錄
QRELS_FILE="data/qrels/qrels.rag24.test-umbrela-all.txt"
RUN_FILE="runs/retrieval/bm25_run.txt"
EVAL_DIR="runs/evaluation"
LOG_DIR="logs/evaluation"

# 創建必要的目錄
mkdir -p $EVAL_DIR
mkdir -p $LOG_DIR

# 記錄開始時間
echo "開始評估: $(date)" | tee -a $LOG_DIR/evaluation.log

# 使用 pytrec_eval 進行評估
python -c "
import pytrec_eval
import json

# 讀取 qrels 和 run 檔案
with open('$QRELS_FILE', 'r') as f:
    qrels = pytrec_eval.parse_qrel(f)
with open('$RUN_FILE', 'r') as f:
    run = pytrec_eval.parse_run(f)

# 設定評估指標
metrics = {
    'map': 'map',
    'ndcg_cut_10': 'ndcg_cut_10',
    'recall_1000': 'recall_1000'
}

# 進行評估
evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
results = evaluator.evaluate(run)

# 計算平均值
avg_results = {}
for metric in metrics:
    avg_results[metric] = sum(result[metric] for result in results.values()) / len(results)

# 輸出結果
print('評估結果:')
for metric, value in avg_results.items():
    print(f'{metric}: {value:.4f}')
" | tee -a $LOG_DIR/evaluation.log

# 檢查評估是否成功
if [ $? -eq 0 ]; then
    echo "評估完成: $(date)" | tee -a $LOG_DIR/evaluation.log
else
    echo "評估失敗: $(date)" | tee -a $LOG_DIR/evaluation.log
    exit 1
fi 