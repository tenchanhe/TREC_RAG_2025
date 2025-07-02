import pytrec_eval
import json

# 讀取 qrels 和 run 檔案
with open('/tmp2/TREC_RAG2025/qrels/qrels.rag24.test-umbrela-all.txt', 'r') as f:
    qrels = pytrec_eval.parse_qrel(f)
with open('./runs/retrieval/bm25_run.txt', 'r') as f:
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