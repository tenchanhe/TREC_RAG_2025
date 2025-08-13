import pytrec_eval

# 讀 qrels
with open("data/qrels/qrels.rag24.test-umbrela-all.txt", "r") as f_qrel:
# with open("data/qrels/10q_qrels.txt", "r") as f_qrel:
    qrels = pytrec_eval.parse_qrel(f_qrel)

# 讀 run
with open("runs/retrieval/dense_10q.txt", "r") as f_run:
    run = pytrec_eval.parse_run(f_run)

oracle_run = {}

for topic_id in run:
    doc_ids = list(run[topic_id].keys())

    # 對於這些 doc，分成 relevant 跟 non-relevant
    relevant_docs = []
    non_relevant_docs = []

    for doc_id in doc_ids:
        if topic_id in qrels and doc_id in qrels[topic_id] and qrels[topic_id][doc_id] > 0:
            relevant_docs.append(doc_id)
        else:
            non_relevant_docs.append(doc_id)

    # Oracle 排序：relevant 放前面，non-relevant 放後面
    sorted_docs = relevant_docs + non_relevant_docs

    # 建立一個新的 run 分數（高分在前）
    oracle_run[topic_id] = {doc_id: float(len(sorted_docs) - i) for i, doc_id in enumerate(sorted_docs)}

metrics = {
    'map_cut_10', 'map_cut_100', 'map_cut_1000', 'ndcg_cut_10', 'ndcg_cut_100', 'ndcg_cut_1000',
    'recall_10', 'recall_100', 'recall_1000', 'P_10', 'P_100', 'P_1000'
}
evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
results = evaluator.evaluate(oracle_run)

# 平均分數
avg_results = {metric: sum([v[metric] for v in results.values()]) / len(results) for metric in metrics}
print("Oracle Rerank 分數（Upper Bound）:")
for k in sorted(avg_results.keys()):
    print(f"{k:15}: {avg_results[k]:.4f}")
