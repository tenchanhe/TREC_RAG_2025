import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import tqdm
import re
import sys
import os
from collections import defaultdict
import gc

def highest_score_per_docid(sentence_scores, sentencestr_to_docid):
    """對每個 docid 只保留最高分"""
    docid_scores = defaultdict(float)
    for sentence, score in sentence_scores.items():
        doc_id = sentencestr_to_docid[sentence]
        for id in doc_id:
            if id not in docid_scores or score > docid_scores[id]:
                docid_scores[id] = score
    return docid_scores

def mean_score_per_docid(sentence_scores, sentencestr_to_docid):
    """每個 docid 的所有句子分數取平均"""
    docid_score_list = defaultdict(list)
    for sentence, score in sentence_scores.items():
        doc_id = sentencestr_to_docid[sentence]
        for id in doc_id:
            docid_score_list[id].append(score)
    docid_scores = {id: sum(scores)/len(scores) for id, scores in docid_score_list.items() if scores}
    return docid_scores

def sum_score_per_docid(sentence_scores, sentencestr_to_docid):
    """每個 docid 的所有句子分數累加"""
    docid_score_list = defaultdict(list)
    for sentence, score in sentence_scores.items():
        doc_id = sentencestr_to_docid[sentence]
        for id in doc_id:
            docid_score_list[id].append(score)
    docid_scores = {id: sum(scores) for id, scores in docid_score_list.items() if scores}
    return docid_scores

def weighted_mean_score_per_docid(sentence_scores, sentencestr_to_docid):
    """每個 docid 的所有句子分數以句子長度為權重做加權平均"""
    docid_weighted_scores = defaultdict(list)
    docid_weights = defaultdict(list)
    for sentence, score in sentence_scores.items():
        doc_id = sentencestr_to_docid[sentence]
        weight = len(sentence.split())  # 以詞數作為權重
        for id in doc_id:
            docid_weighted_scores[id].append(score * weight)
            docid_weights[id].append(weight)
    docid_scores = {}
    for id in docid_weighted_scores:
        total_weight = sum(docid_weights[id])
        if total_weight > 0:
            docid_scores[id] = sum(docid_weighted_scores[id]) / total_weight
    return docid_scores

def batched_embeddings(texts, tokenizer, model, device, batch_size=128):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = get_embeddings(batch, tokenizer, model, device)
        all_embeddings.append(emb)
        torch.cuda.empty_cache()
    return torch.cat(all_embeddings, dim=0)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(texts, tokenizer, model, device):
    if not texts:
        return torch.tensor([])
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128).to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def load_queries(query_filepath):
    queries = []
    with open(query_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'(\S+)\s+(.*)', line)
            if match:
                query_id = match.group(1)
                query_text = match.group(2)
                queries.append({"id": query_id, "text": query_text})
            else:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    queries.append({"id": parts[0], "text": parts[1]})
                else:
                    print(f"Warning: Could not parse query line: {line}", file=sys.stderr)
    return queries


def main():
    # grouped_desc_dir = "data/kg/grouped_descriptions_dense"
    # run_name = "dense"
    # grouped_desc_dir = "data/kg/grouped_descriptions_own"
    # run_name = "own"
    grouped_desc_dir = "data/kg/grouped_descriptions_et_dense"
    run_name = "et_dense"
    # grouped_desc_dir = "data/kg/grouped_descriptions_et_own"
    # run_name = "et_own"

    query_file = "data/topics/top10_topic.txt"
    output_dir = "runs/retrieval/group_description/"
    model_name = "pretrained_model/sentence-transformers/all-MiniLM-L6-v2/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_ranking_file = os.path.join(output_dir, f"{run_name}.txt")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    queries = load_queries(query_file)
    print(f"Loaded {len(queries)} queries.")

    retrieval_results = []
    print("Starting retrieval for each query using grouped descriptions...")
    for query_index, query_item in enumerate(tqdm.tqdm(queries, desc="Processing queries")):
        query_id = query_item["id"]
        query_text = query_item["text"]
        grouped_file = os.path.join(grouped_desc_dir, f"query_{query_index+1}.json")
        if not os.path.exists(grouped_file):
            print(f"Warning: grouped file not found for query {query_id}: {grouped_file}", file=sys.stderr)
            continue
        with open(grouped_file, "r", encoding="utf-8") as f:
            grouped_data = json.load(f)
        # grouped_data: {"keywords1": {"sentence": [...], "id": [...]}}
        sentences = []
        doc_ids = []
        for key, value in grouped_data.items():
            sentences.append(" ".join(value.get("sentence")))
            doc_ids.append(value.get("id"))
        if not sentences or not doc_ids or len(sentences) != len(doc_ids):
            print(f"Warning: Invalid grouped data for query {query_id}", file=sys.stderr)
            continue
        # 構建 sentence -> docid 映射
        sentencestr_to_docid = {sent: docid for sent, docid in zip(sentences, doc_ids)}
        unique_sentence_group = sentences

        query_embedding = get_embeddings([query_text], tokenizer, model, device)
        keyword_embeddings = batched_embeddings(unique_sentence_group, tokenizer, model, device, batch_size=128)

        scores = torch.matmul(query_embedding, keyword_embeddings.transpose(0, 1)).squeeze(0)
        sentence_scores = defaultdict(float)
        for i, score in enumerate(scores):
            sentence = unique_sentence_group[i]
            sentence_scores[sentence] = score.item()

        docid_scores = highest_score_per_docid(sentence_scores, sentencestr_to_docid)
        # docid_scores = mean_score_per_docid(sentence_scores, sentencestr_to_docid)
        # docid_scores = sum_score_per_docid(sentence_scores, sentencestr_to_docid)
        # docid_scores = weighted_mean_score_per_docid(sentence_scores, sentencestr_to_docid)

        sorted_docid_scores = sorted(docid_scores.items(), key=lambda item: item[1], reverse=True)
        # breakpoint()
        retrieval_results.append({
            "query_id": query_id,
            "retrieved_documents": sorted_docid_scores
        })

        del query_embedding, keyword_embeddings, scores
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Saving retrieval results to '{output_ranking_file}'...")
    with open(output_ranking_file, 'w', encoding='utf-8') as f:
        for result in retrieval_results:
            query_id = result["query_id"]
            for rank, (doc_id, score) in enumerate(result["retrieved_documents"][:1000]):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

    print("Retrieval process completed successfully.")
    print(f"Results saved in TREC format at: {output_ranking_file}")

if __name__ == "__main__":
    main()
