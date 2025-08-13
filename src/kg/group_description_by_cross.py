import json
import jsonlines
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
import re
import os
import argparse
from collections import defaultdict
import gc
from src.utils.normalized_list import normalize_to_list

def filter_subsentences(sentences, ids):
    sorted_items = sorted(zip(sentences, ids), key=lambda x: len(x[0]), reverse=True)
    filtered_sentences = []
    filtered_ids = []
    for sent, id_ in sorted_items:
        if not any(sent in s for s in filtered_sentences):
            filtered_sentences.append(sent)
            filtered_ids.append(id_)
    return filtered_sentences, filtered_ids

def split_into_sentences(text):
    sentences = re.split(r'[.!?]+\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    return sentences

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(texts, tokenizer, model, device):
    if not texts:
        return torch.tensor([], device=device)
    texts = [t if isinstance(t, str) else "" for t in texts]
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=256).to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def batched_embeddings(texts, tokenizer, model, device, batch_size=128):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="生成嵌入向量 (Embeddings)"):
        batch = texts[i:i+batch_size]
        emb = get_embeddings(batch, tokenizer, model, device)
        all_embeddings.append(emb.cpu())
        torch.cuda.empty_cache()
    return torch.cat(all_embeddings, dim=0)

def load_keywords_map(keywords_file: str) -> dict:
    print(f"正在從 {keywords_file} 預先載入所有關鍵字...")
    keywords_map = defaultdict(list)
    with jsonlines.open(keywords_file) as reader:
        for item in tqdm(reader, desc="預載入關鍵字"):
            doc_id = item.get("id")
            if doc_id:
                kw_data = item.get("keywords", [])
                keywords_map[doc_id].extend(normalize_to_list(kw_data))
    print(f"完成！共找到 {len(keywords_map)} 個文件ID的關鍵字。")
    return keywords_map

def read_segments_in_chunks(segment_file: str, chunk_size: int):
    try:
        with jsonlines.open(segment_file) as reader:
            chunk = []
            for item in reader:
                chunk.append(item)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
    except FileNotFoundError:
        print(f"錯誤: Segment 檔案 {segment_file} 不存在。")
        return

def compute_cross_scores(model, tokenizer, queries, documents, device, batch_size=32):
    scores = []
    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start:start+batch_size]
        batch_queries = queries[start:start+batch_size]
        features = tokenizer(batch_queries, batch_docs, padding=True, truncation=True, return_tensors="pt")
        for key in features:
            features[key] = features[key].to(device)
        with torch.no_grad():
            logits = model(**features).logits.squeeze(-1)
            scores.extend(logits.cpu().tolist())
    return scores

def process_chunk(chunk_data, keywords_map, dense_tokenizer, dense_model, cross_tokenizer, cross_model, device, batch_size):
    sentence_id_pairs = []
    current_keywords = set()
    segment_ids_in_chunk = {item.get("id") for item in chunk_data if item.get("id")}
    for item in chunk_data:
        doc_id = item.get("id")
        content = item.get("contents", "")
        if doc_id and content:
            split_sents = split_into_sentences(content)
            if split_sents:
                for s in split_sents:
                    if s.strip():
                        sentence_id_pairs.append((s.strip(), doc_id))
    for seg_id in segment_ids_in_chunk:
        if seg_id in keywords_map:
            current_keywords.update(keywords_map[seg_id])
    unique_sentences_map = {}
    for sentence, doc_id in sentence_id_pairs:
        if sentence not in unique_sentences_map:
            unique_sentences_map[sentence] = doc_id
    if not unique_sentences_map or not current_keywords:
        print("警告：目前區塊找不到句子或對應的關鍵字，跳過處理。")
        return None
    unique_sentences = list(unique_sentences_map.keys())
    corresponding_ids = list(unique_sentences_map.values())
    unique_keywords = list(current_keywords)
    print(f"本區塊找到 {len(unique_sentences)} 句不重複的句子和 {len(unique_keywords)} 個不重複的關鍵字。")
    results = defaultdict(lambda: {"sentence": [], "id": []})
    print(f"正在為 {len(unique_keywords)} 個關鍵字檢索 Top 100 句子 (MiniLM-L6-v2) 並用cross-encoder rerank Top 10...")
    sent_emb = batched_embeddings(unique_sentences, dense_tokenizer, dense_model, device, batch_size)
    sent_emb = sent_emb.to(device)
    for i in tqdm(range(len(unique_keywords)), desc="檢索中"):
        keyword = unique_keywords[i]
        # 1. dense 檢索 top100
        query_emb = get_embeddings([keyword], dense_tokenizer, dense_model, device)
        # sent_emb = batched_embeddings(unique_sentences, dense_tokenizer, dense_model, device, batch_size)
        # 確保兩者都在同一 device
        query_emb = query_emb.to(device)
        # sent_emb = sent_emb.to(device)

        dense_scores = torch.matmul(query_emb, sent_emb.T).squeeze(0)
        dense_scores_tensor = dense_scores.cpu()
        top100 = min(100, len(unique_sentences))
        top100_scores, top100_indices = torch.topk(dense_scores_tensor, k=top100)
        top100_sentences = [unique_sentences[idx] for idx in top100_indices]
        top100_ids = [corresponding_ids[idx] for idx in top100_indices]
        # 2. cross-encoder rerank top10
        queries = [keyword] * len(top100_sentences)
        documents = top100_sentences
        cross_scores = compute_cross_scores(cross_model, cross_tokenizer, queries, documents, device, batch_size)
        cross_scores_tensor = torch.tensor(cross_scores)
        top10 = min(10, len(top100_sentences))
        top10_scores, top10_indices = torch.topk(cross_scores_tensor, k=top10)
        top_sentences = [top100_sentences[idx] for idx in top10_indices]
        top_ids = [top100_ids[idx] for idx in top10_indices]
        filtered_sentences, filtered_ids = filter_subsentences(top_sentences, top_ids)
        if len(filtered_sentences) < top10:
            selected_set = set(filtered_sentences)
            for sent, id_ in zip(top_sentences, top_ids):
                if sent not in selected_set:
                    filtered_sentences.append(sent)
                    filtered_ids.append(id_)
                if len(filtered_sentences) >= top10:
                    break
        results[keyword]["sentence"].extend(filtered_sentences)
        results[keyword]["id"].extend(filtered_ids)
        gc.collect()
        torch.cuda.empty_cache()
    return dict(results)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"正在載入 dense 模型: pretrained_model/sentence-transformers/all-MiniLM-L6-v2 ...")
    dense_tokenizer = AutoTokenizer.from_pretrained("pretrained_model/sentence-transformers/all-MiniLM-L6-v2/")
    dense_model = AutoModel.from_pretrained("pretrained_model/sentence-transformers/all-MiniLM-L6-v2/").to(device).eval()
    print(f"正在載入 cross-encoder 模型: pretrained_model/cross-encoder/ms-marco-MiniLM-L6-v2 ...")
    cross_tokenizer = AutoTokenizer.from_pretrained("pretrained_model/cross-encoder/ms-marco-MiniLM-L6-v2")
    cross_model = AutoModelForSequenceClassification.from_pretrained("pretrained_model/cross-encoder/ms-marco-MiniLM-L6-v2").to(device).eval()
    keywords_map = load_keywords_map(args.keywords_file)
    segment_chunks = read_segments_in_chunks(args.segment_file, args.chunk_size)
    for i, chunk in enumerate(segment_chunks):
        chunk_num = i + 1
        print(f"--- 正在處理區塊 {chunk_num} (共 {len(chunk)} 筆 segments) ---")
        results = process_chunk(
            chunk_data=chunk,
            keywords_map=keywords_map,
            dense_tokenizer=dense_tokenizer,
            dense_model=dense_model,
            cross_tokenizer=cross_tokenizer,
            cross_model=cross_model,
            device=device,
            batch_size=args.batch_size
        )
        if results:
            output_file = os.path.join(args.output_dir, f"query_{i+1}.json")
            print(f"正在將區塊 {chunk_num} 的結果儲存至 {output_file}...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"區塊 {chunk_num} 處理完成！🎉")
        else:
            print(f"區塊 {chunk_num} 沒有生成結果，已跳過。")
    print("所有區塊處理完畢！")

if __name__ == "__main__":
    class Args:
        keywords_file = "data/kg/entities_sentence.jsonl"
        output_dir = "data/kg/grouped_descriptions_et_cross/"
        segment_file = "data/segment/dense_10q.jsonl"
        top_k = 10
        batch_size = 2048
        chunk_size = 1000
    args = Args()
    main(args)
