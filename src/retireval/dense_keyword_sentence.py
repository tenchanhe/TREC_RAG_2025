import jsonlines
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import tqdm
import re
import sys
import os
import pickle
import gc
import unicodedata
import argparse
from src.utils.normalized_list import normalize_to_list


def split_into_sentences(text):
    # 使用正則表達式分句，考慮多種結尾符號
    sentences = re.split(r'[.!?]+\s+', text.strip())
    
    # 過濾掉空字串和太短的句子
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    return sentences

def clean_text(text: str) -> str:
    # 基本清理
    text = text.strip()
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)

    # 去除 HTML 標籤
    text = re.sub(r'<.*?>', '', text)

    # 正規化 Unicode（處理全形字、特殊空格等）
    text = unicodedata.normalize('NFKC', text)

    # 去除多餘標點（視情況）
    text = re.sub(r'[^\w\s.,!?]', '', text)

    return text

# --- embedding 輔助函數 ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sentence_embeddings(sentences, tokenizer, model, device):
    """計算給定句子的 sentence embeddings。"""
    if not sentences:
        return torch.tensor([])
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

# --- 資料載入函數 ---
def load_queries(query_filepath):
    """從 TXT 檔案載入查詢。"""
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
                print(f"警告: 無法解析查詢行: {line}")
    return queries


def get_keywords_list(segment_id, keyword_items):
    keywords_list = []
    for item in keyword_items:
        if item['id'] == segment_id:
            keywords_list.append(item['keywords'])

    return keywords_list

def load_jsonl_batch_with_sentences(content_file, keyword_file, start_idx, batch_size):
    """從 JSONL 檔案載入指定範圍的資料批次，並處理句子分割。"""
    content_data = []
    
    with jsonlines.open(content_file) as content_reader, \
         jsonlines.open(keyword_file) as keyword_reader:
        
        content_items = list(content_reader)
        keyword_items = list(keyword_reader)
        
        end_idx = min(start_idx + batch_size, len(content_items))
        
        for i in range(start_idx, end_idx):
            if i >= len(content_items):
                break
                
            content_item = content_items[i]
            
            sentences = split_into_sentences(content_item["contents"])
            
            keywords_list = get_keywords_list(content_item['id'], keyword_items)
            
            if len(keywords_list) != len(sentences):
                print(f"警告: 文件 {content_item['id']} 的句子數量 ({len(sentences)}) 與關鍵字數量 ({len(keywords_list)}) 不匹配")
                min_len = min(len(sentences), len(keywords_list))
                sentences = sentences[:min_len]
                keywords_list = keywords_list[:min_len]
            
            for j, (sentence, keywords) in enumerate(zip(sentences, keywords_list)):
                sentence_data = {
                    "doc_id": content_item["id"],
                    "sentence_idx": j,
                    "sentence_id": f"{content_item['id']}_sent_{j}",
                    "sentence": sentence,
                    "keywords": keywords if isinstance(keywords, list) else [keywords]
                }
                content_data.append(sentence_data)
    
    return content_data

def count_jsonl_lines(filepath):
    """計算 JSONL 檔案的總行數。"""
    count = 0
    with jsonlines.open(filepath) as reader:
        for _ in reader:
            count += 1
    return count

def save_embeddings_cache(embeddings, cache_file):
    """將 embeddings 儲存到快取檔案。"""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings.cpu().numpy(), f)

def load_embeddings_cache(cache_file):
    """從快取檔案載入 embeddings。"""
    if not os.path.exists(cache_file):
        return None
    with open(cache_file, 'rb') as f:
        embeddings_np = pickle.load(f)
        return torch.from_numpy(embeddings_np)

def process_sentence_embeddings_in_batches(content_file, keyword_file, embedding_type, 
                                         tokenizer, model, device, cache_dir, 
                                         batch_size=50, force_recompute=False):
    """分批處理句子級別的 embeddings。"""
    
    cache_file = os.path.join(cache_dir, f"sentence_{embedding_type}_embeddings.pkl")
    metadata_cache_file = os.path.join(cache_dir, f"sentence_metadata.pkl")
    
    if not force_recompute and os.path.exists(cache_file) and os.path.exists(metadata_cache_file):
        print(f"載入句子級別 {embedding_type} embeddings 快取...")
        embeddings = load_embeddings_cache(cache_file)
        with open(metadata_cache_file, 'rb') as f:
            metadata = pickle.load(f)
        return embeddings, metadata
    
    print(f"計算句子級別 {embedding_type} embeddings...")
    
    total_content_lines = count_jsonl_lines(content_file)
    all_embeddings = []
    all_metadata = []
    
    for start_idx in tqdm.tqdm(range(0, total_content_lines, batch_size), 
                              desc=f"處理句子 {embedding_type} 批次"):
        sentence_batch = load_jsonl_batch_with_sentences(content_file, keyword_file, start_idx, batch_size)
        
        if not sentence_batch:
            break
        
        if embedding_type == "content":
            texts = [item["sentence"] for item in sentence_batch]
        elif embedding_type == "keyword":
            texts = []
            for item in sentence_batch:
                text = normalize_to_list(item["keywords"])
                try:
                    texts.append(", ".join(text))
                except:
                    print("這個有問題：", text)
        else:
            raise ValueError(f"未支援的 embedding 類型: {embedding_type}")
        
        if texts:
            batch_embeddings = get_sentence_embeddings(texts, tokenizer, model, device)
            all_embeddings.append(batch_embeddings.cpu())
            
            for item in sentence_batch:
                all_metadata.append({
                    "doc_id": item["doc_id"],
                    "sentence_idx": item["sentence_idx"],
                    "sentence_id": item["sentence_id"]
                })
        
        torch.cuda.empty_cache()
        gc.collect()
    
    if all_embeddings:
        final_embeddings = torch.cat(all_embeddings, dim=0)
        save_embeddings_cache(final_embeddings, cache_file)
        with open(metadata_cache_file, 'wb') as f:
            pickle.dump(all_metadata, f)
        return final_embeddings, all_metadata
    else:
        return torch.tensor([]), []

def retrieve_for_single_query_sentence_based(query_embedding, sentence_content_embeddings, 
                                           sentence_keyword_embeddings, sentence_metadata, 
                                           start_idx, end_idx, content_weight, keyword_weight):
    """為單一查詢執行基於句子的檢索。"""
    if start_idx >= len(sentence_metadata):
        return []
        
    end_idx = min(end_idx, len(sentence_metadata))
    
    target_content_embeddings = sentence_content_embeddings[start_idx:end_idx]
    target_keyword_embeddings = sentence_keyword_embeddings[start_idx:end_idx]
    target_metadata = sentence_metadata[start_idx:end_idx]
    
    if target_content_embeddings.size(0) == 0:
        return []
    
    device = query_embedding.device
    target_content_embeddings = target_content_embeddings.to(device)
    target_keyword_embeddings = target_keyword_embeddings.to(device)
    
    content_scores = torch.matmul(query_embedding, target_content_embeddings.transpose(0, 1)).squeeze(0)
    keyword_scores = torch.matmul(query_embedding, target_keyword_embeddings.transpose(0, 1)).squeeze(0)
    
    combined_scores = (content_weight * content_scores) + (keyword_weight * keyword_scores)
    
    doc_scores = {}
    doc_sentence_info = {}
    
    for j, metadata in enumerate(target_metadata):
        doc_id = metadata["doc_id"]
        sentence_score = combined_scores[j].item()
        
        if doc_id not in doc_scores or sentence_score > doc_scores[doc_id]:
            doc_scores[doc_id] = sentence_score
            doc_sentence_info[doc_id] = {
                "sentence_id": metadata["sentence_id"],
                "sentence_idx": metadata["sentence_idx"],
                "score": sentence_score
            }
    
    scored_documents = []
    for doc_id, score in doc_scores.items():
        scored_documents.append({
            "doc_id": doc_id,
            "score": score,
            "best_sentence_id": doc_sentence_info[doc_id]["sentence_id"],
            "best_sentence_idx": doc_sentence_info[doc_id]["sentence_idx"]
        })
    
    scored_documents.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_documents

def main():
    parser = argparse.ArgumentParser(description="基於句子和關鍵字的混合檢索，可調整權重。")
    parser.add_argument('--content_weight', type=float, default=1.0, help='內容分數的權重')
    parser.add_argument('--keyword_weight', type=float, default=1.0, help='關鍵字分數的權重')
    args = parser.parse_args()

    # --- 設定檔案路徑和參數 ---
    query_file = "data/topics/top10_topic.txt"
    content_jsonl_file = "data/segment/dense_10q.jsonl"
    keyword_jsonl_file = "data/kg/keywords_sentence.jsonl"
    output_dir = "runs/retrieval"
    cache_dir = "cache/keywords_sentence"
    
    output_ranking_file = os.path.join(
        output_dir, 
        f"dense_cw_{args.content_weight}_kw_{args.keyword_weight}.txt"
    )
    
    model_name = 'pretrained_model/sentence-transformers/all-MiniLM-L6-v2/'
    
    embedding_batch_size = 50
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    print(f"載入模型: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    print("載入查詢檔案...")
    queries = load_queries(query_file)
    print(f"載入 {len(queries)} 條查詢。")
    
    print("處理句子級別的內容和關鍵字 embeddings...")
    sentence_content_embeddings, sentence_metadata = process_sentence_embeddings_in_batches(
        content_jsonl_file, keyword_jsonl_file, "content", tokenizer, model, device, 
        cache_dir, embedding_batch_size
    )
    
    sentence_keyword_embeddings, _ = process_sentence_embeddings_in_batches(
        content_jsonl_file, keyword_jsonl_file, "keyword", tokenizer, model, device, 
        cache_dir, embedding_batch_size
    )
    
    print(f"總共處理了 {len(sentence_metadata)} 個句子")
    
    unique_doc_ids = set(meta["doc_id"] for meta in sentence_metadata)
    print(f"涵蓋 {len(unique_doc_ids)} 個獨特文件")
    
    retrieval_results = []
    print("開始處理每條查詢...")
    
    sentences_per_query = len(sentence_metadata) // len(queries) if queries else 0
    
    for i, query_item in enumerate(tqdm.tqdm(queries, desc="處理查詢")):
        query_id = query_item["id"]
        query_text = query_item["text"]
        
        query_embedding = get_sentence_embeddings([query_text], tokenizer, model, device)
        
        start_idx = i * sentences_per_query
        if i == len(queries) - 1:
            end_idx = len(sentence_metadata)
        else:
            end_idx = start_idx + sentences_per_query
        
        if start_idx >= len(sentence_metadata):
            print(f"警告: 查詢 {query_id} 的起始索引超出句子範圍，跳過。", file=sys.stderr)
            retrieval_results.append({
                "query_id": query_id,
                "retrieved_documents": []
            })
            continue
        
        print(f"查詢 {query_id}: 處理句子範圍 {start_idx}-{end_idx} (共 {end_idx-start_idx} 個句子)")
        
        scored_documents = retrieve_for_single_query_sentence_based(
            query_embedding, sentence_content_embeddings, sentence_keyword_embeddings, 
            sentence_metadata, start_idx, end_idx,
            args.content_weight, args.keyword_weight
        )
        
        retrieval_results.append({
            "query_id": query_id,
            "retrieved_documents": scored_documents
        })
        
        del query_embedding
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"將檢索結果儲存到 '{output_ranking_file}'...")
    os.makedirs(os.path.dirname(output_ranking_file), exist_ok=True)
    
    run_name = f"dense_cw_{args.content_weight}_kw_{args.keyword_weight}"
    
    with open(output_ranking_file, 'w', encoding='utf-8') as f:
        for result in retrieval_results:
            query_id = result["query_id"]
            for rank, doc in enumerate(result["retrieved_documents"], 1):
                doc_id = doc["doc_id"]
                score = doc["score"]
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")
    
    print("所有檢索結果已成功保存！")
    print(f"快取檔案儲存在: {cache_dir}")
    print(f"結果已以 TREC 格式儲存於 {output_ranking_file}")

if __name__ == "__main__":
    main()
