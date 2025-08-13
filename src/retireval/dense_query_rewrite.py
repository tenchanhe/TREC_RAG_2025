import jsonlines, json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import tqdm
import re
import sys
import os
import pickle
import gc, ast
from src.utils.normalized_list import normalize_to_list

# --- embedding 輔助函數 ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sentence_embeddings(sentences, tokenizer, model, device):
    """計算給定句子的 sentence embeddings"""
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
    queries = []
    with open(query_filepath, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    return queries

def load_jsonl_batch(filepath, start_idx, batch_size):
    """從 JSONL 檔案載入指定範圍的資料批次"""
    data = []
    with jsonlines.open(filepath) as reader:
        for i, item in enumerate(reader):
            if i < start_idx:
                continue
            if i >= start_idx + batch_size:
                break
            data.append(item)
    return data

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

def process_embeddings_in_batches(data_file, embedding_type, tokenizer, model, device, 
                                cache_dir, batch_size=100, force_recompute=False):
    """分批處理並儲存 embeddings。"""
    
    cache_file = os.path.join(cache_dir, f"{embedding_type}_embeddings.pkl")
    
    # # 如果快取存在且不強制重新計算，則載入快取
    # if not force_recompute and os.path.exists(cache_file):
    #     print(f"載入 {embedding_type} embeddings 快取...")
    #     return load_embeddings_cache(cache_file)
    
    print(f"計算 {embedding_type} embeddings...")
    total_lines = count_jsonl_lines(data_file)
    all_embeddings = []
    
    for start_idx in tqdm.tqdm(range(0, total_lines, batch_size), 
                              desc=f"處理 {embedding_type} 批次"):
        # 載入當前批次
        batch_data = load_jsonl_batch(data_file, start_idx, batch_size)
        if not batch_data:
            break
            
        # 根據類型提取文本
        if embedding_type == "content":
            texts = [item["contents"] for item in batch_data]
        elif embedding_type == "keyword":
            texts = []
            for item in batch_data:
                text = normalize_to_list(item["keywords"])
                try:
                    texts.append(", ".join(text))
                except:
                    print("這個有問題：", text)
            # texts = [", ".join(parse_string_list(item["keywords"])) for item in batch_data]
        else:
            raise ValueError(f"未支援的 embedding 類型: {embedding_type}")
        
        # 計算當前批次的 embeddings
        batch_embeddings = get_sentence_embeddings(texts, tokenizer, model, device)
        all_embeddings.append(batch_embeddings.cpu())  # 移到 CPU 以節省 GPU 記憶體
        
        # 清理 GPU 記憶體
        torch.cuda.empty_cache()
        gc.collect()
    
    # 合併所有批次的 embeddings
    if all_embeddings:
        final_embeddings = torch.cat(all_embeddings, dim=0)
        # 儲存到快取
        save_embeddings_cache(final_embeddings, cache_file)
        return final_embeddings
    else:
        return torch.tensor([])

def retrieve_for_single_query(query_embedding, subquery_embedding, content_embeddings, keyword_embeddings, 
                            content_ids, start_idx, end_idx):
    """為單一查詢執行檢索。"""
    # 取得目標範圍的 embeddings
    target_content_embeddings = content_embeddings[start_idx:end_idx]
    target_keyword_embeddings = keyword_embeddings[start_idx:end_idx]
    target_ids = content_ids[start_idx:end_idx]
    
    if target_content_embeddings.size(0) == 0:
        return []
    
    # 將 embeddings 移到與 query_embedding 相同的設備
    device = query_embedding.device
    target_content_embeddings = target_content_embeddings.to(device)
    target_keyword_embeddings = target_keyword_embeddings.to(device)
    
    # 計算相似度分數
    content_scores = torch.matmul(query_embedding, target_content_embeddings.transpose(0, 1)).squeeze(0)
    keyword_scores = torch.matmul(query_embedding, target_keyword_embeddings.transpose(0, 1)).squeeze(0)
    sub_content_scores = torch.matmul(subquery_embedding, target_content_embeddings.transpose(0, 1)).squeeze(0)
    sub_keyword_scores = torch.matmul(subquery_embedding, target_keyword_embeddings.transpose(0, 1)).squeeze(0)
    
    # 合併分數 (可以調整權重)
    combined_scores = content_scores*1 + keyword_scores*0 + sub_content_scores*1 + sub_keyword_scores*0
    
    # 創建分數文件對
    scored_documents = []
    for j, doc_id in enumerate(target_ids):
        scored_documents.append({
            "doc_id": doc_id,
            "score": combined_scores[j].item()
        })
    
    # 按分數降序排序
    scored_documents.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_documents

def main():
    # breakpoint()
    # --- 設定檔案路徑和參數 ---
    query_file = "data/topics/top10_query_rewrite.json"
    content_jsonl_file = "data/segment/dense_10q.jsonl"
    keyword_jsonl_file = "data/kg/keywords2.jsonl"
    output_ranking_file = "runs/retrieval/dense_query_rewrite.txt"
    cache_dir = "cache/embeddings"  # 快取目錄
    
    model_name = 'pretrained_model/sentence-transformers/all-MiniLM-L6-v2/'
    
    # 批次處理參數
    embedding_batch_size = 50  # 可根據記憶體大小調整
    docs_per_query_segment = 1000
    
    # 檢查設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 載入模型
    print(f"載入模型: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # --- 載入查詢 ---
    print("載入查詢檔案...")
    queries = load_queries(query_file)
    print(f"載入 {len(queries)} 條查詢。")
    
    # --- 檢查資料檔案 ---
    total_content_lines = count_jsonl_lines(content_jsonl_file)
    total_keyword_lines = count_jsonl_lines(keyword_jsonl_file)
    print(f"內容檔案總行數: {total_content_lines}")
    print(f"關鍵字檔案總行數: {total_keyword_lines}")
    
    if total_content_lines != total_keyword_lines:
        print("警告: content.jsonl 和 keyword.jsonl 的資料筆數不一致。", file=sys.stderr)
    
    # --- 分批計算並快取 embeddings ---
    print("處理內容 embeddings...")
    content_embeddings = process_embeddings_in_batches(
        content_jsonl_file, "content", tokenizer, model, device, 
        cache_dir, embedding_batch_size
    )
    
    print("處理關鍵字 embeddings...")
    keyword_embeddings = process_embeddings_in_batches(
        keyword_jsonl_file, "keyword", tokenizer, model, device, 
        cache_dir, embedding_batch_size
    )
    
    # --- 載入文件 ID ---
    print("載入文件 ID...")
    content_ids = []
    with jsonlines.open(content_jsonl_file) as reader:
        for item in reader:
            content_ids.append(item["id"])
    
    print(f"總共 {len(content_ids)} 個文件 ID")
    
    # --- 處理查詢 ---
    retrieval_results = []
    print("開始處理每條查詢...")
    
    for i, query_item in enumerate(tqdm.tqdm(queries, desc="處理查詢")):
        query_id = query_item["id"]
        query_text = query_item["query"]
        query_sub = query_item["sub_object"]
        
        # 計算查詢 embedding
        query_embedding = get_sentence_embeddings([query_text], tokenizer, model, device)
        subquery_embedding = get_sentence_embeddings(query_sub, tokenizer, model, device)
        # breakpoint()
        
        # 確定檢索範圍
        start_idx = i * docs_per_query_segment
        end_idx = min(start_idx + docs_per_query_segment, len(content_ids))
        print(i, start_idx, end_idx)
        
        if start_idx >= len(content_ids):
            print(f"警告: 查詢 {query_id} 的起始索引超出資料範圍，跳過。", file=sys.stderr)
            retrieval_results.append({
                "query_id": query_id,
                "retrieved_documents": []
            })
            continue
        
        # 執行檢索
        scored_documents = retrieve_for_single_query(
            query_embedding, subquery_embedding, content_embeddings, keyword_embeddings, 
            content_ids, start_idx, end_idx
        )
        
        retrieval_results.append({
            "query_id": query_id,
            "retrieved_documents": scored_documents
        })
        
        # 清理記憶體
        del query_embedding
        torch.cuda.empty_cache()
        gc.collect()
    
    with open(output_ranking_file, 'w', encoding='utf-8') as f:
        for result in retrieval_results:
            query_id = result["query_id"]
            for rank, doc in enumerate(result["retrieved_documents"], 1):
                doc_id = doc["doc_id"]
                score = doc["score"]
                # 格式：query_id Q0 doc_id rank score run
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} keywords_run\n")
    
    
    print("所有檢索結果已成功保存！")
    print(f"快取檔案儲存在: {cache_dir}")

if __name__ == "__main__":
    main()