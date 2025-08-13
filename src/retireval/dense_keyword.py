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
import argparse
from src.utils.normalized_list import normalize_to_list

# --- embedding 輔助函數 ---
def mean_pooling(model_output, attention_mask):
    """對模型的輸出進行平均池化，以獲得句子級別的 embedding。"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sentence_embeddings(sentences, tokenizer, model, device):
    """計算給定句子列表的 sentence embeddings。"""
    if not sentences:
        return torch.tensor([])
    # 將句子編碼為模型可接受的格式
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    # 在不計算梯度的情況下執行模型推斷
    with torch.no_grad():
        model_output = model(**encoded_input)
    # 進行平均池化和正規化
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

# --- 資料載入與快取函數 ---
def load_queries(query_filepath):
    """從 TXT 檔案載入查詢，格式為 'ID  TEXT'。"""
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
                print(f"警告: 無法解析查詢行: {line}", file=sys.stderr)
    return queries

def load_jsonl_batch(filepath, start_idx, batch_size):
    """從 JSONL 檔案載入指定範圍的資料批次。"""
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
    """計算 JSONL 檔案的總行數，用於進度條顯示。"""
    print(f"正在計算 {os.path.basename(filepath)} 的總行數...")
    # breakpoint()
    count = 0
    with jsonlines.open(filepath) as reader:
        for _ in tqdm.tqdm(reader, desc="計數中"):
            count += 1
    return count

def save_embeddings_cache(embeddings, cache_file):
    """將 embeddings 儲存到 pickle 快取檔案。"""
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
                                cache_dir, batch_size=100, force_recompute=False, return_ids=False):
    """
    分批處理並計算 embeddings，同時支援快取和選擇性返回 ID。
    """
    cache_file = os.path.join(cache_dir, f"{embedding_type}_embeddings.pkl")
    
    # 如果快取存在且不強制重新計算，則載入快取
    if not force_recompute and os.path.exists(cache_file):
        print(f"從快取載入 {embedding_type} embeddings: {cache_file}")
        embeddings = load_embeddings_cache(cache_file)
        if return_ids:
            print("從快取載入 embeddings，現在需要讀取一次文件以獲取 IDs...")
            ids = [item['id'] for item in jsonlines.open(data_file)]
            return embeddings, ids
        return embeddings

    print(f"開始計算 {embedding_type} embeddings (將會儲存至 {cache_file})...")
    total_lines = count_jsonl_lines(data_file)
    print("total lines", total_lines)
    all_embeddings = []
    all_ids = []
    
    for start_idx in tqdm.tqdm(range(0, total_lines, batch_size), 
                              desc=f"處理 {embedding_type} 批次"):
        batch_data = load_jsonl_batch(data_file, start_idx, batch_size)
        if not batch_data:
            break
            
        if return_ids:
            all_ids.extend([item["id"] for item in batch_data])

        if embedding_type == "content":
            texts = [item["contents"] for item in batch_data]
        elif embedding_type == "keyword":
            texts = [", ".join(normalize_to_list(item.get("keywords", []))) for item in batch_data]
        else:
            raise ValueError(f"未支援的 embedding 類型: {embedding_type}")
        
        batch_embeddings = get_sentence_embeddings(texts, tokenizer, model, device)
        all_embeddings.append(batch_embeddings.cpu())
        
        torch.cuda.empty_cache()
        gc.collect()
    
    if not all_embeddings:
        final_embeddings = torch.tensor([])
    else:
        final_embeddings = torch.cat(all_embeddings, dim=0)
        save_embeddings_cache(final_embeddings, cache_file)
        print(f"{embedding_type} embeddings 已儲存至快取。")

    if return_ids:
        return final_embeddings, all_ids
    return final_embeddings

# --- 檢索核心函數 ---
def retrieve_for_single_query(query_embedding, content_embeddings, keyword_embeddings, 
                            content_ids, start_idx, end_idx, content_weight, keyword_weight):
    """為單一查詢執行檢索，並使用可配置的權重合併分數。"""
    target_content_embeddings = content_embeddings[start_idx:end_idx]
    target_keyword_embeddings = keyword_embeddings[start_idx:end_idx]
    target_ids = content_ids[start_idx:end_idx]
    
    if target_content_embeddings.size(0) == 0:
        return []
    
    device = query_embedding.device
    target_content_embeddings = target_content_embeddings.to(device)
    target_keyword_embeddings = target_keyword_embeddings.to(device)
    
    content_scores = torch.matmul(query_embedding, target_content_embeddings.transpose(0, 1)).squeeze(0)
    keyword_scores = torch.matmul(query_embedding, target_keyword_embeddings.transpose(0, 1)).squeeze(0)
    
    # 使用可配置的權重合併分數
    combined_scores = (content_scores * content_weight) + (keyword_scores * keyword_weight)
    
    scored_documents = [
        {"doc_id": doc_id, "score": score.item()}
        for doc_id, score in zip(target_ids, combined_scores)
    ]
    
    scored_documents.sort(key=lambda x: x["score"], reverse=True)
    return scored_documents

def write_results_to_file(retrieval_results, output_filepath):
    """將檢索結果寫入 TREC 格式的檔案。"""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for result in retrieval_results:
            query_id = result["query_id"]
            for rank, doc in enumerate(result["retrieved_documents"], 1):
                doc_id = doc["doc_id"]
                score = doc["score"]
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} keywords_run\n")

def main():
    parser = argparse.ArgumentParser(description="使用內容和關鍵字進行密集檢索。")
    # --- 檔案路徑參數 ---
    parser.add_argument("--query_file", type=str, default="data/topics/top10_topic.txt", help="查詢檔案路徑 (TXT 格式)。")
    parser.add_argument("--content_jsonl_file", type=str, default="data/segment/dense_10q.jsonl", help="文件內容檔案路徑 (JSONL 格式)。")
    parser.add_argument("--keyword_jsonl_file", type=str, default="data/kg/keywords.jsonl", help="文件關鍵字檔案路徑 (JSONL 格式)。")
    parser.add_argument("--output", type=str, required=True, help="輸出排名檔案的路徑。")
    parser.add_argument("--cache_dir", type=str, default="cache/embeddings_refactored", help="儲存 embeddings 快取的目錄。")
    
    # --- 模型與設備參數 ---
    parser.add_argument("--model_name", type=str, default='pretrained_model/sentence-transformers/all-MiniLM-L6-v2/', help="預訓練模型的名稱或路徑。")
    
    # --- 處理參數 ---
    parser.add_argument("--embedding_batch_size", type=int, default=50, help="計算 embeddings 時的批次大小。")
    parser.add_argument("--docs_per_query_segment", type=int, default=1000, help="每個查詢對應的文件區段大小。")
    parser.add_argument("--force_recompute", action="store_true", help="強制重新計算 embeddings，忽略現有快取。")
    
    # --- 檢索權重參數 ---
    parser.add_argument("--content_weight", type=float, default=1.0, help="內容相似度分數的權重。")
    parser.add_argument("--keyword_weight", type=float, default=1.0, help="關鍵字相似度分數的權重。")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    print(f"載入模型: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()
    
    print("載入查詢檔案...")
    queries = load_queries(args.query_file)
    print(f"載入 {len(queries)} 條查詢。")
    
    # --- 處理 Embeddings ---
    content_embeddings, content_ids = process_embeddings_in_batches(
        args.content_jsonl_file, "content", tokenizer, model, device, 
        args.cache_dir, args.embedding_batch_size, args.force_recompute, return_ids=True
    )
    
    keyword_embeddings = process_embeddings_in_batches(
        args.keyword_jsonl_file, "keyword", tokenizer, model, device, 
        args.cache_dir, args.embedding_batch_size, args.force_recompute
    )
    
    if content_embeddings.size(0) != keyword_embeddings.size(0):
        print(f"警告: 內容 embeddings ({content_embeddings.size(0)}) 和關鍵字 embeddings ({keyword_embeddings.size(0)}) 的數量不一致。", file=sys.stderr)
        # 這裡可以選擇退出或嘗試繼續
        # sys.exit(1)

    print(f"總共 {len(content_ids)} 個文件 ID 被載入。")
    
    # --- 處理查詢與檢索 ---
    retrieval_results = []
    print("開始處理每條查詢...")
    
    for i, query_item in enumerate(tqdm.tqdm(queries, desc="處理查詢")):
        query_id = query_item["id"]
        query_text = query_item["text"]
        
        query_embedding = get_sentence_embeddings([query_text], tokenizer, model, device)
        
        start_idx = i * args.docs_per_query_segment
        end_idx = min(start_idx + args.docs_per_query_segment, len(content_ids))
        
        if start_idx >= len(content_ids):
            print(f"警告: 查詢 {query_id} 的起始索引超出資料範圍，跳過。", file=sys.stderr)
            continue
        
        scored_documents = retrieve_for_single_query(
            query_embedding, content_embeddings, keyword_embeddings, 
            content_ids, start_idx, end_idx,
            args.content_weight, args.keyword_weight
        )
        
        retrieval_results.append({
            "query_id": query_id,
            "retrieved_documents": scored_documents
        })
        
        del query_embedding
        torch.cuda.empty_cache()
        gc.collect()
    
    # --- 寫入結果 ---
    write_results_to_file(retrieval_results, args.output)
    
    print("所有檢索結果已成功保存！")
    print(f"排名檔案儲存在: {args.output}")
    print(f"快取檔案儲存在: {args.cache_dir}")

if __name__ == "__main__":
    main()
