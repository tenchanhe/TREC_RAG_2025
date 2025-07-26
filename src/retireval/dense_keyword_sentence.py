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
    keyword_data = []
    
    # 同時讀取兩個檔案
    with jsonlines.open(content_file) as content_reader, \
         jsonlines.open(keyword_file) as keyword_reader:
        
        content_items = list(content_reader)
        keyword_items = list(keyword_reader)
        
        end_idx = min(start_idx + batch_size, len(content_items))
        
        for i in range(start_idx, end_idx):
            if i >= len(content_items) or i >= len(keyword_items):
                break
                
            content_item = content_items[i]
            # keyword_item = keyword_items[i]
            
            # 拆分內容為句子
            sentences = split_into_sentences(content_item["contents"])
            
            # 確保關鍵字數量與句子數量匹配
            keywords_list = get_keywords_list(content_item['id'], keyword_items)
            # breakpoint()
            
            if len(keywords_list) != len(sentences):
                print(f"警告: 文件 {content_item['id']} 的句子數量 ({len(sentences)}) 與關鍵字數量 ({len(keywords_list)}) 不匹配")
                # 調整到最小長度
                min_len = min(len(sentences), len(keywords_list))
                sentences = sentences[:min_len]
                keywords_list = keywords_list[:min_len]
            
            # 為每個句子創建記錄
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
    
    # # 如果快取存在且不強制重新計算，則載入快取
    # if not force_recompute and os.path.exists(cache_file) and os.path.exists(metadata_cache_file):
    #     print(f"載入句子級別 {embedding_type} embeddings 快取...")
    #     embeddings = load_embeddings_cache(cache_file)
    #     with open(metadata_cache_file, 'rb') as f:
    #         metadata = pickle.load(f)
    #     return embeddings, metadata
    
    print(f"計算句子級別 {embedding_type} embeddings...")
    
    total_content_lines = count_jsonl_lines(content_file)
    all_embeddings = []
    all_metadata = []
    
    for start_idx in tqdm.tqdm(range(0, total_content_lines, batch_size), 
                              desc=f"處理句子 {embedding_type} 批次"):
        # 載入當前批次的句子資料
        sentence_batch = load_jsonl_batch_with_sentences(content_file, keyword_file, start_idx, batch_size)
        
        if not sentence_batch:
            break
        
        # 根據類型提取文本
        if embedding_type == "content":
            texts = [item["sentence"] for item in sentence_batch]
        elif embedding_type == "keyword":
            texts = [", ".join(item["keywords"]) for item in sentence_batch]
        else:
            raise ValueError(f"未支援的 embedding 類型: {embedding_type}")
        
        # 計算當前批次的 embeddings
        if texts:  # 確保有文本要處理
            batch_embeddings = get_sentence_embeddings(texts, tokenizer, model, device)
            all_embeddings.append(batch_embeddings.cpu())
            
            # 儲存元數據
            for item in sentence_batch:
                all_metadata.append({
                    "doc_id": item["doc_id"],
                    "sentence_idx": item["sentence_idx"],
                    "sentence_id": item["sentence_id"]
                })
        
        # 清理 GPU 記憶體
        torch.cuda.empty_cache()
        gc.collect()
    
    # 合併所有批次的 embeddings
    if all_embeddings:
        final_embeddings = torch.cat(all_embeddings, dim=0)
        # 儲存到快取
        save_embeddings_cache(final_embeddings, cache_file)
        with open(metadata_cache_file, 'wb') as f:
            pickle.dump(all_metadata, f)
        return final_embeddings, all_metadata
    else:
        return torch.tensor([]), []

def retrieve_for_single_query_sentence_based(query_embedding, sentence_content_embeddings, 
                                           sentence_keyword_embeddings, sentence_metadata, 
                                           start_idx, end_idx):
    """為單一查詢執行基於句子的檢索。"""
    if start_idx >= len(sentence_metadata):
        return []
        
    # 調整結束索引
    end_idx = min(end_idx, len(sentence_metadata))
    
    # 取得目標範圍的 embeddings 和元數據
    target_content_embeddings = sentence_content_embeddings[start_idx:end_idx]
    target_keyword_embeddings = sentence_keyword_embeddings[start_idx:end_idx]
    target_metadata = sentence_metadata[start_idx:end_idx]
    
    if target_content_embeddings.size(0) == 0:
        return []
    
    # 將 embeddings 移到與 query_embedding 相同的設備
    device = query_embedding.device
    target_content_embeddings = target_content_embeddings.to(device)
    target_keyword_embeddings = target_keyword_embeddings.to(device)
    
    # 計算句子級別的相似度分數
    content_scores = torch.matmul(query_embedding, target_content_embeddings.transpose(0, 1)).squeeze(0)
    keyword_scores = torch.matmul(query_embedding, target_keyword_embeddings.transpose(0, 1)).squeeze(0)
    
    # 合併分數 (可以調整權重)
    combined_scores = content_scores + keyword_scores
    
    # 將句子分數聚合到文件級別
    doc_scores = {}  # doc_id -> max_score
    doc_sentence_info = {}  # doc_id -> 最佳句子資訊
    
    for j, metadata in enumerate(target_metadata):
        doc_id = metadata["doc_id"]
        sentence_score = combined_scores[j].item()
        
        # 使用最高分數的句子來代表該文件
        if doc_id not in doc_scores or sentence_score > doc_scores[doc_id]:
            doc_scores[doc_id] = sentence_score
            doc_sentence_info[doc_id] = {
                "sentence_id": metadata["sentence_id"],
                "sentence_idx": metadata["sentence_idx"],
                "score": sentence_score
            }
    
    # 創建文件級別的分數列表
    scored_documents = []
    for doc_id, score in doc_scores.items():
        scored_documents.append({
            "doc_id": doc_id,
            "score": score,
            "best_sentence_id": doc_sentence_info[doc_id]["sentence_id"],
            "best_sentence_idx": doc_sentence_info[doc_id]["sentence_idx"]
        })
    
    # 按分數降序排序
    scored_documents.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_documents

def main():
    # --- 設定檔案路徑和參數 ---
    query_file = "data/topics/top10_topic.txt"
    content_jsonl_file = "data/segment/dense_10q.jsonl"
    keyword_jsonl_file = "data/kg/keywords_sentence.jsonl"
    output_ranking_file = "runs/retrieval/dense_keyword_sentence.txt"
    cache_dir = "cache/keywords_sentence"  # 快取目錄
    
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
    
    # # --- 檢查資料檔案 ---
    # total_content_lines = count_jsonl_lines(content_jsonl_file)
    # total_keyword_lines = count_jsonl_lines(keyword_jsonl_file)
    # print(f"內容檔案總行數: {total_content_lines}")
    # print(f"關鍵字檔案總行數: {total_keyword_lines}")
    
    # if total_content_lines != total_keyword_lines:
    #     print("警告: content.jsonl 和 keyword.jsonl 的資料筆數不一致。", file=sys.stderr)
    
    # --- 分批計算並快取句子級別的 embeddings ---
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
    
    # 統計文件數量
    unique_doc_ids = set(meta["doc_id"] for meta in sentence_metadata)
    print(f"涵蓋 {len(unique_doc_ids)} 個獨特文件")
    
    # --- 處理查詢 ---
    retrieval_results = []
    print("開始處理每條查詢...")
    
    # 計算每個查詢應該檢索的句子範圍
    sentences_per_query = len(sentence_metadata) // len(queries) if queries else 0
    
    for i, query_item in enumerate(tqdm.tqdm(queries, desc="處理查詢")):
        query_id = query_item["id"]
        query_text = query_item["text"]
        
        # 計算查詢 embedding
        query_embedding = get_sentence_embeddings([query_text], tokenizer, model, device)
        
        # 確定檢索範圍（基於句子索引）
        start_idx = i * sentences_per_query
        if i == len(queries) - 1:  # 最後一個查詢處理剩餘所有句子
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
        
        # 執行基於句子的檢索
        scored_documents = retrieve_for_single_query_sentence_based(
            query_embedding, sentence_content_embeddings, sentence_keyword_embeddings, 
            sentence_metadata, start_idx, end_idx
        )
        
        retrieval_results.append({
            "query_id": query_id,
            "retrieved_documents": scored_documents
        })
        
        # 清理記憶體
        del query_embedding
        torch.cuda.empty_cache()
        gc.collect()
    
    # --- 儲存結果 ---
    print(f"將檢索結果儲存到 '{output_ranking_file}'...")
    os.makedirs(os.path.dirname(output_ranking_file), exist_ok=True)
    
    # 以 TREC 格式儲存結果：query_id Q0 doc_id 1 score run
    run_name = "dense_keyword"  # 可以自訂 run 名稱
    
    with open(output_ranking_file, 'w', encoding='utf-8') as f:
        for result in retrieval_results:
            query_id = result["query_id"]
            for rank, doc in enumerate(result["retrieved_documents"], 1):
                doc_id = doc["doc_id"]
                score = doc["score"]
                # 格式：query_id Q0 doc_id rank score run
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")
    
    print("所有檢索結果已成功保存！")
    print(f"快取檔案儲存在: {cache_dir}")
    print(f"結果已以 TREC 格式儲存，共 {sum(len(r['retrieved_documents']) for r in retrieval_results)} 筆記錄")

if __name__ == "__main__":
    main()