import os
import gzip
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import numpy as np
from tqdm import tqdm
import heapq

# --- 來自您範例的程式碼 (無變動) ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# --- 主要邏輯修改 ---

def search_in_chunks(topics_file, model_name, temp_output_folder, top_k, run_name, output_path):
    """
    分塊載入索引進行搜尋，無需合併暫存檔。
    """
    # 檢查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"已删除旧文件: {output_path}")

    # 從 HuggingFace Hub 加載模型
    print("Loading model for query encoding...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # 1. 準備索引檔案路徑列表
    print(f"Scanning for index chunks in '{temp_output_folder}'...")
    chunk_paths = []
    temp_files = sorted(os.listdir(temp_output_folder))
    json_files = [f for f in temp_files if f.endswith('.ids.json')]
    
    for json_filename in json_files:
        base_name = json_filename[:-9]
        embed_filename = f"{base_name}.embed.npy"
        
        json_path = os.path.join(temp_output_folder, json_filename)
        embed_path = os.path.join(temp_output_folder, embed_filename)

        if os.path.exists(embed_path):
            chunk_paths.append({'ids': json_path, 'embed': embed_path})
            
    if not chunk_paths:
        print("Error: No index chunks found in the temp folder.")
        return
    
    # breakpoint()
    print(f"Found {len(chunk_paths)} index chunks to search through.")

    # 2. 逐一處理查詢
    print(f"Processing queries from '{topics_file}'...")
    with open(topics_file, 'r', encoding='utf-8') as f:
        queries = f.readlines()

    for line in tqdm(queries, desc="Total Queries"):
        line = line.strip()
        if not line:
            continue
        
        query_id, query_text = line.split('\t', 1)
        # breakpoint()
        
        # --- 編碼查詢 ---
        encoded_input = tokenizer([query_text], padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        query_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        query_embedding = F.normalize(query_embedding, p=2, dim=1)

        # --- 維護一個 top-k 的最小堆 ---
        # 儲存 (score, doc_id) 的元組
        top_k_heap = []

        # 3. 遍歷所有索引區塊
        for chunk in chunk_paths:
            # 載入當前區塊的 IDs 和 Embeddings
            with open(chunk['ids'], 'r', encoding='utf-8') as f_ids:
                chunk_ids = json.load(f_ids)
            chunk_embeddings = torch.from_numpy(np.load(chunk['embed'])).to(device)

            # --- 進行向量搜尋 ---
            cos_scores = torch.mm(query_embedding, chunk_embeddings.T)[0].cpu()

            # --- 更新最小堆 ---
            for i in range(len(chunk_ids)):
                score = cos_scores[i].item()
                doc_id = chunk_ids[i]
                
                # 如果堆還沒滿，直接加入
                if len(top_k_heap) < top_k:
                    heapq.heappush(top_k_heap, (score, doc_id))
                # 如果新分數比堆中最小的分數還大，則替換掉最小的
                else:
                    # heappushpop 比分開的 heappush 和 heappop 更有效率
                    heapq.heappushpop(top_k_heap, (score, doc_id))
        
        # 4. 格式化並輸出結果
        # 此時 top_k_heap 中是全域的 top-k 結果，但需要排序
        # 將最小堆轉換為列表並按分數降序排序
        sorted_results = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
        
        with open(output_path, 'a', encoding='utf-8') as f:
            # 添加進度條            
            for rank, (score, doc_id) in enumerate(sorted_results, 1):
                line = f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}\n"
                f.write(line)

if __name__ == '__main__':
    # --- 設定 ---
    # TOPICS_FILE = '/tmp2/TREC_RAG2025/topics/topics.rag24.test.txt'
    TOPICS_FILE = 'data/topics/test_topic.txt'
    MODEL_NAME = 'pretrained_model/sentence-transformers/all-MiniLM-L6-v2'
    OUTPUT_PATH = "runs/retrieval/dense_10q.txt"
    
    TEMP_OUTPUT_FOLDER = 'index_dense/'
    TOP_K = 1000
    
    # 執行搜尋
    search_in_chunks(
        topics_file=TOPICS_FILE,
        model_name=MODEL_NAME,
        temp_output_folder=TEMP_OUTPUT_FOLDER,
        top_k=TOP_K,
        run_name="MiniLM-run",
        output_path=OUTPUT_PATH
    )