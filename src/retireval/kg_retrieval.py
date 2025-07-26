import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import heapq

def mean_pooling(model_output, attention_mask):
    """平均池化函數"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def relation_to_text(relation):
    """將關係三元組轉換為文本"""
    head = relation['head']
    rel_type = relation['type']
    tail = relation['tail']
    return f"{head} {rel_type} {tail}"

def create_triplet_embeddings(triplets_file, model_name, output_folder):
    """為三元組創建嵌入向量"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加載模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # 讀取三元組數據
    with open(triplets_file, 'r', encoding='utf-8') as f:
        triplets_data = json.load(f)
    
    # 創建輸出文件夾
    os.makedirs(output_folder, exist_ok=True)
    
    # 處理每個文檔的三元組
    all_doc_ids = []
    all_embeddings = []
    
    batch_size = 32  # 批次處理大小
    
    print("Processing triplets...")
    for doc_data in tqdm(triplets_data, desc="Documents"):
        doc_id = doc_data['docid']
        relations = doc_data['relations']
        
        # 將所有關係轉換為文本
        relation_texts = [relation_to_text(rel) for rel in relations]
        
        # 如果沒有關係，跳過
        if not relation_texts:
            continue
        
        # 批次編碼關係文本
        doc_embeddings = []
        for i in range(0, len(relation_texts), batch_size):
            batch_texts = relation_texts[i:i+batch_size]
            
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = model(**encoded_input)
                batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                doc_embeddings.append(batch_embeddings.cpu())
        
        # 合併所有嵌入向量
        if doc_embeddings:
            doc_embeddings = torch.cat(doc_embeddings, dim=0)
            # 使用平均池化來代表整個文檔
            doc_embedding = torch.mean(doc_embeddings, dim=0, keepdim=True)
            
            all_doc_ids.append(doc_id)
            all_embeddings.append(doc_embedding)
    
    # 保存嵌入向量和ID
    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # 保存為numpy格式
        np.save(os.path.join(output_folder, 'triplets.embed.npy'), all_embeddings.numpy())
        
        # 保存文檔ID
        with open(os.path.join(output_folder, 'triplets.ids.json'), 'w', encoding='utf-8') as f:
            json.dump(all_doc_ids, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(all_doc_ids)} document embeddings to {output_folder}")
    else:
        print("No valid triplets found to process")

def search_json_queries(queries_file, triplets_embeddings_folder, model_name, top_k, output_path):
    """基於JSON查詢搜尋相關文檔"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 清理輸出文件
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"已删除旧文件: {output_path}")
    
    # 加載模型
    print("Loading model for query encoding...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # 載入三元組嵌入向量
    embed_path = os.path.join(triplets_embeddings_folder, 'triplets.embed.npy')
    ids_path = os.path.join(triplets_embeddings_folder, 'triplets.ids.json')
    
    if not os.path.exists(embed_path) or not os.path.exists(ids_path):
        print("Error: Triplet embeddings not found. Please run create_triplet_embeddings first.")
        return
    
    print("Loading triplet embeddings...")
    doc_embeddings = torch.from_numpy(np.load(embed_path)).to(device)
    with open(ids_path, 'r', encoding='utf-8') as f:
        doc_ids = json.load(f)
    
    # 讀取查詢數據
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    
    print(f"Processing {len(queries_data)} queries...")
    
    # 處理每個查詢
    for query_idx, query_data in enumerate(tqdm(queries_data, desc="Queries")):
        query_text = query_data['docid']  # 使用docid作為查詢文本
        query_id = query_data['qid']  # 生成查詢ID
        
        # 也可以將查詢的關係納入考慮
        query_relations = query_data.get('relations', [])
        if query_relations:
            relation_texts = [relation_to_text(rel) for rel in query_relations]
            query_text += " " + " ".join(relation_texts)
        
        # 編碼查詢
        encoded_input = tokenizer([query_text], padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
            query_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
        
        # 計算相似度分數
        cos_scores = torch.mm(query_embedding, doc_embeddings.T)[0].cpu()
        
        # 找到top-k結果
        top_k_scores, top_k_indices = torch.topk(cos_scores, min(top_k, len(doc_ids)), largest=True)
        
        # 寫入結果
        with open(output_path, 'a', encoding='utf-8') as f:
            for rank, (score, idx) in enumerate(zip(top_k_scores, top_k_indices), 1):
                doc_id = doc_ids[idx.item()]
                line = f"{query_id} Q0 {doc_id} {rank} {score:.4f} JSON-triplet-run\n"
                f.write(line)
    
    print(f"Search results saved to: {output_path}")

def main():
    """主函數"""
    # 設定檔案路徑
    QUERIES_FILE = 'data/kg/topic_10q.json'  # 您的查詢JSON檔案
    TRIPLETS_FILE = 'data/kg/relation_bm25.json'  # 您的三元組JSON檔案
    MODEL_NAME = 'pretrained_model/sentence-transformers/all-MiniLM-L6-v2'
    EMBEDDINGS_FOLDER = 'index_triplets/'
    OUTPUT_PATH = "runs/retrieval/kg_10q.txt"
    TOP_K = 1000
    
    # 步驟1：為三元組創建嵌入向量（只需要運行一次）
    print("Step 1: Creating triplet embeddings...")
    create_triplet_embeddings(TRIPLETS_FILE, MODEL_NAME, EMBEDDINGS_FOLDER)
    
    # 步驟2：執行搜尋
    print("\nStep 2: Searching for relevant documents...")
    search_json_queries(QUERIES_FILE, EMBEDDINGS_FOLDER, MODEL_NAME, TOP_K, OUTPUT_PATH)
    
    print("\nDone!")

if __name__ == '__main__':
    main()