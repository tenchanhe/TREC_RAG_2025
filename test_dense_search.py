import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json

# --- 來自您範例的程式碼 ---

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# --- 主程式開始 ---

def search_topics(topics_file, model_name, corpus_embeddings_path, corpus_ids_path, top_k=1000, run_name="Gemini-MiniLM-run"):
    """
    加載查詢，與語料庫嵌入進行比較，並以 TREC 格式輸出 top-k 結果。
    """
    # # 檢查是否有可用的 GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    device = "cpu"

    # 從 HuggingFace Hub 加載模型
    print("Loading model for query encoding...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # 加載預先計算好的語料庫嵌入和 ID
    print("Loading corpus index...")
    corpus_embeddings = torch.from_numpy(np.load(corpus_embeddings_path)).to(device)
    with open(corpus_ids_path, 'r', encoding='utf-8') as f:
        corpus_ids = json.load(f)
    
    print(f"Loaded {len(corpus_ids)} document embeddings.")

    # 讀取並處理查詢
    print(f"Processing queries from '{topics_file}'...")
    with open(topics_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 分割 queryid 和 query text
            query_id, query_text = line.split('\t', 1)
            # breakpoint()
            
            # --- 編碼單個查詢 ---
            encoded_input = tokenizer([query_text], padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = model(**encoded_input)
            query_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
            query_embedding = F.normalize(query_embedding, p=2, dim=1)

            # --- 進行向量搜尋 ---
            # 使用矩陣乘法高效計算餘弦相似度 (因為向量都已標準化)
            # (1, dim) @ (dim, N) -> (1, N)
            cos_scores = torch.mm(query_embedding, corpus_embeddings.T)[0]

            # 獲取 top-k 結果
            top_results = torch.topk(cos_scores, k=min(top_k, len(corpus_ids)), largest=True)

            # --- 以 TREC 格式輸出 ---
            # 格式: query_id Q0 document_id rank score run_name
            for rank, (score, idx) in enumerate(zip(top_results.values, top_results.indices), 1):
                doc_id = corpus_ids[idx.item()]
                # print(f"{query_id} Q0 {doc_id} {rank} {score.item():.4f} {run_name}")
                # 直接寫入檔案，避免大量print造成效能問題
                print(f"{query_id} Q0 {doc_id} {rank} {score.item():.4f} {run_name}", flush=False)


if __name__ == '__main__':
    # --- 設定 ---
    TOPICS_FILE = '/tmp2/TREC_RAG2025/topics/topics.rag24.test.txt'
    MODEL_NAME = 'pretrained_model/sentence-transformers/all-MiniLM-L6-v2'
    CORPUS_EMBEDDINGS_FILE = './test/test_embeddings.npy'
    CORPUS_IDS_FILE = './test/test_corpus_ids.json'
    TOP_K = 1000
    
    search_topics(
        topics_file=TOPICS_FILE,
        model_name=MODEL_NAME,
        corpus_embeddings_path=CORPUS_EMBEDDINGS_FILE,
        corpus_ids_path=CORPUS_IDS_FILE,
        top_k=TOP_K
    )