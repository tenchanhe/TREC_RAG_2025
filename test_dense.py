import os
import gzip
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

# --- 來自您範例的程式碼 ---

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# --- 主程式開始 ---

def create_corpus_embeddings(corpus_file, model_name, output_embeddings_path, output_ids_path, batch_size=32):
    """
    遍歷語料庫文件夾，為每個文件內容創建嵌入，並將結果保存到磁碟。
    """
    # 檢查是否有可用的 GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    device = "cpu"

    # 從 HuggingFace Hub 加載模型
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval() # 設置為評估模式

    # # 找到所有 .json.gz 文件
    # filepaths = [os.path.join(dirpath, filename)
    #              for dirpath, _, filenames in os.walk(corpus_folder)
    #              for filename in filenames if filename.endswith('.json.gz')]

    # print(f"Found {len(filepaths)} .json.gz files in '{corpus_folder}'.")

    all_doc_ids = []
    all_embeddings = []
    
    # 使用 tqdm 顯示進度條
    # pbar = tqdm(total=len(filepaths), desc="Processing files", position=0)

    contents_batch = []
    ids_batch = []
    try:
        with gzip.open(corpus_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="處理內容"):
                try:
                    data = json.loads(line)
                    # 確保 'id' 和 'contents' 欄位存在
                    if 'id' in data and 'contents' in data and data['contents']:
                        ids_batch.append(data['id'])
                        contents_batch.append(data['contents'])

                        # 當批次達到指定大小時，進行編碼
                        if len(contents_batch) >= batch_size:
                            process_batch(contents_batch, ids_batch, tokenizer, model, device, all_doc_ids, all_embeddings)
                            contents_batch, ids_batch = [], [] # 清空批次

                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {corpus_file}")
                    continue
        
        # 處理檔案中剩餘的最後一個批次
        if contents_batch:
            process_batch(contents_batch, ids_batch, tokenizer, model, device, all_doc_ids, all_embeddings)

    except Exception as e:
        print(f"Error processing file {corpus_file}: {e}")
            
    #     pbar.update(1)

    # pbar.close()

    # 將所有嵌入轉換為 NumPy 陣列並保存
    print("Saving embeddings and IDs...")
    final_embeddings = torch.cat(all_embeddings).cpu().numpy()
    np.save(output_embeddings_path, final_embeddings)

    # 保存 IDs
    with open(output_ids_path, 'w', encoding='utf-8') as f:
        json.dump(all_doc_ids, f)

    print(f"Indexing complete. Embeddings saved to '{output_embeddings_path}', IDs saved to '{output_ids_path}'.")


def process_batch(contents, ids, tokenizer, model, device, all_doc_ids, all_embeddings):
    """
    對一個批次的文本進行編碼處理。
    """
    # Tokenize sentences
    encoded_input = tokenizer(contents, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    all_embeddings.append(sentence_embeddings.cpu())
    all_doc_ids.extend(ids)


if __name__ == '__main__':
    # --- 設定 ---
    CORPUS_FOLDER = './test_data/test.json.gz'
    MODEL_NAME = 'pretrained_model/sentence-transformers/all-MiniLM-L6-v2'
    OUTPUT_EMBEDDINGS_FILE = 'test/test_embeddings.npy'
    OUTPUT_IDS_FILE = 'test/test_corpus_ids.json'
    BATCH_SIZE = 1024

    create_corpus_embeddings(
        corpus_file=CORPUS_FOLDER,
        model_name=MODEL_NAME,
        output_embeddings_path=OUTPUT_EMBEDDINGS_FILE,
        output_ids_path=OUTPUT_IDS_FILE,
        batch_size=BATCH_SIZE
    )