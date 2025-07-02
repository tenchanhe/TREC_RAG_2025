import os
import gzip
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import shutil

# --- 來自您範例的程式碼 (無變動) ---
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def process_batch(contents, tokenizer, model, device):
    """
    對一個批次的文本進行編碼處理，返回嵌入向量。
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
    return sentence_embeddings.cpu()

# --- 主要邏輯修改 ---

def create_corpus_embeddings_resumable(
    corpus_folder, model_name, temp_output_folder, batch_size=32
):
    """
    可接續執行的索引建立流程。
    為每個輸入文件在暫存資料夾中創建對應的嵌入檔案。
    """
    # 建立暫存資料夾 (如果不存在)
    os.makedirs(temp_output_folder, exist_ok=True)
    
    # 檢查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 從 HuggingFace Hub 加載模型
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # 1. 決定需要處理的檔案
    all_source_files = {
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(corpus_folder)
        for filename in filenames if filename.endswith('.json.gz')
    }
    
    processed_files = set()
    for filename in os.listdir(temp_output_folder):
        if filename.endswith('.done'):
            # 從 .done 標記檔還原原始路徑
            original_path = filename[:-5].replace('___', '/')
            processed_files.add(original_path)

    files_to_process = sorted(list(all_source_files - processed_files))
    
    if not files_to_process:
        print("All files have already been processed and indexed.")
        return

    print(f"Total files: {len(all_source_files)}. Already processed: {len(processed_files)}. Files to process: {len(files_to_process)}")

    # 2. 處理剩餘的檔案
    for filepath in tqdm(files_to_process, desc="Processing files"):
        file_embeddings = []
        file_doc_ids = []
        
        contents_batch = []
        ids_batch = []

        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'id' in data and 'contents' in data and data['contents']:
                            ids_batch.append(data['id'])
                            contents_batch.append(data['contents'])

                            if len(contents_batch) >= batch_size:
                                batch_embeddings = process_batch(contents_batch, tokenizer, model, device)
                                file_embeddings.append(batch_embeddings)
                                file_doc_ids.extend(ids_batch)
                                contents_batch, ids_batch = [], []

                    except (json.JSONDecodeError, KeyError):
                        continue
            
            # 處理檔案中剩餘的最後一個批次
            if contents_batch:
                batch_embeddings = process_batch(contents_batch, tokenizer, model, device)
                file_embeddings.append(batch_embeddings)
                file_doc_ids.extend(ids_batch)

            # 3. 儲存該檔案的暫存結果
            if file_doc_ids:
                # 將檔案路徑轉換為安全的檔名
                safe_filename = filepath.replace('/', '___')
                
                # 儲存 IDs
                with open(os.path.join(temp_output_folder, f"{safe_filename}.ids.json"), 'w', encoding='utf-8') as f_ids:
                    json.dump(file_doc_ids, f_ids)
                
                # 儲存 embeddings
                final_file_embeddings = torch.cat(file_embeddings).numpy()
                np.save(os.path.join(temp_output_folder, f"{safe_filename}.embed.npy"), final_file_embeddings)

                # 建立一個完成標記檔
                with open(os.path.join(temp_output_folder, f"{safe_filename}.done"), 'w') as f_done:
                    f_done.write('done')

        except Exception as e:
            print(f"\nAn error occurred while processing {filepath}: {e}")
            print("The script will stop. You can run it again to resume.")
            return # 發生錯誤時停止，以便下次可以從此檔案繼續

    print("All individual files have been processed.")


def merge_temp_files(temp_output_folder, output_embeddings_path, output_ids_path):
    """
    合併暫存資料夾中的所有結果，生成最終的索引檔。
    """
    print(f"\nMerging temporary files from '{temp_output_folder}'...")
    all_embeddings_list = []
    all_ids_list = []

    # 按檔名排序以確保一致性
    temp_files = sorted(os.listdir(temp_output_folder))
    
    # 找出所有對應的 ids.json 和 embed.npy 檔案
    json_files = [f for f in temp_files if f.endswith('.ids.json')]

    if not json_files:
        print("No temporary files to merge.")
        return

    for json_filename in tqdm(json_files, desc="Merging files"):
        base_name = json_filename[:-9] # 移除 '.ids.json'
        embed_filename = f"{base_name}.embed.npy"
        
        json_path = os.path.join(temp_output_folder, json_filename)
        embed_path = os.path.join(temp_output_folder, embed_filename)

        if os.path.exists(embed_path):
            # 載入 IDs
            with open(json_path, 'r', encoding='utf-8') as f:
                ids = json.load(f)
                all_ids_list.extend(ids)
            
            # 載入 Embeddings
            embeddings = np.load(embed_path)
            all_embeddings_list.append(embeddings)
    
    if not all_ids_list:
        print("No data found in temporary files.")
        return

    # 合併成一個大的 NumPy 陣列
    final_embeddings = np.concatenate(all_embeddings_list, axis=0)
    
    # 儲存最終結果
    print(f"Saving final merged files...")
    np.save(output_embeddings_path, final_embeddings)
    with open(output_ids_path, 'w', encoding='utf-8') as f:
        json.dump(all_ids_list, f)
        
    print(f"\nMerge complete!")
    print(f"Final embeddings saved to '{output_embeddings_path}' ({final_embeddings.shape})")
    print(f"Final IDs saved to '{output_ids_path}' ({len(all_ids_list)} documents)")
    print(f"You can now run search.py. The temporary folder '{temp_output_folder}' can be deleted if desired.")


if __name__ == '__main__':
    # --- 設定 ---
    CORPUS_FOLDER = 'data/corpus/processed/'
    MODEL_NAME = 'pretrained_model/sentence-transformers/all-MiniLM-L6-v2/'
    
    # 暫存與最終輸出路徑
    TEMP_OUTPUT_FOLDER = 'temp_output'
    FINAL_EMBEDDINGS_FILE = 'corpus_embeddings.npy'
    FINAL_IDS_FILE = 'corpus_ids.json'
    
    BATCH_SIZE = 1024

    # --- 執行流程 ---
    
    # 步驟 1: 處理所有單獨的檔案，支援中斷續傳
    create_corpus_embeddings_resumable(
        corpus_folder=CORPUS_FOLDER,
        model_name=MODEL_NAME,
        temp_output_folder=TEMP_OUTPUT_FOLDER,
        batch_size=BATCH_SIZE
    )
    
    # 步驟 2: 合併所有暫存檔為最終的索引
    merge_temp_files(
        temp_output_folder=TEMP_OUTPUT_FOLDER,
        output_embeddings_path=FINAL_EMBEDDINGS_FILE,
        output_ids_path=FINAL_IDS_FILE
    )