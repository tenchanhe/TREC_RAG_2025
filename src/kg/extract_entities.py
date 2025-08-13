import ollama
import jsonlines
import re, unicodedata, tqdm
import os
import glob

PROMPT_TEMPLETE = \
"""Extract {min_entities} to {max_entities} named entities or concept-level entities from the following text.
Return only entities that could be nodes in a knowledge graph.
Format the output strictly as a Python list of strings.
Do NOT include any introductory phrases, explanations, or additional text.
Each entity should be distinct and expressed in its canonical form.
Example output format: ["entity one", "entity two", "entity three"]

Text:
{content}

Entities:"""

def split_into_sentences(text, split):
    if split:
        # 使用正則表達式分句，考慮多種結尾符號
        sentences = re.split(r'[.!?]+\s+', text.strip())
        
        # 過濾掉空字串和太短的句子
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    # breakpoint()
    return [clean_text(text)]

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

def call_llm(prompt, ollama_model_name):
    try:
        response = ollama.generate(model=ollama_model_name, prompt=prompt, stream=False)
        
        # Ollama 的 generate 返回的 'response' 包含一個 'response' 鍵
        # 這裡假設模型會嚴格按照要求返回逗號分隔的關鍵字
        response = response['response'].strip()
        return response
        
        # # 將抽取到的關鍵字字串分割成列表，並去除可能的多餘空格
        # keywords_list = [kw.strip() for kw in raw_keywords.split(',') if kw.strip()]
        
    except Exception as e:
        print(f"call llm 時發生錯誤: {e}")

def extract_keywords_from_jsonl(input_filepath, output_filepath, ollama_model_name, split_segment=False):
    results = []

    print(f"正在從 '{input_filepath}' 讀取資料並抽取關鍵字...")

    with jsonlines.open(input_filepath) as reader:
        for item in tqdm.tqdm(reader, desc="處理中"):
            item_id = item.get("doc_id")
            up_content = item.get("content")
            content = up_content.get("contents")
            # print("Processing", item_id)

            if item_id is None or content is None:
                print(f"警告：跳過無效的記錄，缺少 'id' 或 'contents': {item}")
                continue

            if split_segment:
                min_keywords=1
                max_keywords=3
                content = split_into_sentences(content, True)
                for sentence in content:
                    prompt = PROMPT_TEMPLETE.format(min_entities=min_keywords, max_entities=max_keywords, content=sentence)
                    response = call_llm(prompt, ollama_model_name)
                    results.append({
                        "id": item_id,
                        "keywords": response,
                        "sentence": sentence
                    })

            else:
                min_keywords=5
                max_keywords=10
                prompt = PROMPT_TEMPLETE.format(min_entities=min_keywords, max_entities=max_keywords, content=content)
                response = call_llm(prompt, ollama_model_name)
                results.append({
                    "id": item_id,
                    "keywords": response
                })

    print(f"關鍵字抽取完成。正在將結果寫入 '{output_filepath}'...")
    
    with jsonlines.open(output_filepath, mode='w') as writer:
        for res in results:
            writer.write(res)
    
    print("所有結果已成功保存！")

if __name__ == "__main__":
    # 定義輸入資料夾和輸出檔案的路徑
    input_folder = "data/segment/2025_dense_top1000/"
    output_folder = "data/kg/2025_entities/"
    MODEL = "phi4-mini:latest"
    split = True

    # 取得資料夾中所有的 .jsonl 檔案
    input_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    
    if not input_files:
        print(f"在資料夾 '{input_folder}' 中沒有找到 .jsonl 檔案")
        exit(1)
    
    print(f"找到 {len(input_files)} 個檔案要處理：")
    # for file in input_files:
    #     print(f"  - {file}")
    
    all_results = []
    
    # 處理每個檔案
    for input_file in tqdm.tqdm(input_files, desc="處理中"):
        output_file = output_folder + input_file.split('/')[-1].replace('.jsonl', '_entities.jsonl')
        if os.path.exists(output_file):
            print(f"檔案已存在，跳過: {output_file}")
            continue
        print(f"\n正在處理檔案: {input_file}")
        extract_keywords_from_jsonl(input_file, output_file, ollama_model_name=MODEL, split_segment=split)
    
    print("所有結果已成功保存！")