import ollama
import jsonlines
import re, unicodedata, tqdm

PROMPT_TEMPLETE = \
"""Extract factual knowledge triples from the following text. 
Each triple should represent a relationship between two entities and be suitable for use in a knowledge graph.

Format the output strictly as a Python list of triples. Each triple should be a tuple of the form (subject, relation, object).
Only include clear and factual relationships. Avoid vague or implied statements.

Example output format: [("entity", "relation", "entity"), ("entity", "relation", "entity"), ("entity", "relation", "entity")]

Text:
{content}

Triples:"""

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

def extract_keywords_from_jsonl(input_filepath, output_filepath, ollama_model_name="llama3", split_segment=False):
    results = []

    print(f"正在從 '{input_filepath}' 讀取資料並抽取關鍵字...")

    with jsonlines.open(input_filepath) as reader:
        for item in tqdm.tqdm(reader, desc="處理中"):
            item_id = item.get("id")
            content = item.get("contents")
            print("Processing", item_id)

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
                        "triple": response
                    })

            else:
                min_keywords=5
                max_keywords=10
                prompt = PROMPT_TEMPLETE.format(min_entities=min_keywords, max_entities=max_keywords, content=content)
                response = call_llm(prompt, ollama_model_name)
                results.append({
                    "id": item_id,
                    "triple": response
                })

    print(f"關鍵字抽取完成。正在將結果寫入 '{output_filepath}'...")
    
    with jsonlines.open(output_filepath, mode='w') as writer:
        for res in results:
            writer.write(res)
    
    print("所有結果已成功保存！")

if __name__ == "__main__":
    # 定義輸入和輸出檔案的路徑
    input_file = "data/segment/dense_10q.jsonl"
    output_file = "data/kg/triple.jsonl"
    MODEL = "llama3.2"
    split = False

    extract_keywords_from_jsonl(input_file, output_file, MODEL, split)