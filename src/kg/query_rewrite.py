import ollama
import json
import re, unicodedata, tqdm

PROMPT_TEMPLETE = """Please break down the process of answering the question into as few subobjectives as possible based on semantic analysis.
Here is an example: 
Q: Which of the countries in the Caribbean has the smallest country calling code?
Output: ['Search the countries in the Caribbean', 'Search the country calling code for each Caribbean country', 'Compare the country calling codes to find the smallest one']

Now you need to directly output subobjectives of the following question in list format without other information or notes. 
Q:{query}"""

def call_llm(prompt, ollama_model_name):
    try:
        response = ollama.generate(model=ollama_model_name, prompt=prompt, stream=False)
        
        # Ollama 的 generate 返回的 'response' 包含一個 'response' 鍵
        # 這裡假設模型會嚴格按照要求返回逗號分隔的關鍵字
        response = response['response'].strip()
        return response
        
    except Exception as e:
        print(f"call llm 時發生錯誤: {e}")

def split_query(input_filepath, output_filepath, ollama_model_name="llama3", split_segment=False):
    results = []

    with open(input_filepath, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            query_id = line.split('\t')[0].strip()
            query = line.split('\t')[1].strip()

            prompt = PROMPT_TEMPLETE.format(query=query)

            response = call_llm(prompt, ollama_model_name)
            results.append(
                {"id": query_id,
                 "query": query,
                 "sub_object": response}
            )

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("所有結果已成功保存！")

if __name__ == "__main__":
    # 定義輸入和輸出檔案的路徑
    input_file = "data/topics/top10_topic.txt"
    output_file = "data/topics/query_rewrite.json"
    MODEL = "qwen2.5:32b"
    split = True

    split_query(input_file, output_file, MODEL, split)