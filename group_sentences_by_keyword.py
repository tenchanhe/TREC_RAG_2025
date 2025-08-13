import json
import jsonlines
from collections import defaultdict
import os
import ast
from tqdm import tqdm
import re
import unicodedata
from src.utils.normalized_list import normalize_to_list

# Following the project's conventions, we assume the input file is a .jsonl file.
# This script expects each line to be a JSON object with "sentence" and "keywords" fields.
KEYWORDS_FILE = 'data/kg/keywords_sentence.jsonl'
SEGMENTS_FILE = 'data/segment/dense_10q.jsonl'
OUTPUT_FILE = 'grouped_by_keyword.json'

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

def group_sentences_by_keyword(segments_filepath, keywords_filepath, output_filepath):
    """
    Groups sentences from segments file by their keywords from keywords file.
    Each segment is split into sentences, then matched to keywords by 'id' and order.
    Saves the result to a JSON file.
    """
    keyword_groups = defaultdict(list)

    # 檢查檔案是否存在
    if not os.path.exists(segments_filepath) or not os.path.exists(keywords_filepath):
        print(f"Error: File not found. Please check '{segments_filepath}' and '{keywords_filepath}'.")
        return

    # 讀取 keywords 資料，根據 id 分組
    keywords_by_id = defaultdict(list)
    with jsonlines.open(keywords_filepath) as reader:
        for item in reader:
            sid = item.get('id')
            # sentence = item.get('sentence')
            keywords_data = normalize_to_list(item.get('keywords'))
            if not keywords_data:
                keywords_by_id[sid].append({
                    "keywords": []  # 如果沒有關鍵字，仍然需要保持結構
                })
            else:
                keywords_by_id[sid].append({
                    "keywords": keywords_data
                })
    # breakpoint()

    # 讀取 segments 檔案，拆分句子並根據 id 和順序對應 keywords
    with jsonlines.open(segments_filepath) as reader:
        for item in tqdm(reader, desc="Grouping sentences"):
            sid = item.get('id')
            text = item.get('contents')
            if not sid or not text:
                continue
            sentences = split_into_sentences(text, split=True)
            keyword_items = keywords_by_id.get(sid, [])
            # breakpoint()
            for idx, sentence in enumerate(sentences):
                cleaned_sentence = clean_text(sentence)
                if idx < len(keyword_items):
                    keywords = keyword_items[idx]["keywords"]
                    for keyword in keywords:
                        if keyword:
                            keyword_groups[keyword].append(cleaned_sentence)
                else:
                    breakpoint()
                    raise ValueError(f"Not enough keywords for sentence {idx} in segment {sid}. Expected {len(sentences)}, got {len(keyword_items)}.")
    # 儲存結果
    with open(output_filepath, 'w', encoding='utf-8') as f_out:
        json.dump(keyword_groups, f_out, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    group_sentences_by_keyword(SEGMENTS_FILE, KEYWORDS_FILE, OUTPUT_FILE)
