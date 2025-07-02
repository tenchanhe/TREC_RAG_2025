#!/usr/bin/env python3

import json
import gzip
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def convert_document(doc):
    """將文檔轉換為 Pyserini 期望的格式"""
    return {
        "id": doc["docid"],
        "contents": doc["segment"],
        "title": doc["title"],
        "headings": doc["headings"],
        "url": doc["url"]
    }

def process_file(input_file):
    """處理單個文件"""
    output_file = Path("data/corpus/processed") / input_file.name
    
    # 計算總行數
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in:
        with gzip.open(output_file, 'wt', encoding='utf-8') as f_out:
            for line in tqdm(f_in, total=total_lines, desc=f"處理 {input_file.name}"):
                doc = json.loads(line)
                converted_doc = convert_document(doc)
                f_out.write(json.dumps(converted_doc) + '\n')

def main():
    # 設定目錄
    input_dir = Path("/tmp2/TREC_RAG2025/corpus/msmarco_v2.1_doc_segmented")
    output_dir = Path("data/corpus/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 獲取所有需要處理的文件
    input_files = list(input_dir.glob("*.json.gz"))
    print(f"找到 {len(input_files)} 個文件需要處理")

    # 設定進程數為 16
    num_processes = 16
    print(f"使用 {num_processes} 個進程進行處理")

    # 使用進程池處理文件
    with Pool(num_processes) as pool:
        list(tqdm(
            pool.imap(process_file, input_files),
            total=len(input_files),
            desc="總體進度"
        ))

if __name__ == "__main__":
    main() 