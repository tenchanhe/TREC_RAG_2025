import json
import gzip
import os
from typing import List, Dict

def extract_filename_and_docid(line: str) -> tuple:
    """
    從bm25_run.txt的行中提取檔案名和docid
    
    Args:
        line: 格式如 "2024-145979 Q0 msmarco_v2.1_doc_13_1647729865#0_3617397938 1 21.865700 Anserini"
    
    Returns:
        tuple: (filename, docid)
    """
    parts = line.strip().split()
    if len(parts) < 3:
        return None, None
    
    docid = parts[2]  # msmarco_v2.1_doc_13_1647729865#0_3617397938
    
    # 找到最後一個 '#' 的位置
    last_hash_pos = docid.rfind('#')
    if last_hash_pos == -1:
        return None, None
    
    # 提取檔案名部分 (msmarco_v2.1_doc_13)
    filename_part = docid[:last_hash_pos]
    second_hash_pos = filename_part.rfind('#')
    if second_hash_pos != -1:
        filename_part = filename_part[:second_hash_pos]
    
    # 找到最後一個 '_' 的位置，取到該位置之前的部分
    last_underscore_pos = filename_part.rfind('_')
    if last_underscore_pos != -1:
        filename_base = filename_part[:last_underscore_pos]
    else:
        filename_base = filename_part
    
    # breakpoint()
    filename = filename_base[:-2] + 'segmented_' + filename_base[-2:] + '.json.gz'
    
    return filename, docid

def read_json_gz_file(filepath: str) -> List[Dict]:
    """
    讀取JSON.gz檔案
    
    Args:
        filepath: JSON.gz檔案路徑
    
    Returns:
        List[Dict]: JSON物件列表
    """
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = []
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
            return data
    except Exception as e:
        print(f"讀取檔案 {filepath} 時發生錯誤: {e}")
        return []

def find_segment_by_docid(data: List[Dict], target_docid: str) -> str:
    """
    在JSON資料中找到指定docid的segment
    
    Args:
        data: JSON資料列表
        target_docid: 要搜尋的docid
    
    Returns:
        str: 找到的segment內容，如果沒找到則返回None
    """
    for item in data:
        if item.get('docid') == target_docid:
            return item.get('segment', '')
    return None

def process_bm25_run_streaming_simple(bm25_file: str, output_file: str, corpus_dir: str = '.'):
    """
    處理BM25運行結果文件，串流式提取segments並以簡單格式寫入輸出文件
    每行一個segment，適合非常大的文件
    
    Args:
        bm25_file: bm25_run.txt檔案路徑
        output_file: 輸出文件路徑
        corpus_dir: JSON.gz檔案所在目錄
    """
    processed_files = {}  # 快取已讀取的檔案
    segments_found = 0
    total_processed = 0
    
    try:
        # 開啟輸出文件，每行寫一個segment
        with open(output_file, 'w', encoding='utf-8') as out_f:
            with open(bm25_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    total_processed += 1
                    
                    filename, docid = extract_filename_and_docid(line)
                    if not filename or not docid:
                        print(f"第 {line_num} 行格式錯誤: {line}")
                        continue
                    
                    filepath = os.path.join(corpus_dir, filename)
                    
                    # 檢查檔案是否存在
                    if not os.path.exists(filepath):
                        print(f"檔案不存在: {filepath}")
                        continue
                    
                    # 如果檔案還沒被讀取過，則讀取並快取
                    if filename not in processed_files:
                        print(f"讀取檔案: {filename}")
                        processed_files[filename] = read_json_gz_file(filepath)
                        # 限制快取大小以節省記憶體
                        if len(processed_files) > 5:  # 限制快取最多5個檔案
                            oldest_file = next(iter(processed_files))
                            del processed_files[oldest_file]
                            print(f"清除快取: {oldest_file}")
                    
                    # 在資料中搜尋segment
                    segment = find_segment_by_docid(processed_files[filename], docid)
                    if segment:
                        segments_found += 1
                        
                        # 將segment寫入文件（每行一個JSON物件）
                        segment_data = {
                            "docid": docid,
                            "segment": segment
                        }
                        out_f.write(json.dumps(segment_data, ensure_ascii=False) + '\n')
                        
                        print(f"找到segment {segments_found} (docid: {docid}): {segment[:100]}...")
                        
                        # 每找到100個segments就刷新一次文件緩衝區
                        if segments_found % 100 == 0:
                            out_f.flush()
                            print(f"已處理 {total_processed} 行，找到 {segments_found} 個segments")
                    else:
                        print(f"未找到docid: {docid}")
        
        print(f"\n處理完成！")
        print(f"總共處理了 {total_processed} 行")
        print(f"找到了 {segments_found} 個segments")
        print(f"結果已保存到: {output_file}")
        
    except Exception as e:
        print(f"處理檔案 {bm25_file} 時發生錯誤: {e}")

if __name__ == "__main__":
    bm25_file = "runs/retrieval/bm25_run_test1.txt"
    corpus_dir = "/tmp2/TREC_RAG2025/corpus/msmarco_v2.1_doc_segmented"
    # output_file = "extracted_segments2.jsonl"
    
    # print("選擇輸出格式:")
    # print("1. JSON陣列格式 (適合中等大小的檔案)")
    # print("2. 每行一個JSON物件格式 (適合非常大的檔案)")
    
    choice = 2
    
    process_bm25_run_streaming_simple(bm25_file, output_file, corpus_dir)