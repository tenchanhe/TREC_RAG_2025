import gzip
import json
from collections import defaultdict

def analyze_msmarco_file(file_path, sample_size=5):
    """
    分析 MSMARCO JSON.GZ 文件
    
    Args:
        file_path (str): 文件路徑
        sample_size (int): 要顯示的樣本數量
    """
    try:
        # 讀取壓縮的JSON文件
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            print(f"開始分析文件: {file_path}")
            
            # 初始化統計變量
            total_records = 0
            attribute_counts = defaultdict(int)
            sample_records = []
            
            # 逐行分析
            for line in f:
                try:
                    record = json.loads(line.strip())
                    total_records += 1
                    
                    # 收集樣本
                    if total_records <= sample_size:
                        sample_records.append(record)
                    
                    # 統計屬性出現次數
                    for key in record.keys():
                        attribute_counts[key] += 1
                        
                except json.JSONDecodeError:
                    continue  # 跳過無效行
            
            # 顯示基本統計信息
            print("\n=== 基本統計 ===")
            print(f"總記錄數: {total_records}")
            print(f"發現的屬性: {list(attribute_counts.keys())}")
            
            # 顯示屬性出現頻率
            print("\n=== 屬性出現統計 ===")
            for attr, count in attribute_counts.items():
                print(f"{attr}: {count} 次 ({count/total_records*100:.1f}%)")
            
            # 顯示樣本記錄
            print(f"\n=== 前 {sample_size} 條記錄樣本 ===")
            for i, record in enumerate(sample_records, 1):
                print(f"\n記錄 {i}:")
                for key, value in record.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}... (長度: {len(value)})")
                    else:
                        print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"分析文件時出錯: {str(e)}")

if __name__ == "__main__":
    file_path = "./data/corpus/processed/msmarco_v2.1_doc_segmented_00.json.gz"
    # file_path = "/tmp2/TREC_RAG2025/corpus/msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_00.json.gz"
    analyze_msmarco_file(file_path)
