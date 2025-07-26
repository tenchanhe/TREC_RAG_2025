from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import json, torch, re, unicodedata
from labels import entity_labels

def split_into_sentences(text, split):
    """
    將文本拆分成句子
    """
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


def process_jsonl_file(input_file_path, output_file_path, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    
    # 初始化輸出檔案，寫入 JSON 陣列開頭
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write('[\n')
    
    processed_count = 0
    
    print("開始處理 JSONL 檔案...")
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            try:
                # 解析每一行 JSON
                data = json.loads(line.strip())
                # docid = data['docid']
                # segment = data['segment']
                docid = data['id']
                segment = data['contents']
                
                print(f"處理第 {line_num} 行，docid: {docid}")
                
                # 將 segment 拆分成句子
                # sentences = split_into_sentences(segment, True)
                sentences = split_into_sentences(segment, False)
                print(f"  拆分成 {len(sentences)} 個句子")
                
                all_entities = []
                all_type = []
                for sent_idx, sentence in enumerate(sentences):
                    print(f"  處理句子 {sent_idx + 1}/{len(sentences)}: {sentence[:50]}...")
                    
                    ner_results = nlp(sentence)
                    all_entities.extend([word['word'] for word in ner_results])
                    all_type.extend([word['entity'] for word in ner_results])
                    # breakpoint()
                
                # 建立結果物件
                result = {
                    "docid": docid,
                    "segment": segment,
                    "entities": all_entities,
                    "type": all_type
                }
                
                # 立即寫入檔案
                with open(output_file_path, 'a', encoding='utf-8') as output_file:
                    if processed_count > 0:
                        output_file.write(',\n')
                    json.dump(result, output_file, ensure_ascii=False, indent=2)
                
                processed_count += 1
                
                # 清理記憶體
                del all_entities, result
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解析錯誤: {e}")
                continue
            except Exception as e:
                print(f"第 {line_num} 行處理錯誤: {e}")
                continue



if __name__ == "__main__":
    # 設定檔案路徑
    input_file = "data/segment/dense_10q.jsonl"  # 您的輸入 JSONL 檔案路徑
    output_file = "data/kg/entities_bertlarge_full.json"  # 輸出 JSON 檔案路徑
    # model_path = "pretrained_model/dslim/bert-base-NER/"
    model_path = "pretrained_model/dslim/bert-large-NER/"
    
    # 執行處理
    process_jsonl_file(input_file, output_file, model_path)