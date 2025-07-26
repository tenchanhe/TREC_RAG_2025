
import json, os, torch, re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def split_into_sentences(text):
    """
    將文本拆分成句子
    """
    # 使用正則表達式分句，考慮多種結尾符號
    sentences = re.split(r'[.!?]+\s+', text.strip())
    
    # 過濾掉空字串和太短的句子
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    return sentences

def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets


def process_topics_file(input_file_path, output_file_path, model_path):
   
    # 載入模型和 tokenizer
    print("載入模型和 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # 設定生成參數
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 5,
        "num_return_sequences": 3,
    }
    
    # 初始化輸出檔案，寫入 JSON 陣列開頭
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write('[\n')
    
    processed_count = 0
    
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            try:
                query = line.split('\t')[1].strip()
                # breakpoint()
                                
                # 將 segment 拆分成句子
                sentences = split_into_sentences(query)
                print(f"  拆分成 {len(sentences)} 個句子")
                
                # 對每個句子進行關係抽取
                all_relations = []
                for sent_idx, sentence in enumerate(sentences):
                    print(f"  處理句子 {sent_idx + 1}/{len(sentences)}: {sentence[:50]}...")
                    
                    # 對句子進行 tokenization
                    model_inputs = tokenizer(sentence, max_length=256, padding=True, truncation=True, return_tensors='pt')
                    
                    # 生成關係抽取結果
                    generated_tokens = model.generate(
                        model_inputs["input_ids"].to(model.device),
                        attention_mask=model_inputs["attention_mask"].to(model.device),
                        **gen_kwargs,
                    )
                    
                    # 解碼預測結果
                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
                    
                    # 抽取三元組
                    for pred in decoded_preds:
                        triplets = extract_triplets(pred)
                        all_relations.extend(triplets)
                    # print(sentence)
                    # print(decoded_preds)
                    # print()

                    # 清理這個句子的記憶體
                    del model_inputs, generated_tokens, decoded_preds
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # 建立結果物件
                result = {
                    "docid": query,
                    "relations": all_relations
                }
                
                # 立即寫入檔案
                with open(output_file_path, 'a', encoding='utf-8') as output_file:
                    if processed_count > 0:
                        output_file.write(',\n')
                    json.dump(result, output_file, ensure_ascii=False, indent=2)
                
                processed_count += 1
                
                # 清理記憶體
                del all_relations, result
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解析錯誤: {e}")
                continue
            except Exception as e:
                print(f"第 {line_num} 行處理錯誤: {e}")
                continue
    
    # 結束 JSON 陣列
    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        output_file.write('\n]')
    
    print(f"處理完成！共處理 {processed_count} 筆資料")
    print(f"結果已儲存到 {output_file_path}")

if __name__ == "__main__":
    # 設定檔案路徑
    input_file = "data/topics/test_topic.txt"  # 您的輸入 JSONL 檔案路徑
    output_file = "data/kg/topic_2024.json"  # 輸出 JSON 檔案路徑
    model_path = "./pretrained_model/Babelscape/rebel-large/"
    
    # 執行處理
    process_topics_file(input_file, output_file, model_path)