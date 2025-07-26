
import json, os, torch, re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def split_into_sentences(text):
    """
    將文本拆分成句子
    """
    # 使用正則表達式分句，考慮多種結尾符號
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    
    # 過濾掉空字串和太短的句子
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    # breakpoint()
    
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
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 設定生成參數
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 5,
        "num_return_sequences": 3,
    }

    # 載入 JSON 檔（格式：list of dicts with 'id' and 'narrative'）
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化輸出檔案，寫入 JSON 陣列開頭
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write('[\n')

    # data = [{
    #     "id": "464",
    #     "narrative": "I want a thorough understanding of what makes up a community, including its definitions in various contexts like science and what it means to be a 'civilized community.' I'm also interested in related terms like 'grassroots organizations,' how communities set boundaries and priorities, and their roles in important areas such as preparedness and nation-building."
    # }]
    processed_count = 0

    for item in data:
        try:
            doc_id = item["id"]
            narrative = item["narrative"].strip()

            # 拆分句子
            sentences = split_into_sentences(narrative)
            print(f"[{doc_id}] 拆分成 {len(sentences)} 個句子")

            # 對每個句子進行關係抽取
            all_relations = []
            for sent_idx, sentence in enumerate(sentences):
                print(f"  處理句子 {sent_idx + 1}/{len(sentences)}: {sentence[:50]}...")

                # tokenization
                model_inputs = tokenizer(sentence, max_length=256, padding=True, truncation=True, return_tensors='pt')
                model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

                # generate
                generated_tokens = model.generate(
                    model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    **gen_kwargs,
                )

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

                for pred in decoded_preds:
                    triplets = extract_triplets(pred)
                    all_relations.extend(triplets)

                del model_inputs, generated_tokens, decoded_preds
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # 寫入結果
            result = {
                "docid": doc_id,
                "narrative": narrative,
                "relations": all_relations
            }

            with open(output_file_path, 'a', encoding='utf-8') as output_file:
                if processed_count > 0:
                    output_file.write(',\n')
                json.dump(result, output_file, ensure_ascii=False, indent=2)

            processed_count += 1

            del all_relations, result
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"[{item.get('id', '??')}] 處理錯誤: {e}")
            continue

    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        output_file.write('\n]')

    print(f"✅ 處理完成！共處理 {processed_count} 筆資料")
    print(f"📄 結果儲存到：{output_file_path}")

if __name__ == "__main__":
    # 設定檔案路徑
    input_file = "data/topics/trec_rag_2025_queries.jsonl"  # 您的輸入 JSONL 檔案路徑
    output_file = "data/kg/topic_2025.json"  # 輸出 JSON 檔案路徑
    model_path = "./pretrained_model/Babelscape/rebel-large/"
    
    # 執行處理
    process_topics_file(input_file, output_file, model_path)