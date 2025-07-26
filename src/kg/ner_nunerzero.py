import json, torch, re
from gliner import GLiNER
from labels import entity_labels

def split_into_sentences(text):
    """
    將文本拆分成句子
    """
    # 使用正則表達式分句，考慮多種結尾符號
    sentences = re.split(r'[.!?]+\s+', text.strip())
    
    # 過濾掉空字串和太短的句子
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    return sentences

# def merge_entities(entities, text):
#     if not entities:
#         return []
#     merged = []
#     current = entities[0]
#     for next_entity in entities[1:]:
#         if next_entity['label'] == current['label'] and (next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']):
#             current['text'] = text[current['start']: next_entity['end']].strip()
#             current['end'] = next_entity['end']
#         else:
#             merged.append(current)
#             current = next_entity
#     # Append the last entity
#     merged.append(current)
#     return merged

# def process_jsonl_file(input_file_path, output_file_path, model_path):
#     model = GLiNER.from_pretrained(model_path)

#     # NuZero requires labels to be lower-cased!
#     labels = entity_labels
#     labels = [l.lower() for l in labels]
    
#     # 初始化輸出檔案，寫入 JSON 陣列開頭
#     with open(output_file_path, 'w', encoding='utf-8') as output_file:
#         output_file.write('[\n')
    
#     processed_count = 0
    
#     print("開始處理 JSONL 檔案...")
#     with open(input_file_path, 'r', encoding='utf-8') as file:
#         for line_num, line in enumerate(file, 1):
#             try:
#                 # 解析每一行 JSON
#                 data = json.loads(line.strip())
#                 # docid = data['docid']
#                 # segment = data['segment']
#                 docid = data['id']
#                 segment = data['contents']
                
#                 print(f"處理第 {line_num} 行，docid: {docid}")
                
#                 # 將 segment 拆分成句子
#                 sentences = split_into_sentences(segment)
#                 print(f"  拆分成 {len(sentences)} 個句子")
                
#                 all_entities = []
#                 for sent_idx, sentence in enumerate(sentences):
#                     print(f"  處理句子 {sent_idx + 1}/{len(sentences)}: {sentence[:50]}...")
                    
#                     entities = model.predict_entities(sentence, labels)
#                     entities = merge_entities(entities, sentence)
#                     all_entities.extend(entities)
                
#                 # 建立結果物件
#                 result = {
#                     "docid": docid,
#                     "segment": segment,
#                     "entities": all_entities
#                 }
                
#                 # 立即寫入檔案
#                 with open(output_file_path, 'a', encoding='utf-8') as output_file:
#                     if processed_count > 0:
#                         output_file.write(',\n')
#                     json.dump(result, output_file, ensure_ascii=False, indent=2)
                
#                 processed_count += 1
                
#                 # 清理記憶體
#                 del all_entities, result
#                 torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
#             except json.JSONDecodeError as e:
#                 print(f"第 {line_num} 行 JSON 解析錯誤: {e}")
#                 continue
#             except Exception as e:
#                 print(f"第 {line_num} 行處理錯誤: {e}")
#                 continue



# if __name__ == "__main__":
#     # 設定檔案路徑
#     input_file = "data/segment/dense_10q.jsonl"  # 您的輸入 JSONL 檔案路徑
#     output_file = "data/kg/entities_nunerzero.json"  # 輸出 JSON 檔案路徑
#     model_path = "pretrained_model/numind/NuNerZero/"
    
#     # 執行處理
#     process_jsonl_file(input_file, output_file, model_path)






from gliner import GLiNER
from labels import entity_labels


def merge_entities(entities):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity['label'] == current['label'] and (next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']):
            current['text'] = text[current['start']: next_entity['end']].strip()
            current['end'] = next_entity['end']
        else:
            merged.append(current)
            current = next_entity
    # Append the last entity
    merged.append(current)
    return merged


model = GLiNER.from_pretrained("pretrained_model/numind/NuNerZero")

# NuZero requires labels to be lower-cased!
labels = entity_labels
labels = [l.lower() for l in labels]

# text = "At the annual technology summit, the keynote address was delivered by a senior member of the Association for Computing Machinery Special Interest Group on Algorithms and Computation Theory, which recently launched an expansive initiative titled 'Quantum Computing and Algorithmic Innovations: Shaping the Future of Technology'. This initiative explores the implications of quantum mechanics on next-generation computing and algorithm design and is part of a broader effort that includes the 'Global Computational Science Advancement Project'. The latter focuses on enhancing computational methodologies across scientific disciplines, aiming to set new benchmarks in computational efficiency and accuracy."

text = "Some cultures in this region were very similar and share certain elements, such as the importance of salmon to their cultures, while others differed. Prior to contact, and for a brief time after colonisation, some of these groups regularly conducted war against each other through raids and attacks. Through warfare they gathered captives for slavery. The creation of beautiful and practical objects (for all tribal communities) served as a means of transmitting stories, history, wisdom and property from generation to generation. Art provided Indigenous people with a tie to the land by depicting their histories on totem poles the Big (Plank) Houses of the Pacific Northwest coast – the symbols depicted were a constant reminder of their birth places, lineages and nations. Due to the abundance of natural resources and the affluence of most Northwest tribes, there was plenty of leisure time to create art. Many works of art served practical purposes, such as clothing, tools, weapons of war and hunting, transportation, cooking, and shelter; but others were purely aesthetic. The First Nations people of the Northwest Coast have a very rich tradition of hats, cloaks, body armour, and masks. Before Europeans arrived and introduced cloth, most coastal people wore minimal clothing."

sent = split_into_sentences(text)

for s in sent:
    entities = model.predict_entities(s, labels)

    entities = merge_entities(entities)

    for entity in entities:
        print(entity["text"], "=>", entity["label"])
