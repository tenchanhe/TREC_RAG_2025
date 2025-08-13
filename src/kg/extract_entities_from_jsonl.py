import jsonlines
import tqdm
from src.kg.extract_entities import call_llm, PROMPT_TEMPLETE

INPUT_FILE = "data/kg/entities_sentence.jsonl"
OUTPUT_FILE = "data/kg/entities_sentence_extracted.jsonl"
MODEL = "phi4-mini:latest"  # 可根據你的環境調整
START_LINE = 80864
MIN_ENTITIES = 1
MAX_ENTITIES = 3

results = []

with jsonlines.open(INPUT_FILE) as reader:
    for idx, item in enumerate(tqdm.tqdm(reader, desc="抽取entity")):
        print("Processing item", idx)
        if idx < START_LINE-1:
            continue
        sentence = item.get("sentence")
        item_id = item.get("id")
        if not sentence:
            continue
        prompt = PROMPT_TEMPLETE.format(min_entities=MIN_ENTITIES, max_entities=MAX_ENTITIES, content=sentence)
        entities = call_llm(prompt, MODEL)
        results.append({
            "id": item_id,
            "sentence": sentence,
            "entities": entities
        })

with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
    for res in results:
        writer.write(res)

print(f"已完成抽取，結果寫入 {OUTPUT_FILE}")
