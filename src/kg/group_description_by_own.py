import jsonlines
import json
import tqdm
import re
import sys
import os
from collections import defaultdict
from src.utils.normalized_list import normalize_to_list

def load_all_sentence_items(keyword_filepath):
    """Loads all sentence items from the keyword file into a list."""
    print(f"Loading all sentence items from {keyword_filepath}...")
    with jsonlines.open(keyword_filepath) as reader:
        return list(tqdm.tqdm(reader, desc="Loading sentences"))

def group_keywords_from_slice(sentence_items, slice_size, start_index):
    """
    Groups sentences and their source IDs by keyword from a given slice of items.
    """
    group = set()
    keyword_groups = defaultdict(lambda: {"id": [], "sentence": []})
    copy_items = sentence_items[start_index:] 
    for item in copy_items:
        start_index += 1
        if len(group) >= slice_size:
            break
        keywords = item.get('keywords', [])
        keywords = normalize_to_list(keywords)
        doc_id = item.get("id")
        sentence = item.get("sentence")
        group.add(doc_id)
        if not doc_id or not sentence or not keywords:
            # print(f"Invalid item structure: {item}")
            continue
        # print(keywords)
        for kw in keywords:
            if isinstance(kw, str):
                keyword_groups[kw]["id"].append(doc_id)
                keyword_groups[kw]["sentence"].append(sentence)
            if isinstance(kw, list):
                for sub_kw in kw:
                    if isinstance(sub_kw, str):
                        keyword_groups[sub_kw]["id"].append(doc_id)
                        keyword_groups[sub_kw]["sentence"].append(sentence)
    # 資料清洗
    keyword_groups = clean_keyword_groups(keyword_groups)
    return keyword_groups, start_index

def clean_keyword_groups(keyword_groups):
    """簡單資料清洗：去除重複句子"""
    for kw, group in keyword_groups.items():
        seen = set()
        new_sentences = []
        new_ids = []
        for sent, docid in zip(group["sentence"], group["id"]):
            sent_norm = sent.strip()
            if sent_norm not in seen:
                seen.add(sent_norm)
                new_sentences.append(sent)
                new_ids.append(docid)
        group["sentence"] = new_sentences
        group["id"] = new_ids
    return keyword_groups

def main():
    keywords_file = "data/kg/entities_sentence.jsonl"
    output_dir = "data/kg/grouped_descriptions_et_own/"
    # keywords_file = "data/kg/keywords_sentences.jsonl"
    # output_dir = "data/kg/grouped_descriptions_own/"
    model_name = "pretrained_model/sentence-transformers/all-MiniLM-L6-v2/"
    slice_size = 1000
    
    all_sentence_items = load_all_sentence_items(keywords_file)
    print(f"Loaded {len(all_sentence_items)} total sentences.")

    start_index = 0
    query_id = 1
    while start_index < len(all_sentence_items):
        keyword_groups, start_index = group_keywords_from_slice(all_sentence_items, slice_size, start_index)
        
        # save keyword_groups to json file
        with open(os.path.join(output_dir, f"query_{query_id}.json"), 'w', encoding='utf-8') as f:
            json.dump(keyword_groups, f, ensure_ascii=False, indent=4)

        query_id += 1


    
if __name__ == "__main__":
    main()
