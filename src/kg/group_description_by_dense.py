import json
import jsonlines
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import re
import os
import argparse
from collections import defaultdict
import gc
from src.utils.normalized_list import normalize_to_list
import string
import math
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def normalize_entity(entity):
    """
    實體規範化處理
    Args:
        entity (str): 原始實體字串
    Returns:
        str: 規範化後的實體
    """
    # 小寫處理
    normalized = entity.lower().strip()
    
    # 去除尾部標點符號
    normalized = normalized.rstrip(string.punctuation)

    import re
    # 將日期格式統一為 YYYY-MM-DD
    date_pattern = r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})'
    normalized = re.sub(date_pattern, r'\3-\1-\2', normalized)
    
    return normalized

def calculate_entity_weights(entities, total_docs=1000):
    """
    計算實體的IDF權重（逆實體頻率）
    Args:
        entities (list): 實體列表
        total_docs (int): 總文檔數量（可調整）
    Returns:
        dict: 實體到權重的映射
    """
    entity_counts = Counter(entities)
    weights = {}
    
    for entity, count in entity_counts.items():
        # IDF計算：log(N/df_e)，其中N是總文檔數，df_e是包含實體e的文檔數
        idf_weight = math.log(total_docs / (count + 1))  # +1避免除零
        weights[entity] = idf_weight
    
    return weights

def process_entities_ned(entities_list, context_sentences=None):
    """
    實體抽取、規範化與消歧（NED）處理
    Args:
        entities_list (list): 原始實體列表
        context_sentences (list): 上下文句子列表（用於消歧）
    Returns:
        dict: 處理後的實體資訊，包含權重和消歧分數
    """
    if not entities_list:
        return {}
    
    # 1. 規範化處理
    normalized_entities = []
    original_to_normalized = {}
    
    for entity in entities_list:
        if isinstance(entity, str) and entity.strip():
            normalized = normalize_entity(entity.strip())
            if normalized:  # 確保規範化後不為空
                normalized_entities.append(normalized)
                original_to_normalized[entity] = normalized
    
    # 去除重複
    unique_entities = list(set(normalized_entities))
    
    if not unique_entities:
        return {}
    
    # 2. 計算IDF權重
    entity_weights = calculate_entity_weights(normalized_entities)
    
    # 3. 上下文消歧（如果有上下文）
    entity_scores = {}
    for entity in unique_entities:
        final_score = entity_weights.get(entity, 0.0)
        entity_scores[entity] = {
            'weight': entity_weights.get(entity, 0.0),
            'final_score': final_score,
            'original_forms': [k for k, v in original_to_normalized.items() if v == entity]
        }
    
    return entity_scores


def extract_relation_triggers(sentence):
    """
    抽取句子中的關係觸發詞/謂詞
    Args:
        sentence (str): 輸入句子
    Returns:
        list: 關係觸發詞列表
    """
    # 常見的關係觸發詞列表（可以擴展）
    relation_patterns = {
        'acquisition': ['併購', '收購', '購買', 'acquired', 'bought', 'purchased'],
        'location': ['位於', '坐落於', '地處', 'located', 'situated', 'based'],
        'release': ['發布', '推出', '發表', 'released', 'launched', 'published'],
        'investment': ['投資', '入股', 'invested', 'funding', 'financed'],
        'partnership': ['合作', '夥伴', 'partnership', 'collaboration', 'alliance'],
        'employment': ['僱用', '聘請', 'hired', 'employed', 'appointed'],
        'ownership': ['擁有', '持有', 'owns', 'possesses', 'holds'],
        'foundation': ['成立', '創立', 'founded', 'established', 'created']
    }
    
    found_triggers = []
    sentence_lower = sentence.lower()
    
    for relation_type, triggers in relation_patterns.items():
        for trigger in triggers:
            if trigger.lower() in sentence_lower:
                found_triggers.append((relation_type, trigger))
    
    return found_triggers

def calculate_pmi_weights(entity_pairs, total_docs=1000):
    """
    計算實體對的點互信息（PMI）權重
    Args:
        entity_pairs (list): 實體對列表 [(entity1, entity2), ...]
        total_docs (int): 總文檔數量
    Returns:
        dict: 實體對到PMI權重的映射
    """
    pair_counts = Counter(entity_pairs)
    single_counts = Counter()
    
    # 計算單個實體出現頻率
    for (e1, e2), count in pair_counts.items():
        single_counts[e1] += count
        single_counts[e2] += count
    
    pmi_weights = {}
    for (e1, e2), pair_count in pair_counts.items():
        # PMI = log(P(e1,e2) / (P(e1) * P(e2)))
        p_e1_e2 = pair_count / total_docs
        p_e1 = single_counts[e1] / total_docs
        p_e2 = single_counts[e2] / total_docs
        
        if p_e1 > 0 and p_e2 > 0:
            pmi = math.log(p_e1_e2 / (p_e1 * p_e2))
            pmi_weights[(e1, e2)] = max(0, pmi)  # 只保留正PMI
    
    return pmi_weights

def mmr_selection(sentences, sentence_embeddings, query_embedding, k=5, lambda_param=0.7):
    """
    使用最大邊際相關性（MMR）進行多樣性取樣
    Args:
        sentences (list): 候選句子列表
        sentence_embeddings (torch.Tensor): 句子嵌入向量
        query_embedding (torch.Tensor): 查詢（關鍵字）嵌入向量
        k (int): 選擇的句子數量
        lambda_param (float): 相關性與多樣性的平衡參數 [0,1]
    Returns:
        list: 選中的句子索引
    """
    if len(sentences) <= k:
        return list(range(len(sentences)))
    
    # 轉換為numpy格式進行計算
    sent_emb_np = sentence_embeddings.cpu().numpy()
    query_emb_np = query_embedding.cpu().numpy().reshape(1, -1)
    
    # 計算與查詢的相似度
    query_similarities = cosine_similarity(sent_emb_np, query_emb_np).flatten()
    
    selected_indices = []
    remaining_indices = list(range(len(sentences)))
    
    # 選擇與查詢最相似的句子作為起始
    first_idx = np.argmax(query_similarities)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # MMR迭代選擇
    for _ in range(min(k-1, len(remaining_indices))):
        best_score = -float('inf')
        best_idx = None
        
        for idx in remaining_indices:
            # 與查詢的相關性
            relevance = query_similarities[idx]
            
            # 與已選擇句子的最大相似度（多樣性懲罰）
            if selected_indices:
                selected_emb = sent_emb_np[selected_indices]
                current_emb = sent_emb_np[idx:idx+1]
                max_similarity = np.max(cosine_similarity(current_emb, selected_emb))
            else:
                max_similarity = 0
            
            # MMR分數
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    return selected_indices

def control_description_length(sentences, max_tokens=150):
    """
    控制描述長度，避免過長或過短
    Args:
        sentences (list): 句子列表
        max_tokens (int): 最大token數量
    Returns:
        list: 長度控制後的句子列表
    """
    controlled_sentences = []
    
    for sentence in sentences:
        # 簡單的token計數（用空格分割）
        tokens = sentence.split()
        
        if len(tokens) > max_tokens:
            # 截斷過長的句子
            truncated = ' '.join(tokens[:max_tokens])
            # 嘗試在句號處截斷以保持完整性
            last_period = truncated.rfind('.')
            if last_period > max_tokens * 0.7:  # 至少保留70%的內容
                truncated = truncated[:last_period + 1]
            controlled_sentences.append(truncated)
        elif len(tokens) >= 10:  # 過濾太短的句子
            controlled_sentences.append(sentence)
    
    return controlled_sentences

def weighted_embedding_composition(embeddings, weights):
    """
    加權嵌入合成
    Args:
        embeddings (torch.Tensor): 嵌入向量矩陣
        weights (list): 權重列表
    Returns:
        torch.Tensor: 加權合成後的嵌入向量
    """
    if len(embeddings) == 0:
        return None
    
    weights_tensor = torch.tensor(weights, device=embeddings.device).unsqueeze(1)
    weighted_emb = torch.sum(embeddings * weights_tensor, dim=0)
    
    # 正規化
    weighted_emb = F.normalize(weighted_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
    
    return weighted_emb

def process_sentence_retrieval_with_denoising(sentences, sentence_embeddings, sentence_ids, 
                                            keyword_embedding, entities_in_sentences=None,
                                            top_k=10, mmr_k=5):
    """
    處理句子檢索並進行實體描述去噪
    Args:
        sentences (list): 候選句子列表
        sentence_embeddings (torch.Tensor): 句子嵌入向量
        sentence_ids (list): 句子對應的ID
        keyword_embedding (torch.Tensor): 關鍵字嵌入向量
        entities_in_sentences (list): 每個句子中的實體列表（可選）
        top_k (int): 初始檢索的句子數量
        mmr_k (int): MMR多樣性取樣的句子數量
    Returns:
        dict: 處理後的結果，包含句子、ID、關係觸發詞、鄰居實體等
    """
    if len(sentences) == 0:
        raise ValueError("句子列表不能為空")
    
    # 1. 初始相似度計算和Top-K選擇
    scores = torch.matmul(keyword_embedding.unsqueeze(0), sentence_embeddings.T).squeeze(0)
    k_val = min(top_k, len(sentences))
    top_k_scores, top_k_indices = torch.topk(scores, k=k_val)
    
    # 提取top-k句子和嵌入
    top_sentences = [sentences[idx] for idx in top_k_indices]
    top_embeddings = sentence_embeddings[top_k_indices]
    top_ids = [sentence_ids[idx] for idx in top_k_indices]
    
    # 2. 長度控制
    controlled_sentences = control_description_length(top_sentences, max_tokens=150)
    
    # 更新索引（過濾掉被截斷或移除的句子）
    valid_indices = []
    final_sentences = []
    for i, (original, controlled) in enumerate(zip(top_sentences, controlled_sentences)):
        if controlled:  # 如果句子沒有被完全移除
            valid_indices.append(i)
            final_sentences.append(controlled)
    
    if not final_sentences:
        return {"sentences": [], "ids": [], "relations": [], "neighbors": [], "weights": []}
    
    # 更新嵌入和ID
    valid_embeddings = top_embeddings[valid_indices]
    valid_ids = [top_ids[i] for i in valid_indices]
    
    # 3. MMR多樣性取樣
    mmr_k = min(mmr_k, len(final_sentences))
    mmr_indices = mmr_selection(final_sentences, valid_embeddings, keyword_embedding, k=mmr_k)
    
    # 最終選擇的句子
    selected_sentences = [final_sentences[i] for i in mmr_indices]
    selected_ids = [valid_ids[i] for i in mmr_indices]
    selected_embeddings = valid_embeddings[mmr_indices]
    
    # # 4. 抽取關係觸發詞
    # relations_info = []
    # for sentence in selected_sentences:
    #     triggers = extract_relation_triggers(sentence)
    #     relations_info.append(triggers)
    
    # 5. 計算IDF權重（基於句子長度和稀有詞）
    sentence_weights = []
    for sentence in selected_sentences:
        # 簡單的IDF權重：基於句子長度的逆向權重
        tokens = sentence.split()
        length_weight = 1.0 / (1.0 + math.log(len(tokens) + 1))  # 長句子權重較低
        sentence_weights.append(length_weight)
    
    # 6. 鄰居實體分析（如果提供了實體信息）
    neighbors_info = []
    if entities_in_sentences:
        for i, sentence_idx in enumerate([top_k_indices[valid_indices[mmr_idx]] for mmr_idx in mmr_indices]):
            if sentence_idx < len(entities_in_sentences) and entities_in_sentences[sentence_idx]:
                # 計算實體對的PMI
                sentence_entities = entities_in_sentences[sentence_idx]
                entity_pairs = [(e1, e2) for i, e1 in enumerate(sentence_entities) 
                              for e2 in sentence_entities[i+1:]]
                neighbors_info.append({
                    'entities': sentence_entities,
                    'pairs': entity_pairs
                })
            else:
                neighbors_info.append({'entities': [], 'pairs': []})
    
    # 7. 加權嵌入合成（可選，用於後續聚合）
    if len(selected_embeddings) > 1:
        composed_embedding = weighted_embedding_composition(selected_embeddings, sentence_weights)
    else:
        composed_embedding = selected_embeddings[0] if len(selected_embeddings) > 0 else None
    
    return {
        "sentences": selected_sentences,
        "ids": selected_ids,
        # "relations": relations_info,
        "neighbors": neighbors_info,
        "weights": sentence_weights,
        "composed_embedding": composed_embedding
    }


def filter_subsentences(sentences, ids):
    """
    過濾掉重複或為其他句子子句的句子，只保留較長的那個。
    Args:
        sentences (list[str]): 句子列表
        ids (list[str]): 對應的ID列表
    Returns:
        tuple: (過濾後的句子列表, 對應的ID列表)
    """
    # 先根據長度排序，長的在前
    sorted_items = sorted(zip(sentences, ids), key=lambda x: len(x[0]), reverse=True)
    filtered_sentences = []
    filtered_ids = []
    for sent, id_ in sorted_items:
        # 如果這個句子不是任何已保留句子的子句，則保留
        if not any(sent in s for s in filtered_sentences):
            filtered_sentences.append(sent)
            filtered_ids.append(id_)
    return filtered_sentences, filtered_ids


def split_into_sentences(text, sentences_per_group=1):
    """
    將文本分句，並可以設定要把幾個句子連在一起。
    Args:
        text (str): 輸入文本
        sentences_per_group (int): 每組包含的句子數量，預設為1
    Returns:
        list[str]: 分組後的句子列表
    """
    # 使用正則表達式分句，考慮多種結尾符號
    sentences = re.split(r'[.!?]+\s+', text.strip())
    # 過濾掉空字串和太短的句子
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    # 如果句子數量為1，直接返回
    if sentences_per_group <= 1:
        return sentences
    
    # 將句子分組連接
    grouped_sentences = []
    for i in range(0, len(sentences), sentences_per_group):
        group = sentences[i:i+sentences_per_group]
        # 將組內的句子用句號空格連接
        grouped_text = '. '.join(group)
        if grouped_text:
            grouped_sentences.append(grouped_text)
    
    return grouped_sentences

def mean_pooling(model_output, attention_mask):
    """對 token embeddings 進行平均池化"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(texts, tokenizer, model, device):
    """計算一批文本的嵌入向量"""
    if not texts:
        return torch.tensor([], device=device)
    texts = [t if isinstance(t, str) else "" for t in texts]
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=256).to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def batched_embeddings(texts, tokenizer, model, device, batch_size=128):
    """分批次計算所有文本的嵌入向量，以節省記憶體"""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="生成嵌入向量 (Embeddings)"):
        batch = texts[i:i+batch_size]
        emb = get_embeddings(batch, tokenizer, model, device)
        all_embeddings.append(emb.cpu())
        torch.cuda.empty_cache()
    return torch.cat(all_embeddings, dim=0)

# --- 資料讀取函式 (重構) ---

def load_keywords_map(keywords_file: str) -> dict[str, list[str]]:
    """
    從 keywords 檔案讀取所有關鍵字，並建立一個 id -> keywords 的映射。
    這避免了在處理每個區塊時重複讀取整個關鍵字檔案。
    """
    print(f"正在從 {keywords_file} 預先載入所有關鍵字...")
    keywords_map = defaultdict(list)
    with jsonlines.open(keywords_file) as reader:
        for item in tqdm(reader, desc="預載入關鍵字"):
            doc_id = item.get("id")
            if doc_id:
                kw_data = item.get("keywords", [])
                keywords_map[doc_id].extend(normalize_to_list(kw_data))
    print(f"完成！共找到 {len(keywords_map)} 個文件ID的關鍵字。")
    return keywords_map

def read_segments_in_chunks(segment_file: str, chunk_size: int):
    """
    一個生成器函式，用來分塊讀取 segment 檔案。
    """
    try:
        with jsonlines.open(segment_file) as reader:
            chunk = []
            for item in reader:
                chunk.append(item)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
    except FileNotFoundError:
        print(f"錯誤: Segment 檔案 {segment_file} 不存在。")
        return


# --- 主要處理邏輯 ---
def process_chunk(chunk_data, keywords_map, tokenizer, model, device, top_k, batch_size, sentences_per_group=1):
    """
    處理單一資料區塊的函式。
    """
    # 1. 從區塊中提取 (句子, segment_id) 對和對應的關鍵字
    sentence_id_pairs = []
    current_keywords = set()
    segment_ids_in_chunk = {item.get("id") for item in chunk_data if item.get("id")}

    for item in chunk_data:
        doc_id = item.get("id")
        content = item.get("contents", "")
        if doc_id and content:
            # 假設 split_into_sentences 返回一個句子列表
            split_sents = split_into_sentences(content, sentences_per_group)
            if split_sents:
                for s in split_sents:
                    if s.strip():
                        sentence_id_pairs.append((s.strip(), doc_id))

    for seg_id in segment_ids_in_chunk:
        if seg_id in keywords_map:
            current_keywords.update(keywords_map[seg_id])
    
    # 處理entities - 實體規範化與消歧（NED）
    if current_keywords:
        
        # 執行NED處理
        processed_entities = process_entities_ned(list(current_keywords))
        
        # 根據處理結果更新關鍵字集合，保留高分實體
        filtered_keywords = set()
        for entity, info in processed_entities.items():
            # 可以設定閾值來過濾低分實體
            if info['final_score'] > 0.1:  # 目前沒差
                filtered_keywords.add(entity)
        
        current_keywords = filtered_keywords
        print(f"NED處理後保留 {len(current_keywords)} 個實體（原始: {len(keywords_map)} 個關鍵字映射）")
    else:
        raise ValueError("當前區塊沒有找到任何關鍵字，請檢查輸入資料。")

    # 2. 去除重複的句子，同時保留對應的 ID (保留第一次出現的)
    unique_sentences_map = {}
    for sentence, doc_id in sentence_id_pairs:
        if sentence not in unique_sentences_map:
            unique_sentences_map[sentence] = doc_id
    
    if not unique_sentences_map or not current_keywords:
        print("警告：目前區塊找不到句子或對應的關鍵字，跳過處理。")
        return None

    unique_sentences = list(unique_sentences_map.keys())
    corresponding_ids = list(unique_sentences_map.values())
    unique_keywords = list(current_keywords)

    print(f"本區塊找到 {len(unique_sentences)} 句不重複的句子和 {len(unique_keywords)} 個不重複的關鍵字。")

    # 3. 計算嵌入向量
    sentence_embeddings = batched_embeddings(unique_sentences, tokenizer, model, device, batch_size=batch_size).to(device)
    keyword_embeddings = batched_embeddings(unique_keywords, tokenizer, model, device, batch_size=batch_size).to(device)

    # 4. 為每個關鍵字檢索 Top K 句子（使用新的去噪處理）
    results = defaultdict(lambda: {"sentence": [], "id": [], "relations": [], "neighbors": [], "weights": []})
    print(f"正在為 {len(unique_keywords)} 個關鍵字檢索 Top {top_k} 句子...")
    
    for i in tqdm(range(len(unique_keywords)), desc="檢索中"):
        keyword = unique_keywords[i]
        kw_embedding = keyword_embeddings[i]
        
        # 使用新的句子檢索和去噪函數
        retrieval_result = process_sentence_retrieval_with_denoising(
            sentences=unique_sentences,
            sentence_embeddings=sentence_embeddings,
            sentence_ids=corresponding_ids,
            keyword_embedding=kw_embedding,
            entities_in_sentences=None,  # 可以後續添加實體信息
            top_k=top_k,
            mmr_k=min(10, top_k)  # MMR取樣數量
        )
        
        # 將結果存儲到最終結果中
        results[keyword]["sentence"] = retrieval_result["sentences"]
        results[keyword]["id"] = retrieval_result["ids"]
        results[keyword]["relations"] = retrieval_result["relations"]
        results[keyword]["neighbors"] = retrieval_result["neighbors"]
        results[keyword]["weights"] = retrieval_result["weights"]

    # 5. 清理記憶體
    del sentence_embeddings, keyword_embeddings
    gc.collect()
    torch.cuda.empty_cache()

    return dict(results)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"正在載入模型: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    # 預先載入所有關鍵字到一個映射中
    keywords_map = load_keywords_map(args.keywords_file)

    # 分塊處理 segments
    segment_chunks = read_segments_in_chunks(args.segment_file, args.chunk_size)
    
    for i, chunk in enumerate(segment_chunks):
        chunk_num = i + 1
        print(f"--- 正在處理區塊 {chunk_num} (共 {len(chunk)} 筆 segments) ---")

        # 處理單一區塊
        results = process_chunk(
            chunk_data=chunk,
            keywords_map=keywords_map,
            tokenizer=tokenizer,
            model=model,
            device=device,
            top_k=args.top_k,
            batch_size=args.batch_size,
            sentences_per_group=args.sentences_per_group
        )

        if results:
            # 為每個區塊設定獨立的輸出檔案
            output_file = os.path.join(args.output_dir, f"query_{i+1}.json")
            print(f"正在將區塊 {chunk_num} 的結果儲存至 {output_file}...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"區塊 {chunk_num} 處理完成！🎉")
        else:
            print(f"區塊 {chunk_num} 沒有生成結果，已跳過。")

    print("所有區塊處理完畢！")

if __name__ == "__main__":
    # 直接用變數設定參數
    class Args:
        # keywords_file = "data/kg/keywords_sentences.jsonl"
        # output_dir = "data/kg/grouped_descriptions_dense/"
        keywords_file = "data/kg/entities_sentence.jsonl"
        output_dir = "data/kg/grouped_descriptions_et_dense/"
        segment_file = "data/segment/dense_10q.jsonl"
        model_name = "pretrained_model/sentence-transformers/all-MiniLM-L6-v2/"
        top_k = 10
        batch_size = 1024
        chunk_size = 1000
        sentences_per_group = 1

    args = Args()
    main(args)
