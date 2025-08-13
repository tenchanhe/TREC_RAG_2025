#!/usr/bin/env python3
import argparse, json, os, sys, re
from pathlib import Path
from openai import OpenAI

# === 固定載入你的段落查找函式 ===
MODULE_DIR = Path("/Users/stud113/Documents/TREC_RAG2025/src").resolve()
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
try:
    from segmentid_to_data_abs import search_for_original_data
except Exception as e:
    raise RuntimeError(f"無法載入 search_for_original_data：{e}")

def load_topics(path, limit=None):
    text = Path(path).read_text(encoding="utf-8")
    # 1) JSON array
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return arr[:limit] if limit else arr
    except json.JSONDecodeError:
        pass
    # 2) JSONL or純文字
    topics = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s in ("[", "]"):
            continue
        s_clean = s.rstrip(",")
        try:
            obj = json.loads(s_clean)
            topics.append(obj)
            if limit and len(topics) >= limit:
                break
            continue
        except json.JSONDecodeError:
            pass
        parts = s.split(None, 1)
        if len(parts) == 2:
            tid, narr = parts
            topics.append({"id": tid, "narrative": narr})
            if limit and len(topics) >= limit:
                break
    return topics

def load_run(run_path, limit_topics=None):
    """讀入 TSV run file，回傳 { topic_id: [segment_id,…] }，並保留原始順序（即索引。"""
    runs = {}
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            tid, _, seg, rank, score, _ = line.strip().split()
            runs.setdefault(tid, []).append(seg)
    if limit_topics:
        runs = {tid: runs[tid] for tid in list(runs)[:limit_topics]}
    return runs

def build_prompt(narrative, numbered_candidates):
    """
    numbered_candidates: list[(idx(int), seg_id(str), contents(str))]
    產生 user prompt，嚴格要求 citations 回傳 0-based indices。
    """
    # 這段會被寫進 metadata["prompt"]，供檢視
    meta_prompt = (
        "Based on trusted sources, provide a factual and comprehensive answer to the following narrative: "
        f"{narrative}"
    )

    # 提供給 LLM 的完整指令（含編號參考）
    user_prompt_lines = []
    user_prompt_lines.append("You are doing retrieval-augmented generation using the following numbered references.")
    user_prompt_lines.append("Write ONE coherent answer split into sentences; each sentence must cite the indices")
    user_prompt_lines.append("of the supporting references from the numbered list (0-based).")
    user_prompt_lines.append("")
    user_prompt_lines.append(f"NARRATIVE:\n{narrative}\n")
    user_prompt_lines.append("REFERENCES-LIST (0-based indices match the `references` array we will output):")
    for idx, seg_id, contents in numbered_candidates:
        user_prompt_lines.append(f"{idx}: {seg_id}\n{contents}\n")
    user_prompt_lines.append(
        """
Return ONLY a JSON object (no markdown, no code fences) with this exact schema:
{
  "answer": [
    {"text": "<sentence_1>", "citations": [<int>, <int>, ...]},
    {"text": "<sentence_2>", "citations": [<int>, ...]},
    ...
  ]
}
Hard constraints:
- "citations" must be an array of 0-based integer indices into the REFERENCES-LIST above.
- Every sentence MUST have at least one citation; omit any sentence that cannot be grounded.
- Do not include any extra keys. Do not include segment IDs in citations. Integers only.
"""
    )
    return "\n".join(user_prompt_lines), meta_prompt

def compress_references_and_reindex(items, seg_ids):
    """
    只保留被引用到的 segments，並將 citations 依新 references 重新映射為 0-based。
    - items: [{"text": str, "citations": [int, ...]}, ...]  # 目前以舊 references 索引為準
    - seg_ids: 原始的 references（20 個）
    回傳: (new_items, new_references)
    """
    # 蒐集所有被用到的舊索引
    used_set = set()
    for it in items:
        for idx in it.get("citations", []):
            if isinstance(idx, int) and 0 <= idx < len(seg_ids):
                used_set.add(idx)

    # 以「原始 references 順序」做穩定過濾（符合你範例 03,19,45 的順序）
    used_sorted = [i for i in range(len(seg_ids)) if i in used_set]

    # 若你想要「依句子首次出現順序」重排，改用下面這行：
    # used_sorted = []
    # seen = set()
    # for it in items:
    #     for idx in it.get("citations", []):
    #         if 0 <= idx < len(seg_ids) and idx not in seen:
    #             seen.add(idx); used_sorted.append(idx)

    # 舊索引 -> 新索引 的對照
    old2new = {old: new for new, old in enumerate(used_sorted)}
    new_refs = [seg_ids[old] for old in used_sorted]

    # 逐句重映射 citations，並去重、過濾
    new_items = []
    for it in items:
        text = (it.get("text") or "").strip()
        seen = set()
        new_cites = []
        for old in it.get("citations", []):
            if old in old2new:
                new_idx = old2new[old]
                if new_idx not in seen:
                    seen.add(new_idx)
                    new_cites.append(new_idx)
        if text and new_cites:
            new_items.append({"text": text, "citations": new_cites})

    return new_items, new_refs

def split_sentences(text):
    sentences = re.split(r'(?<=[。\.!?])\s*', text)
    return [s.strip() for s in sentences if s.strip()]

def normalize_citations(raw_cites, id2idx, k):
    """
    將 citation 清為 0-based indices：
      - 若是 int/數字字串：保留且過濾越界
      - 若是不小心給了 segment_id：用映射轉 index
      - 去重、保序
    """
    out, seen = [], set()
    for c in raw_cites:
        idx = None
        if isinstance(c, int):
            idx = c
        elif isinstance(c, str):
            if c.isdigit():
                idx = int(c)
            else:
                idx = id2idx.get(c)  # 當成 segment_id 嘗試映射
        if isinstance(idx, int) and 0 <= idx < k and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--team_id", type=str, default="clip2025")
    ap.add_argument("--run_id", type=str, default="run1")
    args = ap.parse_args()

    client = OpenAI(api_key="Authorization: Bearer sk-RIpti_ZbkoMs0B5XgJnEbw",
                    base_url="https://towel.cs.nccu.edu.tw:4000")

    topics = load_topics(args.topics, args.limit)
    runs = load_run(args.run, args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 為了讓空格/換行格式一致：一行一 JSON，逗號與冒號後固定一個空格
    JSON_SEPARATORS = (",", ": ")

    with out_path.open("w", encoding="utf-8") as fout:
        for topic in topics:
            tid = str(topic["id"])
            narrative = topic.get("narrative") or ""

            # 取出 top-k references（保持次序，作為 0-based index 來源）
            seg_ids = runs.get(tid, [])[: args.top_k]
            id2idx = {sid: i for i, sid in enumerate(seg_ids)}

            # 拉原文內容以提供給 LLM（不進輸出，只用於生成）
            candidates = []
            for i, sid in enumerate(seg_ids):
                obj = search_for_original_data(sid)
                contents = obj["contents"] if obj else ""
                candidates.append((i, sid, contents))

            user_prompt, meta_prompt = build_prompt(narrative, candidates)

            system_prompt = (
                'You are a careful research assistant. Output ONLY valid JSON for the requested schema. '
                'No markdown. No explanations.'
            )

            resp = client.chat.completions.create(
                model="phi4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            raw = (resp.choices[0].message.content or "").strip()

            # 嘗試 JSON 解析；若失敗再用簡單回退規則
            answer_items = []
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and "answer" in obj and isinstance(obj["answer"], list):
                    answer_items = obj["answer"]
            except json.JSONDecodeError:
                # 回退：嘗試依句子+方括號數字解析
                tmp = []
                for sent in split_sentences(raw):
                    m = re.match(r"(.+?)\s*\[([0-9 ,]+)\]\s*$", sent)
                    if m:
                        t = m.group(1).strip()
                        idxs = [int(x) for x in re.split(r"[ ,]+", m.group(2)) if x.isdigit()]
                        tmp.append({"text": t, "citations": idxs})
                answer_items = tmp

            # 正常化 citations -> 0-based indices，過濾空句
            k = len(seg_ids)
            norm_items = []
            for it in answer_items:
                text = (it.get("text") or "").strip()
                cites = it.get("citations") or []
                cites = normalize_citations(cites, id2idx, k)
                if text and cites:
                    norm_items.append({"text": text, "citations": cites})

            # 若模型沒生成任何有效句子，至少給一個空安全回覆（可依需求改）
            if not norm_items:
                if k > 0:
                    norm_items = [{"text": "No grounded answer could be generated from the provided references.", "citations": [99]}]
                else:
                    norm_items = []

            remapped_items, filtered_refs = compress_references_and_reindex(norm_items, seg_ids)

            out_obj = {
                "metadata": {
                    "team_id": args.team_id,
                    "run_id": args.run_id,
                    "type": "automatic",
                    "narrative_id": tid,
                    "narrative": narrative,
                    "prompt": meta_prompt,
                },
                "references": filtered_refs,          # 這個順序即 citations 的 0-based 索引對應
                "answer": remapped_items,           # 單一答案，分句 + 0-based citations
            }

            # JSONL：固定分隔與單一換行，確保空格/換行一致
            fout.write(json.dumps(out_obj, ensure_ascii=False, separators=JSON_SEPARATORS, indent=4) + "\n")

    print(f"Done. Wrote JSONL to {out_path}")

if __name__ == "__main__":
    main()
