import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# --- Configuration ---
RELATION_FILE = 'data/kg/relation_bm25.json'
SEGMENT_FILE = 'data/segment/bm25_10q.jsonl'
OUTPUT_DESCRIPTION_FILE = 'data/kg/entity_descriptions.json'
MODEL_NAME = 'pretrained_model/google-t5/t5-base/' # Using t5-base
MAX_SEGMENT_LENGTH = 256 # Limit segment length to avoid excessively long inputs for T5
BATCH_SIZE = 8 # Adjust based on your GPU memory

# --- Load Model and Tokenizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Loading T5 model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval() # Set model to evaluation mode

# --- 1. Load Data ---
print(f"Loading relations from {RELATION_FILE}...")
with open(RELATION_FILE, 'r', encoding='utf-8') as f:
    relations_data = json.load(f)

print(f"Loading segments from {SEGMENT_FILE}...")
segments_data = {}
with open(SEGMENT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        segment = json.loads(line)
        segments_data[segment['id']] = segment['contents']

# --- 2. Identify Unique Entities and gather their contexts ---
# entity_contexts: stores {entity_name: [(docid, segment_text), ...]}
# This maps each unique entity to all segments where it appears in a triplet.
entity_contexts = {}

for entry in tqdm(relations_data, desc="Gathering entity contexts"):
    docid = entry['docid']
    segment_text = segments_data.get(docid, "") # Get segment content by docid
    
    if not segment_text:
        continue # Skip if segment content not found (e.g., segment.jsonl is incomplete)

    for relation in entry.get('relations', []):
        head = relation['head']
        tail = relation['tail']

        if head not in entity_contexts:
            entity_contexts[head] = set()
        entity_contexts[head].add((docid, segment_text)) # Use set to avoid duplicate (docid, text) pairs

        if tail not in entity_contexts:
            entity_contexts[tail] = set()
        entity_contexts[tail].add((docid, segment_text)) # Add tail entity context too

print(f"Found {len(entity_contexts)} unique entities.")

# --- 3. Generate Descriptions using T5 ---
generated_descriptions = {} # To store {entity_name: description}

# Prepare inputs for batch processing
# Each item will be (entity_name, prompt_text)
t5_inputs_to_process = []
for entity, contexts in entity_contexts.items():
    # For simplicity, we'll just take the first context for now.
    # For better results, you might concatenate multiple relevant contexts,
    # or select the most relevant one using embedding similarity.
    if contexts:
        # Sort contexts to ensure deterministic behavior and potentially use a "primary" context
        sorted_contexts = sorted(list(contexts))
        context_text = sorted_contexts[0][1] # Get segment text
        
        # Truncate context if too long
        if len(context_text) > MAX_SEGMENT_LENGTH:
            context_text = context_text[:MAX_SEGMENT_LENGTH] + "..."

        # T5 prompt engineering
        prompt = f"describe entity: {entity} in context: {context_text}"
        t5_inputs_to_process.append((entity, prompt))
    else:
        # If no context found for an entity, you could generate a generic description
        # or skip it. For this example, we'll skip for now.
        pass

print(f"Preparing {len(t5_inputs_to_process)} T5 inference requests...")

# Batch processing loop
for i in tqdm(range(0, len(t5_inputs_to_process), BATCH_SIZE), desc="Generating descriptions"):
    batch = t5_inputs_to_process[i:i + BATCH_SIZE]
    entity_names = [item[0] for item in batch]
    prompts = [item[1] for item in batch]

    # Tokenize inputs
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate descriptions
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50, # Max length for the generated description
            min_length=10,
            num_beams=5,  # Use beam search for better quality
            early_stopping=True
        )

    # Decode and store results
    for j, output_ids in enumerate(outputs):
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        generated_descriptions[entity_names[j]] = generated_text

    breakpoint()
    
print("\nDescription generation complete.")
print(f"Generated descriptions for {len(generated_descriptions)} entities.")

# --- Save Descriptions ---
print(f"Saving generated descriptions to {OUTPUT_DESCRIPTION_FILE}...")
with open(OUTPUT_DESCRIPTION_FILE, 'w', encoding='utf-8') as f:
    json.dump(generated_descriptions, f, ensure_ascii=False, indent=2)

print("Process finished successfully!")