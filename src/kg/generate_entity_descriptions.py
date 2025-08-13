import ollama
import jsonlines
import tqdm
# import ast
from src.utils.normalized_list import normalize_to_list

# Prompt template for generating entity descriptions
PROMPT_TEMPLATE = """Given the following text and an entity, generate a concise one-sentence description of the entity based on the context provided in the text.
Focus only on the information available in the text.
Do not add any information from outside the text.
Output only the description as a single string.

Text:
{context}

Entity:
{entity}

Description:"""

def call_llm(prompt, ollama_model_name):
    """Calls the OLLAMA API to generate text."""
    try:
        response = ollama.generate(model=ollama_model_name, prompt=prompt, stream=False)
        return response['response'].strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def generate_entity_descriptions(entities_filepath, segments_filepath, output_filepath, ollama_model_name):
    """
    Generates descriptions for entities based on text segments using an LLM.

    Args:
        entities_filepath (str): Path to the JSONL file containing entities (from extract_entities.py).
        segments_filepath (str): Path to the JSONL file containing the original text segments.
        output_filepath (str): Path to write the output JSONL file with entity descriptions.
        ollama_model_name (str): The name of the Ollama model to use.
    """
    # 1. Read segments and store them in a dictionary for easy lookup
    print(f"Reading segments from '{segments_filepath}'...")
    segments = {item['id']: item['contents'] for item in jsonlines.open(segments_filepath)}
    print(f"Found {len(segments)} segments.")

    results = []
    print(f"Reading entities from '{entities_filepath}' and generating descriptions...")

    # 2. Read entities and generate descriptions
    flag = 1
    with jsonlines.open(entities_filepath) as reader:
        for item in tqdm.tqdm(reader, desc="Generating Descriptions"):
            segment_id = item.get("id")
            raw_entities = item.get("keywords") # Assuming 'keywords' contains the entities

            if not segment_id or not raw_entities or segment_id not in segments:
                print(f"Skipping invalid record: {item}")
                continue

            context = segments[segment_id]

            try:
                # The output from the previous script is a string representation of a list
                entities = normalize_to_list(raw_entities)
                if not isinstance(entities, list):
                    raise ValueError
            except (ValueError, SyntaxError):
                print(f"Warning: Could not parse entities for ID {segment_id}. Content: '{raw_entities}'. Skipping.")
                continue


            # 3. For each entity, call the LLM to generate a description
            for entity in entities:
                prompt = PROMPT_TEMPLATE.format(context=context, entity=entity)
                description = call_llm(prompt, ollama_model_name)
                print("flag= ", flag)
                flag += 1

                if description:
                    results.append({
                        "segment_id": segment_id,
                        "entity": entity,
                        "description": description
                    })

    # 4. Write results to the output file
    print(f"Writing {len(results)} entity descriptions to '{output_filepath}'...")
    with jsonlines.open(output_filepath, mode='w') as writer:
        writer.write_all(results)

    print("Successfully generated and saved all entity descriptions.")

if __name__ == "__main__":
    # Define file paths
    # Assumes the output of extract_entities.py is the input here
    entities_input_file = "data/kg/keywords.jsonl"
    # This should be the same file used as input for extract_entities.py
    segments_input_file = "data/segment/dense_10q.jsonl"
    output_file = "data/kg/entity_descriptions.json"
    MODEL = "phi4-mini:latest"

    generate_entity_descriptions(entities_input_file, segments_input_file, output_file, MODEL)