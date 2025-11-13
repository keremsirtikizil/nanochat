import os
import json
from datasets import load_dataset
from tqdm import tqdm
import sys

# --- Configuration ---
DATASET_NAME = "AlicanKiraz0/Turkish-SFT-Dataset-v1.0"
OUTPUT_FILE = "turkish_sft_data.jsonl"
OUTPUT_PATH = os.path.join(os.getcwd(), OUTPUT_FILE) # Save in the current nanochat directory

# This script must run successfully after 'huggingface-cli login'
# Ensure you are authenticated to avoid errors

def format_and_save():
    """Loads the Hugging Face dataset and converts it to nanochat's JSONL format."""
    print(f"Loading dataset: {DATASET_NAME}")
    try:
        # Load the dataset (assuming it is small enough to load fully)
        dataset = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Ensure you are authenticated via 'huggingface-cli login'.")
        sys.exit(1)

    print(f"Dataset loaded. Total rows: {len(dataset)}")
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for row in tqdm(dataset, desc="Formatting and saving data"):
            
            # The dataset has 'instruction', 'input', and 'output' columns.
            # We must map this to the {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]} format.
            
            user_prompt = row['instruction']
            if row['input'] and row['input'].strip():
                user_prompt += "\n" + row['input'].strip()

            conversation = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": row['output']}
                ]
            }
            
            # Write the JSON object as a single line (JSON Lines format)
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
            
    print("\n" + "="*50)
    print(f"Successfully formatted and saved data to: {OUTPUT_PATH}")
    print(f"Next step: Modify scripts/mid_train.py to use this file.")
    print("="*50)

if __name__ == "__main__":
    format_and_save()