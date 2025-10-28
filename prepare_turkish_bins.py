import os
import glob
import numpy as np
import pandas as pd
from nanochat.tokenizer import get_tokenizer # Assumes tokenizer is saved in default location
from tqdm import tqdm
import sys # For flushing output

# --- Configuration ---
input_parquet_dir = os.path.expanduser("~/.cache/nanochat/base_data")
output_bin_dir = os.path.expanduser("~/.cache/nanochat/turkish_nanochat_data") # Choose where to save .bin files
val_split_shards = 16  # How many parquet shards to use for validation (adjust if needed based on total downloaded)
train_filename = "train.bin"
val_filename = "val.bin"
# --- End Configuration ---

os.makedirs(output_bin_dir, exist_ok=True)

print("Loading Turkish tokenizer...")
try:
    # This function should load the tokenizer you trained in Phase 3
    tokenizer = get_tokenizer() 
except FileNotFoundError:
    print("Error: Tokenizer file not found in the default location.")
    print("Make sure Phase 3 (train_tokenizer_only.sh) completed successfully.")
    sys.exit(1)

# Get the ID for a separator token (e.g., Beginning Of Sequence)
separator_token_id = tokenizer.get_bos_token_id() 
if separator_token_id is None:
    # Fallback if BOS is not defined, adjust if needed
    print("Warning: BOS token not found, using 0 as separator. Check your tokenizer.")
    separator_token_id = 0 

print(f"Reading parquet shards from: {input_parquet_dir}")
parquet_files = sorted(glob.glob(os.path.join(input_parquet_dir, "shard_*.parquet")))

if not parquet_files:
    print(f"Error: No parquet files found in {input_parquet_dir}. Did Phase 2 complete?")
    sys.exit(1)

# Adjust validation split if fewer files than expected were downloaded
actual_val_split = min(val_split_shards, max(1, len(parquet_files) // 10)) # Use at least 1, max 10% or specified number
if len(parquet_files) <= actual_val_split:
     print(f"Error: Not enough shards ({len(parquet_files)}) to create even a small validation split ({actual_val_split}).")
     print("Consider downloading more data or reducing val_split_shards.")
     sys.exit(1)
if actual_val_split != val_split_shards:
    print(f"Warning: Using {actual_val_split} shards for validation split instead of {val_split_shards} due to limited total shards.")

train_files = parquet_files[:-actual_val_split]
val_files = parquet_files[-actual_val_split:]

print(f"Processing {len(train_files)} shards for {train_filename}")
print(f"Processing {len(val_files)} shards for {val_filename}")

def process_files_incrementally(file_list, output_filename):
    output_path = os.path.join(output_bin_dir, output_filename)
    total_tokens_written = 0
    
    # Open file in binary append mode ('ab'). Create if not exists ('wb' first, then 'ab')
    # If the file exists, delete it first to start fresh
    if os.path.exists(output_path):
        print(f"Output file {output_path} exists, removing it first.")
        os.remove(output_path)
        
    print(f"Starting incremental processing for {output_filename}...")
    
    with open(output_path, 'ab') as f: # Open in append binary mode
        for file_path in tqdm(file_list, desc=f"Processing {output_filename}"):
            try:
                df = pd.read_parquet(file_path)
                current_chunk_ids = []
                for text in df['text']:
                    if isinstance(text, str) and text.strip(): # Ensure it's a non-empty string
                        tokens = tokenizer.encode(text)
                        current_chunk_ids.extend(tokens)
                        current_chunk_ids.append(separator_token_id) # Add separator
                
                # Convert chunk to numpy uint16 and write to file
                if current_chunk_ids:
                    chunk_array = np.array(current_chunk_ids, dtype=np.uint16)
                    f.write(chunk_array.tobytes())
                    total_tokens_written += len(current_chunk_ids)
                    
            except Exception as e:
                print(f"Warning: Could not process file {file_path}. Error: {e}")
                continue
            # Optionally flush buffer to disk periodically if needed, though OS usually handles this.
            # f.flush()

    print(f"Finished processing for {output_filename}.")
    print(f"Total tokens written: {total_tokens_written:,}")
    print(f"Saved to {output_path}")

# Process training files
process_files_incrementally(train_files, train_filename)

# Process validation files
process_files_incrementally(val_files, val_filename)

print("\nBinary data preparation complete.")
print(f"Train data: {os.path.join(output_bin_dir, train_filename)}")
print(f"Validation data: {os.path.join(output_bin_dir, val_filename)}")