import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
max_gb_limit = 25  # Set your limit here
output_dir = os.path.expanduser("~/.cache/nanochat/base_data")
dataset_name = "uonlp/CulturaX"
language_split = "tr"
shard_size = 100000  # Number of documents per parquet file
auth_token = True    # Assumes you ran 'huggingface-cli login'
# ---------------------

max_bytes_limit = max_gb_limit * (1024**3)
total_bytes_saved = 0

os.makedirs(output_dir, exist_ok=True)

print(f"Loading {dataset_name} dataset, split '{language_split}'...")
print(f"Will stop after downloading approximately {max_gb_limit} GB of data.")

dataset = load_dataset(
    dataset_name, 
    language_split, 
    streaming=True, 
    split="train", 
    token=auth_token
)

shard_num = 0
buffer = []

try:
    for doc in tqdm(dataset, desc="Downloading documents"):
        buffer.append({"text": doc["text"]})
        
        if len(buffer) >= shard_size:
            df = pd.DataFrame(buffer)
            shard_name = f"shard_{shard_num:04d}.parquet"
            shard_path = os.path.join(output_dir, shard_name)
            
            df.to_parquet(shard_path, index=False)
            
            # Check file size and update total
            shard_bytes = os.path.getsize(shard_path)
            total_bytes_saved += shard_bytes
            current_gb = total_bytes_saved / (1024**3)
            
            print(f"Saved {shard_path} ({len(buffer)} docs). Total downloaded: {current_gb:.2f} GB")
            
            shard_num += 1
            buffer = []
            
            # Check if we've hit the limit
            if total_bytes_saved >= max_bytes_limit:
                print(f"Hit limit of {max_gb_limit} GB. Halting download.")
                break

except Exception as e:
    print(f"An error occurred (this is common if the stream ends): {e}")

finally:
    # Save any remaining documents in the last shard
    if buffer and total_bytes_saved < max_bytes_limit:
        df = pd.DataFrame(buffer)
        shard_name = f"shard_{shard_num:04d}.parquet"
        shard_path = os.path.join(output_dir, shard_name)
        df.to_parquet(shard_path, index=False)
        
        shard_bytes = os.path.getsize(shard_path)
        total_bytes_saved += shard_bytes
        current_gb = total_bytes_saved / (1024**3)
        print(f"Saved final shard {shard_path} ({len(buffer)} docs). Total downloaded: {current_gb:.2f} GB")

print(f"\nDownload complete. Total data saved: {total_bytes_saved / (1024**3):.2f} GB in {output_dir}")