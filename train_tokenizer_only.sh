#!/bin/bash

# This script is modified to train ONLY the tokenizer using pre-downloaded data
# and then exit.

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR




# -----------------------------------------------------------------------------
# Report setup (kept for reference)
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer Training Section

echo "Setting up Rust environment..."
# Install Rust / Cargo (if not already installed - safe to run again)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
echo "Rust environment ready."

echo "Building the rustbpe Tokenizer..."
# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
echo "Tokenizer built."

# --- DATA DOWNLOAD SKIPPED ---
# We assume the Turkish .parquet files are already in $NANOCHAT_BASE_DIR/base_data/
echo "Skipping data download. Using existing data in $NANOCHAT_BASE_DIR/base_data/"
# --- DATA DOWNLOAD SKIPPED ---

echo "Training the tokenizer..."
# train the tokenizer with vocab size 2**16 = 65536
# It will use the data found in the cache, up to the max_chars limit
python -m scripts.tok_train --max_chars=2000000000
echo "Tokenizer training finished."

echo "Evaluating the tokenizer..."
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval
echo "Tokenizer evaluation finished."

# --- STOP THE SCRIPT HERE ---
echo "Tokenizer training and evaluation complete. Exiting script as requested."
exit 0
# --- END OF MODIFICATIONS ---

# -----------------------------------------------------------------------------
# Base model (pretraining) - THIS SECTION WILL NOT RUN

echo "This section (Base model pretraining) will not run because the script exited."

# ... (rest of the original script remains below but won't be executed) ...