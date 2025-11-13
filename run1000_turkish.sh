#!/bin/bash

# ... [other setup] ...

export OMP_NUM_THREADS=1

# === MODIFICATION 1 ===
# Point the base directory to your new dataset root.
# The scripts will use this to find/save tokenizer files
# and tokenized data.
export NANOCHAT_BASE_DIR="/usr/nanochat/dataset"

mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
python -m nanochat.report reset
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# === MODIFICATION 2 ===
# Comment out all default data download commands.
# curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
# python -m nanochat.dataset -n 16
# python -m nanochat.dataset -n 800 &

# ... [tokenizer training] ...
# NOTE: This assumes your /usr/nanochat/dataset/base
# data is in the format expected by the 'parquets_iter_batched'
# function in 'nanochat/dataset.py'. You may need to
# modify that Python module if it's not found.
python -m scripts.tok_train --max_chars=4000000000
python -m scripts.tok_eval

# ... [hyperparameter comments] ...

# === MODIFICATION 3 ===
# No changes are needed to the *command* for base_train,
# as it automatically uses $NANOCHAT_BASE_DIR to find
# the tokenized data (e.g., /usr/nanochat/dataset/tokenized_data)
# that 'tok_train.py' should have just created.
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=32 --device_batch_size=8
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# ... [midtrain and sft] ...
# These commands do not need to be changed in the shell script,
# but the Python files they call ('mid_train.py', 'chat_sft.py')
# MUST be modified as described in the next section.

torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=8 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# ... [reporting and chat] ...
python -m nanochat.report generate
python -m scripts.chat_web