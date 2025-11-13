#!/usr/bin/env python
"""
interact_base_model.py

The final, corrected standalone script to interact with a pre-trained nanochat BASE model.

- Dynamically loads the configuration from the model's meta.json file.
- Includes the necessary monkey-patch for Mac (CPU/MPS) execution.
- Loops through an array of Turkish prompts for text completion testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os
import sys
import math
import numpy as np
import json 

# --- 1. Import Nanochat Libraries ---
# We need these to load the tokenizer and set up the device/paths
try:
    import nanochat.common
    from nanochat.tokenizer import get_tokenizer
    # Import the correct functions from the provided file
    from nanochat.checkpoint_manager import get_base_dir, find_last_step
except ModuleNotFoundError:
    print("Error: 'nanochat' module not found.")
    print("Please run this script from the root of the nanochat directory,")
    print("and make sure your environment is activated and PYTHONPATH is set:")
    print("  export PYTHONPATH=$(pwd):$PYTHONPATH")
    sys.exit(1)

# --- 2. Copied Model Class Definitions (from nanochat/gpt.py) ---
#
# Definitions must come BEFORE they are used.

@dataclass
class GPTConfig:
    sequence_len: int = 256
    vocab_size: int = 65536
    n_layer: int = 6
    n_head: int = 3
    n_kv_head: int = 3
    n_embd: int = 256

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= config.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2); Tk = k.size(2)
        enable_gqa = self.n_head != self.n_kv_head
        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
    def forward(self, x, cos_sin, kv_cache=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, device=torch.device('cpu'))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
             try: device = next(self.parameters()).device
             except StopIteration: device = torch.device('cpu')
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = self.get_device()
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, device=device)
        self.cos = cos
        self.sin = sin
        return self

    def get_device(self):
        return self.transformer.wte.weight.device

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        if idx.device != self.cos.device:
             self.cos = self.cos.to(idx.device)
             self.sin = self.sin.to(idx.device)
        assert T <= self.cos.size(1), f"Seq len {T} > rotary cache {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"idx device {idx.device} != cos device {self.cos.device}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)
        softcap = 15
        if targets is not None:
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list), f"Input 'tokens' must be a list, got {type(tokens)}"
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids) 
            logits = logits[:, -1, :] 
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature == 0:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
# --- End of Copied Class Definitions ---


# --- User Configuration ---
# Set the tag for the model you want to load (e.g., "d6" from your test run)
MODEL_TAG = "d6" 
# Set to a specific step number (e.g., 50 or 100) or None to use the latest
CHECKPOINT_STEP = None 

PROMPTS_TO_TEST = [
    "Sabancı Üniversitesi",
    "Nasılsınız?",
    "Ders",
    "Hayat"
]
MAX_NEW_TOKENS = 10 
TEMPERATURE = 0.9  
TOP_K = 50          
# --- End Configuration ---


# --- MONKEY PATCH SECTION for compute_init ---
print("Applying monkey patch for compute_init...")
def patched_compute_init():
    #
    ddp = int(os.environ.get('RANK', -1)) != -1 
    if ddp:
        print("Warning: DDP detected, patch might not cover this case correctly without CUDA.")
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0))
        ddp_world_size = int(os.environ.get('WORLD_SIZE', 1))
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        if not torch.cuda.is_available(): sys.exit("ERROR: DDP requires CUDA")
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
    else:
        ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
        master_process = True
        seed_offset = 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Patched init: Using MPS device")
        device = torch.device('mps')
    elif torch.cuda.is_available():
         print("Patched init: Using CUDA device")
         device = torch.device('cuda')
    else:
        print("Patched init: Using CPU device")
        device = torch.device('cpu')
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, seed_offset, device
# Overwrite the original function in the loaded module
nanochat.common.compute_init = patched_compute_init
print("Monkey patch applied.")
# --- END MONKEY PATCH SECTION ---


# --- Main Script Logic ---
def main():
    print("Initializing device...")
    try:
        # Call our patched function
        *_, device = nanochat.common.compute_init()
    except Exception as e:
        print(f"Error during patched compute_init, trying fallback: {e}")
        device = torch.device('mps' if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    try:
        tokenizer = get_tokenizer() #
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

    print(f"Loading configuration for model tag: '{MODEL_TAG}'")
    try:
        # Define the directory for base model checkpoints
        base_checkpoints_dir = os.path.join(get_base_dir(), "base_checkpoints", MODEL_TAG)
        
        # Find the step to load
        step_to_load = CHECKPOINT_STEP
        if step_to_load is None:
            print(f"No checkpoint step specified, finding latest step in {base_checkpoints_dir}...")
            step_to_load = find_last_step(base_checkpoints_dir)
            print(f"Found latest step: {step_to_load}")
        
        # Load the meta.json file to get the model configuration
        meta_path = os.path.join(base_checkpoints_dir, f"meta_{step_to_load:06d}.json")
        print(f"Loading model config from: {meta_path}")
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
        
        model_config_kwargs = meta_data["model_config"]
        print(f"Building model with loaded config: {model_config_kwargs}")
        
        # Create the GPTConfig object from the loaded data
        config = GPTConfig(**model_config_kwargs)
        
        # Verify tokenizer vocab size matches
        if tokenizer.get_vocab_size() != config.vocab_size:
            print(f"Warning: Tokenizer vocab size ({tokenizer.get_vocab_size()}) != loaded config vocab size ({config.vocab_size}).")
            config.vocab_size = tokenizer.get_vocab_size()

        print(f"Initializing model structure (depth={config.n_layer})...")
        model = GPT(config)
        model.to(device) 

        # Now load the model weights (.pt file)
        checkpoint_path = os.path.join(base_checkpoints_dir, f"model_{step_to_load:06d}.pt")
        print(f"Loading model weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle weights being nested in 'model' key or having prefixes
        state_dict = checkpoint.get('model', checkpoint)
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"Model load result: {load_result}")
        model.eval()

    except FileNotFoundError as e:
        print(f"Error: Checkpoint file not found. {e}")
        print("Did base_train.py run and save checkpoints?")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*50)
    print("Model and tokenizer loaded successfully.")
    print("Starting batch text completion...")
    print("="*50 + "\n")

    for prompt_text in PROMPTS_TO_TEST:
        try:
            print(f"\n--- PROMPT ---")
            print(prompt_text)
            start_ids = tokenizer.encode(prompt_text)
            
            token_generator = model.generate(
                start_ids, 
                MAX_NEW_TOKENS, 
                temperature=TEMPERATURE, 
                top_k=TOP_K
            )

            print("--- GENERATING ---")
            generated_ids_list = list(token_generator)
            
            full_ids = start_ids + generated_ids_list
            full_text = tokenizer.decode(full_ids)
            generated_text = tokenizer.decode(generated_ids_list)

            print("\n--- FULL OUTPUT ---")
            print(full_text)
            print("\n--- GENERATED PART ONLY ---")
            print(generated_text)
            print("-"*(len("--- GENERATED PART ONLY ---")))
        except Exception as e:
            print(f"\nAn error occurred during generation for prompt '{prompt_text}': {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    current_dir = os.path.abspath(os.getcwd())
    python_path = os.environ.get('PYTHONPATH', '')
    if current_dir not in python_path:
         print(f"Warning: Current directory '{current_dir}' not in PYTHONPATH.")
         print("If you get a ModuleNotFoundError, please run:")
         print(f"  export PYTHONPATH={current_dir}:$PYTHONPATH")
         print("Then re-run the script.")
    
    main()