"""
Fixed research harness for the autonomous nanochat experiment loop.

This file is the *read-only* contract that makes experiments comparable:
it pins the data, the tokenizer, the evaluation metric (val_bpb), the
sequence length, and the training time budget. The experiment agent does
NOT edit this file. It edits `research/train.py` (and, fair game, the real
model/optimizer source in `nanochat/gpt.py` and `nanochat/optim.py`).

The single metric is val_bpb (validation bits-per-byte), lower is better.
It is vocab-size independent, so architecture changes that alter the vocab
or tokenization are still compared apples-to-apples.
"""

import time

import torch

from nanochat.common import compute_init, get_peak_flops
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

# -----------------------------------------------------------------------------
# FIXED CONSTANTS (do not modify — they define the benchmark)
TIME_BUDGET = 300          # seconds of *training* wall-clock (excludes startup/compile)
SEQ_LEN = 2048             # fixed context length, used for BOTH training and eval
EVAL_TOKENS = 10 * 524288  # ~5.2M tokens evaluated for val_bpb (fast + stable)
EVAL_BATCH_SIZE = 16       # device batch used during the fixed eval
SEED = 42
# -----------------------------------------------------------------------------


def init():
    """Initialize compute and return (device, tokenizer, vocab_size, token_bytes)."""
    _, _, _, _, device = compute_init(device_type="cuda")
    torch.manual_seed(SEED)
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    vocab_size = tokenizer.get_vocab_size()
    return device, tokenizer, vocab_size, token_bytes


def make_train_loader(tokenizer, batch_size, device):
    """Training data iterator yielding (inputs, targets) at the fixed SEQ_LEN."""
    return tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, batch_size, SEQ_LEN, split="train", device=device
    )


@torch.no_grad()
def evaluate(model, token_bytes, device, batch_size=EVAL_BATCH_SIZE):
    """Run the fixed val_bpb evaluation. Lower is better."""
    model.eval()
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        get_tokenizer(), batch_size, SEQ_LEN, split="val", device=device
    )
    steps = EVAL_TOKENS // (batch_size * SEQ_LEN)
    val_bpb = evaluate_bpb(model, val_loader, steps, token_bytes)
    model.train()
    return val_bpb


def peak_flops():
    """Peak BF16 FLOP/s of the active GPU (for MFU), or None if unknown."""
    try:
        return get_peak_flops(torch.cuda.get_device_name())
    except Exception:
        return None


def report(*, val_bpb, training_seconds, total_seconds, flops_per_token,
           num_steps, total_batch_size, num_params, depth):
    """Print the FIXED summary block. Metric extraction greps `^val_bpb:`."""
    peak_mem_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available() else 0.0
    )
    total_tokens = num_steps * total_batch_size
    pf = peak_flops()
    if pf and training_seconds > 0:
        mfu = 100.0 * flops_per_token * total_tokens / training_seconds / pf
    else:
        mfu = 0.0
    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_mem_mb:.1f}")
    print(f"mfu_percent:      {mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {num_steps}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {depth}")
