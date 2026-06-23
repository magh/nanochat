"""
EDITABLE training driver for the autonomous nanochat research loop.

This is the file the agent hacks on to improve val_bpb. Everything here is
fair game: the architecture knobs, the optimizer learning rates, the batch
size, the LR schedule, the training loop itself. You may ALSO edit the real
model/optimizer source in `nanochat/gpt.py` and `nanochat/optim.py`.

What you must NOT touch: `research/harness.py` (the fixed data/eval/budget
contract). Run this file with:

    uv run python -m research.train > run.log 2>&1

It trains for a fixed time budget (harness.TIME_BUDGET seconds of training
wall-clock, excluding startup/compile) and then prints a summary block whose
val_bpb is the metric to minimize.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time

import torch

from nanochat.gpt import GPT, GPTConfig
from research import harness

# =============================================================================
# HYPERPARAMETERS — edit these directly (no CLI flags)
# =============================================================================
# Model architecture
DEPTH = 8                  # number of transformer blocks
ASPECT_RATIO = 64          # model_dim = DEPTH * ASPECT_RATIO (nudged to a multiple of HEAD_DIM)
HEAD_DIM = 128             # attention head dimension
WINDOW_PATTERN = "L"       # "L" = all full-causal (fast flash path on ROCm); "SSSL" = sliding windows

# Optimization
DEVICE_BATCH_SIZE = 16     # per-step micro-batch (rows); tokens/micro = DEVICE_BATCH_SIZE * SEQ_LEN
TOTAL_BATCH_SIZE = 2 ** 16 # tokens per optimizer step (grad accumulation fills this)
MATRIX_LR = 0.02           # Muon LR for hidden matrices
EMBEDDING_LR = 0.2         # AdamW LR for token + value embeddings
UNEMBEDDING_LR = 0.004     # AdamW LR for the lm_head
SCALAR_LR = 0.5            # AdamW LR for scalar/lambda params
WEIGHT_DECAY = 0.0         # cautious weight decay (Muon)

# LR schedule (as a fraction of the time budget)
WARMUP_FRAC = 0.05         # linear warmup over the first 5% of the budget
WARMDOWN_FRAC = 0.20       # linear warmdown over the last 20% of the budget
FINAL_LR_FRAC = 0.0        # LR multiplier at the very end
# =============================================================================


def lr_multiplier(frac):
    """LR multiplier as a function of elapsed training-time fraction in [0, 1]."""
    if frac < WARMUP_FRAC:
        return (frac + 1e-9) / WARMUP_FRAC
    if frac > 1.0 - WARMDOWN_FRAC:
        progress = (1.0 - frac) / WARMDOWN_FRAC
        return progress * 1.0 + (1.0 - progress) * FINAL_LR_FRAC
    return 1.0


def build_model(vocab_size):
    base_dim = DEPTH * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    config = GPTConfig(
        sequence_len=harness.SEQ_LEN, vocab_size=vocab_size,
        n_layer=DEPTH, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cuda")
    model.init_weights()
    return model


def main():
    t_start = time.time()
    device, tokenizer, vocab_size, token_bytes = harness.init()
    print(f"Vocab size: {vocab_size:,}")

    # Build + compile the model
    orig_model = build_model(vocab_size)
    num_params = orig_model.num_scaling_params()["total"]
    flops_per_token = orig_model.estimate_flops()
    print(f"Depth {DEPTH} | params {num_params/1e6:.1f}M | flops/token {flops_per_token:.3e}")
    model = torch.compile(orig_model, dynamic=False)

    # Optimizer
    optimizer = orig_model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR, embedding_lr=EMBEDDING_LR,
        matrix_lr=MATRIX_LR, scalar_lr=SCALAR_LR, weight_decay=WEIGHT_DECAY,
    )

    # Gradient accumulation to reach TOTAL_BATCH_SIZE tokens per step
    tokens_per_micro = DEVICE_BATCH_SIZE * harness.SEQ_LEN
    assert TOTAL_BATCH_SIZE % tokens_per_micro == 0, "TOTAL_BATCH_SIZE must be divisible by device_batch*seq_len"
    grad_accum = TOTAL_BATCH_SIZE // tokens_per_micro
    print(f"Tokens/micro {tokens_per_micro:,} | grad_accum {grad_accum} | total_batch {TOTAL_BATCH_SIZE:,}")

    train_loader = harness.make_train_loader(tokenizer, DEVICE_BATCH_SIZE, device)
    x, y = next(train_loader)

    # ----------------------------------------------------------------------
    # Training loop (fixed time budget; step 0 absorbs compile and is untimed)
    model.train()
    step = 0
    train_time = 0.0
    while train_time < harness.TIME_BUDGET:
        t0 = time.time()

        for _ in range(grad_accum):
            loss = model(x, y)
            (loss / grad_accum).backward()
            x, y = next(train_loader)

        lrm = lr_multiplier(min(train_time / harness.TIME_BUDGET, 1.0))
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = time.time() - t0
        if step > 0:  # exclude the first (compile) step from the budget
            train_time += dt
        if step % 10 == 0:
            print(f"step {step:05d} | loss {loss.item():.4f} | lrm {lrm:.3f} | "
                  f"dt {dt*1000:.0f}ms | train_time {train_time:.1f}s")
        step += 1

    # ----------------------------------------------------------------------
    # Final fixed evaluation + report
    val_bpb = harness.evaluate(model, token_bytes, device)
    harness.report(
        val_bpb=val_bpb,
        training_seconds=train_time,
        total_seconds=time.time() - t_start,
        flops_per_token=flops_per_token,
        num_steps=step,
        total_batch_size=TOTAL_BATCH_SIZE,
        num_params=num_params,
        depth=DEPTH,
    )


if __name__ == "__main__":
    main()
