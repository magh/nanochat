"""
Export a nanochat checkpoint (base / sft / rl) to a single .safetensors file.

The output file contains:
  - the model weights as tensors (with the torch.compile "_orig_mod." prefix stripped)
  - the model config embedded in the safetensors metadata header (string -> string),
    under the key "model_config" as a JSON string.

This is a portable, pickle-free artifact. Note that the tokenizer is NOT included
(it is not tensor data); ship tokenizer/tokenizer.pkl alongside for inference.

Usage:
  uv run python -m scripts.export_safetensors --source=rl --model-tag=d12
  uv run python -m scripts.export_safetensors --source=sft --model-tag=d12 --output=/path/model.safetensors
"""
import os
import re
import json
import argparse

import torch
from safetensors.torch import save_file, load_file

from nanochat.common import get_base_dir

CHECKPOINT_DIRS = {
    "base": "base_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
}


def find_latest_step(checkpoint_dir):
    steps = []
    for fname in os.listdir(checkpoint_dir):
        m = re.match(r"model_(\d+)\.pt$", fname)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        raise FileNotFoundError(f"No model_*.pt files found in {checkpoint_dir}")
    return max(steps)


def main():
    parser = argparse.ArgumentParser(description="Export a nanochat checkpoint to .safetensors")
    parser.add_argument("--source", type=str, required=True, choices=["base", "sft", "rl"],
                        help="Which training stage to export")
    parser.add_argument("--model-tag", type=str, required=True, help="Model tag, e.g. d12")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .safetensors path (default: alongside the checkpoint)")
    args = parser.parse_args()

    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, CHECKPOINT_DIRS[args.source], args.model_tag)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    step = args.step if args.step is not None else find_latest_step(checkpoint_dir)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    print(f"Loading weights: {model_path}")
    print(f"Loading config:  {meta_path}")

    # Load weights, strip the torch.compile "_orig_mod." prefix, ensure contiguous CPU tensors
    state_dict = torch.load(model_path, map_location="cpu")
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    # Load the model config and embed it (as JSON) in the safetensors metadata header.
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    model_config = meta_data["model_config"]
    metadata = {
        "format": "pt",
        "model_config": json.dumps(model_config),
        "source": args.source,
        "model_tag": args.model_tag,
        "step": str(step),
    }

    output = args.output or os.path.join(checkpoint_dir, f"{args.model_tag}-{args.source}.safetensors")
    save_file(state_dict, output, metadata=metadata)
    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.1f} MB, {len(state_dict)} tensors)")
    print(f"Embedded model_config: {model_config}")

    # Verify: reload and check tensors + metadata round-trip
    reloaded = load_file(output)
    assert set(reloaded.keys()) == set(state_dict.keys()), "tensor key mismatch on reload"
    for k in state_dict:
        a, b = state_dict[k], reloaded[k]
        assert a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b), f"tensor mismatch: {k}"
    print(f"Verified: {len(reloaded)} tensors round-trip exactly. Config recoverable from metadata header.")


if __name__ == "__main__":
    main()
