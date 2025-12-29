#!/usr/bin/env python3
"""
Script to generate text completions from a pretrained model.

Usage:
    uv run python scripts/generate.py checkpoints/run_name/checkpoint_latest.pt "The answer is" 50
    uv run python scripts/generate.py checkpoints/run_name/checkpoint_best.pt "Once upon a time" 100 --tokenizer out/tokenizer-mix-512.json
    uv run python scripts/generate.py checkpoints/run_name/checkpoint_latest.pt "Hello" 30 --temperature 0.8 --top-k 40
"""

import argparse
import torch

from atomiclm.model.decoder import Decoder
from atomiclm.tokenizer.tokenizer import BasicTokenizer
from atomiclm.training.config import _resolve_device
from atomiclm.training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Generate text from a pretrained model")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint (.pt file)")
    parser.add_argument("prompt", type=str, help="Starting prompt")
    parser.add_argument("steps", type=int, help="Number of tokens to generate")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer JSON (auto-detect from checkpoint config if not provided)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (default: 0, no filtering)")
    parser.add_argument("--eos", type=str, default=None, help="Stop token (any special token name, e.g. '<|endoftext|>')")
    parser.add_argument("--pad", type=str, default=None, help="Pad token for finished sequences (defaults to --eos token)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu, auto-detect if not provided)")
    args = parser.parse_args()

    # Resolve device
    device = torch.device(_resolve_device(args.device or "auto"))
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]

    # Load tokenizer
    if args.tokenizer is None:
        args.tokenizer = config_dict["data"]["tokenizer_path"]
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = BasicTokenizer()
    tokenizer.load(args.tokenizer)

    # Create model (vocab_size from checkpoint config, set during training)
    print("Initializing model...")
    model_config = config_dict["model"]
    model = Decoder(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        max_seq_len=model_config["max_seq_len"],
        num_kv_heads=model_config.get("num_kv_heads", model_config["num_heads"]),
        dropout=model_config.get("dropout", 0.0),
        qkv_bias=model_config.get("qkv_bias", False),
        use_gated_ffn=model_config.get("use_gated_ffn", False),
        rope_base=model_config.get("rope_base", 10000.0),
    )
    model.to(device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Resolve EOS / pad token IDs
    eos_token_id = None
    pad_token_id = None
    if args.eos is not None:
        if args.eos not in tokenizer.special_tokens:
            raise ValueError(
                f"'{args.eos}' not in tokenizer special tokens: "
                f"{list(tokenizer.special_tokens.keys())}"
            )
        eos_token_id = tokenizer.special_tokens[args.eos]
        if args.pad is not None:
            if args.pad not in tokenizer.special_tokens:
                raise ValueError(
                    f"'{args.pad}' not in tokenizer special tokens: "
                    f"{list(tokenizer.special_tokens.keys())}"
                )
            pad_token_id = tokenizer.special_tokens[args.pad]
        else:
            pad_token_id = eos_token_id

    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)

    # Generate
    print(f"\nGenerating {args.steps} tokens with temperature={args.temperature}")
    print("-" * 60)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    # Decode
    result = tokenizer.decode(output_ids[0].tolist())
    print(result)
    print("-" * 60)


if __name__ == "__main__":
    main()
