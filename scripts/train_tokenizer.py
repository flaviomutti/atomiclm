#!/usr/bin/env python3
r"""
Script to train a BPE tokenizer.

Usage:
    uv run python scripts/train_tokenizer.py data/mix.txt 512 --output out/tokenizer-512
    uv run python scripts/train_tokenizer.py data/the-verdict.txt 300 --special "<|endoftext|>" "<|pad|>"
    uv run python scripts/train_tokenizer.py data/mix.txt 1024 --method basic --pattern "[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
"""

import argparse
import os
import time
from pathlib import Path

from atomiclm.tokenizer.tokenizer import BasicTokenizer


def _format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def read_chunks(file_path, chunk_size=1024 * 1024):
    """Read file in chunks for iterator-based training."""
    with open(file_path, encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("data", type=str, help="Path to training text file")
    parser.add_argument("vocab_size", type=int, help="Target vocabulary size (includes 256 base bytes)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path (without .json extension)")
    parser.add_argument("--special", nargs="+", default=None,
                        help="Special tokens (e.g., --special '<|endoftext|>' '<|pad|>' '<|im_start|>')")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Regex pattern for text splitting (defaults to GPT-4 pattern)")
    parser.add_argument("--method", choices=["heap", "basic"], default="heap",
                        help="Training method: heap (O(n log n), default) or basic (O(n²))")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress training progress output")
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        data_stem = Path(args.data).stem
        args.output = f"out/tokenizer-{data_stem}-{args.vocab_size}"

    num_merges = args.vocab_size - 256
    file_size = os.path.getsize(args.data)
    verbose = not args.quiet

    print(f"Data:         {args.data}  ({_format_size(file_size)})")
    print(f"Target vocab: {args.vocab_size}  ({num_merges} merges)")
    print(f"Method:       {args.method}")
    if args.pattern:
        print(f"Pattern:      {args.pattern}")
    print("-" * 60)

    tokenizer = BasicTokenizer(pattern=args.pattern)
    method = None if args.method == "heap" else "basic"

    t0 = time.time()
    tokenizer.train_from_iterator(
        read_chunks(args.data),
        vocab_size=args.vocab_size,
        verbose=verbose,
        method=method,
    )
    elapsed = time.time() - t0

    # Register special tokens if provided
    if args.special:
        special_tokens = {}
        next_id = len(tokenizer.vocab)
        for token in args.special:
            special_tokens[token] = next_id
            next_id += 1
        tokenizer.register_special_tokens(special_tokens)

    # Save
    print("-" * 60)
    print(f"Saving to {args.output}.json")
    tokenizer.save(args.output)

    print(f"Done.  Vocab: {len(tokenizer.vocab)}  |  {elapsed:.1f}s  ({num_merges / elapsed:.0f} merges/s)")
    if tokenizer.special_tokens:
        print(f"Special tokens: {list(tokenizer.special_tokens.keys())}")


if __name__ == "__main__":
    main()
