#!/usr/bin/env python3
"""
Pretraining script for decoder-only language model.

Usage:
    uv run python scripts/pretrain.py local/tiny.json
    uv run python scripts/pretrain.py local/tiny.json --resume checkpoints/run/checkpoint_latest.pt
"""

import argparse

import torch

from atomiclm.model.decoder import Decoder
from atomiclm.tokenizer.tokenizer import BasicTokenizer
from atomiclm.training.config import TrainingConfig
from atomiclm.training.data import create_dataloaders
from atomiclm.training.train import Trainer


def main():
    parser = argparse.ArgumentParser(description="Pretrain a language model")
    parser.add_argument("config", type=str, help="Path to config JSON file")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    config = TrainingConfig.from_json(args.config)
    if args.resume is not None:
        config.resume_from = args.resume

    # Seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load tokenizer
    print(f"Loading tokenizer from {config.data.tokenizer_path}")
    tokenizer = BasicTokenizer()
    tokenizer.load(config.data.tokenizer_path)

    # Override vocab_size from tokenizer (include special tokens)
    config.model.vocab_size = len(tokenizer.vocab) + len(tokenizer.special_tokens)
    print(f"Vocab size: {config.model.vocab_size}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config.data, tokenizer)
    print(f"Train tokens: {len(train_loader.dataset.token_ids):,}")
    if val_loader is not None:
        print(f"Val tokens: {len(val_loader.dataset.token_ids):,}")

    # Create model
    print("Initializing model...")
    model = Decoder(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        num_kv_heads=config.model.num_kv_heads,
        dropout=config.model.dropout,
        qkv_bias=config.model.qkv_bias,
        use_gated_ffn=config.model.use_gated_ffn,
        rope_base=config.model.rope_base,
    )

    # Train
    trainer = Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    trainer.train()


if __name__ == "__main__":
    main()
