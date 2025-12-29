# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

From-scratch implementation of a decoder-only transformer language model in Python/PyTorch. The goal is explicit control over every component — no black-box wrappers. Implements a BPE tokenizer, full transformer decoder with modern architecture choices (GQA, RoPE, RMSNorm, SwiGLU), and training infrastructure with checkpointing.

## Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_tokenizer.py::test_encode_decode_roundtrip

# Run tests with output
uv run pytest -s
```

## Architecture

### Tokenizer (`src/atomiclm/tokenizer/`)

Byte Pair Encoding tokenizer built on raw bytes (base vocab = 256 byte tokens).

- `base.py` — Abstract `Tokenizer` interface (train/encode/decode/save/load)
- `tokenizer.py` — `BasicTokenizer` implementation with two training algorithms:
  - `_train_heap` (default) — heap-based O(n log n) merging, tracks pair positions for incremental updates
  - `_train_basic` — simple quadratic approach, rescans all stats each merge
- `bpe_ops.py` — Low-level merge and pair-counting primitives (`merge`, `get_stats`)
- `constants.py` — `TOKENIZER_BASE_LENGTH = 256`, version string
- `patterns.py` — GPT-4 compatible regex split pattern (used by default)

Key design: regex pre-splits text into chunks before BPE (like GPT-4/tiktoken). Iterator-based training (`train_from_iterator`) handles carry-over across chunk boundaries so split points don't affect the result. Merges and merge_ranks are maintained in parallel — ranks determine encoding priority.

### Model (`src/atomiclm/model/`)

Decoder-only transformer with modern architecture choices.

- `decoder.py` — `Decoder` top-level module: token embeddings, N transformer blocks, final RMSNorm, LM head with weight tying. Supports KV-cache for autoregressive `generate()` with temperature and top-k sampling.
- `block.py` — `TransformerBlock` pre-norm decoder block (RMSNorm → MHA → residual, RMSNorm → FFN → residual)
- `attention.py` — `MultiHeadAttention` with separate Q/K/V projections, GQA support (`num_kv_heads`), head-first layout `(b, h, t, d)`, KV-cache, and FlashAttention via `F.scaled_dot_product_attention`
- `feedforward.py` — `FeedForward` with standard (GELU) and gated (SwiGLU) variants
- `norm.py` — `RMSNorm` (no mean subtraction, no bias)
- `rope.py` — `RotaryEmbedding` with complex-number rotation (LLaMA style)

### Training (`src/atomiclm/training/`)

- `config.py` — Nested dataclasses (`ModelConfig`, `DataConfig`, `OptimConfig`, `TrainingConfig`) loaded from JSON. Auto-detects device (cuda > mps > cpu).
- `data.py` — `TextDataset` for token-level language modeling, `create_dataloaders` with train/val split
- `train.py` — `Trainer` with AdamW (separate weight-decay groups for residual projections), cosine LR schedule with warmup, periodic validation and generation samples
- `checkpoint.py` — Save/load checkpoints including model state, optimizer state, RNG state, and config

### Scripts (`scripts/`)

- `train_tokenizer.py` — Train BPE tokenizer from text files
- `pretrain.py` — Pretrain decoder model from JSON config
- `generate.py` — Generate text from a checkpoint

### Data

- `data/` — Sample corpora for tokenizer training (not included in repo)
- `out/` — Saved tokenizer vocabularies (JSON)
- `notebooks/` — Jupyter notebooks demonstrating tokenizer usage, tiktoken export, RoPE, and GQA

## Conventions

- Package uses `regex` (not `re`) for Unicode-aware pattern matching
- Tokenizer persistence is JSON-based (merges stored as `[left, right, id]` triples)
- Tests are in `tests/` using pytest; tokenizer tests cover roundtrip, determinism, iterator equivalence, and edge cases
- Python >=3.12 required; managed with `uv`

## Dependency Management

**IMPORTANT**: Before importing any new library in code, verify that it's actually installed in the project:

```bash
# Check if package is in dependencies
uv pip list | grep <package-name>

# Or check pyproject.toml
cat pyproject.toml | grep <package-name>
```

If the package is not present, add it to `pyproject.toml` under `dependencies` or `dev` (for test/dev-only packages) and run `uv sync`. Do not assume packages are available just because they're common — this project has a minimal dependency footprint by design.
