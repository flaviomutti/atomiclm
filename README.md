# AtomicLM

From-scratch implementation of a decoder-only transformer language model in Python/PyTorch. Every component is written explicitly — no HuggingFace wrappers, no pre-built transformer layers.

The codebase covers three stages: training a BPE tokenizer on your own data, pretraining a transformer model, and generating text from a checkpoint. Trained tokenizers can be exported to tiktoken for fast inference.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
# git clone https://github.com/flaviomutti/atomiclm.git
cd atomiclm
uv sync
```

## Usage

### 1. Train a tokenizer

Point the script at any UTF-8 text file and choose a target vocabulary size:

```bash
uv run python scripts/train_tokenizer.py data/corpus.txt 512 --output out/tokenizer-512
```

You can add special tokens and choose between the default heap-based training (O(n log n)) or the simpler quadratic method:

```bash
uv run python scripts/train_tokenizer.py data/corpus.txt 1024 --special "<|endoftext|>" "<|pad|>"
uv run python scripts/train_tokenizer.py data/corpus.txt 1024 --method basic
```

The tokenizer saves as JSON and can be reloaded without retraining.

### 1b. Export to tiktoken

`export_mergeable_ranks()` yields `(bytes, rank)` pairs in the format tiktoken expects. After wrapping, `enc` is a drop-in replacement for any tiktoken encoding backed by a compiled Rust extension.

```bash
uv sync --extra tiktoken
```

```python
from atomiclm.tokenizer import BasicTokenizer
import tiktoken

tokenizer = BasicTokenizer()
tokenizer.load('out/tokenizer.json')

enc = tiktoken.Encoding(
    name='my_tokenizer',
    pat_str=tokenizer.pattern,
    mergeable_ranks=tokenizer.export_mergeable_ranks(),
    special_tokens=tokenizer.special_tokens,
)

enc.encode('hello world')  # same output as tokenizer.encode('hello world')
```

See `notebooks/tokenizer_tiktoken.ipynb` for a full walkthrough including a correctness check.

### 2. Pretrain a model

Create a JSON config file. All fields and their defaults are in `src/atomiclm/training/config.py`.

```json
{
    "model": {
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 4,
        "d_ff": 512,
        "max_seq_len": 256,
        "num_kv_heads": 2,
        "use_gated_ffn": true
    },
    "data": {
        "train_path": "data/corpus.txt",
        "tokenizer_path": "out/tokenizer-512.json",
        "max_seq_len": 256
    },
    "optim": {
        "lr": 3e-4,
        "num_epochs": 10,
        "batch_size": 32
    }
}
```

```bash
uv run python scripts/pretrain.py local/config.json
uv run python scripts/pretrain.py local/config.json --resume checkpoints/run/checkpoint_latest.pt
```

### 3. Generate text

```bash
uv run python scripts/generate.py checkpoints/run/checkpoint_latest.pt "The answer is" 50
uv run python scripts/generate.py checkpoints/run/checkpoint_latest.pt "Hello" 30 --temperature 0.8 --top-k 40
```

## What's Inside

### Tokenizer (`src/atomiclm/tokenizer/`)

BPE tokenizer operating on raw bytes with GPT-4 regex pre-splitting. Supports iterator-based training for large files, special tokens, and JSON persistence.

### Model (`src/atomiclm/model/`)

Decoder-only transformer following LLaMA-style design: Grouped Query Attention (configurable `num_kv_heads`), RoPE, RMSNorm, SwiGLU (optional), KV-cache, FlashAttention, and weight tying between embeddings and the LM head.

### Training (`src/atomiclm/training/`)

JSON config with nested dataclasses, AdamW with separate weight-decay groups, cosine LR with warmup, and checkpointing with full RNG state for reproducible resumption. Device auto-detection (cuda > mps > cpu).

## Project Structure

```
src/atomiclm/
    tokenizer/
        base.py             # Abstract Tokenizer interface
        tokenizer.py        # BasicTokenizer (BPE)
        bpe_ops.py          # Merge and pair-counting primitives
        patterns.py         # GPT-4 regex split pattern
        constants.py        # Base vocab size, version
    model/
        decoder.py          # Decoder (top-level LM)
        block.py            # TransformerBlock (pre-norm)
        attention.py        # MultiHeadAttention (GQA, KV-cache)
        feedforward.py      # FeedForward (GELU / SwiGLU)
        norm.py             # RMSNorm
        rope.py             # RotaryEmbedding
    training/
        config.py           # ModelConfig, DataConfig, OptimConfig, TrainingConfig
        data.py             # TextDataset, create_dataloaders
        train.py            # Trainer
        checkpoint.py       # Save/load checkpoints
scripts/
    train_tokenizer.py      # Train BPE tokenizer from text
    pretrain.py             # Pretrain model from JSON config
    generate.py             # Generate text from checkpoint
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `basic_tokenizer.ipynb` | BPE training, encoding/decoding, save/load |
| `tokenizer_tiktoken.ipynb` | Export trained tokenizer to tiktoken; correctness check |
| `tokenizer_special_tokens.ipynb` | Special token registration and encoding |
| `rope_explained.ipynb` | RoPE: 2D rotations, complex implementation, relative position encoding |
| `gqa_explained.ipynb` | GQA: MHA vs GQA vs MQA, KV-cache memory, repeat_kv |

## Roadmap

See [TODO.md](TODO.md).

## License

[MIT](LICENSE)

---

Developed with the assistance of Claude Code.
