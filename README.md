# ⚡ AtomicLM

From-scratch decoder-only transformer in Python/PyTorch. No `.from_pretrained()`, no framework wrappers, no mystery layers — every component written explicitly and meant to be read.

Train a BPE tokenizer → pretrain a transformer → generate text. That's the whole pipeline.

## 🧠 About

AtomicLM is a learning-first codebase. The goal isn't a competitive model — it's a codebase where you can trace every operation, understand every gradient, and learn by breaking things.

- **🔍 No black boxes.** Any failure traces to a specific line. If you can't follow the code, that's a bug in the code.
- **📖 Readable over clever.** Ten explicit lines beat one cryptic abstraction.
- **🪶 Minimal footprint.** PyTorch and a regex library. Nothing else at runtime.
- **🆓 Free and open-source.** MIT — use it, fork it, publish papers with it.

## ⚙️ Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) — if you don't have it: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## 🚀 Quickstart

```bash
git clone https://github.com/flaviomutti/atomiclm.git
cd atomiclm && uv sync

# 1. Train a BPE tokenizer on your corpus
uv run python scripts/train_tokenizer.py data/corpus.txt 512

# 2. Pretrain — write a config.json first (see Usage below)
uv run python scripts/pretrain.py config.json

# 3. Generate
uv run python scripts/generate.py checkpoints/run/checkpoint_latest.pt "The answer is" 50
```

## 🛠️ Usage

### 🔤 Tokenizer

Train a BPE tokenizer on any UTF-8 text file. Pick vocabulary size based on corpus size — 512 for small experiments, 32k+ for real data.

```bash
uv run python scripts/train_tokenizer.py data/corpus.txt 512 --output out/tokenizer-512
```

Optional flags:

```bash
# Register special tokens (they bypass BPE and get reserved IDs)
uv run python scripts/train_tokenizer.py data/corpus.txt 1024 \
    --special "<|endoftext|>" "<|pad|>"

# Swap to the simpler O(n²) algorithm — useful for debugging or tiny corpora
uv run python scripts/train_tokenizer.py data/corpus.txt 1024 --method basic
```

Output is a JSON file — merges stored as `[left, right, id]` triples, human-readable, reloadable without retraining.

### 📦 Export to tiktoken

The trained tokenizer can wrap as a tiktoken `Encoding` — useful for fast inference or dropping into pipelines that already expect tiktoken's API.

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

enc.encode('hello world')  # identical output to tokenizer.encode('hello world')
```

See `notebooks/tokenizer_tiktoken.ipynb` for a roundtrip correctness check.

### 🏋️ Pretrain

Write a JSON config. All available fields and defaults live in `src/atomiclm/training/config.py` — it's short, read it.

```json
{
    "model": {
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "d_ff": 512,
        "max_seq_len": 256,
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
        "batch_size": 32,
        "grad_accum_steps": 4
    }
}
```

```bash
uv run python scripts/pretrain.py local/config.json

# Resume from a checkpoint (restores model, optimizer, and RNG state)
uv run python scripts/pretrain.py local/config.json --resume checkpoints/run/checkpoint_latest.pt
```

`grad_accum_steps` lets you simulate a larger effective batch without extra GPU memory — 4 micro-batches × 32 = effective batch of 128, same gradient as if you fit all 128 in one shot.

### ✍️ Generate

```bash
# checkpoint path, prompt string, max new tokens
uv run python scripts/generate.py checkpoints/run/checkpoint_latest.pt "The answer is" 50

# dial in randomness
uv run python scripts/generate.py checkpoints/run/checkpoint_latest.pt "Hello" 100 \
    --temperature 0.8 --top-k 40
```

`--temperature` scales the logit distribution before sampling (lower = greedier, higher = more chaotic). `--top-k` restricts sampling to the k highest-probability tokens.

## 🔬 Architecture

### 🔤 Tokenizer (`src/atomiclm/tokenizer/`)

Byte Pair Encoding on raw bytes (base vocab = 256 byte tokens). Uses GPT-4's regex pre-split pattern to chunk text before merging — same design as tiktoken, so the split boundary doesn't contaminate merge statistics. Training is O(n log n) via a heap that tracks pair positions for incremental updates.

Key files:

| File | What it does |
|------|-------------|
| `tokenizer.py` | `BasicTokenizer` — `train`, `encode`, `decode`, `save`, `load` |
| `bpe_ops.py` | Low-level `merge()` and `get_stats()` primitives |
| `patterns.py` | GPT-4 compatible Unicode-aware split regex |

### 🤖 Model (`src/atomiclm/model/`)

LLaMA-style decoder-only transformer:

| Component | Details |
|-----------|---------|
| Attention | Multi-head + GQA (`num_kv_heads`), RoPE, KV-cache, FlashAttention via `F.scaled_dot_product_attention` |
| FFN | GELU (default) or SwiGLU (`use_gated_ffn: true`) |
| Norm | RMSNorm — pre-norm, no mean subtraction, no bias |
| Positional | Rotary embeddings (complex-number rotation, LLaMA style) |
| LM head | Weight-tied with the token embedding matrix |

KV-cache is wired up in `Decoder.generate()` so autoregressive decoding doesn't recompute past keys and values.

### 🏋️ Training (`src/atomiclm/training/`)

| Feature | Implementation |
|---------|---------------|
| Config | Nested dataclasses from JSON; device auto-detected (cuda > mps > cpu) |
| Optimizer | AdamW with separate weight-decay groups — residual projections decay, biases/norms don't |
| Schedule | Cosine LR with linear warmup |
| Gradient accumulation | `grad_accum_steps` micro-batches per optimizer step |
| Checkpointing | Saves model + optimizer + RNG state — exact resumption, not approximate |

## 🗂️ Project Structure

```
src/atomiclm/
    tokenizer/
        base.py             # Abstract Tokenizer interface
        tokenizer.py        # BasicTokenizer (BPE)
        bpe_ops.py          # Merge and pair-counting primitives
        patterns.py         # GPT-4 regex split pattern
        constants.py        # Base vocab size, version
    model/
        decoder.py          # Decoder — top-level LM + generate()
        block.py            # TransformerBlock (pre-norm)
        attention.py        # MultiHeadAttention (GQA, KV-cache, FlashAttn)
        feedforward.py      # FeedForward (GELU / SwiGLU)
        norm.py             # RMSNorm
        rope.py             # RotaryEmbedding
    training/
        config.py           # ModelConfig, DataConfig, OptimConfig, TrainingConfig
        data.py             # TextDataset, create_dataloaders
        train.py            # Trainer
        checkpoint.py       # Save/load checkpoints
scripts/
    train_tokenizer.py      # CLI: train BPE tokenizer
    pretrain.py             # CLI: pretrain model
    generate.py             # CLI: generate from checkpoint
```

## 📓 Notebooks

Good starting points if you want to understand one component before diving into the source:

| Notebook | What you'll learn |
|----------|------------------|
| `basic_tokenizer.ipynb` | BPE from scratch: training, encoding, decoding, save/load |
| `tokenizer_tiktoken.ipynb` | Export to tiktoken and verify identical output |
| `tokenizer_special_tokens.ipynb` | How special tokens bypass BPE |
| `rope_explained.ipynb` | RoPE: 2D rotations, complex numbers, relative positions |
| `gqa_explained.ipynb` | GQA vs MHA vs MQA, KV-cache memory math, `repeat_kv` |

## 🗺️ Roadmap

See [TODO.md](TODO.md).

## 📄 License

[MIT](LICENSE)

---

Developed with the assistance of Claude Code.
