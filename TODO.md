# Roadmap

## Active

- Streaming generation (yield tokens incrementally)
- EOS handling with continuous batching

## Planned

- Code data pipeline (The Stack / GitHub)
- Gradient accumulation + AMP
- RL fine-tuning (reward head, GRPO)
- HuggingFace model/tokenizer export
- Evaluation harness (HumanEval, lm-eval)
- Tokenizer Rust backend

## Completed

- Decoder-only transformer (GQA, RoPE, RMSNorm, SwiGLU, KV-cache, FlashAttention, weight tying)
- Training infrastructure (JSON config, AdamW with param grouping, cosine LR, checkpointing with RNG state)
- BPE tokenizer (GPT-4 regex, heap-based O(n log n) training, iterator training, special tokens)
- Tiktoken export
- Training and generation scripts
