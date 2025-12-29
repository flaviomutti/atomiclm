import math
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .block import TransformerBlock
from .norm import RMSNorm
from .rope import RotaryEmbedding


class Decoder(nn.Module):
    """
    Decoder-only transformer language model.

    Wires together token embeddings, rotary positional embeddings (RoPE),
    N transformer blocks, a final RMSNorm, and an LM head with weight tying.
    Supports KV-cache for efficient autoregressive decoding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 4096,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        use_gated_ffn: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.rope = RotaryEmbedding(
            head_dim=d_model // num_heads,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
                qkv_bias=qkv_bias,
                use_gated_ffn=use_gated_ffn,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: embedding and LM head share weights
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _residual_projections(self):
        """Yield the output projection modules that feed into residual adds."""
        for block in self.blocks:
            yield block.attn.out_proj
            if block.ffn.use_gated:
                yield block.ffn.down_proj
            else:
                yield block.ffn.fc2

    def _init_weights(self):
        """
        Initialize weights following GPT-2 / LLaMA conventions.

        - Linear layers: Normal(0, 0.02)
        - Embeddings: Normal(0, 0.02)
        - Biases: zeros
        - Residual output projections (out_proj, fc2/down_proj):
          scaled by 1/sqrt(2 * num_layers) to stabilize the residual stream.
        """
        residual_std = 0.02 / math.sqrt(2 * self.num_layers)
        residual_proj_ids = {id(m) for m in self._residual_projections()}

        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = residual_std if id(module) in residual_proj_ids else 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: Tensor,
        kv_cache: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Tensor, Optional[List[Dict[str, Tensor]]]]:
        """
        Forward pass through the full model.

        Args:
            token_ids: (batch, seq_len) long tensor of token indices.
            kv_cache: Optional list of N cache dicts (one per layer).

        Returns:
            logits: (batch, seq_len, vocab_size) float tensor.
            updated_cache: Updated KV cache list, or None if no cache was passed.
        """
        b, t = token_ids.shape

        # 1. Token embedding
        x = self.tok_emb(token_ids)  # (b, t, d_model)

        # 2. Compute RoPE frequencies with correct offset
        offset = kv_cache[0]["pos"] if kv_cache is not None else 0
        freqs_cis = self.rope(t, offset=offset)

        # 3. Pass through transformer blocks
        updated_cache: Optional[List[Dict[str, Tensor]]] = (
            [None] * self.num_layers if kv_cache is not None else None
        )
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_layer_cache = block(
                x, kv_cache=layer_cache, freqs_cis=freqs_cis,
            )
            if updated_cache is not None:
                updated_cache[i] = new_layer_cache

        # 4. Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits, updated_cache

    def make_cache(
        self,
        batch_size: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> List[Dict[str, Tensor]]:
        """
        Create an empty KV cache for autoregressive decoding.

        Cache uses num_kv_heads (not num_heads) for memory efficiency with GQA.

        Returns:
            List of N dicts, each with keys "k", "v" (zero tensors)
            and "pos" (int 0).
        """
        cache = []
        for _ in range(self.num_layers):
            cache.append({
                "k": torch.zeros(
                    batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim,
                    device=device, dtype=dtype,
                ),
                "v": torch.zeros(
                    batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim,
                    device=device, dtype=dtype,
                ),
                "pos": 0,
            })
        return cache

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Autoregressive generation with KV-cache.

        Args:
            token_ids: (batch, prompt_len) long tensor of prompt token ids.
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature (1.0 = no change).
            top_k: If > 0, keep only top-k logits before sampling.
            eos_token_id: If set, stop generating per-sequence when EOS is
                produced. Requires pad_token_id to be set as well.
            pad_token_id: Token used to pad finished sequences. Required
                when eos_token_id is set.

        Returns:
            (batch, prompt_len + generated_len) long tensor of all token ids.
            generated_len <= max_new_tokens (may be shorter if all sequences
            hit EOS early). Finished sequences are padded with pad_token_id.
        """
        if eos_token_id is not None and pad_token_id is None:
            raise ValueError("pad_token_id is required when eos_token_id is set")

        self.eval()
        b = token_ids.shape[0]
        device = token_ids.device
        dtype = self.tok_emb.weight.dtype

        cache = self.make_cache(b, device=device, dtype=dtype)
        generated = token_ids

        # Per-sequence finished tracking
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        # Prefill: process entire prompt
        logits, cache = self.forward(token_ids, kv_cache=cache)

        # Generate new tokens one at a time
        for _ in range(max_new_tokens):
            # Sample from last position
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                # Zero out everything outside top-k
                top_values, _ = next_logits.topk(top_k, dim=-1)
                threshold = top_values[:, -1].unsqueeze(-1)
                next_logits = next_logits.where(
                    next_logits >= threshold,
                    torch.full_like(next_logits, float("-inf")),
                )

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (b, 1)

            # EOS handling: replace output with pad for already-finished seqs
            if eos_token_id is not None:
                next_id = next_id.where(
                    ~finished.unsqueeze(1),
                    torch.tensor(pad_token_id, device=device),
                )
                finished = finished | (next_id.squeeze(1) == eos_token_id)

            generated = torch.cat([generated, next_id], dim=1)

            if finished.all():
                break

            # Forward single token with cache
            logits, cache = self.forward(next_id, kv_cache=cache)

        return generated
