from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rope import RotaryEmbedding


"""
Production-oriented multi-head self-attention with Grouped Query Attention (GQA),
replacing nn.MultiheadAttention for explicit control over memory layout, kernel
boundaries, and inference behavior. Supports MHA, GQA, and MQA via a single
num_kv_heads parameter with separate Q/K/V projections for memory-efficient
KV-cache. Uses a head-first tensor layout for FlashAttention-compatible kernels
and includes KV-cache support for autoregressive decoding. Masking and projection
logic are separated from attention math for extensibility and maintainability.
"""


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA) support.

    Features:
    - Supports MHA (num_kv_heads = num_heads), GQA (1 < num_kv_heads < num_heads),
      and MQA (num_kv_heads = 1) via a single num_kv_heads parameter
    - Separate Q/K/V projections for memory-efficient KV-cache
    - Optional attention mask (causal or arbitrary)
    - KV-cache–ready API for autoregressive decoding
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        causal: bool = True,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # GQA support: num_kv_heads defaults to num_heads (MHA)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )
        self.num_groups = self.num_heads // self.num_kv_heads

        # Split QKV projection: Q uses num_heads, K/V use num_kv_heads
        self.q_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.k_proj = nn.Linear(d_in, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(d_in, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        self.is_causal = causal

    @staticmethod
    def _repeat_kv(x: Tensor, num_groups: int) -> Tensor:
        """
        Expand K/V from (b, num_kv_heads, t, d_h) to (b, num_heads, t, d_h).

        Each KV head is repeated num_groups times to match the number of query heads.
        When num_groups == 1 (MHA), this is a no-op.

        Args:
            x: Tensor of shape (b, num_kv_heads, t, d_h)
            num_groups: Number of query heads per KV head

        Returns:
            Tensor of shape (b, num_kv_heads * num_groups, t, d_h)
        """
        if num_groups == 1:
            return x

        b, num_kv_heads, t, d_h = x.shape
        # Expand: (b, num_kv_heads, 1, t, d_h) -> (b, num_kv_heads, num_groups, t, d_h)
        x = x.unsqueeze(2).expand(b, num_kv_heads, num_groups, t, d_h)
        # Reshape to merge kv_heads and groups: (b, num_heads, t, d_h)
        return x.reshape(b, num_kv_heads * num_groups, t, d_h)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Compute multi-head self-attention with optional KV caching.

        This module supports both training (full-sequence attention) and
        autoregressive inference (incremental decoding with KV-cache).
        The attention computation is delegated to PyTorch's
        scaled_dot_product_attention, enabling FlashAttention-compatible
        kernels when possible.

        Args:
            x:
                Input tensor of shape (batch_size, seq_len, d_in).

            attn_mask:
                Optional attention mask of shape (seq_len, seq_len).
                Accepts two formats:
                    - Float (additive): added to attention logits before
                      softmax; 0 = attend, -inf = block.
                    - Bool: True = attend, False = block (PyTorch convention).
                      Converted to additive form internally.
                When provided, FlashAttention is disabled and the generic
                SDPA backend is used. If None and `causal=True`, a causal
                mask is applied internally.

            kv_cache:
                Optional dictionary used for autoregressive decoding.
                Expected keys:
                    - "k":   Cached keys of shape
                            (batch_size, num_kv_heads, cache_capacity, head_dim)
                    - "v":   Cached values of shape
                            (batch_size, num_kv_heads, cache_capacity, head_dim)
                    - "pos": Integer write pointer indicating the current
                            sequence position in the cache.

                When provided, the module appends the newly computed keys and
                values to the cache in-place and attends over the cached
                prefix plus the current timestep(s). KV caching is only
                supported in eval mode.

        Returns:
            output:
                Attention output of shape (batch_size, seq_len, d_out).

            new_cache:
                The updated KV cache when `kv_cache` is provided, otherwise
                None.
        """

        # ------------------------------------------------------------
        # Input
        # ------------------------------------------------------------
        # x: (batch, time, d_in)
        b, t, _ = x.shape

        # ------------------------------------------------------------
        # QKV projection (separate for GQA)
        # ------------------------------------------------------------
        # q: (b, t, d_out) = (b, t, num_heads * head_dim)
        # k, v: (b, t, num_kv_heads * head_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to separate heads
        # q: (b, t, num_heads, head_dim)
        # k, v: (b, t, num_kv_heads, head_dim)
        q = q.view(b, t, self.num_heads, self.head_dim)
        k = k.view(b, t, self.num_kv_heads, self.head_dim)
        v = v.view(b, t, self.num_kv_heads, self.head_dim)

        # Transpose to attention layout
        # q: (b, num_heads, t, head_dim)
        # k, v: (b, num_kv_heads, t, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ------------------------------------------------------------
        # Rotary positional embedding (RoPE)
        # ------------------------------------------------------------
        # Applied BEFORE caching so cached K's have positional info baked in.
        if freqs_cis is not None:
            q = RotaryEmbedding.apply_rotary(q, freqs_cis)
            k = RotaryEmbedding.apply_rotary(k, freqs_cis)

        # ------------------------------------------------------------
        # KV-cache validation & handling (autoregressive decoding)
        # ------------------------------------------------------------
        if kv_cache is not None:
            # KV-cache is for inference / decoding only
            assert not self.training
            assert isinstance(kv_cache, dict)

            required_keys = {"k", "v", "pos"}
            assert (
                kv_cache.keys() >= required_keys
            ), f"kv_cache must contain keys {required_keys}"

            cache_k: Tensor = kv_cache["k"]
            cache_v: Tensor = kv_cache["v"]
            pos: int = kv_cache["pos"]

            # dtype / device invariants
            assert cache_k.device == x.device
            assert cache_k.dtype == x.dtype

            # shape invariants
            assert cache_k.shape[:2] == (b, self.num_kv_heads)
            assert cache_k.shape[-1] == self.head_dim

            # capacity check
            assert pos + t <= cache_k.size(2), "KV-cache capacity exceeded"

            # write new keys / values in-place
            cache_k[:, :, pos : pos + t, :] = k
            cache_v[:, :, pos : pos + t, :] = v

            # use cached prefix for attention
            k = cache_k[:, :, : pos + t, :]
            v = cache_v[:, :, : pos + t, :]

            kv_cache["pos"] = pos + t
            new_cache = kv_cache
        else:
            new_cache = None

        # ------------------------------------------------------------
        # GQA: Expand K/V heads to match Q heads
        # ------------------------------------------------------------
        # k, v are (b, num_kv_heads, t_kv, head_dim)
        # Expand to (b, num_heads, t_kv, head_dim) for attention
        k = self._repeat_kv(k, self.num_groups)
        v = self._repeat_kv(v, self.num_groups)

        # ------------------------------------------------------------
        # Attention mask normalization / validation
        # ------------------------------------------------------------
        if attn_mask is not None:
            # Custom masks disable FlashAttention
            assert not self.is_causal, (
                "Custom attention masks disable FlashAttention; "
                "set causal=False explicitly"
            )

            # Convert boolean mask -> additive mask (True=attend, False=block)
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.zeros_like(attn_mask, dtype=x.dtype).masked_fill(
                    ~attn_mask, float("-inf")
                )

            assert attn_mask.dtype in (
                torch.float16,
                torch.float32,
                torch.bfloat16,
            )

        # ------------------------------------------------------------
        # Scaled Dot-Product Attention (FLASHABLE)
        # ------------------------------------------------------------
        # q, k, v: (b, h, t, d_h)

        # is_causal=True in SDPA assumes L==S (PyTorch < 2.5), so only
        # safe for the full-sequence path without cache.
        use_causal = self.is_causal and attn_mask is None and kv_cache is None

        # Multi-token cached prefill: build explicit prefix-aware causal mask.
        # Each new query attends to ALL cached keys plus causally within the
        # new chunk.  Single-token steps (t=1) need no mask — the lone query
        # legitimately attends to every cached key.
        if kv_cache is not None and self.is_causal and t > 1:
            prefix_len = k.size(2) - t
            causal_block = torch.triu(
                torch.full((t, t), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            attn_mask = torch.cat(
                [torch.zeros(t, prefix_len, device=x.device, dtype=x.dtype), causal_block],
                dim=1,
            )

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,  # must be None for FlashAttention
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=use_causal,  # enables FlashAttention causal kernel
        )

        # context: (b, h, t, d_h)

        # ------------------------------------------------------------
        # Merge heads
        # ------------------------------------------------------------
        context = context.transpose(1, 2).contiguous().view(b, t, self.d_out)

        # ------------------------------------------------------------
        # Final projection
        # ------------------------------------------------------------
        output = self.out_proj(context)

        return output, new_cache
