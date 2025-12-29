from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .norm import RMSNorm
from .feedforward import FeedForward


"""
Pre-norm decoder transformer block.

Composes multi-head self-attention with a position-wise feed-forward network,
using RMSNorm and residual connections. Follows the pre-norm architecture
used in LLaMA and other modern decoder-only LLMs:

    residual = x
    x = RMSNorm(x) -> MHA(x) -> dropout + residual
    residual = x
    x = RMSNorm(x) -> FFN(x) + residual

KV-cache for autoregressive decoding passes through to the attention layer.
"""


class TransformerBlock(nn.Module):
    """
    Pre-norm decoder transformer block.

    Architecture: RMSNorm -> MHA -> residual -> RMSNorm -> FFN -> residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        use_gated_ffn: bool = False,
    ):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(
            d_in=d_model,
            d_out=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            qkv_bias=qkv_bias,
            causal=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            use_gated=use_gated_ffn,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

            attn_mask:
                Optional attention mask forwarded to MultiHeadAttention.
                Note: passing a mask with the default causal=True will raise
                (custom masks and causal are mutually exclusive in MHA).

            kv_cache:
                Optional KV cache dict for autoregressive decoding.
                Passed through to MultiHeadAttention and returned.

            freqs_cis:
                Optional RoPE frequency tensor forwarded to MultiHeadAttention.

        Returns:
            output: Tensor of shape (batch, seq_len, d_model).
            new_cache: Updated KV cache or None.
        """
        # ------------------------------------------------------------
        # Self-attention with pre-norm and residual
        # ------------------------------------------------------------
        residual = x
        x = self.norm1(x)
        x, new_cache = self.attn(
            x, attn_mask=attn_mask, kv_cache=kv_cache, freqs_cis=freqs_cis,
        )
        x = self.attn_dropout(x) + residual

        # ------------------------------------------------------------
        # Feed-forward with pre-norm and residual
        # ------------------------------------------------------------
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual

        return x, new_cache
