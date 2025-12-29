import torch
import torch.nn as nn
from torch import Tensor


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Precomputes a frequency table of complex exponentials and applies
    2D rotations to query/key pairs so that their dot product naturally
    encodes relative position distance. Uses complex multiplication
    for compactness (LLaMA style).
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        # freq_i = 1 / (base^(2i/d)) for i in [0, d/2)
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # positions: [0, 1, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len).float()

        # outer product: (max_seq_len, head_dim/2)
        angles = torch.outer(positions, freqs)

        # complex exponentials: e^(i*theta) = cos(theta) + i*sin(theta)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        # Register as buffer (not a parameter — no gradient)
        self.register_buffer("freqs_cis", freqs_cis)

    @staticmethod
    def apply_rotary(x: Tensor, freqs_cis: Tensor) -> Tensor:
        """
        Apply rotary embedding to a tensor.

        Args:
            x: (batch, num_heads, seq_len, head_dim) real tensor
            freqs_cis: (seq_len, head_dim/2) complex tensor

        Returns:
            Rotated tensor with same shape as x.
        """
        # View as complex: (b, h, t, d/2) complex
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

        # Broadcast freqs_cis to (1, 1, t, d/2)
        freqs = freqs_cis.unsqueeze(0).unsqueeze(0)

        # Rotate via complex multiplication
        x_rotated = x_complex * freqs

        # Back to real: (b, h, t, d)
        return torch.view_as_real(x_rotated).flatten(-2).to(x.dtype)

    def forward(self, seq_len: int, offset: int = 0) -> Tensor:
        """
        Return frequency table for positions [offset, offset + seq_len).

        Args:
            seq_len: Number of positions to return.
            offset: Starting position (for cached decoding).

        Returns:
            Complex tensor of shape (seq_len, head_dim/2).
        """
        return self.freqs_cis[offset : offset + seq_len]
