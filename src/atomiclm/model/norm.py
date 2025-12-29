import torch
import torch.nn as nn


"""
Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

Normalizes activations by their root mean square without mean subtraction,
using a learnable scale parameter and no bias. Simpler and faster than
standard LayerNorm; used in LLaMA and other modern decoder-only LLMs.
"""


class RMSNorm(nn.Module):
    """
    RMSNorm: normalizes by RMS of activations, with learnable gain.

    Unlike LayerNorm, there is no mean subtraction and no bias term.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Normalized tensor of the same shape.
        """
        # ------------------------------------------------------------
        # RMS normalization
        # ------------------------------------------------------------
        # norm: (b, t, 1)
        norm = x.pow(2).mean(-1, keepdim=True) + self.eps
        x = x * torch.rsqrt(norm)

        # ------------------------------------------------------------
        # Learnable scale
        # ------------------------------------------------------------
        return x * self.weight
