import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Position-wise feed-forward network for transformer decoder blocks.

Supports two variants:
  - Standard: Linear -> GELU -> Dropout -> Linear -> Dropout
  - Gated (SwiGLU): silu(gate) * up -> Dropout -> down -> Dropout

The gated variant follows the LLaMA convention (no bias on projections).
"""


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Supports standard (GELU) and gated (SwiGLU) variants.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        use_gated: bool = False,
    ):
        super().__init__()
        self.use_gated = use_gated

        if use_gated:
            # SwiGLU: silu(gate_proj(x)) * up_proj(x) -> down_proj
            self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
            self.up_proj = nn.Linear(d_model, d_ff, bias=False)
            self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        else:
            # Standard: fc1 -> GELU -> fc2
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        if self.use_gated:
            # ------------------------------------------------------------
            # Gated (SwiGLU) path
            # ------------------------------------------------------------
            hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
            output = self.down_proj(self.dropout(hidden))
        else:
            # ------------------------------------------------------------
            # Standard (GELU) path
            # ------------------------------------------------------------
            hidden = F.gelu(self.fc1(x))
            output = self.fc2(self.dropout(hidden))

        return self.dropout(output)
