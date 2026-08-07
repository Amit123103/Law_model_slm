"""
Feed-Forward Networks supporting SwiGLU, GELU, and ReLU activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    SwiGLU (Swish Gated Linear Unit) Feed-Forward Network (Shazeer, 2020).
    FFN_SwiGLU(x) = (SiLU(x W_gate) * (x W_up)) W_down
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = False
    ) -> None:
        """
        Initializes SwiGLU projection weights.

        Args:
            d_model: Input and output feature dimension.
            d_ff: Inner feed-forward hidden dimension.
            dropout: Dropout probability.
            bias: Whether linear layers contain bias vectors.
        """
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initializes weight parameters."""
        nn.init.normal_(self.w_gate.weight, std=0.02)
        nn.init.normal_(self.w_up.weight, std=0.02)
        nn.init.normal_(self.w_down.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying SwiGLU activation and projection down.
        """
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        output = self.w_down(hidden)
        return self.dropout(output)


class StandardFFN(nn.Module):
    """
    Standard Feed-Forward Network supporting GELU or ReLU activations.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = False
    ) -> None:
        """
        Initializes Standard FFN layers.
        """
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.act(self.w1(x))
        output = self.w2(hidden)
        return self.dropout(output)


def build_feedforward(
    d_model: int,
    d_ff: int,
    activation: str = "swiglu",
    dropout: float = 0.1,
    bias: bool = False
) -> nn.Module:
    """
    Factory function building feed-forward network matching requested activation type.
    """
    if activation == "swiglu":
        return SwiGLUFFN(d_model=d_model, d_ff=d_ff, dropout=dropout, bias=bias)
    else:
        return StandardFFN(d_model=d_model, d_ff=d_ff, activation=activation, dropout=dropout, bias=bias)
