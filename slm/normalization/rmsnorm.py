"""
Root Mean Square Layer Normalization (RMSNorm) implemented from scratch.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Normalizes input tensor by its root mean square across the feature dimension.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """
        Initializes RMSNorm parameters.

        Args:
            dim: Feature hidden dimension (d_model).
            eps: Epsilon parameter for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes RMS scaling factor and normalizes input tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying RMSNorm and learnable scaling parameter weight gamma.

        Args:
            x: Input tensor [..., dim].

        Returns:
            Normalized tensor [..., dim].
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
