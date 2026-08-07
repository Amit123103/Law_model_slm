"""
Layer Normalization (LayerNorm) implemented from scratch.
"""

import torch
import torch.nn as nn


class CustomLayerNorm(nn.Module):
    """
    Standard Layer Normalization implemented from PyTorch primitives.
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True) -> None:
        """
        Initializes LayerNorm parameters.

        Args:
            dim: Feature hidden dimension.
            eps: Epsilon denominator value for stability.
            bias: Whether to include learnable bias parameter beta.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying mean-centering, variance scaling, and affine transform.
        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        output = x_norm * self.weight
        if self.bias is not None:
            output = output + self.bias
            
        return output
