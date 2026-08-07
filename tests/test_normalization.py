"""
Unit tests for RMSNorm and LayerNorm modules.
"""

import torch
import pytest

from slm.normalization.rmsnorm import RMSNorm
from slm.normalization.layernorm import CustomLayerNorm


def test_rmsnorm_forward_and_backward():
    batch, seq, dim = 2, 8, 64
    x = torch.randn(batch, seq, dim, requires_grad=True)

    norm = RMSNorm(dim=dim)
    out = norm(x)

    assert out.shape == (batch, seq, dim)
    
    # Check that root mean square across last dimension is close to 1.0 (before scaling)
    rms_val = torch.sqrt(out.pow(2).mean(dim=-1))
    assert torch.allclose(rms_val, torch.ones_like(rms_val), atol=1e-2)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_layernorm_forward_and_backward():
    batch, seq, dim = 2, 8, 64
    x = torch.randn(batch, seq, dim, requires_grad=True)

    norm = CustomLayerNorm(dim=dim)
    out = norm(x)

    assert out.shape == (batch, seq, dim)

    # Check mean close to 0 and std close to 1
    mean = out.mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-3)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
