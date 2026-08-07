"""
Unit tests for Multi-Head Causal Attention and RoPE embeddings.
"""

import torch
import pytest

from slm.attention.causal_attention import MultiHeadCausalAttention
from slm.embeddings.positional import RotaryPositionEmbedding


def test_rope_embedding_shape():
    dim = 16
    max_seq_len = 64
    rope = RotaryPositionEmbedding(dim=dim, max_seq_len=max_seq_len)

    batch, n_heads, seq_len, head_dim = 2, 4, 32, dim
    q = torch.randn(batch, n_heads, seq_len, head_dim)
    k = torch.randn(batch, n_heads, seq_len, head_dim)

    q_rot, k_rot = rope(q, k, seq_len=seq_len)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_multihead_causal_attention():
    d_model = 64
    n_heads = 4
    batch, seq_len = 2, 16

    mha = MultiHeadCausalAttention(d_model=d_model, n_heads=n_heads, use_rope=True)
    x = torch.randn(batch, seq_len, d_model)

    out = mha(x)
    assert out.shape == (batch, seq_len, d_model)


def test_causal_mask_property():
    """Verifies that future token modifications do not affect past position outputs."""
    d_model = 32
    n_heads = 2
    seq_len = 8

    mha = MultiHeadCausalAttention(d_model=d_model, n_heads=n_heads, use_rope=False)
    mha.eval()

    x = torch.randn(1, seq_len, d_model)
    out1 = mha(x)

    # Modify future tokens (position 5 onwards)
    x_modified = x.clone()
    x_modified[0, 5:, :] += 10.0
    out2 = mha(x_modified)

    # Past position outputs (positions 0..4) must remain identical
    assert torch.allclose(out1[0, :5, :], out2[0, :5, :], atol=1e-5)
