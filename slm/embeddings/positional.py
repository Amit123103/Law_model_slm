"""
Positional Encodings: Learned, Sinusoidal, and Rotary Position Embeddings (RoPE).
"""

import math
from typing import Tuple
import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    """
    Trainable 1D Positional Embedding matrix [max_seq_len, d_model].
    """

    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        return self.dropout(self.embedding(positions))


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Fixed Sinusoidal Positional Encoding (Vaswani et al., 2017).
    """

    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: [1, max_seq_len, d_model]

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return self.dropout(self.pe[:, :seq_len, :])


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) applied to Queries and Keys (Su et al., 2021).
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        """
        Initializes RoPE frequency matrices.

        Args:
            dim: Dimension per attention head.
            max_seq_len: Maximum context length.
            base: Frequency base constant (10000.0).
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Inverse frequencies theta_i = base^(-2(i-1)/dim)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        """Builds pre-computed cos and sin frequency cache tensors."""
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate frequencies to match head dimension
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half dimensions: [-x2, x1]."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies Rotary Embeddings to Query and Key tensors.

        Args:
            q: Queries tensor of shape [batch_size, n_heads, seq_len, head_dim].
            k: Keys tensor of shape [batch_size, n_heads, seq_len, head_dim].
            seq_len: Current sequence length.

        Returns:
            Tuple of (rotated_q, rotated_k).
        """
        cos = self.cos_cached[:seq_len, :].to(q.device)  # [seq_len, head_dim]
        sin = self.sin_cached[:seq_len, :].to(q.device)  # [seq_len, head_dim]

        # Reshape for broadcasting over batch and heads: [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed
