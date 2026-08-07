"""
Scaled Dot-Product Attention and Multi-Head Causal Self-Attention implemented from scratch.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.embeddings.positional import RotaryPositionEmbedding


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism with causal masking.
    Attention(Q, K, V) = softmax((Q @ K.T) / sqrt(d_k) + Mask) @ V
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for scaled dot-product attention.

        Args:
            q: Queries tensor of shape [batch_size, n_heads, seq_len_q, head_dim].
            k: Keys tensor of shape [batch_size, n_heads, seq_len_k, head_dim].
            v: Values tensor of shape [batch_size, n_heads, seq_len_v, head_dim].
            mask: Optional causal or padding mask tensor.

        Returns:
            Tuple of (output tensor [batch_size, n_heads, seq_len_q, head_dim], attn_weights).
        """
        head_dim = q.size(-1)
        scale = 1.0 / math.sqrt(head_dim)

        # Compute raw dot-product scores: [batch, n_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention probabilities
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values: [batch, n_heads, seq_len_q, head_dim]
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadCausalAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention module supporting RoPE position embeddings.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        bias: bool = False,
        use_rope: bool = True,
        rope_base: float = 10000.0
    ) -> None:
        """
        Initializes MultiHeadCausalAttention.

        Args:
            d_model: Hidden dimension.
            n_heads: Number of parallel attention heads.
            max_seq_len: Maximum sequence context length.
            dropout: Dropout probability.
            bias: Whether linear projection layers contain bias.
            use_rope: Whether to apply Rotary Position Embeddings to Q and K.
            rope_base: RoPE base frequency scaling factor.
        """
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope

        # Linear projections for Queries, Keys, Values, and Output
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_engine = ScaledDotProductAttention(dropout=dropout)
        self.out_dropout = nn.Dropout(dropout)

        if self.use_rope:
            self.rope = RotaryPositionEmbedding(dim=self.head_dim, max_seq_len=max_seq_len, base=rope_base)
        else:
            self.rope = None

        # Build causal lower-triangular mask matrix
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len), persistent=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for Multi-Head Causal Attention.

        Args:
            x: Input feature Tensor [batch_size, seq_len, d_model].
            attention_mask: Optional external padding mask [batch_size, 1, 1, seq_len].

        Returns:
            Output Tensor [batch_size, seq_len, d_model].
        """
        batch_size, seq_len, _ = x.size()

        # Project and reshape into heads: [batch_size, n_heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if enabled
        if self.use_rope and self.rope is not None:
            q, k = self.rope(q, k, seq_len=seq_len)

        # Slice lower-triangular causal mask for current sequence length
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        if attention_mask is not None:
            mask = mask & attention_mask.bool()

        # Scaled dot-product attention
        attn_out, _ = self.attn_engine(q, k, v, mask=mask)

        # Concatenate heads: [batch_size, seq_len, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final output projection
        output = self.out_proj(attn_out)
        return self.out_dropout(output)
