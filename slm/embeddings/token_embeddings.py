"""
Token Embeddings module with weight initialization and dropout.
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token Embedding mapping discrete integer token IDs to continuous dense vectors.
    """

    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1) -> None:
        """
        Initializes token embedding weights.

        Args:
            vocab_size: Total vocabulary size.
            d_model: Hidden embedding dimension.
            dropout: Embedding dropout rate.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initializes weight matrix with normal distribution N(0, 0.02)."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting token IDs to embedding vectors.

        Args:
            input_ids: Long Tensor of shape [batch_size, seq_len].

        Returns:
            Embedded representation Tensor of shape [batch_size, seq_len, d_model].
        """
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        return self.dropout(x)

    def get_weight(self) -> torch.Tensor:
        """Returns weight matrix for LM Head weight tying."""
        return self.embedding.weight
