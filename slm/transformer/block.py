"""
Transformer Block implementing Pre-Normalization, Causal Self-Attention, and SwiGLU/FFN.
"""

from typing import Optional
import torch
import torch.nn as nn

from slm.config.model_config import ModelConfig
from slm.attention.causal_attention import MultiHeadCausalAttention
from slm.normalization.rmsnorm import RMSNorm
from slm.normalization.layernorm import CustomLayerNorm
from slm.feedforward.mlp import build_feedforward


class TransformerBlock(nn.Module):
    """
    Single Decoder-Only Transformer Layer combining Pre-Norm, Multi-Head Causal Attention,
    Residual Connections, and Feed-Forward Network.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initializes TransformerBlock with model configuration options.

        Args:
            config: ModelConfig instance.
        """
        super().__init__()
        self.config = config

        # Pre-Normalization 1 (Attention)
        if config.norm_type == "rmsnorm":
            self.norm1 = RMSNorm(dim=config.d_model, eps=config.layer_norm_eps)
            self.norm2 = RMSNorm(dim=config.d_model, eps=config.layer_norm_eps)
        else:
            self.norm1 = CustomLayerNorm(dim=config.d_model, eps=config.layer_norm_eps, bias=config.bias)
            self.norm2 = CustomLayerNorm(dim=config.d_model, eps=config.layer_norm_eps, bias=config.bias)

        # Multi-Head Causal Attention
        use_rope = (config.pos_encoding_type == "rope")
        self.attn = MultiHeadCausalAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            bias=config.bias,
            use_rope=use_rope,
            rope_base=config.rope_base
        )

        # Feed-Forward Network
        self.ffn = build_feedforward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            activation=config.activation,
            dropout=config.dropout,
            bias=config.bias
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass applying Pre-Norm Attention Residual followed by Pre-Norm FFN Residual.

        Args:
            x: Hidden state Tensor [batch_size, seq_len, d_model].
            attention_mask: Optional attention mask Tensor.

        Returns:
            Updated hidden state Tensor [batch_size, seq_len, d_model].
        """
        # 1. Causal Attention Sub-layer with Residual Connection
        h = x + self.attn(self.norm1(x), attention_mask=attention_mask)

        # 2. Feed-Forward Sub-layer with Residual Connection
        out = h + self.ffn(self.norm2(h))

        return out
