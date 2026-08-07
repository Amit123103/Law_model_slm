"""
Decoder-Only Small Language Model (SLM) for Causal Language Modeling.
"""

from typing import Optional, Dict, Any, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.config.model_config import ModelConfig
from slm.embeddings.token_embeddings import TokenEmbedding
from slm.embeddings.positional import LearnedPositionalEmbedding, SinusoidalPositionalEmbedding
from slm.normalization.rmsnorm import RMSNorm
from slm.normalization.layernorm import CustomLayerNorm
from slm.transformer.block import TransformerBlock
from slm.utils.logger import get_logger
from slm.utils.utils import count_parameters, format_number

logger = get_logger("slm.model")


class SLMForCausalLM(nn.Module):
    """
    Decoder-Only Small Language Model (SLM) architecture for autoregressive text generation.
    Implemented completely from scratch using PyTorch tensor operations.
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initializes SLM architecture layers.

        Args:
            config: ModelConfig instance defining architecture parameters.
        """
        super().__init__()
        self.config = config

        # Token Embedding Matrix
        self.token_embeddings = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            dropout=config.dropout
        )

        # Optional Positional Encodings (if not using RoPE inside attention layers)
        if config.pos_encoding_type == "learned":
            self.pos_embeddings = LearnedPositionalEmbedding(
                max_seq_len=config.max_seq_len,
                d_model=config.d_model,
                dropout=config.dropout
            )
        elif config.pos_encoding_type == "sinusoidal":
            self.pos_embeddings = SinusoidalPositionalEmbedding(
                max_seq_len=config.max_seq_len,
                d_model=config.d_model,
                dropout=config.dropout
            )
        else:  # 'rope'
            self.pos_embeddings = None

        # Stack of Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config=config) for _ in range(config.n_layers)
        ])

        # Final Pre-head Normalization Layer
        if config.norm_type == "rmsnorm":
            self.final_norm = RMSNorm(dim=config.d_model, eps=config.layer_norm_eps)
        else:
            self.final_norm = CustomLayerNorm(dim=config.d_model, eps=config.layer_norm_eps, bias=config.bias)

        # Language Model Projection Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights between token embeddings and LM head projection if enabled
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.embedding.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initializes non-tied weights."""
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward propagation through decoder-only Transformer layers.

        Args:
            input_ids: Long Tensor of token IDs [batch_size, seq_len].
            targets: Optional ground-truth target token IDs [batch_size, seq_len].
            target_ids: Alternative ground-truth target token IDs.
            labels: Alternative ground-truth target token IDs.
            attention_mask: Optional attention mask Tensor [batch_size, 1, 1, seq_len].

        Returns:
            Tuple of (logits [batch_size, seq_len, vocab_size], loss (Scalar Tensor or None)).
        """
        if targets is None:
            targets = target_ids if target_ids is not None else labels

        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # 1. Look up token embeddings: [batch_size, seq_len, d_model]
        h = self.token_embeddings(input_ids)

        # 2. Add positional embeddings if enabled (learned or sinusoidal)
        if self.pos_embeddings is not None:
            pos_emb = self.pos_embeddings(seq_len, device=device)
            h = h + pos_emb

        # 3. Forward through transformer blocks
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)

        # 4. Final normalization
        h = self.final_norm(h)

        # 5. Project to vocabulary logits: [batch_size, seq_len, vocab_size]
        logits = self.lm_head(h)

        # 6. Compute cross-entropy loss if targets provided
        loss = None
        if targets is not None:
            # Flatten tensors for cross-entropy loss computation
            # logits: [batch_size * seq_len, vocab_size]
            # targets: [batch_size * seq_len]
            flat_logits = logits.view(-1, self.config.vocab_size)
            flat_targets = targets.view(-1)
            loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)

        return logits, loss

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Calculates total or non-embedding parameter count.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_embeddings is not None and hasattr(self.pos_embeddings, "embedding"):
            n_params -= self.pos_embeddings.embedding.weight.numel()
        return n_params

    def estimate_memory_mb(self, batch_size: int = 16, seq_len: int = 512) -> Dict[str, float]:
        """
        Estimates VRAM memory required for model weights, activations, and gradients.
        """
        params = self.get_num_params(non_embedding=False)
        bytes_per_param = 4  # float32
        weight_mem_mb = (params * bytes_per_param) / (1024 ** 2)
        grad_mem_mb = weight_mem_mb  # gradient memory equals weight memory

        # Rough activation memory estimation per layer
        activation_elements = batch_size * seq_len * self.config.d_model * self.config.n_layers * 4
        activation_mem_mb = (activation_elements * bytes_per_param) / (1024 ** 2)

        return {
            "weight_memory_mb": round(weight_mem_mb, 2),
            "grad_memory_mb": round(grad_mem_mb, 2),
            "activation_memory_mb": round(activation_mem_mb, 2),
            "total_estimated_mb": round(weight_mem_mb * 3 + activation_mem_mb, 2)  # weights + grads + opt state
        }

    def summary(self) -> str:
        """Generates architectural summary report."""
        param_counts = count_parameters(self)
        lines = [
            "=" * 60,
            f"  SLMForCausalLM Architecture Summary",
            "=" * 60,
            f"  Vocab Size:          {self.config.vocab_size}",
            f"  Embedding Dim (d):   {self.config.d_model}",
            f"  Attention Heads (h): {self.config.n_heads}",
            f"  Head Dim:            {self.config.head_dim}",
            f"  Layers (N):          {self.config.n_layers}",
            f"  FFN Dim (d_ff):      {self.config.d_ff}",
            f"  Max Context Len:     {self.config.max_seq_len}",
            f"  Norm Type:           {self.config.norm_type}",
            f"  Activation:          {self.config.activation}",
            f"  Positional Encoding: {self.config.pos_encoding_type}",
            f"  Weight Tying:        {self.config.tie_word_embeddings}",
            "-" * 60,
            f"  Total Parameters:    {format_number(param_counts['total'])} ({param_counts['total']:,})",
            f"  Trainable Params:    {format_number(param_counts['trainable'])} ({param_counts['trainable']:,})",
            "=" * 60,
        ]
        return "\n".join(lines)
