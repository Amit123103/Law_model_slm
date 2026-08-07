"""
Model Configuration schema for Small Language Model architecture.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ModelConfig:
    """
    Configuration dataclass defining Small Language Model hyperparameters.
    """
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    max_seq_len: int = 1024
    norm_type: str = "rmsnorm"  # 'rmsnorm' or 'layernorm'
    activation: str = "swiglu"  # 'swiglu', 'gelu', or 'relu'
    pos_encoding_type: str = "rope"  # 'rope', 'sinusoidal', or 'learned'
    dropout: float = 0.1
    bias: bool = False
    tie_word_embeddings: bool = True
    rope_base: float = 10000.0
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        """
        Validates hyperparameter constraints.
        """
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.norm_type not in ("rmsnorm", "layernorm"):
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")
        if self.activation not in ("swiglu", "gelu", "relu"):
            raise ValueError(f"Unsupported activation: {self.activation}")
        if self.pos_encoding_type not in ("rope", "sinusoidal", "learned"):
            raise ValueError(f"Unsupported pos_encoding_type: {self.pos_encoding_type}")

    @property
    def head_dim(self) -> int:
        """Computes dimension per attention head."""
        return self.d_model // self.n_heads

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "norm_type": self.norm_type,
            "activation": self.activation,
            "pos_encoding_type": self.pos_encoding_type,
            "dropout": self.dropout,
            "bias": self.bias,
            "tie_word_embeddings": self.tie_word_embeddings,
            "rope_base": self.rope_base,
            "layer_norm_eps": self.layer_norm_eps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Instantiates ModelConfig from dictionary."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
