"""
Unit tests for SLMForCausalLM model architecture.
"""

import torch
import pytest

from slm.config.model_config import ModelConfig
from slm.model.transformer_lm import SLMForCausalLM


def test_slm_model_forward_and_loss():
    config = ModelConfig(
        vocab_size=500,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=128,
        norm_type="rmsnorm",
        activation="swiglu",
        pos_encoding_type="rope"
    )

    model = SLMForCausalLM(config)
    batch, seq = 2, 16

    input_ids = torch.randint(0, config.vocab_size, (batch, seq))
    targets = torch.randint(0, config.vocab_size, (batch, seq))

    logits, loss = model(input_ids, targets=targets)

    assert logits.shape == (batch, seq, config.vocab_size)
    assert loss is not None
    assert isinstance(loss.item(), float)
    assert loss.item() > 0.0


def test_weight_tying():
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        n_heads=2,
        n_layers=1,
        tie_word_embeddings=True
    )
    model = SLMForCausalLM(config)
    assert torch.equal(model.lm_head.weight, model.token_embeddings.embedding.weight)
