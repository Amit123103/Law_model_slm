"""
Unit tests for QualityChecker.
"""

import os
import tempfile
import torch
import torch.nn as nn
import pytest
from slm.config.model_config import ModelConfig
from slm.model.transformer_lm import SLMForCausalLM
from slm.utils.quality import QualityChecker


def test_quality_checker_tensor_and_weights():
    # 1. Healthy tensor test
    t = torch.tensor([1.0, 2.0, 3.0])
    assert QualityChecker.check_tensor_sanities(t)

    # 2. NaN tensor test
    nan_t = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(ValueError, match="NaN detected"):
        QualityChecker.check_tensor_sanities(nan_t)

    # 3. Model weight check
    config = ModelConfig(vocab_size=100, d_model=32, n_heads=2, n_layers=1)
    model = SLMForCausalLM(config)
    health = QualityChecker.check_model_weights(model)
    assert health["is_healthy"] is True

    # 4. Gradient norm test
    x = torch.randint(0, 100, (2, 10))
    logits, loss = model(x, target_ids=x)
    loss.backward()
    grad_norm = QualityChecker.check_gradient_norms(model)
    assert grad_norm >= 0.0


def test_quality_checker_checkpoint_and_loss():
    # Loss health test
    assert QualityChecker.check_loss_health(2.45)
    with pytest.raises(ValueError, match="non-finite"):
        QualityChecker.check_loss_health(float("nan"))

    # Checkpoint verification test
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "valid_ckpt.pt")
        torch.save({"model_state_dict": {}, "step": 10}, ckpt_path)
        assert QualityChecker.verify_checkpoint(ckpt_path)
