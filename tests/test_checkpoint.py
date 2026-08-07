"""
Unit tests for CheckpointManager saving and loading functionality.
"""

import os
import torch
import pytest

from slm.config.model_config import ModelConfig
from slm.config.train_config import TrainConfig
from slm.model.transformer_lm import SLMForCausalLM
from slm.checkpoint.manager import CheckpointManager


def test_checkpoint_manager_save_and_load(tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    manager = CheckpointManager(output_dir=ckpt_dir)

    model_config = ModelConfig(vocab_size=100, d_model=32, n_heads=2, n_layers=1)
    train_config = TrainConfig(output_dir=ckpt_dir)
    model = SLMForCausalLM(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        step=10,
        epoch=1,
        loss=2.5,
        model_config=model_config,
        train_config=train_config,
        is_best=True
    )

    assert os.path.exists(ckpt_path)

    # Test loading into a new model instance
    new_model = SLMForCausalLM(model_config)
    loaded_meta = manager.load_checkpoint(ckpt_path, new_model)

    assert loaded_meta["step"] == 10
    assert loaded_meta["epoch"] == 1
    assert loaded_meta["loss"] == 2.5
    assert loaded_meta["model_config"]["vocab_size"] == 100

    # Verify weight restoration match
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)
