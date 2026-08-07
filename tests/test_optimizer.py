"""
Unit tests for CustomAdamW and CustomLion optimizers.
"""

import torch
import pytest

from slm.optimizer.adamw import CustomAdamW
from slm.optimizer.lion import CustomLion


def test_custom_adamw_step():
    weights = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    optimizer = CustomAdamW([weights], lr=0.1, weight_decay=0.01)

    loss = (weights ** 2).sum()
    loss.backward()

    optimizer.step()

    # Weights should decrease after step
    assert weights[0].item() < 1.0
    assert weights[1].item() < 2.0


def test_custom_lion_step():
    weights = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    optimizer = CustomLion([weights], lr=0.01, weight_decay=0.01)

    loss = (weights ** 2).sum()
    loss.backward()

    optimizer.step()

    assert weights[0].item() < 1.0
