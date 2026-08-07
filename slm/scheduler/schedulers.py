"""
Learning Rate Schedulers: Cosine with Warmup, Linear with Warmup, and OneCycle.
"""

import math
from typing import List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineWithWarmupLR(_LRScheduler):
    """
    Cosine Annealing Learning Rate Scheduler with Linear Warmup.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ) -> None:
        """
        Initializes CosineWithWarmupLR.

        Args:
            optimizer: PyTorch optimizer instance.
            warmup_steps: Number of linear warmup steps.
            max_steps: Total maximum training steps.
            min_lr: Floor minimum learning rate.
            last_epoch: The index of last epoch/step.
        """
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps = max(warmup_steps + 1, max_steps)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup ratio from 0 to 1
            alpha = float(step) / float(self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        elif step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]
        else:
            # Cosine decay from max_lr down to min_lr
            progress = float(step - self.warmup_steps) / float(self.max_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]


class LinearWithWarmupLR(_LRScheduler):
    """
    Linear Learning Rate Scheduler with Linear Warmup and Linear Decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ) -> None:
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps = max(warmup_steps + 1, max_steps)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch

        if step < self.warmup_steps:
            alpha = float(step) / float(self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        elif step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]
        else:
            progress = float(step - self.warmup_steps) / float(self.max_steps - self.warmup_steps)
            linear_decay = 1.0 - progress
            return [self.min_lr + (base_lr - self.min_lr) * linear_decay for base_lr in self.base_lrs]


def build_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = "cosine",
    warmup_steps: int = 500,
    max_steps: int = 10000,
    min_lr: float = 1e-6
) -> _LRScheduler:
    """
    Factory function instantiating LR scheduler.
    """
    if scheduler_name == "cosine":
        return CosineWithWarmupLR(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr=min_lr
        )
    elif scheduler_name == "linear":
        return LinearWithWarmupLR(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr=min_lr
        )
    else:
        raise ValueError(f"Unsupported scheduler_name: {scheduler_name}")
