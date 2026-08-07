"""
AdamW Optimizer with decoupled weight decay implemented from scratch.
"""

import math
from typing import List, Dict, Any, Tuple, Optional, Callable
import torch
from torch.optim import Optimizer


class CustomAdamW(Optimizer):
    """
    AdamW Optimizer with decoupled weight decay implemented from PyTorch Optimizer primitives.
    Ref: Loshchilov & Hutter, 2019 (Decoupled Weight Decay Regularization).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True
    ) -> None:
        """
        Initializes CustomAdamW optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            betas: Coefficients for computing running averages of gradient and its square.
            eps: Term added to denominator for numerical stability.
            weight_decay: Weight decay coefficient (decoupled).
            correct_bias: Whether to apply bias correction terms.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure: A closure function that re-evaluates the model and returns loss.

        Returns:
            Optional loss value.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CustomAdamW does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                step = state["step"]

                # 1. Update exponential moving average of gradient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 2. Update exponential moving average of squared gradient
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group["correct_bias"]:
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    step_size = group["lr"] * (math.sqrt(bias_correction2) / bias_correction1)
                else:
                    step_size = group["lr"]

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # 3. Apply decoupled weight decay
                if group["weight_decay"] > 0.0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                # 4. Update parameter weights
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
