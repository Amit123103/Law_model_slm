"""
Lion (EvoLved Sign Momentum) Optimizer implemented from scratch.
"""

from typing import Tuple, Optional, Callable
import torch
from torch.optim import Optimizer


class CustomLion(Optimizer):
    """
    Lion Optimizer (EvoLved Sign Momentum, Chen et al., 2023).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0
    ) -> None:
        """
        Initializes CustomLion optimizer.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas parameters: {betas}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.
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
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # 1. Update sign update vector: update = sign(beta1 * exp_avg + (1 - beta1) * grad)
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign_()

                # 2. Apply decoupled weight decay
                if group["weight_decay"] > 0.0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                # 3. Update parameter weights
                p.add_(update, alpha=-group["lr"])

                # 4. Update momentum buffer
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
