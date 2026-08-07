"""
General utility functions for seed setting, device resolution, and parameter formatting.
"""

import os
import random
from typing import Dict, Any, Union
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets reproducible random seeds across Python random, NumPy, PyTorch CPU, and PyTorch CUDA.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device_arg: str = "auto") -> torch.device:
    """
    Resolves execution device based on hardware availability.

    Args:
        device_arg: Device choice ('auto', 'cuda', 'cpu', 'mps').

    Returns:
        torch.device instance.
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_arg)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Calculates total and trainable parameter count of a PyTorch module.

    Args:
        model: Target PyTorch module.

    Returns:
        Dictionary containing total, trainable, and non-trainable parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable
    }


def format_number(num: Union[int, float]) -> str:
    """
    Formats large numbers into human-readable metric prefixes (K, M, B).

    Args:
        num: Input numerical value.

    Returns:
        Formatted string (e.g. 1.23M).
    """
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    if num >= 1e6:
        return f"{num / 1e6:.2f}M"
    if num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return str(num)


def get_memory_stats(device: torch.device) -> Dict[str, Any]:
    """
    Retrieves CUDA GPU memory usage statistics if available.

    Args:
        device: Active torch device.

    Returns:
        Dictionary of allocated and reserved VRAM in MB.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        return {
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "max_allocated_mb": round(max_allocated, 2)
        }
    return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
