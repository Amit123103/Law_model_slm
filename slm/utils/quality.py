"""
Automated Quality Checker & Anomaly Detector for SLM training and inference.
Verifies model weights, gradient norms, loss stability, checkpoint integrity, and tokenizer consistency.
"""

import os
import math
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
from slm.tokenizer.bpe import BPETokenizer
from slm.utils.logger import get_logger

logger = get_logger("slm.utils.quality")


class QualityChecker:
    """Performs pre-flight, in-flight, and post-flight quality checks and anomaly detection."""

    @staticmethod
    def check_tensor_sanities(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """
        Verifies that a tensor contains no NaN or Inf values.
        Raises ValueError if anomalies are found.
        """
        if torch.isnan(tensor).any():
            raise ValueError(f"Quality Check Failed: NaN detected in {name}!")
        if torch.isinf(tensor).any():
            raise ValueError(f"Quality Check Failed: Inf detected in {name}!")
        return True

    @classmethod
    def check_model_weights(cls, model: nn.Module) -> Dict[str, Any]:
        """Inspects all model parameters for NaNs, Infs, and extreme values."""
        total_params = 0
        total_nan = 0
        total_inf = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                if torch.isnan(param).any():
                    total_nan += 1
                if torch.isinf(param).any():
                    total_inf += 1

        if total_nan > 0 or total_inf > 0:
            raise ValueError(f"Quality Check Failed: Model weights contained {total_nan} NaN tensors and {total_inf} Inf tensors!")

        return {
            "total_parameters": total_params,
            "nan_count": total_nan,
            "inf_count": total_inf,
            "is_healthy": True
        }

    @classmethod
    def check_gradient_norms(cls, model: nn.Module, max_norm_threshold: float = 100.0) -> float:
        """
        Computes total gradient norm across all model parameters.
        Raises ValueError if gradient explosion (norm > threshold) or NaN/Inf is detected.
        """
        total_sq_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                cls.check_tensor_sanities(p.grad, name="param_gradient")
                param_norm = p.grad.data.norm(2)
                total_sq_norm += param_norm.item() ** 2

        total_norm = math.sqrt(total_sq_norm)

        if math.isnan(total_norm) or math.isinf(total_norm):
            raise ValueError("Quality Check Failed: Gradient norm is NaN or Inf!")
        if total_norm > max_norm_threshold:
            logger.warning(f"High gradient norm detected: {total_norm:.4f} > threshold {max_norm_threshold}")

        return total_norm

    @staticmethod
    def check_loss_health(loss_val: float, history: Optional[List[float]] = None) -> bool:
        """Checks if current loss value is finite and not exploding relative to history."""
        if math.isnan(loss_val) or math.isinf(loss_val):
            raise ValueError(f"Quality Check Failed: Training loss is non-finite ({loss_val})!")

        if history and len(history) >= 5:
            avg_recent = sum(history[-5:]) / 5.0
            if loss_val > avg_recent * 4.0 and loss_val > 10.0:
                logger.warning(f"Loss Spike Detected: Current loss {loss_val:.4f} is 4x recent average {avg_recent:.4f}")

        return True

    @staticmethod
    def verify_tokenizer_consistency(tokenizer: BPETokenizer, test_sample: str = "LawSLM test sentence.") -> bool:
        """Verifies tokenizer encode/decode consistency."""
        ids = tokenizer.encode(test_sample, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        if decoded.strip() != test_sample.strip():
            logger.warning(f"Tokenizer mismatch! Input: {test_sample} != Decoded: {decoded}")
            return False
        return True

    @staticmethod
    def verify_checkpoint(filepath: str) -> bool:
        """Verifies that a checkpoint file exists, is non-empty, and loadable by PyTorch."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        if os.path.getsize(filepath) == 0:
            raise ValueError(f"Checkpoint file is empty: {filepath}")

        try:
            checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
                raise ValueError(f"Invalid checkpoint format in {filepath}: missing 'model_state_dict'")
        except Exception as e:
            raise ValueError(f"Corrupted checkpoint file {filepath}: {str(e)}")

        return True
