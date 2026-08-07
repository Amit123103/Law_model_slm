"""
Checkpoint Manager for saving, resuming, and managing model and training states.
"""

import glob
import os
import random
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from slm.config.model_config import ModelConfig
from slm.config.train_config import TrainConfig
from slm.tokenizer.bpe import BPETokenizer
from slm.utils.logger import get_logger

logger = get_logger("slm.checkpoint")


class CheckpointManager:
    """
    Handles state serialization, checkpoint rotation, and full training resumption.
    """

    def __init__(self, output_dir: str = "checkpoints", max_to_keep: int = 5) -> None:
        """
        Initializes CheckpointManager.

        Args:
            output_dir: Directory path to store saved checkpoints.
            max_to_keep: Maximum number of recent checkpoints to retain before pruning.
        """
        self.output_dir = output_dir
        self.max_to_keep = max_to_keep
        os.makedirs(output_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        step: int,
        epoch: int,
        loss: float,
        model_config: ModelConfig,
        train_config: TrainConfig,
        tokenizer: Optional[BPETokenizer] = None,
        is_best: bool = False
    ) -> str:
        """
        Saves a complete training checkpoint file (.pt) and updates tokenizer.

        Returns:
            Saved checkpoint file path string.
        """
        ckpt_name = f"checkpoint_step_{step:07d}.pt"
        ckpt_path = os.path.join(self.output_dir, ckpt_name)

        checkpoint_state = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": model_config.to_dict(),
            "train_config": train_config.to_dict(),
            "random_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        }

        torch.save(checkpoint_state, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path} (step={step}, loss={loss:.4f})")

        # Save tokenizer if provided
        if tokenizer is not None:
            tok_dir = os.path.join(self.output_dir, "tokenizer")
            tokenizer.save(tok_dir)

        # Save best model copy if flagged
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint_state, best_path)
            logger.info(f"Updated best model checkpoint: {best_path}")

        # Rotate old checkpoints
        self._rotate_checkpoints()

        return ckpt_path

    def _rotate_checkpoints(self) -> None:
        """Prunes older checkpoint files exceeding max_to_keep limit."""
        pattern = os.path.join(self.output_dir, "checkpoint_step_*.pt")
        ckpt_files = sorted(glob.glob(pattern))

        if len(ckpt_files) > self.max_to_keep:
            to_remove = ckpt_files[:-self.max_to_keep]
            for file_path in to_remove:
                try:
                    os.remove(file_path)
                    logger.info(f"Pruned old checkpoint: {file_path}")
                except OSError as e:
                    logger.warning(f"Error removing old checkpoint {file_path}: {e}")

    def load_latest_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: torch.device = torch.device("cpu")
    ) -> Optional[Dict[str, Any]]:
        """
        Loads state from the most recent step checkpoint file in output_dir.

        Returns:
            Dictionary of loaded checkpoint metadata (step, epoch, loss) or None.
        """
        pattern = os.path.join(self.output_dir, "checkpoint_step_*.pt")
        ckpt_files = sorted(glob.glob(pattern))

        if not ckpt_files:
            logger.warning(f"No checkpoints found matching pattern in {self.output_dir}")
            return None

        latest_path = ckpt_files[-1]
        return self.load_checkpoint(latest_path, model, optimizer, scheduler, device)

    def load_checkpoint(
        self,
        ckpt_path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, Any]:
        """
        Loads model parameters, optimizer, and scheduler states from specified checkpoint file.
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Successfully loaded model weights from {ckpt_path}")

        if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Restored optimizer state.")

        if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Restored scheduler state.")

        # Restore random generator states if available
        if "random_state" in checkpoint:
            rnd = checkpoint["random_state"]
            if "python" in rnd:
                random.setstate(rnd["python"])
            if "numpy" in rnd:
                np.random.set_state(rnd["numpy"])
            if "torch_cpu" in rnd:
                torch.set_rng_state(rnd["torch_cpu"])
            if "torch_cuda" in rnd and rnd["torch_cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rnd["torch_cuda"])

        return {
            "step": checkpoint.get("step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "loss": checkpoint.get("loss", float("inf")),
            "model_config": checkpoint.get("model_config"),
            "train_config": checkpoint.get("train_config"),
        }
