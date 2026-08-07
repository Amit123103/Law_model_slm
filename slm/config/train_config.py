"""
TrainConfig schema for training pipeline settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class TrainConfig:
    """
    Configuration dataclass for model training hyperparameters and execution flags.
    """
    batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    warmup_steps: int = 500
    max_steps: int = 10000
    epochs: int = 5
    grad_accum_steps: int = 2
    optimizer_name: str = "adamw"  # 'adamw', 'lion', or 'sgd'
    scheduler_name: str = "cosine"  # 'cosine', 'linear', or 'onecycle'
    mixed_precision: str = "fp16"  # 'fp16', 'bf16', or 'fp32'
    device: str = "auto"
    seed: int = 42
    
    # Checkpointing and Logging
    output_dir: str = "checkpoints"
    save_interval_steps: int = 1000
    eval_interval_steps: int = 500
    log_interval_steps: int = 50
    early_stopping_patience: int = 10

    # Dataset paths
    train_dataset_path: Optional[str] = None
    val_dataset_path: Optional[str] = None
    tokenizer_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts train config to dictionary."""
        return {
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "adam_eps": self.adam_eps,
            "grad_clip_norm": self.grad_clip_norm,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "epochs": self.epochs,
            "grad_accum_steps": self.grad_accum_steps,
            "optimizer_name": self.optimizer_name,
            "scheduler_name": self.scheduler_name,
            "mixed_precision": self.mixed_precision,
            "device": self.device,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "save_interval_steps": self.save_interval_steps,
            "eval_interval_steps": self.eval_interval_steps,
            "log_interval_steps": self.log_interval_steps,
            "early_stopping_patience": self.early_stopping_patience,
            "train_dataset_path": self.train_dataset_path,
            "val_dataset_path": self.val_dataset_path,
            "tokenizer_path": self.tokenizer_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        """Instantiates TrainConfig from dictionary."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
