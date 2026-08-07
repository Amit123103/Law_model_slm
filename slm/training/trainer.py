"""
Industrial Trainer loop supporting Teacher Forcing, Gradient Accumulation, AMP, and Checkpointing.
"""

import csv
import os
import time
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from slm.config.model_config import ModelConfig
from slm.config.train_config import TrainConfig
from slm.model.transformer_lm import SLMForCausalLM
from slm.optimizer.adamw import CustomAdamW
from slm.optimizer.lion import CustomLion
from slm.scheduler.schedulers import build_scheduler
from slm.checkpoint.manager import CheckpointManager
from slm.tokenizer.bpe import BPETokenizer
from slm.utils.logger import get_logger
from slm.utils.utils import get_device, get_memory_stats, format_number

logger = get_logger("slm.training")


class Trainer:
    """
    Complete Training engine for Small Language Model architecture.
    """

    def __init__(
        self,
        model: SLMForCausalLM,
        train_config: TrainConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        tokenizer: Optional[BPETokenizer] = None
    ) -> None:
        """
        Initializes Trainer engine with model, configuration, and data loaders.
        """
        self.model = model
        self.config = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer

        # Resolve execution device
        self.device = get_device(train_config.device)
        self.model.to(self.device)

        # Build Optimizer
        if train_config.optimizer_name == "adamw":
            self.optimizer = CustomAdamW(
                self.model.parameters(),
                lr=train_config.learning_rate,
                betas=(train_config.beta1, train_config.beta2),
                eps=train_config.adam_eps,
                weight_decay=train_config.weight_decay
            )
        elif train_config.optimizer_name == "lion":
            self.optimizer = CustomLion(
                self.model.parameters(),
                lr=train_config.learning_rate,
                betas=(train_config.beta1, train_config.beta2),
                weight_decay=train_config.weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=train_config.learning_rate,
                momentum=0.9
            )

        # Build Learning Rate Scheduler
        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_name=train_config.scheduler_name,
            warmup_steps=train_config.warmup_steps,
            max_steps=train_config.max_steps,
            min_lr=train_config.min_learning_rate
        )

        # Automatic Mixed Precision GradScaler
        self.use_amp = (train_config.mixed_precision in ("fp16", "bf16") and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=(train_config.mixed_precision == "fp16" and self.use_amp))

        # Checkpoint Manager
        self.ckpt_manager = CheckpointManager(output_dir=train_config.output_dir)

        # CSV log file setup
        os.makedirs(train_config.output_dir, exist_ok=True)
        self.log_file_path = os.path.join(train_config.output_dir, "train_log.csv")
        self._init_csv_log()

        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def _init_csv_log(self) -> None:
        """Initializes CSV log headers."""
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "epoch", "train_loss", "val_loss", "learning_rate", "grad_norm", "step_time_ms"])

    def _log_csv(self, step: int, epoch: int, train_loss: float, val_loss: Optional[float], lr: float, grad_norm: float, step_time_ms: float) -> None:
        """Appends metrics row to CSV log."""
        with open(self.log_file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([step, epoch, round(train_loss, 4), round(val_loss, 4) if val_loss is not None else "", f"{lr:.6e}", round(grad_norm, 4), round(step_time_ms, 2)])

    def evaluate(self) -> float:
        """
        Runs validation loop calculating average cross-entropy loss.

        Returns:
            Validation loss float score.
        """
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for input_ids, target_ids in self.val_loader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    _, loss = self.model(input_ids, targets=target_ids)

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        self.model.train()
        return avg_loss

    def train(self) -> Dict[str, Any]:
        """
        Executes full training loop across steps/epochs.

        Returns:
            Dictionary containing final training history.
        """
        logger.info(f"Starting training on device={self.device} (AMP={self.use_amp})...")
        logger.info(self.model.summary())

        self.model.train()
        start_time = time.time()
        running_loss = 0.0
        step_in_accum = 0

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch + 1

            for input_ids, target_ids in self.train_loader:
                step_start = time.perf_counter()
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                # Forward pass under autocast
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    _, loss = self.model(input_ids, targets=target_ids)
                    scaled_loss = loss / self.config.grad_accum_steps

                # Backward pass
                self.scaler.scale(scaled_loss).backward()
                running_loss += loss.item()
                step_in_accum += 1

                # Optimizer Step upon gradient accumulation threshold
                if step_in_accum % self.config.grad_accum_steps == 0:
                    # Unscale gradients for clipping
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm).item()

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    self.scheduler.step()
                    self.global_step += 1

                    step_end = time.perf_counter()
                    step_time_ms = (step_end - step_start) * 1000.0

                    # Logging interval
                    if self.global_step % self.config.log_interval_steps == 0:
                        avg_train_loss = running_loss / self.config.log_interval_steps
                        running_loss = 0.0
                        curr_lr = self.scheduler.get_last_lr()[0]

                        mem_stats = get_memory_stats(self.device)
                        logger.info(
                            f"[Epoch {self.current_epoch}/{self.config.epochs}] Step {self.global_step}/{self.config.max_steps} | "
                            f"Loss: {avg_train_loss:.4f} | LR: {curr_lr:.2e} | GradNorm: {grad_norm:.2f} | "
                            f"VRAM: {mem_stats['allocated_mb']}MB"
                        )
                        self._log_csv(self.global_step, self.current_epoch, avg_train_loss, None, curr_lr, grad_norm, step_time_ms)

                    # Evaluation interval
                    if self.val_loader is not None and self.global_step % self.config.eval_interval_steps == 0:
                        val_loss = self.evaluate()
                        logger.info(f"=== Validation at Step {self.global_step}: Loss = {val_loss:.4f} ===")
                        is_best = val_loss < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_loss

                        # Save checkpoint
                        self.ckpt_manager.save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            step=self.global_step,
                            epoch=self.current_epoch,
                            loss=val_loss,
                            model_config=self.model.config,
                            train_config=self.config,
                            tokenizer=self.tokenizer,
                            is_best=is_best
                        )

                    # Save regular step checkpoint
                    if self.global_step % self.config.save_interval_steps == 0:
                        self.ckpt_manager.save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            step=self.global_step,
                            epoch=self.current_epoch,
                            loss=running_loss / max(1, step_in_accum),
                            model_config=self.model.config,
                            train_config=self.config,
                            tokenizer=self.tokenizer
                        )

                    if self.global_step >= self.config.max_steps:
                        logger.info(f"Reached max_steps ({self.config.max_steps}). Training finished.")
                        break

            if self.global_step >= self.config.max_steps:
                break

        total_training_time = time.time() - start_time
        logger.info(f"Training finished in {total_training_time:.2f} seconds.")

        # Final checkpoint save
        final_ckpt = self.ckpt_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            epoch=self.current_epoch,
            loss=running_loss / max(1, step_in_accum),
            model_config=self.model.config,
            train_config=self.config,
            tokenizer=self.tokenizer
        )

        return {
            "global_step": self.global_step,
            "final_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "total_time_sec": total_training_time,
            "final_checkpoint": final_ckpt
        }
